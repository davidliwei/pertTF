import time
import torch
import random
import warnings
from pathlib import Path
import copy
import numpy as np

from typing import Dict, Mapping, Optional, Tuple, Any, Union
from typing import List, Tuple

from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from anndata import AnnData
import scanpy as sc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
#multiprocessing.set_start_method('spawn', force=True)
import wandb
from scipy.sparse import issparse

import scgpt as scg
from scgpt.loss import (
    masked_relative_error,
)
from perttf.utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.model import TransformerModel, AdversarialDiscriminator

import matplotlib.pyplot as plt

from perttf.model.train_data_gen import prepare_data, prepare_dataloader
from perttf.utils.set_optimizer import create_optimizer_dict
from perttf.custom_loss import cce_loss, criterion_neg_log_bernoulli, masked_mse_loss
from perttf.utils.plot import process_and_log_umaps
from perttf.utils.misc import init_plot_worker
def train(model: nn.Module,
          loader: DataLoader,
          config,
          vocab,
          optim_dict: Dict,
          epoch = 0,
          logger = scg.logger,
          device = None) -> None:
    """
    Train the model for one epoch.
    """
    criterion = masked_mse_loss 
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    criterion_ps = nn.MSELoss() # this is the loss for predicting PS scores
    #criterion_ps = nn.CrossEntropyLoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_mse_next, total_gepc_next = 0.0, 0.0
    total_error, total_error_next = 0.0, 0.0
    total_dab, total_adv_E, total_adv_D = 0.0, 0.0, 0.0
    total_cls, total_pert, total_ps, total_ps_next = 0.0, 0.0, 0.0, 0.0
    log_interval = config.log_interval
    start_time = time.time()


    scaler=optim_dict["scaler"]
    discriminator=optim_dict["discriminator"]
    optimizer=optim_dict["optimizer"]
    scheduler=optim_dict["scheduler"]
    optimizer_dab=optim_dict["optimizer_dab"]
    scheduler_dab=optim_dict["scheduler_dab"]
    optimizer_E=optim_dict["optimizer_E"]
    scheduler_E=optim_dict["scheduler_E"]
    optimizer_D=optim_dict["optimizer_D"]
    scheduler_D=optim_dict["scheduler_D"]

    # check ps_next_weight. The ps_next prediction is used for predicting the lochness score for a new gene from pert_next label
    if hasattr(config, "pred_lochness_next"):
        has_lochness_next_pred = True
        ps_next_training_weight = config.pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = config.ps_weight * config.next_weight

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        target_values_next = batch_data["target_values_next"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device) #added
        perturbation_labels = batch_data["perturbation_labels"].to(device) #added

        celltype_labels_next = batch_data["celltype_labels_next"].to(device) #added
        perturbation_labels_next = batch_data["perturbation_labels_next"].to(device) #added

        mvc_src = None if config.get('mvc_masked_train', True) else batch_data['full_gene_ids'].to(device)
        if config.ps_weight >0:
            ps_score = batch_data["ps"].to(device)
            ps_score_next = batch_data["ps_next"].to(device) #

        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            #import pdb; pdb.set_trace()

            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_label else None, # if config.DSBN else None,
                pert_labels = perturbation_labels if config.perturbation_input else None,
                pert_labels_next = perturbation_labels_next if (config.next_weight >0 or has_lochness_next_pred )  else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.cell_type_classifier,
                CCE = config.CCE,
                PERTPRED = config.perturbation_classifier_weight > 0,
                PSPRED = config.ps_weight >0,
                mvc_src = mvc_src
            )

            masked_positions = input_values.eq(config.mask_value)  # the postions to predict
            loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            loss = config.this_weight * loss_mse
            metrics_to_log = {"train/mse": loss_mse.item()}

            if config.CCE and len(output_dict["contrastive_dict"]) > 0:
                cce_mode = config.get('cce_mode', 'cell+geno')
                logit_norm = config.get('logit_norm', False)
                if cce_mode == 'cell_geno': ## supervised contrastive loss plus a custom contrastive loss
                    cce_weight = max(config.perturbation_classifier_weight*config.cell_type_classifier_weight, 1)
                    input_labels = celltype_labels*1000+perturbation_labels # x1000 make labels unique combination of celltype and genotype
                    pert_labels = celltype_labels_next*1000+perturbation_labels_next
                    loss_cce = cce_loss(output_dict["contrastive_dict"], input_labels, pert_labels, logit_norm=logit_norm)
                    metrics_to_log["train/cce"] = loss_cce.item()
                    loss += loss_cce * cce_weight

                if cce_mode == 'celltype' or cce_mode == 'cell+geno':
                    loss_cce_celltype = cce_loss(output_dict["contrastive_dict"], celltype_labels, celltype_labels_next, logit_norm=logit_norm)
                    metrics_to_log["train/cce_celltype"] = loss_cce_celltype.item()
                    loss += loss_cce_celltype * max(config.cell_type_classifier_weight, 1)

                if cce_mode == 'genotype' or cce_mode == 'cell+geno':
                    loss_cce_genotype = cce_loss(output_dict["contrastive_dict"], perturbation_labels, perturbation_labels_next, logit_norm=logit_norm)
                    metrics_to_log["train/cce_genotype"] = loss_cce_genotype.item()
                    loss += loss_cce_genotype * max(config.perturbation_classifier_weight,1)
                
                
            # next value?
            loss_mse_next = criterion(
                output_dict["mlm_output"],
                target_values_next, masked_positions
            )
            # disable now
            #loss = loss + config.next_weight * loss_mse_next
            metrics_to_log.update({"train/mse_next": loss_mse_next.item()})

            if config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + config.this_weight *loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                # added
                loss_zero_log_prob_next = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values_next, masked_positions
                )
                #loss = loss + config.next_weight *loss_zero_log_prob_next
                metrics_to_log.update({"train/nzlp_next": loss_zero_log_prob_next.item()})
            if config.GEPC:
                mvc_target_values = target_values if config.get('mvc_masked_train', True) else batch_data["full_expr"].to(device)
                mvc_target_values_next = target_values_next if config.get('mvc_masked_train', True) else batch_data["full_expr_next"].to(device)
                mvc_masked_positions = masked_positions if config.get('mvc_masked_train', True) else None
                loss_gepc = criterion(
                    output_dict["mvc_output"], mvc_target_values, mvc_masked_positions
                )
                loss = loss + config.this_weight *loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
                # added
                loss_gepc_next = criterion(
                    output_dict["mvc_output_next"], mvc_target_values_next, mvc_masked_positions
                )
                loss = loss + config.next_weight * loss_gepc_next
                metrics_to_log.update({"train/mvc_next": loss_gepc_next.item()})
                
                if config.explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], mvc_target_values, mvc_masked_positions
                    )
                    loss = loss + config.this_weight *loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                    # added
                    loss_gepc_zero_log_prob_next = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs_next"], mvc_target_values_next, mvc_masked_positions
                    )
                    loss = loss + config.next_weight * loss_gepc_zero_log_prob_next
                    metrics_to_log.update(
                        {"train/mvc_nzlp_next": loss_gepc_zero_log_prob_next.item()}
                    )
            if config.cell_type_classifier:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss + config.cell_type_classifier_weight * loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})
                # add for next cls prediction
                loss_cls_next = criterion_cls(output_dict["cls_output_next"], celltype_labels_next)
                loss = loss + config.cell_type_classifier_weight * config.next_weight *  loss_cls_next
                metrics_to_log.update({"train/cls_next": loss_cls_next.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

            if config.perturbation_classifier_weight > 0:
                loss_pert = criterion_pert(output_dict["pert_output"], perturbation_labels)
                loss = loss + config.perturbation_classifier_weight * loss_pert
                metrics_to_log.update({"train/pert": loss_pert.item()})
                # add for next pert prediction
                loss_pert_next = criterion_pert(output_dict["pert_output_next"], perturbation_labels_next)
                loss = loss + config.perturbation_classifier_weight * config.next_weight * loss_pert_next
                metrics_to_log.update({"train/pert_next": loss_pert_next.item()})

            if config.ps_weight >0:
                loss_ps = criterion_ps(output_dict["ps_output"], ps_score)
                #import pdb; pdb.set_trace()
                #print(f"loss_ps: {loss_ps}")
                loss = loss + config.ps_weight * loss_ps
                metrics_to_log.update({"train/ps": loss_ps.item()})
                loss_ps_next = criterion_ps(output_dict["ps_output_next"], ps_score_next)
                loss = loss + ps_next_training_weight * loss_ps_next 
                metrics_to_log.update({"train/ps_next": loss_ps_next.item()})

            if config.ecs_thres > 0:
                loss_ecs = config.ecs_weight  * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if config.dab_weight > 0:
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

        model.zero_grad()
        #print(f"loss: {loss}")
        #import pdb; pdb.set_trace()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0 and logger is not None:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        if config.ADV:
            # rerun the model for adversarial training
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.use_batch_label else None, # if config.DSBN else None,
                pert_labels = perturbation_labels if config.perturbation_input else None,
                pert_labels_next = perturbation_labels_next if (config.next_weight >0 or has_lochness_next_pred )  else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.cell_type_classifier,
                #CCE=config.CCE,
                PERTPRED = config.perturbation_classifier_weight > 0,
                PSPRED = config.ps_weight >0,
                #do_sample=config.do_sample_in_train,
                #generative_training=False
            )

            # TRAINING DISCRIMINATOR
            loss_adv_D = config.adv_weight * criterion_adv(
                discriminator(output_dict["cell_emb"].detach()), batch_labels
            )
            if epoch > config.adv_D_delay_epochs:
                discriminator.zero_grad()
                loss_adv_D.backward()
                optimizer_D.step()

            # TRAINING ENCODER
            loss_adv_E = -1 * config.adv_weight * criterion_adv(
                discriminator(output_dict["cell_emb"]), batch_labels
            )
            # NOTE: the loss is negative here because we want to maximize
            # the cross_entropy_loss, in other words, disguise against the discriminator
            if epoch > config.adv_E_delay_epochs:
                model.zero_grad()
                discriminator.zero_grad()
                loss_adv_E.backward()
                optimizer_E.step()

        wandb.log(metrics_to_log)

        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
            mre_next = masked_relative_error(
                output_dict["mlm_output"], target_values_next, masked_positions
            )

        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_mse_next += loss_mse_next.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_gepc_next += loss_gepc_next.item() if config.GEPC else 0.0
        total_error += mre.item()
        total_error_next += mre_next.item()

        total_dab += loss_dab.item() if config.dab_weight >0 else 0.0
        total_adv_E += loss_adv_E.item() if config.ADV else 0.0
        total_adv_D += loss_adv_D.item() if config.ADV else 0.0

        total_cls += loss_cls.item() if config.cell_type_classifier else 0.0
        total_pert += loss_pert.item() if config.perturbation_classifier_weight > 0 else 0.0
        total_ps += loss_ps.item() if config.ps_weight >0 else 0.0
        total_ps_next += loss_ps_next.item() if ps_next_training_weight >0 else 0.0

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_mse_next = total_mse_next / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_gepc_next = total_gepc_next / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            cur_error_next = total_error_next / log_interval
            cur_dab = total_dab / log_interval if config.dab_weight >0 else 0.0
            cur_adv_E = total_adv_E / log_interval if config.ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if config.ADV else 0.0
            cur_cls = total_cls / log_interval if config.cell_type_classifier else 0.0
            cur_pert = total_pert / log_interval if config.perturbation_classifier_weight > 0 else 0.0
            cur_ps = total_ps / log_interval if config.ps_weight >0 else 0.0
            cur_ps_next = total_ps_next / log_interval if ps_next_training_weight >0 else 0.0
            # ppl = math.exp(cur_loss)
            if logger is not None:
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.8f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    f"mse_next {cur_mse_next:5.2f} | mre_next {cur_error_next:5.2f} |"
                    f"cls {cur_cls:5.2f} | pert {cur_pert:5.2f} | ps {cur_ps:5.2f} | ps_next {cur_ps_next:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                    + (f"gepc_next {cur_gepc_next:5.2f} |" if config.GEPC else "")
                    + (f"dab {cur_dab:5.2f} |" if config.dab_weight >0 else "")
                    + (f"adv_E {cur_adv_E:5.2f} |" if config.ADV else "")
                    + (f"adv_D {cur_adv_D:5.2f} |" if config.ADV else "")
                )
            total_loss = 0
            total_mse = 0
            total_mse_next = 0
            total_gepc = 0
            total_gepc_next = 0
            total_error = 0
            total_error_next = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_cls = 0
            total_pert  = 0
            total_ps  = 0
            total_ps_next = 0

            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mse_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/cls", summary="min", step_metric="epoch")
    wandb.define_metric("valid/pert", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mvc", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mvc_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/ps", summary="min", step_metric="epoch")
    wandb.define_metric("valid/ps_next", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module,
            loader: DataLoader,
            config,
            vocab,
            epoch = 0,
            device = None) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    criterion_ps = nn.MSELoss() # this is the loss for predicting PS scores

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_loss = 0.0
    total_loss_next = 0.0
    total_error = 0.0
    total_error_next = 0.0
    total_dab = 0.0
    total_cls = 0.0
    total_pert = 0.0
    total_ps = 0.0
    total_ps_next = 0.0
    total_num = 0
    total_mvc = 0
    total_mvc_next = 0

    if hasattr(config, "pred_lochness_next"):
        has_lochness_next_pred = True
        ps_next_training_weight = config.pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = config.ps_weight * config.next_weight

    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            target_values_next = batch_data["target_values_next"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device) #added
            perturbation_labels = batch_data["perturbation_labels"].to(device) #added
            perturbation_labels_next = batch_data["perturbation_labels_next"].to(device) #added
            ps_score = batch_data["ps"].to(device) #added
            ps_score_next = batch_data["ps_next"].to(device) #added
            
            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
            mvc_src = None if config.get('mvc_masked_train', True) else batch_data['full_gene_ids'].to(device)
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.use_batch_label else None, # if config.DSBN else None,
                    pert_labels = perturbation_labels if config.perturbation_input else None,
                    pert_labels_next = perturbation_labels_next if (config.next_weight >0 or has_lochness_next_pred )  else None,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                    CLS=config.cell_type_classifier,
                    PERTPRED = config.perturbation_classifier_weight > 0,
                    PSPRED = config.ps_weight>0,
                    mvc_src = mvc_src
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(config.mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                #import pdb; pdb.set_trace()
                #print(f"total mask:{sum(masked_positions)}")
                #print(f"output_values_shape: {output_values.shape}")
                #print(output_values * masked_positions )
                #print(f"target_values_shape: {target_values.shape}")
                #print(target_values * masked_positions )

                loss_mse_next = criterion(output_values, target_values_next, masked_positions)
                #print(f"target_values_next_shape: {target_values_next.shape}")
                #print(target_values_next * masked_positions)
                if config.GEPC:
                    mvc_target_values = target_values if config.get('mvc_masked_train', True) else batch_data["full_expr"].to(device)
                    mvc_target_values_next = target_values_next if config.get('mvc_masked_train', True) else batch_data["full_expr_next"].to(device)
                    mvc_masked_positions = masked_positions if config.get('mvc_masked_train', True) else None

                    loss_gepc = criterion(
                        output_dict["mvc_output"], mvc_target_values, mvc_masked_positions
                    )
                    # added
                    loss_gepc_next = criterion(
                        output_dict["mvc_output_next"], mvc_target_values_next, mvc_masked_positions
                    )
                    

                if config.dab_weight > 0:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                if config.cell_type_classifier: #added
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                    # = loss + loss_cls
                if config.perturbation_classifier_weight > 0:
                    loss_pert = criterion_pert(output_dict["pert_output"], perturbation_labels)
                    # = loss + loss_pert

                if config.ps_weight > 0:
                    loss_ps = criterion_ps(output_dict["ps_output"], ps_score)
                    # = loss + loss_pert
                if ps_next_training_weight >0:
                    loss_ps_next = criterion_ps(output_dict["ps_output_next"], ps_score_next)

            total_loss += loss.item() * len(input_gene_ids)
            total_loss_next += loss_mse_next.item() * len(input_gene_ids)
            total_mvc += loss_gepc.item() * len(input_gene_ids)
            total_mvc_next += loss_gepc_next.item() * len(input_gene_ids)
            total_error += masked_relative_error(output_values, target_values, masked_positions).item() * len(input_gene_ids)
            total_error_next += masked_relative_error(output_values, target_values_next, masked_positions).item() * len(input_gene_ids)
            if config.dab_weight > 0:
                total_dab += loss_dab.item() * len(input_gene_ids)
            if config.cell_type_classifier: #added
                total_cls += loss_cls.item() * len(input_gene_ids)
            if config.perturbation_classifier_weight > 0:
                total_pert += loss_pert.item() * len(input_gene_ids)
            if config.ps_weight > 0:
                total_ps += loss_ps.item() * len(input_gene_ids)
            if ps_next_training_weight >0:
                total_ps_next += loss_ps_next.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mse_next": total_loss_next / total_num,
            "valid/mvc": total_mvc / total_num,
            "valid/mvc_next": total_mvc_next / total_num,
            "valid/mre": total_error / total_num,
            "valid/mre_next": total_error_next / total_num,
            "valid/dab": total_dab / total_num,
            "valid/cls": total_cls / total_num,
            "valid/pert": total_pert / total_num,
            "valid/ps": total_ps / total_num,
            "valid/ps_next": total_ps_next / total_num,
            "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)/ total_num,
            "epoch": epoch,
        },
    )

    return (
        total_loss / total_num, 
        total_loss_next / total_num, 
        total_mvc / total_num,
        total_mvc_next / total_num,
        total_error / total_num, 
        total_error_next / total_num, 
        total_dab / total_num, 
        total_cls / total_num, 
        total_pert / total_num,
        total_ps / total_num, 
        total_ps_next / total_num
    )

def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids: List[str],
    train_data_dict: Dict,
    config,
    include_types: List[str] = ["cls","pert"],
    input_layer_key = "X_binned",
    next_layer_key = "X_binned_next",
    logger = scg.logger,
    epoch = 0,
    eval_key = "", # titles for evaluation
    make_plots = True,
    predict_expr = False,
    mvc_full_expr = False
) -> Optional[Dict]: # Returns a dictionary containing the AnnData object
    """
    Evaluate the model on test data and return an AnnData object with embeddings.
    Plotting and UMAP are offloaded to a separate process.
    """
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    cell_type_to_index = train_data_dict["cell_type_to_index"]
    genotype_to_index = train_data_dict["genotype_to_index"]
    vocab=train_data_dict['vocab']

    adata_t = adata_t.copy() # make a copy
    adata_t = adata_t[adata_t.obs['celltype'].isin(cell_type_to_index)]

    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )
    if next_layer_key in adata_t.layers:
        all_counts_next = (
            adata_t.layers[next_layer_key].toarray()
            if issparse(adata_t.layers[next_layer_key])
            else adata_t.layers[next_layer_key]
        )
    else:
        all_counts_next = None

    if "celltype" in adata_t.obs.columns and config.cell_type_classifier:
        celltypes_labels = adata_t.obs["celltype"].tolist()  # make sure count from 0
        celltypes_labels = np.array(celltypes_labels)
        celltypes_labels = np.array([cell_type_to_index[cell_type] for cell_type in celltypes_labels])
    else:
        #celltypes_labels = np.array(random.choices( [0,1], k=adata_t.shape[0]))
        celltypes_labels = None



    if "genotype" in adata_t.obs.columns and (config.perturbation_classifier_weight > 0 or config.perturbation_input):
        perturbation_labels = adata_t.obs["genotype"].tolist()  # make sure count from 0
        perturbation_labels = np.array(perturbation_labels)
        perturbation_labels = np.array([genotype_to_index[perturbation_type] for perturbation_type in perturbation_labels])
    else:
        #perturbation_labels = np.array(random.choices( [0,1], k=adata_t.shape[0]))
        perturbation_labels = None

    perturbation_indexes = perturbation_labels


    # evaluate the next prediction?
    next_cell_prediction = False
    perturbation_labels_next = None
    if config.next_cell_pred_type == "pert":
        if "genotype_next" in adata_t.obs.columns:
            # this is the pred_next  
            if config.perturbation_classifier_weight > 0:
                next_cell_prediction = True
        else:
            logger.warning('next cell pred is set to pert but the provided adata does not have genotype_next column')
            next_cell_prediction = False
        if next_cell_prediction:
            perturbation_labels_next = adata_t.obs["genotype_next"].tolist()  # make sure count from 0
    if config.next_cell_pred_type ==  'lochness':
        if hasattr(config, "pred_lochness_next") and config.pred_lochness_next >0:
            next_cell_prediction = True
        else:
            next_cell_prediction = False
        if next_cell_prediction:
            perturbation_labels_next = adata_t.obs["genotype_next"].tolist()  # make sure count from 0

    if next_cell_prediction:
        perturbation_labels_next = np.array(perturbation_labels_next)
        perturbation_labels_next = np.array([genotype_to_index[perturbation_type] for perturbation_type in perturbation_labels_next])
    else:
        #perturbation_labels_next = random.choices( [0,1], k=adata_t.shape[0])
        perturbation_labels_next = None
    perturbation_indexes_next = perturbation_labels_next

    if "batch_id" in adata_t.obs.columns: # and config.DSBN:
        batch_ids = adata_t.obs["batch_id"].tolist()
    else:
        batch_ids=random.choices( [0,1], k=adata_t.shape[0])

    batch_ids = np.array(batch_ids)

    if mvc_full_expr: # if we want to get full expression from mvc decoder
        cls_gene_ids = np.insert(gene_ids, 0, vocab[config.cls_token]) # default should always be to insert a cls token at the front
        full_gene_ids = torch.stack([torch.from_numpy(cls_gene_ids).long() for i in range(adata_t.shape[0])], dim = 0)
    else:
        full_gene_ids = None
    # Evaluate cls cell embeddings
    if "cls" in include_types:
        sampling_mode = config.get('sampling_mode', 'simple')
        hvg_inds = None
        max_seq_len = config.max_seq_len
        if sampling_mode == 'expressed':
            max_seq_len = 10000 #10k genes should include all expressed genes for most 10X data 
        elif sampling_mode == 'hvg':
            hvg_col = config.get('hvg_col', 'highly_variable')
            assert hvg_col in adata_t.var.keys(), 'adata must have calculated HVGs or adata.var must have hvg_col'
            hvg_inds = (np.where(adata_t.var[hvg_col])[0], np.where(~adata_t.var[hvg_col])[0])
            max_seq_len = adata_t.var[hvg_col].sum()+config.get('non_hvg_size', 1000)

        tokenized_all, gene_idx_list= tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
            sampling_mode = config.get('sampling_mode', 'simple'),
            nonzero_prop = config.get('nonzero_prop', 0.7),
            fix_nonzero_prop =  config.get('fix_nonzero_prop', False),
            hvg_inds = hvg_inds, 
            non_hvg_size= config.get('non_hvg_size', 1000)
        )


        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        if logger is not None:
            logger.info(f"Evaluating cls cell embeddings using {all_gene_ids.shape} for each cell")

        if next_layer_key in adata_t.layers:
            tokenized_all_next, _ = tokenize_and_pad_batch(
                all_counts_next,
                gene_ids,
                max_len=max_seq_len,
                vocab=vocab,
                pad_token=config.pad_token,
                pad_value=config.pad_value,
                append_cls=True,  # append <cls> token at the beginning
                include_zero_gene=True,
                sample_indices=gene_idx_list,
                sampling_mode = config.get('sampling_mode', 'simple'),
                nonzero_prop = config.get('nonzero_prop', 0.7),
                fix_nonzero_prop =  config.get('fix_nonzero_prop', False),
                hvg_inds = hvg_inds, 
                non_hvg_size= config.get('non_hvg_size', 1000)
            )
            all_gene_ids_next, all_values_next = tokenized_all_next["genes"], tokenized_all_next["values"]

        src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            #cell_embeddings = model.encode_batch(all_gene_ids,all_values.float(),
            #    src_key_padding_mask=src_key_padding_mask,
            #    batch_size=config.batch_size,
            #    batch_labels=torch.from_numpy(batch_ids).long() if config.use_batch_label else None, # if config.DSBN else None,
            #    time_step=0,
            #    return_np=True,
            #)
            cell_embeddings, cell_embeddings_next, pert_preds, cls_preds, ps_preds, ps_preds_next, expr_dict = model.encode_batch_with_perturb(all_gene_ids,all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long() if config.use_batch_label else None, # if config.DSBN else None,
                pert_labels = torch.from_numpy(perturbation_indexes).long() if config.perturbation_input else None,
                pert_labels_next = torch.from_numpy(perturbation_indexes_next).long() if next_cell_prediction else None,
                time_step=0,
                return_np=True,
                predict_expr = predict_expr,
                mvc_src = full_gene_ids
            )

        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        cell_embeddings_next = cell_embeddings_next / np.linalg.norm(
            cell_embeddings_next, axis=1, keepdims=True
        )
        adata_t.obsm["X_scGPT"] = cell_embeddings

        adata_t.obsm["X_scGPT_next"] = cell_embeddings_next
        #adata_t.obsm["X_pert_pred"] = pert_preds
        if config.ps_weight >0:
            adata_t.obsm["ps_pred"] = ps_preds
        if config.next_cell_pred_type ==  'lochness':
            adata_t.obsm["ps_pred_next"] = ps_preds_next 
        for k in expr_dict:

            adata_t.obsm[k] = expr_dict[k][0]
            if len(expr_dict[k]) > 1:
                adata_t.obsm[k+'_zero'] =  expr_dict[k][1]
        # require: genotype_to_index

        # Assuming ret_adata.obsm['X_pert_pred'] is a numpy array or can be converted to one


        # Convert logits to probabilities using softmax
        X_pert_pred_probs = np.exp(pert_preds) / np.sum(np.exp(pert_preds), axis=1, keepdims=True)

        # Assign the probabilities back to the AnnData object
        adata_t.obsm['X_pert_pred_probs'] = X_pert_pred_probs

        # prompt: convert X_pert_pred_probs, which is the probabilities of each label, into label predictions, whose order is defined in genotype_to_index

        # Convert probabilities to predicted labels
        label_predictions = np.argmax(X_pert_pred_probs, axis=1)

        # Map predicted indices back to genotypes using genotype_to_index
        # Assuming genotype_to_index is a dictionary where keys are indices and values are genotypes
        index_to_genotype = {v: k for k, v in genotype_to_index.items()}
        predicted_genotypes = [index_to_genotype[i] for i in label_predictions]

        # Add the predicted genotypes to the AnnData object
        adata_t.obs['predicted_genotype'] = predicted_genotypes

        X_pert_cls_probs = np.exp(cls_preds) / np.sum(np.exp(cls_preds), axis=1, keepdims=True)
        adata_t.obsm['X_cls_pred_probs'] = X_pert_cls_probs
        label_predictions_cls = np.argmax(X_pert_cls_probs, axis=1)
        index_to_celltype = {v: k for k, v in cell_type_to_index.items()}
        predicted_celltypes = [index_to_celltype[i] for i in label_predictions_cls]
        adata_t.obs['predicted_celltype'] = predicted_celltypes
    return adata_t


def wrapper_train(model, config, data_gen,
                  logger = scg.logger,
                  save_dir = None,
                  device = None,
                  eval_adata_dict: Dict = {}):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batch_types = data_gen['num_batch_types']
    vocab = data_gen['vocab']

    optimizer_dict = create_optimizer_dict(model, device, config, num_batch_types)
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    define_wandb_metrcis()

    if save_dir is None:
        save_dir = Path(f"./save/dev_{config.dataset_name}-{time.strftime('%b%d-%H-%M')}/")
        save_dir.mkdir(parents=True, exist_ok=True)

    # save the current configurations before epoch starts
    torch.save(vocab, save_dir / "vocab.pt")
    running_parameters={
     'cell_type_to_index': data_gen["cell_type_to_index"],
     'genotype_to_index': data_gen["genotype_to_index"],
     'genes': data_gen["genes"], # genes,
     'gene_ids': data_gen["gene_ids"], # gene_ids,
     'ps_names': data_gen["ps_names"],
     'config': config.as_dict(), # config as dictionary
    }
    torch.save(running_parameters, save_dir / "running_parameters.pt")
    import json
    json.dump(config.as_dict(), open(save_dir / "config.json", "w"))
    # later, use the following to load json file
    #config_data = json.load(open(save_dir / 'config.json', 'r'))
    train_loader, valid_loader = data_gen['train_loader'], data_gen['valid_loader']
    executor = ProcessPoolExecutor(
        max_workers=4,
        initializer=init_plot_worker,
        mp_context=multiprocessing.get_context('forkserver') 
        )
    evaltest_processes = []

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        remaining_processes = []
        for p in evaltest_processes:
            if p.done():
                try:
                    result = p.result()
                    metrics_to_log = result['metrics']
                    for key, img_path in result['images'].items():
                        metrics_to_log[key]= wandb.Image(img_path)  
                    if metrics_to_log:
                        wandb.log(metrics_to_log)
                    logger.info(f'Finished {result["eval_dict_key"]} UMAP for epoch {result["epoch"]}')
                except Exception as e:
                    logger.warning(f'UMAP process failed for epoch {result["epoch"]} due to: {e}')
            else:
                remaining_processes.append(p)
         # Joins the process to release resources
        evaltest_processes = remaining_processes
        logger.info(f"Active UMAP processes: {len( evaltest_processes)}")

        if config.do_train:
            train(
                model,
                train_loader,
                config,
                vocab,
                optimizer_dict,
                epoch = epoch,
                logger = logger,
                device = device
            )
        val_loss, val_loss_next,  val_mvc, val_mvc_next, val_mre, val_mre_next, val_dab, val_cls, val_pert, val_ps, val_ps_next = evaluate(
            model,
            loader=valid_loader,
            config=config,
            vocab = vocab,
            epoch = epoch,
            device = device
        )
        elapsed = time.time() - epoch_start_time
        if logger is not None:
            logger.info("-" * 89)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mvc {val_mvc:5.4f} | "
                f"valid loss/mse_next {val_loss_next:5.4f} | mvc_next {val_mvc_next:5.4f} | "
                f"valid dab {val_dab:5.4f} | valid cls {val_cls:5.4f} | valid pert {val_pert:5.4f} |"
                f"valid ps {val_ps:5.4f} | valid ps_next {val_ps_next:5.4f} |"
            )
            logger.info("-" * 89)
        loss = val_loss + val_loss_next + val_mvc + val_mvc_next + val_cls + val_pert
        if loss < best_val_loss:
            best_val_loss = loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            if logger is not None:
                logger.info(f"Best model with score {best_val_loss:5.4f}")

        #if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        if epoch % 2 == 1:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / "best_model.pt")
            torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")
            save_dir2 = save_dir / f'e{epoch}_imgs'
            save_dir2.mkdir(parents=True, exist_ok=True)
            
            for eval_dict_key, eval_adata in eval_adata_dict.items():
                # Step 1: Get AnnData with embeddings from the main process
                results = eval_testdata(
                    #best_model,
                    model, # use current model
                    adata_t = eval_adata, #adata_t=data_gen['adata_sorted'], # if config.per_seq_batch_sample else adata,
                    gene_ids = data_gen['gene_ids'],
                    train_data_dict = data_gen,
                    config=config,
                    include_types=["cls"],
                    logger=logger,
                    epoch=epoch,
                    eval_key=eval_dict_key,
                )
                adata_with_embeddings = results
                
                # Step 2: Save the data to a temporary file for the child process
                #temp_adata_path = save_dir2 / f"temp_adata_{eval_dict_key}_e{epoch}.h5ad"
                #adata_with_embeddings.write_h5ad(temp_adata_path)

                # Step 3: Create and start the background process
                logger.info(f"Starting background process for UMAP on epoch {epoch} for '{eval_dict_key}'")
                
                # Pass data_gen['ps_names'] if it exists, otherwise None
                ps_names = data_gen.get('ps_names', None)
                #p = multiprocessing.Process(
                 #   target=process_and_log_umaps,
                  #  args=(adata_with_embeddings, config, epoch, eval_dict_key, save_dir2, ps_names)
                #)
                #p.start()           
                p = executor.submit(
                    process_and_log_umaps,
                    adata_with_embeddings, SimpleNamespace(**config) , epoch, eval_dict_key, save_dir2, ps_names
                )
                evaltest_processes.append(p)

            #metrics_to_log["test/best_model_epoch"] = best_model_epoch
            wandb.log({"test/best_model_epoch":best_model_epoch})
            # wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        optimizer_dict['scheduler'].step()

        if optimizer_dict['DAB_separate_optim']:
            optimizer_dict['scheduler_dab'].step()
        if config.ADV:
            optimizer_dict['scheduler_D'].step()
            optimizer_dict['scheduler_E'].step()

    # save the best model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")

    return best_model




