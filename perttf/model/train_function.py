import time
import torch
import random
import warnings
from pathlib import Path
import copy
from contextlib import nullcontext
import numpy as np
import pandas as pd
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from typing import List, Tuple

from torch import nn, Tensor
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from omegaconf import OmegaConf 
#multiprocessing.set_start_method('spawn', force=True)
import wandb


from ..utils.logger import create_logger
import matplotlib.pyplot as plt

from ..utils.set_optimizer import create_optimizer_dict
from ..custom_loss import (
    cce_loss, 
    criterion_neg_log_bernoulli, 
    masked_mse_loss, 
    masked_relative_error,
    GenerativeExpressionLoss
)
from ..utils.plot import process_and_log_umaps
from ..utils.misc import append_tensor as _append_tensor
from ..utils.misc import concatenate_outputs as _concat_outputs
from ..utils.misc import get_config_value as _cfg
from ..utils.misc import init_plot_worker
from ..utils.misc import to_numpy as _to_numpy
from ..utils.pert_data_loader import PertBatchCollator, PertTFDataset
from ..utils.pert_metrics import (
    GroupMoments,
    compute_perturbation_metrics,
    labels_to_names,
    normalize_expression,
    prediction_scale,
)
from .expr_sampler import DistributionGenerator


def train(model: nn.Module,
          loader: DataLoader,
          config,
          vocab,
          optim_dict: Dict,
          epoch = 0,
          logger = None,
          device = None) -> None:
    """
    Train the model for one epoch.
    """
    logger = create_logger() if logger is None else logger
    criterion = masked_mse_loss 
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()  # consider using label smoothing
    criterion_ps = nn.MSELoss() # this is the loss for predicting PS scores
    criterion_mvc = GenerativeExpressionLoss()
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
        sf = batch_data['sf'].to(device)
        sf_next = batch_data['sf_next'].to(device)
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
                sf = sf,
                sf_next = sf_next,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.get('cell_type_classifier', True),
                CCE = config.CCE,
                PERTPRED = config.get('genotype_classifier', True),
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
                loss_gepc = criterion_mvc(
                        output_dict["mvc_output"], 
                        mvc_target_values, 
                        mvc_masked_positions,
                        scale_factor = sf,
                    )
                loss = loss + config.this_weight *loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
                # added
                loss_gepc_next = criterion_mvc(
                    output_dict["mvc_output_next"], 
                    mvc_target_values_next, 
                    mvc_masked_positions,
                    scale_factor = sf_next,
                )
                loss = loss + config.next_weight * loss_gepc_next
                metrics_to_log.update({"train/mvc_next": loss_gepc_next.item()})
                
                if config.explicit_zero_prob and config.distribution is None:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_output"]["zero_probs"], mvc_target_values, mvc_masked_positions
                    )
                    loss = loss + config.this_weight *loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                    # added
                    loss_gepc_zero_log_prob_next = criterion_neg_log_bernoulli(
                        output_dict["mvc_output_next"]["zero_probs"], mvc_target_values_next, mvc_masked_positions
                    )
                    loss = loss + config.next_weight * loss_gepc_zero_log_prob_next
                    metrics_to_log.update(
                        {"train/mvc_nzlp_next": loss_gepc_zero_log_prob_next.item()}
                    )
            if config.get('cell_type_classifier', True):
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

            if config.get('genotype_classifier', True):
                loss_pert = criterion_pert(output_dict["pert_output"], perturbation_labels)
                loss = loss + config.perturbation_classifier_weight * loss_pert
                metrics_to_log.update({"train/pert": loss_pert.item()})
                # add for next pert prediction
                loss_pert_next = criterion_pert(output_dict["pert_output_next"], perturbation_labels_next)
                loss = loss + config.perturbation_classifier_weight * config.next_weight * loss_pert_next
                metrics_to_log.update({"train/pert_next": loss_pert_next.item()})

            if config.ps_weight > 0:
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
                sf = sf,
                sf_next = sf_next,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
                CLS=config.get('cell_type_classifier', True),
                #CCE=config.CCE,
                PERTPRED = config.get('genotype_classifier', True),
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

        total_cls += loss_cls.item() if config.get('cell_type_classifier', True) else 0.0
        total_pert += loss_pert.item() if config.get('genotype_classifier', True) else 0.0
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
            cur_cls = total_cls / log_interval if config.get('cell_type_classifier', True) else 0.0
            cur_pert = total_pert / log_interval if config.get('genotype_classifier', True) else 0.0
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


def _run_evaluation_batches(
    model: nn.Module,
    loader: DataLoader,
    config,
    vocab,
    device,
    apply_next_perturbation=False,
    compute_losses=True,
    real_groups=None,
    cell_type_to_index=None,
    genotype_to_index=None,
    prediction_modes=("mean",),
    collect_outputs=False,
    predict_expr=False,
    use_full_mvc_src=False,
    use_size_factor=True,
    output_prediction_mode="mean",
    target_sum=10000.0,
    sample_seed=None,
):
    """Shared DataLoader-backed eval/inference loop used by evaluate() and eval_testdata()."""
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_pert = nn.CrossEntropyLoss()
    criterion_ps = nn.MSELoss()
    criterion_mvc = GenerativeExpressionLoss()
    generator = DistributionGenerator(getattr(model, "distribution", None))
    prediction_modes = [prediction_modes] if isinstance(prediction_modes, str) else list(prediction_modes)
    moments = {mode: GroupMoments() for mode in prediction_modes} if real_groups is not None else {}
    index_to_celltype = {value: key for key, value in (cell_type_to_index or {}).items()}
    index_to_genotype = {value: key for key, value in (genotype_to_index or {}).items()}
    outputs = {} if collect_outputs else None

    total_loss = 0.0
    total_loss_next = 0.0
    total_error = 0.0
    total_error_next = 0.0
    total_dab = 0.0
    total_cls = 0.0
    total_pert = 0.0
    total_ps = 0.0
    total_ps_next = 0.0
    total_mvc = 0.0
    total_mvc_next = 0.0
    total_num = 0

    pred_lochness_next = _cfg(config, "pred_lochness_next", None)
    if pred_lochness_next is not None:
        has_lochness_next_pred = True
        ps_next_training_weight = pred_lochness_next
    else:
        has_lochness_next_pred = False
        ps_next_training_weight = _cfg(config, "ps_weight", 0) * _cfg(config, "next_weight", 0)

    model.eval()
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    autocast_context = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        if sample_seed is not None:
            torch.manual_seed(sample_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(sample_seed)
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device) if "batch_labels" in batch_data else None
            celltype_labels = batch_data["celltype_labels"].to(device) if "celltype_labels" in batch_data else None
            perturbation_labels = batch_data["perturbation_labels"].to(device) if "perturbation_labels" in batch_data else None
            perturbation_labels_next = batch_data["perturbation_labels_next"].to(device) if apply_next_perturbation else None
            sf = batch_data["sf"].to(device) if use_size_factor and "sf" in batch_data else None
            sf_next = batch_data["sf_next"].to(device) if compute_losses and use_size_factor else sf
            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
            mvc_src = batch_data["full_gene_ids"].to(device) if (use_full_mvc_src or not _cfg(config, "mvc_masked_train", True)) and "full_gene_ids" in batch_data else None
            use_mvc = predict_expr or real_groups is not None or _cfg(config, "GEPC", False)

            with autocast_context(enabled=_cfg(config, "amp", False)) if device.type == "cuda" else autocast_context():
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if _cfg(config, "use_batch_label", False) else None,
                    pert_labels=perturbation_labels if _cfg(config, "perturbation_input", False) else None,
                    pert_labels_next=perturbation_labels_next,
                    sf=sf,
                    sf_next=sf_next,
                    MVC=use_mvc,
                    ECS=_cfg(config, "ecs_thres", 0) > 0 and compute_losses,
                    CLS=_cfg(config, "cell_type_classifier", True) or collect_outputs,
                    PERTPRED=_cfg(config, "genotype_classifier", True) or collect_outputs,
                    PSPRED=_cfg(config, "ps_weight", 0) > 0 or collect_outputs,
                    mvc_src=mvc_src,
                )

                batch_size = input_gene_ids.shape[0]
                if compute_losses:
                    target_values = batch_data["target_values"].to(device)
                    target_values_next = batch_data["target_values_next"].to(device)
                    output_values = output_dict["mlm_output"]
                    masked_positions = input_values.eq(config.mask_value)
                    loss = criterion(output_values, target_values, masked_positions)
                    loss_mse_next = criterion(output_values, target_values_next, masked_positions)
                    if _cfg(config, "GEPC", False):
                        mvc_target_values = target_values if _cfg(config, "mvc_masked_train", True) else batch_data["full_expr"].to(device)
                        mvc_target_values_next = target_values_next if _cfg(config, "mvc_masked_train", True) else batch_data["full_expr_next"].to(device)
                        loss_gepc = criterion_mvc(output_dict["mvc_output"], mvc_target_values, scale_factor=sf)
                        loss_gepc_next = criterion_mvc(output_dict["mvc_output_next"], mvc_target_values_next, scale_factor=sf_next)
                    else:
                        loss_gepc = output_values.new_tensor(0.0)
                        loss_gepc_next = output_values.new_tensor(0.0)
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels) if _cfg(config, "dab_weight", 0) > 0 else output_values.new_tensor(0.0)
                    loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels) if _cfg(config, "cell_type_classifier", True) else output_values.new_tensor(0.0)
                    loss_pert = criterion_pert(output_dict["pert_output"], perturbation_labels) if _cfg(config, "genotype_classifier", True) else output_values.new_tensor(0.0)
                    loss_ps = criterion_ps(output_dict["ps_output"], batch_data["ps"].to(device)) if _cfg(config, "ps_weight", 0) > 0 else output_values.new_tensor(0.0)
                    loss_ps_next = criterion_ps(output_dict["ps_output_next"], batch_data["ps_next"].to(device)) if ps_next_training_weight > 0 else output_values.new_tensor(0.0)

            if compute_losses:
                total_loss += loss.item() * batch_size
                total_loss_next += loss_mse_next.item() * batch_size
                total_mvc += loss_gepc.item() * batch_size
                total_mvc_next += loss_gepc_next.item() * batch_size
                total_error += masked_relative_error(output_values, target_values, masked_positions).item() * batch_size
                total_error_next += masked_relative_error(output_values, target_values_next, masked_positions).item() * batch_size
                total_dab += loss_dab.item() * batch_size
                total_cls += loss_cls.item() * batch_size
                total_pert += loss_pert.item() * batch_size
                total_ps += loss_ps.item() * batch_size
                total_ps_next += loss_ps_next.item() * batch_size
                total_num += batch_size

            if real_groups is not None:
                contexts = labels_to_names(batch_data["celltype_labels"], index_to_celltype)
                perturbations = labels_to_names(batch_data["perturbation_labels_next"], index_to_genotype)
                for mode in prediction_modes:
                    predicted = generator.generate(output_dict["mvc_output_next"], sample=mode == "sample", device=device)["pred"][:, 1:]
                    normalized = normalize_expression(
                        _to_numpy(predicted),
                        target_sum=target_sum,
                        input_scale=prediction_scale(getattr(model, "distribution", None)),
                    )
                    moments[mode].update(normalized, contexts, perturbations)

            if collect_outputs:
                cell_embedding = output_dict["transformer_output"][:, 0, :]
                next_emb = output_dict["cell_emb_next"]
                _append_tensor(outputs, "X_scGPT", cell_embedding)
                _append_tensor(outputs, "X_scGPT_next", next_emb)
                _append_tensor(outputs, "pert_logits", output_dict.get("pert_output"))
                _append_tensor(outputs, "cls_logits", output_dict.get("cls_output"))
                _append_tensor(outputs, "ps_pred", output_dict.get("ps_output"))
                _append_tensor(outputs, "ps_pred_next", output_dict.get("ps_output_next"))
                if predict_expr:
                    _append_tensor(outputs, "mlm_expr", output_dict.get("mlm_output")[:, 1:])
                    for prefix, expr_output in (("mvc", output_dict["mvc_output"]), ("mvc_next", output_dict["mvc_output_next"])):
                        generated = generator.generate(expr_output, sample=output_prediction_mode == "sample", device=device)
                        outputs.setdefault(f"{prefix}_expr", []).append(_to_numpy(generated["pred"][:, 1:]))
                        if generated.get("zero_probs") is not None:
                            outputs.setdefault(f"{prefix}_expr_zero", []).append(_to_numpy(generated["zero_probs"][:, 1:]))
                        if generated.get("param2") is not None:
                            outputs.setdefault(f"{prefix}_param2", []).append(_to_numpy(generated["param2"][:, 1:]))

    result = {
        "losses": None,
        "metrics": {},
        "outputs": _concat_outputs(outputs) if collect_outputs else {},
    }
    if compute_losses:
        result["losses"] = (
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
            total_ps_next / total_num,
        )
    if real_groups is not None:
        result["metrics"] = {
            mode: compute_perturbation_metrics(
                real_groups,
                mode_moments.finalize(),
                control_value=_cfg(config, "perturbation_control_value", "WT"),
                fdr_threshold=_cfg(config, "perturbation_metric_fdr", 0.05),
                min_cells=_cfg(config, "perturbation_metric_min_cells", 3),
            )
            for mode, mode_moments in moments.items()
        }
    return result


def evaluate(model: nn.Module,
            loader: DataLoader,
            config,
            vocab,
            epoch = 0,
            device = None,
            perturbation_reference_groups = None,
            cell_type_to_index = None,
            genotype_to_index = None,
            prediction_modes = None,
            target_sum = None,
            sample_seed = None) -> Any:
    """Evaluate the model using the shared DataLoader-backed inference loop."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apply_next_perturbation = _cfg(config, "next_cell_pred_type", "identity") != "identity"
    result = _run_evaluation_batches(
        model,
        loader,
        config,
        vocab,
        device,
        apply_next_perturbation=apply_next_perturbation,
        compute_losses=perturbation_reference_groups is None,
        real_groups=perturbation_reference_groups,
        cell_type_to_index=cell_type_to_index,
        genotype_to_index=genotype_to_index,
        prediction_modes=prediction_modes or _cfg(config, "perturbation_metric_modes", ["mean"]),
        target_sum=_cfg(config, "perturbation_metric_target_sum", 10000.0) if target_sum is None else target_sum,
        sample_seed=sample_seed,
    )
    if perturbation_reference_groups is not None:
        return result
    losses = result["losses"]
    wandb.log(
        {
            "valid/mse": losses[0],
            "valid/mse_next": losses[1],
            "valid/mvc": losses[2],
            "valid/mvc_next": losses[3],
            "valid/mre": losses[4],
            "valid/mre_next": losses[5],
            "valid/dab": losses[6],
            "valid/cls": losses[7],
            "valid/pert": losses[8],
            "valid/ps": losses[9],
            "valid/ps_next": losses[10],
            "valid/sum_mse_dab": losses[0] + config.dab_weight * losses[6],
            "epoch": epoch,
        },
    )
    return losses

def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    gene_ids: List[str],
    train_data_dict: Dict,
    config,
    include_types: List[str] = ["cls","pert"],
    input_layer_key = "X_binned",
    next_layer_key = "X_binned_next",
    logger = None,
    epoch = 0,
    eval_key = "", # titles for evaluation
    make_plots = True,
    predict_expr = False,
    mvc_full_expr = False,
    sizefactor = False,
    sample = False,
    device = None,
) -> AnnData:
    """
    Evaluate the model on test data and return an AnnData object with embeddings.
    Plotting and UMAP are offloaded to a separate process.
    """
    logger = create_logger() if logger is None else logger
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata_t = adata_t.copy()
    cell_type_to_index = train_data_dict["cell_type_to_index"]
    genotype_to_index = train_data_dict["genotype_to_index"]
    vocab = train_data_dict["vocab"]
    shared_genes = adata_t.var.index.isin(list(vocab.stoi.keys()))
    logger.info(f"{sum(shared_genes)} genes shared between model vocab's {len(vocab)} and anndata's {adata_t.shape[1]} genes")
    adata_t = adata_t[:, shared_genes].copy()
    gene_ids = np.array(vocab(adata_t.var.index.tolist()), dtype=int)
    if "genotype_next" in adata_t.obs.keys():
        adata_t = adata_t[adata_t.obs["genotype_next"].isin(genotype_to_index)].copy()

    apply_next_perturbation = False
    if _cfg(config, "next_cell_pred_type", "identity") == "pert":
        apply_next_perturbation = "genotype_next" in adata_t.obs.columns
        if not apply_next_perturbation:
            logger.warning("next cell pred is set to pert but the provided adata does not have genotype_next column")
    elif _cfg(config, "next_cell_pred_type", "identity") == "lochness":
        apply_next_perturbation = _cfg(config, "pred_lochness_next", 0) > 0 and "genotype_next" in adata_t.obs.columns

    sampling_mode = _cfg(config, "sampling_mode", "simple")
    hvg_inds = None
    max_seq_len = _cfg(config, "max_seq_len", 3000)
    if sampling_mode == "expressed":
        max_seq_len = 10000
    elif sampling_mode == "hvg":
        hvg_col = _cfg(config, "hvg_col", "highly_variable")
        assert hvg_col in adata_t.var.keys(), "adata must have calculated HVGs or adata.var must have hvg_col"
        hvg_inds = (np.where(adata_t.var[hvg_col])[0], np.where(~adata_t.var[hvg_col])[0])
        max_seq_len = int(adata_t.var[hvg_col].sum()) + _cfg(config, "non_hvg_size", 1000)
    collator_config = dict(config)
    collator_config.update({
        "max_seq_len": max_seq_len,
        "deterministic": True,
        "prediction_only": True,
    })

    dataset = PertTFDataset(
        adata_t,
        indices=np.arange(adata_t.n_obs),
        expr_layer=input_layer_key,
        cell_type_to_index=cell_type_to_index,
        genotype_to_index=genotype_to_index,
        next_cell_pred=_cfg(config, "next_cell_pred_type", "identity"),
        size_factor_col=_cfg(config, "size_factor_col", None),
        prediction_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=_cfg(config, "batch_size", 32),
        shuffle=False,
        num_workers=0,
        collate_fn=PertBatchCollator(vocab, gene_ids, hvg_inds=hvg_inds, **collator_config),
        pin_memory=True,
    )
    prediction_mode = "sample" if sample is True else "mean"

    result = _run_evaluation_batches(
        model,
        loader,
        config,
        vocab,
        device,
        apply_next_perturbation=apply_next_perturbation,
        compute_losses=False,
        collect_outputs="cls" in include_types,
        predict_expr=predict_expr,
        use_full_mvc_src=mvc_full_expr,
        use_size_factor=sizefactor,
        output_prediction_mode=prediction_mode,
    )
    outputs = result["outputs"]
    if not outputs:
        return adata_t

    cell_embeddings = outputs["X_scGPT"]
    cell_embeddings = cell_embeddings / np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_embeddings_next = outputs["X_scGPT_next"]
    cell_embeddings_next = cell_embeddings_next / np.linalg.norm(cell_embeddings_next, axis=1, keepdims=True)
    adata_t.obsm["X_scGPT"] = cell_embeddings
    adata_t.obsm["X_scGPT_next"] = cell_embeddings_next
    if "ps_pred" in outputs and _cfg(config, "ps_weight", 0) > 0:
        adata_t.obsm["ps_pred"] = outputs["ps_pred"]
    if "ps_pred_next" in outputs and _cfg(config, "next_cell_pred_type", "identity") == "lochness":
        adata_t.obsm["ps_pred_next"] = outputs["ps_pred_next"]
    for key, value in outputs.items():
        if key not in {"X_scGPT", "X_scGPT_next", "pert_logits", "cls_logits", "ps_pred", "ps_pred_next"}:
            adata_t.obsm[key] = value

    pert_preds = outputs["pert_logits"]
    pert_shift = pert_preds - pert_preds.max(axis=1, keepdims=True)
    X_genotype_cls_probs = np.exp(pert_shift) / np.sum(np.exp(pert_shift), axis=1, keepdims=True)
    adata_t.obsm["X_pert_pred_probs"] = X_genotype_cls_probs
    adata_t.obsm["genotype_pred_probs"] = X_genotype_cls_probs
    index_to_genotype = {v: k for k, v in genotype_to_index.items()}
    adata_t.obs["predicted_genotype"] = [index_to_genotype[i] for i in np.argmax(X_genotype_cls_probs, axis=1)]
    if "genotype" in adata_t.obs.columns:
        adata_t.obs["genotype_id"] = adata_t.obs["genotype"].map(genotype_to_index).astype(pd.CategoricalDtype(categories=list(genotype_to_index.values())))

    cls_preds = outputs["cls_logits"]
    cls_shift = cls_preds - cls_preds.max(axis=1, keepdims=True)
    X_celltype_cls_probs = np.exp(cls_shift) / np.sum(np.exp(cls_shift), axis=1, keepdims=True)
    adata_t.obsm["X_cls_pred_probs"] = X_celltype_cls_probs
    adata_t.obsm["celltype_pred_probs"] = X_celltype_cls_probs
    index_to_celltype = {v: k for k, v in cell_type_to_index.items()}
    adata_t.obs["predicted_celltype"] = [index_to_celltype[i] for i in np.argmax(X_celltype_cls_probs, axis=1)]
    if "celltype" in adata_t.obs.columns:
        adata_t.obs["celltype_id"] = adata_t.obs["celltype"].map(cell_type_to_index).astype(pd.CategoricalDtype(categories=list(cell_type_to_index.values())))

    return adata_t


def wrapper_train(model, config, data_gen,
                  logger = None,
                  save_dir = None,
                  device = None,
                  eval_adata_dict: Dict = {}):
    logger = create_logger() if logger is None else logger
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
        mp_context=multiprocessing.get_context('spawn') 
        )
    evaltest_processes = []

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        # Clean up background UMAP and metric calculations on past test-data eval
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
                    logger.warning(f'UMAP process failed due to: {e}')
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
        val_loss, val_loss_next, val_mvc, val_mvc_next, val_mre, val_mre_next, val_dab, val_cls, val_pert, val_ps, val_ps_next = evaluate(
            model,
            loader=valid_loader,
            config=config,
            vocab=vocab,
            epoch=epoch,
            device=device,
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
        loss = val_loss * 0.1
        if config.next_cell_pred_type == 'identity':
            loss += (val_cls*int(config.cell_type_classifier)
                      + val_pert*int(config.genotype_classifier))
        elif config.next_cell_pred_type == 'pert':
            loss += val_mvc_next
        else:
            loss += val_ps + val_ps_next 
        best_model_epoch =0
        if loss < best_val_loss:
            best_val_loss = loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            if logger is not None:
                logger.info(f"Best model with score {best_val_loss:5.4f}")

        #if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        eval_expr_interval = max(round(2e5/len(train_loader.dataset)), abs(config.get('eval_expr_interval', 2))//2*2) # this must be even to match the save interval below
        predict_expr_tmp = True if eval_expr_interval and epoch % eval_expr_interval == 1 and config.get('next_cell_pred_type') == 'pert' else False
        save_eval_interval = config.get('save_eval_interval', 2)
        if epoch % save_eval_interval == 1:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / "best_model.pt")
            torch.save(model.state_dict(), save_dir / f"model_e{epoch}.pt")
        if epoch % eval_expr_interval == 1:
            #logger.info(f"Saving model to {save_dir}")
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
                    predict_expr = predict_expr_tmp,
                    mvc_full_expr= predict_expr_tmp
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
                    adata_with_embeddings, OmegaConf.structured(dict(config)) , epoch, eval_dict_key, save_dir2, ps_names
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
            
    # One final gather for the background processes
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
                    logger.warning(f'UMAP process failed due to: {e}')
            else:
                remaining_processes.append(p)
    # save the best model
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")

    return best_model
