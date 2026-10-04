"""
Shared steps of the LoRA fine-tuning entry points on HFPerturbationTFModel
(run_lora_cls_train, run_lora_pert_train). Each entry point keeps its own objective,
label rules, split and checkpoint score; the steps here are identical for all of them.
"""
import json
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from peft import get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer

from . import train_function
from ..custom_loss import GenerativeExpressionLoss, criterion_neg_log_bernoulli, masked_mse_loss
from ..utils.set_optimizer import create_optimizer_dict


def check_no_adapter(model) -> None:
    """One adapter per loaded model: training a second adapter would wrap an already-wrapped model."""
    if any(isinstance(module, BaseTunerLayer) for module in model.modules()):
        raise ValueError(
            "This model already has a LoRA adapter attached. Load a fresh base model with "
            "HFPerturbationTFModel.from_pretrained(...) for each adapter you train."
        )


def prepare_config(model, adata: AnnData, input_layer_key: str, batch_size: Optional[int], lr: Optional[float],
                   amp: Optional[bool], log_interval: Optional[int], seed: Optional[int]) -> Tuple[object, Optional[list]]:
    """Check required adata fields, build the training config from the checkpoint and seed all RNGs.
    Returns the config and the PS columns present in adata."""
    # Training and inference (eval_testdata) both read these fixed names.
    for obs_col, example in (("genotype", "adata.obs['genotype'] = adata.obs['<perturbation column>']"),
                             ("celltype", "adata.obs['celltype'] = adata.obs['<cell type column>'], or 'all' for a single context")):
        if obs_col not in adata.obs.columns:
            raise ValueError(f"adata.obs has no '{obs_col}' column; create it first, e.g. {example}")
    if input_layer_key not in adata.layers:
        raise ValueError(f"adata.layers has no '{input_layer_key}' layer; set it to the log-normalized expression, e.g. adata.layers['{input_layer_key}'] = adata.X")

    config = model._init_default_train_config_()
    # Avoid duplicate kwarg collision in PertBatchCollator(vocab=..., **config).
    if "vocab" in config:
        del config["vocab"]
    # Keep training flags aligned with the loaded model, not stale checkpoint training_config.
    config.GEPC = hasattr(model, "mvc_decoder")
    config.explicit_zero_prob = bool(getattr(model, "explicit_zero_prob", False))
    config.distribution = getattr(model, "distribution", None)
    if batch_size is not None:
        config.batch_size = batch_size
    if lr is not None:
        config.lr = lr
    if amp is not None:
        config.amp = amp
    if log_interval is not None:
        config.log_interval = log_interval
    config.dataset_name = Path(getattr(adata, "filename", "") or "adata").stem

    # Reproducibility: seed all RNGs before data loading and training.
    effective_seed = seed if seed is not None else config.get("seed", 42)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    ps_columns = getattr(model, "ps_names", None)
    if ps_columns:
        valid_ps_columns = [col for col in ps_columns if col in adata.obs.columns]
        if len(valid_ps_columns) != len(ps_columns):
            print(f"[lora] filtered ps columns to existing obs columns: {valid_ps_columns if valid_ps_columns else 'None'}")
        ps_columns = valid_ps_columns if valid_ps_columns else None
    return config, ps_columns


def label_mappings(model, adata: AnnData, require_known) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Label -> index mappings for celltype and genotype. A label column listed in require_known keeps
    the pretrained mapping and must only contain model labels; for the others, a mapping missing
    adata labels is rebuilt from adata (returned as None), which is harmless when nothing reads it.
    """
    mappings = {}
    for obs_col, attr in (("celltype", "cell_type_to_index"), ("genotype", "genotype_to_index")):
        mapping = getattr(model, attr, None)
        missing = [x for x in adata.obs[obs_col].unique() if mapping is None or x not in mapping]
        if obs_col in require_known and missing:
            raise ValueError(
                f"adata.obs['{obs_col}'] has labels the model does not know: {sorted(map(str, missing))}. "
                f"Filter or rename these cells to the model labels (model.{attr})."
            )
        if missing and mapping is not None:
            print(f"[lora] model {attr} missing labels from current adata; rebuilding {obs_col} mapping from adata.")
            mapping = None
        mappings[obs_col] = mapping
    return mappings["celltype"], mappings["genotype"]


def wrap(model, lora_config, config, device: torch.device, num_batch_types: int):
    """Wrap the model with LoRA and build its optimizer."""
    model.to(device)
    peft_model = get_peft_model(model, lora_config).to(device)
    # train_function logs through wandb by default; keep the LoRA entry points side-effect free.
    if hasattr(train_function, "wandb") and hasattr(train_function.wandb, "log"):
        train_function.wandb.log = lambda *args, **kwargs: None
    optimizer_dict = create_optimizer_dict(peft_model, device, config, num_batch_types)
    return peft_model, optimizer_dict


def fit(peft_model, epochs: int, run_epoch: Callable[[int], float], device: torch.device) -> None:
    """Run run_epoch(epoch) -> validation score (lower is better) for each epoch and restore the lowest-score state."""
    best_score = float("inf")
    best_epoch = 0
    best_state_dict = None
    for epoch in range(1, epochs + 1):
        score = run_epoch(epoch)
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in peft_model.state_dict().items()}
    if best_state_dict is not None:
        peft_model.load_state_dict(best_state_dict)
        peft_model.to(device)
        print(f"Restored best model from epoch {best_epoch}")


def save(peft_model, save_dir: Optional[str], heads_info: Dict) -> None:
    """Save the adapter plus lora_heads.json (mode, any new cls head and the base checkpoint hash), read by load_lora_adapter."""
    if not save_dir:
        return
    adapter_dir = Path(save_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))
    heads_info = {**heads_info, "base_hash": peft_model.get_base_model().base_hash}
    with open(adapter_dir / "lora_heads.json", "w") as f:
        json.dump(heads_info, f, indent=2)
    print(f"Saved PEFT adapter to: {adapter_dir.resolve()}")


def cls_epoch(peft_model, cls_head, loader, is_train: bool, config, device: torch.device, pad_id: int,
              cls_targets: torch.Tensor, cls_task: str, optimizer_dict: Dict) -> Tuple[float, float]:
    """
    One pass of the new cls head loss plus, when config.this_weight > 0, the expression reconstruction
    losses of the cell itself (masked-gene MSE, GEPC/MVC and their zero-probability terms, as in
    train_function.train), all from the same forward pass. Returns the mean head and reconstruction losses.
    """
    peft_model.train(is_train)
    reconstruct = config.this_weight > 0
    criterion_mvc = GenerativeExpressionLoss()
    total_head, total_recon, n_cells = 0.0, 0.0, 0
    for batch_data in loader:
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        sf = batch_data["sf"].to(device)
        mvc_src = None if config.get("mvc_masked_train", True) else batch_data["full_gene_ids"].to(device)
        with torch.set_grad_enabled(is_train), torch.amp.autocast("cuda", enabled=config.amp):
            output_dict = peft_model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=input_gene_ids.eq(pad_id),
                batch_labels=batch_data["batch_labels"].to(device) if config.use_batch_label else None,
                sf=sf,
                MVC=reconstruct and config.GEPC,
                mvc_src=mvc_src,
            )
            # L2-normalised <cls> embedding, the same quantity eval_testdata stores in obsm['X_scGPT']
            cell_emb = F.normalize(output_dict["transformer_output"][:, 0, :], p=2, dim=1)
            cls_pred = cls_head(cell_emb)
            target = cls_targets[batch_data["index"].long()].to(device)
            if cls_task == "classification":
                loss_head = F.cross_entropy(cls_pred, target)
            else:
                loss_head = F.mse_loss(cls_pred.squeeze(1), target)

            loss_recon = torch.zeros((), device=device)
            if reconstruct:
                target_values = batch_data["target_values"].to(device)
                masked_positions = input_values.eq(config.mask_value)
                loss_recon = masked_mse_loss(output_dict["mlm_output"], target_values, masked_positions)
                if config.explicit_zero_prob:
                    loss_recon = loss_recon + criterion_neg_log_bernoulli(output_dict["mlm_zero_probs"], target_values, masked_positions)
                if config.GEPC:
                    mvc_target_values = target_values if config.get("mvc_masked_train", True) else batch_data["full_expr"].to(device)
                    mvc_masked_positions = masked_positions if config.get("mvc_masked_train", True) else None
                    loss_recon = loss_recon + criterion_mvc(output_dict["mvc_output"], mvc_target_values, mvc_masked_positions, scale_factor=sf)
                    if config.explicit_zero_prob and config.distribution is None:
                        loss_recon = loss_recon + criterion_neg_log_bernoulli(
                            output_dict["mvc_output"]["zero_probs"], mvc_target_values, mvc_masked_positions)
            loss = loss_head + config.this_weight * loss_recon
        if is_train:
            optimizer = optimizer_dict["optimizer"]
            scaler = optimizer_dict["scaler"]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        total_head += loss_head.item() * len(target)
        total_recon += loss_recon.item() * len(target)
        n_cells += len(target)
    return total_head / n_cells, total_recon / n_cells
