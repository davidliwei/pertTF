# Multi-perturbation (high-MOI) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a pertTF cell carry a set of perturbations (1 to ~10) in `genotype` / `genotype_next`, opt-in via `pert_delimiter`, with no change to legacy behaviour or checkpoint format.

**Architecture:** Delimited strings in the existing obs columns are parsed by a new `perttf/utils/pert_labels.py` module into padded `(k,)` int arrays with a `-1` sentinel; one `encode_perturbation_set` helper wraps the three existing encoders with a masked sum/mean/sqrt reduction; the classifier switches to BCE on 2-D labels and the contrastive loss to an overlap mask; evaluation can group per set or per single perturbation.

**Tech Stack:** Python 3.10+, PyTorch, AnnData/scanpy, pytest (CPU-only CI).

**Spec:** `docs/superpowers/specs/2026-10-20-multi-perturbation-design.md`

## Global Constraints

- `pert_delimiter=None` (default) must leave every code path, tensor shape, loss and output byte-identical to today (spec §5.1).
- No new parameter or buffer in any model: existing checkpoints must load with `strict=True` (spec §5.2). The pad sentinel is `-1` and never reaches an encoder.
- A width-1 padded label must give the same embedding as the scalar label for `sum`, `mean` and `sqrt` (spec §3.4).
- `pert_aggregation` is a named constructor parameter of both `PerturbationTFModel` and `HFPerturbationTFModel` so it is restored from `config.json` (spec §3.4).
- Config keys and defaults (spec §3.10): `pert_delimiter=None`, `pert_aggregation="sum"`, `perturbation_unit="set"`, `pert_pred_threshold=0.5`, `pert_bce_pos_weight=None`, `max_perts_per_cell=None`.
- Control label is `config['perturbation_control_value']` (default `"WT"`); inside a combination it is dropped, alone it is kept (spec §3.1).
- Canonical label = sorted unique components joined by the delimiter (spec §3.1).
- The `test` extra is `pytest, torch>=2.2, tqdm, omegaconf, safetensors, wandb, peft` on top of the base deps (Task 0); CI installs `.[test]` and nothing else.
- Unit tests of Tasks 1–6 and 8 never import `perttf.model.train_function` or `perttf.model.lora`; only `tests/test_tutorial_smoke.py` reaches them, through the same entry points the notebooks call (spec §6).
- `pytest tests/test_tutorial_smoke.py` is the tutorial-compatibility gate: the Task 0 tests must pass unchanged after every later task; they are extended (never edited) in Tasks 7 and 8.
- Work on a feature branch (`feat/multi-perturbation`), never on `main`.

## Review Focus

1. A label with surrounding whitespace (`"A + B"`) in multi mode: components must not be silently stripped; the validator raises naming the label (test in Task 1).
2. A `genotype_next` column that is pandas categorical: canonicalisation must produce a usable string column and pairing must still find the control (test in Task 5).
3. A cell whose set is entirely control after dropping (`"WT+WT"`) must encode exactly as `"WT"` and be treated as a control by pairing (test in Task 1 and Task 5).
4. In multi mode with `perturbation_unit="single"`, a validation set with 10 perturbations per cell must contribute to 10 groups and the real reference groups must be expanded the same way, otherwise predicted and real groups do not align (test in Task 6).
5. Inference with a legacy checkpoint whose mapping contains `"A+B"` and `pert_delimiter="+"` must raise before any computation, not produce silent garbage (test in Task 8).

---

### Task 0: Tutorial compatibility baseline

**Files:**
- Modify: `perttf/model/config_gen.py:76-85` (`wandb.init` call)
- Modify: `pyproject.toml` (`test` extra)
- Create: `tests/test_tutorial_smoke.py`

**Interfaces:**
- Consumes: today's public entry points only: `generate_config`, `produce_training_datasets`, `PerturbationTFModel(...)` as the notebooks construct it, `wrapper_train`, `eval_testdata`, `HFPerturbationTFModel.{save_pretrained,from_pretrained,predict_perturbations,build_lora_config,run_lora_cls_train,run_lora_pert_train,load_lora_adapter,predict_cls}`, `compute_metrics_from_anndata`, `virtual_screen_src2dest`.
- Produces: the module-level helpers `make_adata`, `split_indices`, `tutorial_hyperparameters`, `pert_hyperparameters`, `build_model`, `train_tutorial_model`, `to_hf_checkpoint` and the `adata` / `pert_checkpoint` fixtures, which Tasks 7 and 8 reuse for the multi-mode variants. The whole file runs on CPU in about 25 s (training runs are ~5 s each; the LoRA tests add the `peft` import).

Why this task exists: the six notebooks under `demos/tutorials/` are the only documentation of the end-to-end contracts (obs/obsm/uns keys, saved files, signatures). They need data downloads and a GPU, so they never run in CI. This file replays each notebook's library calls on a 240-cell synthetic dataset. It was written against the unmodified repo and passes there; running it after every task is what "compatible with existing functionalities" means for this plan. Two things it found in the unmodified repo, fixed here because the baseline must be green on a fresh install:

1. `generate_config` passes `settings=wandb.Settings(start_method="fork")`. wandb has ignored `start_method` since 0.19 (it prints "non-functional and safely ignored") and rejects it from 0.29 on (`pydantic ValidationError: Extra inputs are not permitted`), so every training notebook crashes on a current `pip install wandb`. Verified: 0.19.11 through 0.28.0 accept it, 0.29.0 onward raise.
2. `PertTFUniDataManager._create_loaders_from_dataset` hardcodes `num_workers=8`. On macOS (spawn) each worker re-imports torch and scanpy, about 45 s per training run on CPU. Not a product change in this plan: the test forces in-process loaders through a fixture, and the hardcode is listed under "Left to the builder" in the spec as a follow-up.

- [ ] **Step 1: Make `generate_config` work on current wandb**

In `perttf/model/config_gen.py` delete the line `settings=wandb.Settings(start_method="fork"),` from the `wandb.init(...)` call (lines 77-85). Nothing else changes; `settings` is optional and the kwarg was already a no-op.

- [ ] **Step 2: Extend the `test` extra in `pyproject.toml`**

```toml
test = ["pytest", "torch>=2.2", "tqdm", "omegaconf", "safetensors", "wandb", "peft"]
```

- [ ] **Step 3: Write the tutorial smoke tests**

Create `tests/test_tutorial_smoke.py` with exactly this content (it was run and passes against the unmodified repo in a CPU-only environment with torch 2.14, scanpy 1.12, wandb 0.28 and 0.31 after Step 1, peft current):

```python
"""Minimal end-to-end examples distilled from demos/tutorials/*.ipynb.

Each test replays the library calls of one tutorial on a tiny synthetic
dataset so the whole pipeline runs on CPU in seconds. The tests pin the
public contracts the tutorials rely on (function signatures, obs/obsm/uns
keys, saved files) so a refactor that breaks a tutorial fails here first.

Tutorial -> test map
  feature_pert_encoder_demo / train_pertTF_with__lochNESS  -> test_identity_training_tutorial
  train (pert objective, as the perturb-5k checkpoints)     -> pert_checkpoint fixture, test_pert_training_tutorial
  TUTORIAL_50gene_eval / INFERENCE.md                       -> test_hf_round_trip_and_eval_testdata, test_predict_perturbations_tutorial
  lora_finetuning_tutorial                                   -> test_lora_cls_tutorial, test_lora_pert_tutorial
  virtual_pooled_screen                                      -> test_virtual_screen_tutorial
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

anndata = pytest.importorskip("anndata")
pytest.importorskip("wandb")  # config_gen / train_function import wandb at module level
from omegaconf import OmegaConf

from perttf.model.config_gen import generate_config
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.train_data_gen import produce_training_datasets
from perttf.model.train_function import eval_testdata, wrapper_train
from perttf.utils.pert_metrics import compute_metrics_from_anndata

N_GENES = 60
CELLS_PER_GROUP = 30
GENOTYPES = ["WT", "A", "B", "C"]
CELLTYPES = ["T1", "T2"]
DEVICE = torch.device("cpu")


def make_adata(seed=0):
    """Log-normalized toy Perturb-seq: 4 genotypes x 2 cell types x 30 cells."""
    rng = np.random.default_rng(seed)
    genotype = np.repeat(GENOTYPES, CELLS_PER_GROUP * len(CELLTYPES))
    celltype = np.tile(np.repeat(CELLTYPES, CELLS_PER_GROUP), len(GENOTYPES))
    n_cells = genotype.size
    base = rng.gamma(2.0, 1.0, size=N_GENES)
    rate = np.tile(base, (n_cells, 1))
    for i, g in enumerate(GENOTYPES[1:]):
        rate[genotype == g, i * 10:(i + 1) * 10] *= 3.0  # genotype-specific signature
    rate[celltype == "T2", 40:50] *= 2.5  # cell-type signature
    counts = rng.poisson(rate).astype(np.float32)
    normed = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    adata = anndata.AnnData(
        X=normed,
        obs=pd.DataFrame({"genotype": genotype, "celltype": celltype},
                         index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(N_GENES)]),
    )
    adata.layers["X_binned"] = normed.copy()
    adata.var["highly_variable"] = np.arange(N_GENES) < 40
    return adata


def split_indices(adata, seed=0, valid_per_group=6):
    """Explicit split with every (celltype, genotype) group in both halves."""
    rng = np.random.default_rng(seed)
    valid = []
    for _, idx in adata.obs.groupby(["celltype", "genotype"], observed=True).indices.items():
        valid.extend(rng.choice(idx, valid_per_group, replace=False))
    valid = np.sort(np.array(valid))
    train = np.setdiff1d(np.arange(adata.n_obs), valid)
    return train, valid


def tutorial_hyperparameters(**overrides):
    """The make_config() dict of feature_pert_encoder_demo.ipynb, shrunk for CPU."""
    hp = dict(
        seed=0, dataset_name="tutorial_smoke", do_train=True,
        GEPC=True, ecs_thres=0.0, ecs_weight=1.0, dab_weight=0.0,
        this_weight=1.0, next_weight=0.0, next_cell_pred_type="identity",
        cell_type_classifier=True, cell_type_classifier_weight=1.0,
        genotype_classifier=True, perturbation_classifier_weight=1.0,
        ps_weight=0.0, perturbation_input=False, CCE=False, mask_ratio=0.15,
        epochs=1, n_bins=0, lr=1e-3, batch_size=32,
        layer_size=16, nlayers=1, nhead=2, dropout=0.0,
        schedule_ratio=0.97, schedule_interval=1, log_interval=1000,
        fast_transformer=False, pre_norm=False, amp=False, do_sample_in_train=False,
        ADV=False, adv_weight=0, adv_E_delay_epochs=0, adv_D_delay_epochs=0, lr_ADV=1e-3,
        DSBN=False, per_seq_batch_sample=False, use_batch_label=False, explicit_zero_prob=False,
        n_hvg=N_GENES, mask_value=-1, pad_value=-2, pad_token="<pad>", cls_token="<cls>",
        sampling_mode="simple", non_hvg_size=0, fix_nonzero_prop=True, nonzero_prop=0.0,
        hvg_col="highly_variable", mvc_masked_train=False, distribution=None, use_ot=False,
        save_eval_interval=1, eval_expr_interval=2,
    )
    hp.update(overrides)
    return hp


def pert_hyperparameters(**overrides):
    """Perturbation objective, as the published perttf-tiny-perturb checkpoints were trained."""
    return tutorial_hyperparameters(
        next_cell_pred_type="pert", this_weight=0.0, next_weight=1.0, mask_ratio=0.0,
        perturbation_metric_min_cells=5, perturbation_checkpoint_metric="mvc_next",
        **overrides,
    )


def build_model(data_gen, config):
    """The PerturbationTFModel constructor call shared by the training notebooks."""
    vocab = data_gen["vocab"]
    return PerturbationTFModel(
        data_gen["n_perturb"], 3, len(data_gen["ps_names"]), len(vocab),
        config.layer_size, config.nhead, config.layer_size, config.nlayers,
        vocab=vocab, dropout=config.dropout, pad_token=config.pad_token, pad_value=config.pad_value,
        do_mvc=config.GEPC, do_dab=config.dab_weight > 0, use_batch_labels=config.use_batch_label,
        num_batch_labels=data_gen["num_batch_types"], domain_spec_batchnorm=config.DSBN,
        n_input_bins=config.n_bins, ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob, use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm, n_cls=data_gen["n_cls"], nlayers_cls=3,
    )


def train_tutorial_model(adata, hp, save_dir):
    config, _ = generate_config(hp, wandb_mode="disabled")
    train_idx, valid_idx = split_indices(adata)
    data_gen = produce_training_datasets(
        adata, config, next_cell_pred=config.next_cell_pred_type,
        train_indices=train_idx, valid_indices=valid_idx,
    )
    model = build_model(data_gen, config)
    best = wrapper_train(model, config, data_gen, save_dir=save_dir, device=DEVICE, eval_adata_dict={})
    return best, config, data_gen


def to_hf_checkpoint(model, config, data_gen, save_dir):
    """Package a trained PerturbationTFModel the way the HF checkpoints on the Hub are laid out."""
    hf = HFPerturbationTFModel(
        n_pert=data_gen["n_perturb"], nlayers_pert=3, n_ps=len(data_gen["ps_names"]),
        d_model=config.layer_size, nhead=config.nhead, d_hid=config.layer_size, nlayers=config.nlayers,
        nlayers_cls=3, n_cls=data_gen["n_cls"], vocab=data_gen["vocab"], dropout=config.dropout,
        pad_token=config.pad_token, pad_value=config.pad_value, do_mvc=config.GEPC,
        num_batch_labels=data_gen["num_batch_types"], n_bins=config.n_bins,
        ecs_threshold=config.ecs_thres, use_fast_transformer=False,
        genotype_to_index=data_gen["genotype_to_index"], cell_type_to_index=data_gen["cell_type_to_index"],
        ps_names=data_gen["ps_names"],
    )
    hf.load_state_dict(model.state_dict(), strict=True)
    hf.save_pretrained(str(save_dir), training_config=config.as_dict())
    return hf


@pytest.fixture(scope="module", autouse=True)
def single_process_loaders():
    """The data manager hardcodes num_workers=8; on CPU test runners every worker re-imports
    torch and scanpy (about 45 s per training run on macOS), so run the loaders in-process."""
    from torch.utils.data import DataLoader
    from perttf.utils import pert_data_loader

    mp = pytest.MonkeyPatch()
    mp.setattr(pert_data_loader, "DataLoader",
               lambda *args, **kwargs: DataLoader(*args, **{**kwargs, "num_workers": 0}))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def adata():
    return make_adata()


@pytest.fixture(scope="module")
def pert_checkpoint(adata, tmp_path_factory):
    """One pert-objective training run reused by the inference, LoRA and screen tests."""
    save_dir = tmp_path_factory.mktemp("pert_run")
    best, config, data_gen = train_tutorial_model(adata, pert_hyperparameters(), save_dir)
    hf_dir = tmp_path_factory.mktemp("pert_hf")
    to_hf_checkpoint(best, config, data_gen, hf_dir)
    return {"model": best, "config": config, "data_gen": data_gen, "save_dir": save_dir, "hf_dir": hf_dir}


# --- train_pertTF_with__lochNESS.ipynb / feature_pert_encoder_demo.ipynb -------------------

def test_identity_training_tutorial(adata, tmp_path):
    best, config, data_gen = train_tutorial_model(adata, tutorial_hyperparameters(), tmp_path)
    for name in ("vocab.pt", "running_parameters.pt", "config.json", "best_model.pt"):
        assert (tmp_path / name).exists(), name
    running = torch.load(tmp_path / "running_parameters.pt", weights_only=False)
    assert running["genotype_to_index"] == data_gen["genotype_to_index"]
    assert set(json.load(open(tmp_path / "config.json"))) >= {"next_cell_pred_type", "layer_size"}

    adata_eva = eval_testdata(best, data_gen["adata_sorted"], data_gen["gene_ids"],
                              train_data_dict=data_gen, config=config, device=DEVICE)
    assert adata_eva.obsm["X_scGPT"].shape == (data_gen["adata_sorted"].n_obs, config.layer_size)
    assert adata_eva.obsm["X_scGPT_next"].shape == adata_eva.obsm["X_scGPT"].shape
    assert set(adata_eva.obs["predicted_genotype"]) <= set(data_gen["genotype_to_index"])
    assert set(adata_eva.obs["predicted_celltype"]) <= set(data_gen["cell_type_to_index"])
    assert np.allclose(adata_eva.obsm["genotype_pred_probs"].sum(1), 1.0, atol=1e-5)


# --- perturbation objective (perttf-tiny-perturb-5k-nb style training) ---------------------

def test_pert_training_tutorial(pert_checkpoint):
    data_gen = pert_checkpoint["data_gen"]
    assert data_gen["perturbation_validation"] is True
    assert data_gen["pert_valid_pairs"].shape[1] == 2
    assert data_gen["split_check"]["mode"] == "pert"
    assert (pert_checkpoint["save_dir"] / "best_model.pt").exists()


# --- TUTORIAL_50gene_eval.ipynb / INFERENCE.md ----------------------------------------------

def test_hf_round_trip_and_eval_testdata(adata, pert_checkpoint):
    model = HFPerturbationTFModel.from_pretrained(str(pert_checkpoint["hf_dir"]))
    model.eval()
    assert len(model.vocab) == len(pert_checkpoint["data_gen"]["vocab"])
    assert model.genotype_to_index == pert_checkpoint["data_gen"]["genotype_to_index"]
    assert model.training_config.next_cell_pred_type == "pert"

    # the 50-gene tutorial: control cells asked to become each perturbation
    adata_test = adata[adata.obs["genotype"] == "WT"].copy()
    rng = np.random.default_rng(0)
    adata_test.obs["genotype_next"] = rng.choice(["A", "B", "C"], adata_test.n_obs)
    config = OmegaConf.create(dict(model.training_config))
    config.batch_size = 32
    test_adata = eval_testdata(
        model, adata_t=adata_test, gene_ids=model.vocab(adata_test.var_names.tolist()),
        train_data_dict={"genotype_to_index": model.genotype_to_index,
                         "cell_type_to_index": model.cell_type_to_index, "vocab": model.vocab},
        config=config, include_types=["cls"], predict_expr=True, mvc_full_expr=True,
        sizefactor=True, sample=True, device=DEVICE,
    )
    assert test_adata.obsm["X_scGPT_next"].shape == (adata_test.n_obs, model.d_model)
    assert test_adata.obsm["mvc_next_expr"].shape == adata_test.shape
    assert np.isfinite(test_adata.obsm["mvc_next_expr"]).all()

    # the lora tutorial's metric call on the same outputs
    adata_real = adata[adata.obs["genotype"].isin(["WT", "A", "B", "C"])].copy()
    metrics = compute_metrics_from_anndata(
        adata_real, test_adata, real_layer="X_binned", real_scale="log1p",
        prediction_scale=model.prediction_scale, min_cells=5,
    )
    assert metrics["aggregate"]["n_evaluated_groups"] > 0


def test_predict_perturbations_tutorial(adata, pert_checkpoint):
    model = HFPerturbationTFModel.from_pretrained(str(pert_checkpoint["hf_dir"]))
    query = adata[adata.obs["genotype"] == "WT"].copy()
    query.obs["genotype_next"] = "A"
    result = model.predict_perturbations(query, prediction_mode="mean", device=DEVICE)
    assert result.obs_names.equals(query.obs_names)
    assert result.obsm["mvc_next_expr"].shape == query.shape
    assert result.uns["perttf_prediction"]["native_scale"] == model.prediction_scale
    with pytest.raises(ValueError, match="absent from genotype_to_index"):
        bad = query.copy()
        bad.obs["genotype_next"] = "NOT_A_GENE"
        model.predict_perturbations(bad, device=DEVICE)


# --- lora_finetuning_tutorial.ipynb ----------------------------------------------------------

def test_lora_cls_tutorial(adata, pert_checkpoint, tmp_path):
    pytest.importorskip("peft")
    model = HFPerturbationTFModel.from_pretrained(str(pert_checkpoint["hf_dir"]))
    lora_config = model.build_lora_config(r=4, lora_alpha=8, lora_dropout=0.0)
    train_idx, valid_idx = split_indices(adata, seed=1)
    adapter_dir = tmp_path / "cls_adapter"
    peft_model = model.run_lora_cls_train(
        adata=adata, cls_col="genotype", cls_task="classification",
        train_indices=train_idx, valid_indices=valid_idx, epochs=1, batch_size=32,
        lora_config=lora_config, save_dir=str(adapter_dir), seed=0, device=DEVICE,
    )
    assert set(peft_model.get_base_model().cls_info["cls_to_index"]) == set(GENOTYPES)
    assert (adapter_dir / "lora_heads.json").exists()

    loaded = HFPerturbationTFModel.load_lora_adapter(str(pert_checkpoint["hf_dir"]), str(adapter_dir))
    loaded.eval()
    predicted = loaded.predict_cls(adata)
    assert predicted.index.equals(adata.obs_names)
    assert set(predicted) <= set(GENOTYPES)


def test_lora_pert_tutorial(adata, pert_checkpoint, tmp_path):
    pytest.importorskip("peft")
    model = HFPerturbationTFModel.from_pretrained(str(pert_checkpoint["hf_dir"]))
    # The tutorial's unseen-context split: train on T2, validate on every T1 group. The
    # entry point scores epochs on pearson_delta over validation groups of >= 30 cells.
    is_t1 = (adata.obs["celltype"] == "T1").to_numpy()
    train_idx, valid_idx = np.where(~is_t1)[0], np.where(is_t1)[0]
    adapter_dir = tmp_path / "pert_adapter"
    peft_model = model.run_lora_pert_train(
        adata, train_idx, valid_idx, epochs=1, batch_size=32,
        lora_config=model.build_lora_config(r=4, lora_alpha=8, lora_dropout=0.0),
        save_dir=str(adapter_dir), seed=0, device=DEVICE,
    )
    assert (adapter_dir / "lora_heads.json").exists()
    bm = peft_model.get_base_model()
    query = adata[adata.obs["genotype"] == "WT"].copy()
    query.obs["genotype_next"] = "B"
    peft_model.eval()
    res = eval_testdata(
        peft_model, query, None,
        train_data_dict={"genotype_to_index": bm.genotype_to_index, "vocab": bm.vocab,
                         "cell_type_to_index": bm.cell_type_to_index},
        config=bm.training_config, mvc_full_expr=True, predict_expr=True, device=DEVICE,
    )
    assert res.obsm["mvc_next_expr"].shape == query.shape


# --- virtual_pooled_screen.ipynb ------------------------------------------------------------

def test_virtual_screen_tutorial(adata, pert_checkpoint):
    from perttf.utils.virtual_screen import virtual_screen_src2dest

    model, config = pert_checkpoint["model"], pert_checkpoint["config"]
    running_parameters = torch.load(pert_checkpoint["save_dir"] / "running_parameters.pt", weights_only=False)
    vocab = torch.load(pert_checkpoint["save_dir"] / "vocab.pt", weights_only=False)
    is_t1 = adata.obs["celltype"] == "T1"
    adata_src = adata[is_t1 & (adata.obs["genotype"] == "WT")].copy()
    adata_des = adata[is_t1 & (adata.obs["genotype"] == "A")].copy()
    for a in (adata_src, adata_des):
        a.obs["genotype_next"] = a.obs["genotype"].astype("category")
    results = virtual_screen_src2dest(
        adata_src, adata_des, model, vocab, config, running_parameters,
        screen_genes=["A", "B", "NOT_A_GENE"], device=DEVICE,
        n_expands_per_epoch=1, n_epoch=1, wt_pred_next_label="A",
    )
    ranked = results["sim_ranked"]
    assert set(ranked.index) == {"A", "B"}
    assert {"mean", "median", "percentile_90", "rank"} <= set(ranked.columns)
    assert "X_scGPT_next" in results["adata_last_eva"].obsm
```

- [ ] **Step 4: Run the baseline**

Run: `pytest tests/ -v`
Expected: the two existing tests plus 7 tutorial tests PASS; nothing skipped (if `peft` or `wandb` are missing the LoRA tests or the whole file skip, which means Step 2 was not installed: `pip install -e .[test]`).

- [ ] **Step 5: Commit**

```bash
git add perttf/model/config_gen.py pyproject.toml tests/test_tutorial_smoke.py
git commit -m "test: tutorial-derived CPU smoke tests; drop removed wandb start_method setting"
```

---

### Task 1: Label parsing and validation helpers

**Files:**
- Create: `perttf/utils/pert_labels.py`
- Test: `tests/test_pert_labels.py`

**Interfaces:**
- Produces (all in `perttf/utils/pert_labels.py`):
  - `split_perturbation_label(label: str, delimiter: Optional[str], control: str = "WT") -> tuple[str, ...]` — with `delimiter=None` returns `(label,)`; otherwise unique components in first-occurrence order with `control` removed unless it is the only component; raises `ValueError` on empty or whitespace-only components.
  - `canonical_label(label: str, delimiter: Optional[str], control: str = "WT") -> str` — sorted components joined; with `delimiter=None` returns `label` unchanged.
  - `unique_perturbations(labels: Iterable[str], delimiter: Optional[str], control: str = "WT") -> list[str]` — singles in first-occurrence order, control included once when present in any label (as whole label or component).
  - `labels_in_mapping(labels: Iterable[str], mapping: Mapping[str, int], delimiter: Optional[str], control: str = "WT") -> np.ndarray` — bool per label, True only when every component is a key.
  - `labels_to_padded_array(label: str, mapping: Mapping[str, int], delimiter: Optional[str], width: int, control: str = "WT") -> np.ndarray` — int64 array of shape `(width,)`, components' indices then `-1`; raises `ValueError` if more than `width` components or an unknown component.
  - `max_set_size(labels: Iterable[str], delimiter: Optional[str], control: str = "WT") -> int` — largest component count, at least 1.
  - `validate_perturbation_labels(labels: Iterable[str], delimiter: Optional[str], control: str = "WT", existing_mapping: Optional[Mapping[str, int]] = None, max_perts_per_cell: Optional[int] = None) -> None` — implements spec §3.8a rules 1–3 and 5: legacy-mode `warnings.warn` (once per call) if any label contains `"+"`; multi mode raises `ValueError` listing malformed labels; raises `ValueError` listing `existing_mapping` keys containing the delimiter; raises `ValueError` when a label exceeds `max_perts_per_cell`.
  - `canonicalize_perturbation_columns(adata, delimiter: Optional[str], control: str = "WT", columns: Sequence[str] = ("genotype", "genotype_next")) -> int` — rewrites each present column in place as an object-dtype string column of canonical labels, returns the number of changed cells, prints one line when > 0. With `delimiter=None` touches nothing and returns 0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pert_labels.py
import numpy as np, pytest, anndata as ad, pandas as pd
from perttf.utils.pert_labels import (split_perturbation_label, canonical_label, unique_perturbations,
    labels_in_mapping, labels_to_padded_array, max_set_size, validate_perturbation_labels,
    canonicalize_perturbation_columns)

def test_split_legacy_is_identity():
    assert split_perturbation_label("A+B", None) == ("A+B",)

def test_split_drops_control_and_duplicates_keeps_order():
    assert split_perturbation_label("B+WT+A+B", "+") == ("B", "A")
    assert split_perturbation_label("WT", "+") == ("WT",)
    assert split_perturbation_label("WT+WT", "+") == ("WT",)

@pytest.mark.parametrize("bad", ["A++B", "+A", "A+", "+", "A + B", " +B"])
def test_split_rejects_malformed(bad):
    with pytest.raises(ValueError):
        split_perturbation_label(bad, "+")

def test_canonical_sorted():
    assert canonical_label("B+A", "+") == "A+B"
    assert canonical_label("B+A", None) == "B+A"

def test_unique_perturbations_singles_first_occurrence():
    assert unique_perturbations(["WT", "B+A", "A", "C+WT"], "+") == ["WT", "B", "A", "C"]
    assert unique_perturbations(["WT", "B+A"], None) == ["WT", "B+A"]

def test_labels_in_mapping_requires_all_components():
    m = {"WT": 0, "A": 1, "B": 2}
    assert labels_in_mapping(["A+B", "A+C", "WT"], m, "+").tolist() == [True, False, True]

def test_padded_array():
    m = {"WT": 0, "A": 1, "B": 2}
    assert labels_to_padded_array("B+A", m, "+", width=3).tolist() == [2, 1, -1]
    assert labels_to_padded_array("WT", m, "+", width=2).tolist() == [0, -1]
    with pytest.raises(ValueError):
        labels_to_padded_array("A+B+WT+A", m, "+", width=1)

def test_max_set_size():
    assert max_set_size(["WT", "A+B+C"], "+") == 3
    assert max_set_size(["A+B+C"], None) == 1

def test_validate_legacy_warns_on_plus():
    with pytest.warns(UserWarning):
        validate_perturbation_labels(["A+B"], None)

def test_validate_rejects_mapping_keys_with_delimiter():
    with pytest.raises(ValueError, match="A\\+B"):
        validate_perturbation_labels(["A"], "+", existing_mapping={"A+B": 0, "WT": 1})

def test_validate_max_perts():
    with pytest.raises(ValueError):
        validate_perturbation_labels(["A+B+C"], "+", max_perts_per_cell=2)
    validate_perturbation_labels(["A+B+C"], "+", max_perts_per_cell=3)

def test_canonicalize_columns_in_place_handles_categorical():
    obs = pd.DataFrame({"genotype": pd.Categorical(["B+A", "WT", "WT+WT"]), "genotype_next": ["A+B", "C+B", "WT"]})
    a = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32), obs=obs)
    changed = canonicalize_perturbation_columns(a, "+")
    assert a.obs["genotype"].tolist() == ["A+B", "WT", "WT"]
    assert a.obs["genotype_next"].tolist() == ["A+B", "B+C", "WT"]
    assert changed == 3
    assert canonicalize_perturbation_columns(a, None) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pert_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: perttf.utils.pert_labels`

- [ ] **Step 3: Implement the functions listed in Interfaces in `perttf/utils/pert_labels.py`**

Malformed check: a component is malformed if it is `""` or `component.strip() != component` or `component.strip() == ""`. `unique_perturbations` includes `control` when any label's split result contains it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pert_labels.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add perttf/utils/pert_labels.py tests/test_pert_labels.py
git commit -m "feat: perturbation label parsing and validation helpers for delimited sets"
```

---

### Task 2: Set encoding helper

**Files:**
- Modify: `perttf/model/pert_encoder.py` (append)
- Test: `tests/test_encode_perturbation_set.py`

**Interfaces:**
- Produces: `encode_perturbation_set(encoder: nn.Module, labels: Tensor, aggregation: str = "sum") -> Tensor` in `perttf/model/pert_encoder.py`. 1-D labels → `encoder(labels)` unchanged. 2-D `(batch, k)` with `-1` pads → `(batch, pert_dim)`; `aggregation in {"sum", "mean", "sqrt"}` else `ValueError`; `sqrt` divides the masked sum by `sqrt(n_valid)` (with `n_valid` clamped to ≥ 1); `mean` divides by `n_valid` clamped to ≥ 1. Also accepts already-embedded input via a second function `reduce_perturbation_embeddings(emb: Tensor, labels: Tensor, aggregation: str) -> Tensor` taking `(batch, k, pert_dim)` and the same `(batch, k)` labels; `encode_perturbation_set` is implemented on top of it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_encode_perturbation_set.py
import math, torch, pytest
from perttf.model.modules import PertLabelEncoder, FeaturePertEncoder
from perttf.model.pert_encoder import UnifiedPertEncoder, encode_perturbation_set, reduce_perturbation_embeddings

D, N = 16, 6
MAPPING = {"WT": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

def encoders():
    torch.manual_seed(0)
    yield PertLabelEncoder(N, D)
    yield FeaturePertEncoder(torch.randn(N, 8), D)
    yield UnifiedPertEncoder({"src": (["A", "B"], torch.randn(2, 8).numpy())}, MAPPING, D)

@pytest.mark.parametrize("agg", ["sum", "mean", "sqrt"])
def test_width_one_equals_scalar(agg):
    for enc in encoders():
        enc.eval()
        lab = torch.tensor([1, 2, 0])
        a = encode_perturbation_set(enc, lab, agg)
        b = encode_perturbation_set(enc, lab.unsqueeze(1), agg)
        assert torch.allclose(a, b, atol=1e-6)

def test_pad_contributes_zero_and_sum_is_additive():
    enc = PertLabelEncoder(N, D).eval()
    single = encode_perturbation_set(enc, torch.tensor([1, 2]), "sum")
    pair = encode_perturbation_set(enc, torch.tensor([[1, 2, -1, -1]]), "sum")
    assert torch.allclose(pair[0], single[0] + single[1], atol=1e-6)
    padded = encode_perturbation_set(enc, torch.tensor([[1, -1, -1]]), "sum")
    assert torch.allclose(padded[0], single[0], atol=1e-6)

def test_mean_and_sqrt_scaling():
    enc = PertLabelEncoder(N, D).eval()
    labels = torch.tensor([[1, 2, 3, 4]])
    s = encode_perturbation_set(enc, labels, "sum")
    assert torch.allclose(encode_perturbation_set(enc, labels, "mean"), s / 4, atol=1e-6)
    assert torch.allclose(encode_perturbation_set(enc, labels, "sqrt"), s / math.sqrt(4), atol=1e-6)

def test_unified_never_sees_negative_index():
    enc = list(encoders())[2].eval()
    out = encode_perturbation_set(enc, torch.tensor([[1, 2, -1]]), "sum")
    assert out.shape == (1, D) and torch.isfinite(out).all()

def test_bad_aggregation():
    with pytest.raises(ValueError):
        encode_perturbation_set(PertLabelEncoder(N, D), torch.tensor([[1]]), "max")

def test_reduce_matches_encode():
    enc = PertLabelEncoder(N, D).eval()
    labels = torch.tensor([[1, 2, -1]])
    emb = enc(labels.clamp_min(0))
    assert torch.allclose(reduce_perturbation_embeddings(emb, labels, "sum"),
                          encode_perturbation_set(enc, labels, "sum"), atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_encode_perturbation_set.py -v`
Expected: FAIL with `ImportError: cannot import name 'encode_perturbation_set'`

- [ ] **Step 3: Implement `reduce_perturbation_embeddings` and `encode_perturbation_set` in `perttf/model/pert_encoder.py`**

Mask with `(labels >= 0).unsqueeze(-1).to(emb.dtype)`; the 2-D branch calls `encoder(labels.clamp_min(0))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_encode_perturbation_set.py tests/test_pertTF_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add perttf/model/pert_encoder.py tests/test_encode_perturbation_set.py
git commit -m "feat: masked set aggregation over perturbation embeddings"
```

---

### Task 3: Model and HF constructor wiring

**Files:**
- Modify: `perttf/model/pertTF.py` (constructor kwargs near line 36; call sites at lines ~153, ~250, ~297)
- Modify: `perttf/model/hf.py` (constructor signature near line 71; parent call near line 176; `from_pretrained` config flow)
- Test: `tests/test_multi_pert_model.py`

**Interfaces:**
- Consumes: `encode_perturbation_set`, `reduce_perturbation_embeddings` (Task 2).
- Produces: `PerturbationTFModel(..., pert_aggregation: str = "sum")` kwarg (popped from `kwargs` like `pert_dim`) stored as `self.pert_aggregation`; `forward` accepts `pert_labels` / `pert_labels_next` of shape `(batch,)` or `(batch, k)`; `pert_embeddings_next` of shape `(batch, pert_dim)` or `(batch, k, pert_dim)` (the latter reduced with `reduce_perturbation_embeddings(pert_embeddings_next, pert_labels_next, self.pert_aggregation)`). `HFPerturbationTFModel.__init__(..., pert_aggregation: str = "sum", ...)` named parameter forwarded to the parent.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multi_pert_model.py
import torch, pytest
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.hf import HFPerturbationTFModel
from tests.test_pertTF_model import build_model, N_PERT, NTOKENS, D_MODEL

def batch(B=4, L=12, k=10):
    torch.manual_seed(0)
    src = torch.randint(1, NTOKENS, (B, L)); values = torch.rand(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    labels = torch.randint(0, N_PERT, (B, k)); labels[:, 5:] = -1; labels[0, 1:] = -1
    return src, values, mask, labels

@pytest.mark.parametrize("agg", ["sum", "mean", "sqrt"])
def test_forward_width_ten(agg):
    model = build_model(pert_aggregation=agg)
    src, values, mask, labels = batch()
    with torch.no_grad():
        out = model(src, values, mask, pert_labels=labels, pert_labels_next=labels, CLS=True, MVC=True, PERTPRED=True, PSPRED=True)
    assert out["cell_emb_next"].shape == (4, D_MODEL) and torch.isfinite(out["mvc_output_next"]["pred"]).all()

def test_width_one_matches_scalar_forward():
    model = build_model()
    src, values, mask, labels = batch(k=1); labels = labels.clamp_min(0)
    with torch.no_grad():
        a = model(src, values, mask, pert_labels=labels[:, 0], pert_labels_next=labels[:, 0], MVC=True)
        b = model(src, values, mask, pert_labels=labels, pert_labels_next=labels, MVC=True)
    assert torch.allclose(a["cell_emb_next"], b["cell_emb_next"], atol=1e-6)

def test_pert_embeddings_next_3d_is_reduced():
    model = build_model()
    src, values, mask, labels = batch(k=3)
    emb = torch.randn(4, 3, D_MODEL)
    with torch.no_grad():
        out = model(src, values, mask, pert_labels_next=labels, pert_embeddings_next=emb)
    assert out["cell_emb_next"].shape == (4, D_MODEL)

def test_hf_roundtrip_restores_aggregation(tmp_path):
    vocab = {"<pad>": 0, **{f"g{i}": i for i in range(1, NTOKENS)}}
    from perttf.utils.custom_tokenizer import SimpleVocab
    m = HFPerturbationTFModel(vocab=SimpleVocab.from_dict(vocab), genotype_to_index={"WT": 0, "A": 1},
                              cell_type_to_index={"c": 0}, d_model=16, nhead=2, d_hid=16, nlayers=1,
                              n_ps=0, pert_aggregation="mean")
    m.save_pretrained(tmp_path)
    r = HFPerturbationTFModel.from_pretrained(str(tmp_path))
    assert r.pert_aggregation == "mean"
    assert "pert_aggregation" in r._hub_mixin_config
```

(If `SimpleVocab.from_dict` is named differently, use the constructor the HF test in `hf.py` already relies on; check `perttf/utils/custom_tokenizer.py`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pip install -e .[test] && pytest tests/test_multi_pert_model.py -v`
Expected: FAIL (`TypeError: unexpected keyword 'pert_aggregation'` or shape error on 2-D labels)

- [ ] **Step 3: Implement in `perttf/model/pertTF.py`**

Pop `pert_aggregation` in `__init__` next to `pert_dim`; replace the three `self.pert_encoder(x)` calls with `encode_perturbation_set(self.pert_encoder, x, self.pert_aggregation)`; in `forward`, when `pert_embeddings_next.dim() == 3` reduce it with `reduce_perturbation_embeddings` before the existing `(batch, pert_dim)` shape check.

- [ ] **Step 4: Implement in `perttf/model/hf.py`**

Add `pert_aggregation: str = "sum"` as a named parameter after `pert_dim`, pass `pert_aggregation=pert_aggregation` to `super().__init__`. Add `"omegaconf"` and `"safetensors"` to the `test` extra.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS (including the two legacy tests)

- [ ] **Step 6: Commit**

```bash
git add perttf/model/pertTF.py perttf/model/hf.py tests/test_multi_pert_model.py
git commit -m "feat: accept perturbation sets in PerturbationTFModel and persist pert_aggregation"
```

---

### Task 4: Multi-label classification loss and contrastive overlap mask

**Files:**
- Modify: `perttf/custom_loss.py` (`cce_loss` at line ~207; append new functions)
- Test: `tests/test_multi_pert_loss.py`

**Interfaces:**
- Produces in `perttf/custom_loss.py`:
  - `perturbation_classification_loss(logits: Tensor, labels: Tensor, pos_weight: Optional[float] = None) -> Tensor` — 1-D labels → `F.cross_entropy(logits, labels)`; 2-D → `F.binary_cross_entropy_with_logits(logits, multi_hot, pos_weight=...)` where `multi_hot = multi_hot_targets(labels, logits.shape[1])`.
  - `multi_hot_targets(labels: Tensor, n_classes: int) -> Tensor` — float `(batch, n_classes)`, ignores `-1`.
  - `perturbation_overlap_mask(labels_a: Tensor, labels_b: Tensor, n_classes: int) -> Tensor` — float `(len_a, len_b)`, 1 where the two sets share at least one class (`multi_hot(a) @ multi_hot(b).T > 0`).
  - `cce_loss(contrastive_dict, input_labels, pert_labels, logit_norm=False, pert_mask: Optional[Tensor] = None)` — when `pert_mask` is given, `SUPCON_loss` is called with `mask=pert_mask` and `labels=None`; `pert_mask` must be `(2*batch, 2*batch)` matching the concatenation order of `contrastive_dict` (orig then next, i.e. the same order as `lab_list`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multi_pert_loss.py
import torch, pytest, torch.nn.functional as F
from perttf.custom_loss import perturbation_classification_loss, multi_hot_targets, perturbation_overlap_mask, cce_loss

def test_ce_for_scalar_labels():
    logits = torch.randn(3, 5); labels = torch.tensor([0, 4, 2])
    assert torch.allclose(perturbation_classification_loss(logits, labels), F.cross_entropy(logits, labels))

def test_multi_hot_ignores_pad():
    t = multi_hot_targets(torch.tensor([[1, 3, -1], [0, -1, -1]]), 5)
    assert t.tolist() == [[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]]

def test_bce_for_2d_labels():
    logits = torch.randn(2, 5); labels = torch.tensor([[1, 3, -1], [0, -1, -1]])
    expected = F.binary_cross_entropy_with_logits(logits, multi_hot_targets(labels, 5))
    assert torch.allclose(perturbation_classification_loss(logits, labels), expected)
    assert perturbation_classification_loss(logits, labels, pos_weight=3.0) != expected

def test_overlap_mask():
    a = torch.tensor([[1, 2, -1], [3, -1, -1]]); b = torch.tensor([[2, -1, -1], [0, 4, -1], [3, 1, -1]])
    assert perturbation_overlap_mask(a, b, 5).tolist() == [[1, 0, 1], [0, 0, 1]]

def test_cce_loss_accepts_mask():
    d = {"orig_emb0": torch.randn(4, 8), "next_emb0": torch.randn(4, 8)}
    labels = torch.tensor([[1, 2, -1], [2, -1, -1], [3, -1, -1], [0, -1, -1]])
    both = torch.cat([labels, labels])
    mask = perturbation_overlap_mask(both, both, 5)
    loss = cce_loss(d, None, None, pert_mask=mask)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multi_pert_loss.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement the four functions in `perttf/custom_loss.py`**

`multi_hot_targets` uses `scatter_` on `labels.clamp_min(0)` with a `(labels >= 0)` float source. `cce_loss` keeps its existing code path when `pert_mask is None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_multi_pert_loss.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add perttf/custom_loss.py tests/test_multi_pert_loss.py
git commit -m "feat: multi-label perturbation classification loss and overlap contrastive mask"
```

---

### Task 5: Dataset, data manager and mapping builders

**Files:**
- Modify: `perttf/utils/pert_data_loader.py` (`PertTFDataset.__init__` ~lines 95-180, `__getitem__` ~lines 227-331, `PertTFUniDataManager.__init__` and `set_genotype_index` ~line 523)
- Modify: `perttf/model/pert_emb.py` (`build_perturbation_mapping`, lines 67-81)
- Modify: `perttf/model/lora.py` (`label_mappings`, lines 81-98)
- Test: `tests/test_multi_pert_dataset.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces:
  - `PertTFDataset.__init__(..., pert_delimiter: Optional[str] = None, max_perts_per_cell: Optional[int] = None)`; when the kwarg is `None` it falls back to `config.get('pert_delimiter')` / `config.get('max_perts_per_cell')`. Attributes `self.pert_delimiter`, `self.control_value` (already resolved in `_pairing_config["control_value"]`), `self.max_n_pert: int`.
  - In multi mode, `__init__` calls `validate_perturbation_labels` on both columns (with `existing_mapping=genotype_to_index` when supplied) and `canonicalize_perturbation_columns` **before** `self.genotype_values` / `genotype_next_values` are read and before pairing is built; `self.max_n_pert = max(max_set_size(genotype), max_set_size(genotype_next))`.
  - Samples: `perturbation_labels`, `perturbation_labels_next` are `np.int64` arrays of shape `(max_n_pert,)` in multi mode (scalars in legacy mode); new key `pert_name_next: str` (canonical condition label) emitted in both modes for training samples and when `genotype_next` exists for prediction-only samples; new key `pert_name: str` likewise.
  - Lochness branch: `pert_label_next` built with `labels_to_padded_array(gene, mapping, delimiter, self.max_n_pert)` in multi mode.
  - `PertTFUniDataManager.set_genotype_index` builds the default mapping with `unique_perturbations(self.adata.obs['genotype'], self.pert_delimiter, control)` and validates; `PertTFUniDataManager.__init__` reads `pert_delimiter` / `max_perts_per_cell` from config and canonicalises the columns once.
  - `build_perturbation_mapping(perturbations, sources, control_label="WT", delimiter: Optional[str] = None)` splits dataset labels with `unique_perturbations`.
  - `lora.label_mappings(model, adata, require_known, delimiter: Optional[str] = None)` computes `missing` with `unique_perturbations` for `"genotype"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multi_pert_dataset.py
import numpy as np, pandas as pd, anndata as ad, pytest
from perttf.utils.pert_data_loader import PertTFDataset, PertTFUniDataManager
from perttf.model.pert_emb import build_perturbation_mapping

def tiny(labels, n_genes=5):
    n = len(labels)
    a = ad.AnnData(X=np.random.default_rng(0).poisson(1.0, (n, n_genes)).astype(np.float32),
                   obs=pd.DataFrame({"genotype": pd.Categorical(labels), "celltype": ["c"] * n}, index=[f"cell{i}" for i in range(n)]),
                   var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))
    a.layers["X_binned"] = a.X.copy()
    return a

LABELS = ["WT", "WT", "A", "A+B", "B+A", "WT+WT", "C+B+A"]

def test_legacy_mode_unchanged():
    ds = PertTFDataset(tiny(LABELS), next_cell_pred="pert")
    assert ds.max_n_pert == 1
    s = ds[0]
    assert np.isscalar(s["perturbation_labels"]) or np.ndim(s["perturbation_labels"]) == 0
    assert set(ds.genotype_to_index) == {"WT", "A", "A+B", "B+A", "WT+WT", "C+B+A"}

def test_multi_mode_mapping_width_and_padding():
    a = tiny(LABELS)
    ds = PertTFDataset(a, next_cell_pred="pert", pert_delimiter="+")
    assert ds.genotype_to_index == {"WT": 0, "A": 1, "B": 2, "C": 3}
    assert ds.max_n_pert == 3
    assert a.obs["genotype"].tolist()[3:6] == ["A+B", "A+B", "WT"]   # canonicalised in place, categorical input ok
    s = ds[2]  # cell "A" is perturbed -> identity pair in source-anchored pairing
    assert s["perturbation_labels"].shape == (3,) and s["perturbation_labels"].tolist() == [1, -1, -1]
    assert s["pert_name_next"] == "A"
    assert s["perturbation_labels_next"].dtype == np.int64

def test_multi_mode_control_pairs_to_a_set():
    a = tiny(LABELS)
    ds = PertTFDataset(a, next_cell_pred="pert", pert_delimiter="+")
    names = {ds[i]["pert_name_next"] for i in range(len(ds)) for _ in range(5)}
    assert names <= {"WT", "A", "A+B", "A+B+C"}

def test_wt_plus_wt_is_control_for_pairing():
    a = tiny(["WT+WT", "A"])
    ds = PertTFDataset(a, next_cell_pred="pert", pert_delimiter="+")
    assert ds[0]["pert_name_next"] == "A" and ds[0]["perturbation_labels"].tolist()[0] == ds.genotype_to_index["WT"]

def test_max_perts_per_cell_enforced():
    with pytest.raises(ValueError):
        PertTFDataset(tiny(LABELS), pert_delimiter="+", max_perts_per_cell=2)

def test_existing_mapping_with_delimiter_rejected():
    with pytest.raises(ValueError):
        PertTFDataset(tiny(["WT", "A"]), genotype_to_index={"WT": 0, "A+B": 1}, pert_delimiter="+")

def test_build_perturbation_mapping_splits():
    m = build_perturbation_mapping(["B+A", "WT"], {"src": (["Z"], np.zeros((1, 2)))}, delimiter="+")
    assert m == {"B": 0, "A": 1, "WT": 2, "Z": 3}

def test_prediction_only_emits_padded_next():
    a = tiny(["WT", "WT"]); a.obs["genotype_next"] = ["A+B", "WT"]
    ds = PertTFDataset(a, genotype_to_index={"WT": 0, "A": 1, "B": 2}, prediction_only=True, pert_delimiter="+")
    assert ds[0]["perturbation_labels_next"].tolist() == [1, 2] and ds[0]["pert_name_next"] == "A+B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multi_pert_dataset.py -v`
Expected: FAIL with `TypeError: unexpected keyword 'pert_delimiter'`

- [ ] **Step 3: Implement the dataset changes in `perttf/utils/pert_data_loader.py`**

Add a private `self._to_label(label_str)` returning either `mapping[label]` (legacy) or `labels_to_padded_array(...)` (multi), and use it at the four existing lookup sites in `__getitem__` (lines ~259, ~267, ~287, ~294) plus the two lochness sites; prediction-only path keeps `.get(label, 0)` semantics in legacy mode and uses `_to_label` in multi mode.

- [ ] **Step 4: Implement the manager, `build_perturbation_mapping` and `lora.label_mappings` changes**

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add perttf/utils/pert_data_loader.py perttf/model/pert_emb.py perttf/model/lora.py tests/test_multi_pert_dataset.py
git commit -m "feat: parse delimited perturbation sets in datasets and mapping builders"
```

---

### Task 6: Metrics and split check for `perturbation_unit`

**Files:**
- Modify: `perttf/utils/pert_metrics.py` (`group_moments_from_anndata`, lines 27-52; `GroupMoments.update`)
- Modify: `perttf/utils/split_check.py` (`check_train_valid_split`, perturbation branch lines 81-207)
- Test: `tests/test_multi_pert_metrics.py`

**Interfaces:**
- Consumes: `split_perturbation_label`, `unique_perturbations` (Task 1).
- Produces:
  - `expand_perturbation_rows(values: np.ndarray, contexts: np.ndarray, perturbations: np.ndarray, delimiter: Optional[str], unit: str, control: str = "WT") -> tuple[np.ndarray, np.ndarray, np.ndarray]` in `pert_metrics.py` — with `unit="set"` returns inputs unchanged; with `unit="single"` repeats each row once per component (control rows stay as one row labelled `control`).
  - `group_moments_from_anndata(..., perturbation_unit: str = "set", pert_delimiter: Optional[str] = None, control_value: str = "WT")` applies the expansion per chunk.
  - `check_train_valid_split(..., perturbation_unit: str = "set", pert_delimiter: Optional[str] = None)` — in `"single"` mode the perturbation sets used for `combination_types` are built from components; in `"set"` mode a new category `"unseen_combination"` (all components seen in training, set not seen; only populated when `pert_delimiter` is not None); in `"set"` mode with `pert_delimiter` set, if fewer than half of validation `(context, set)` groups have `>= min_cells` cells, emit `warnings.warn` recommending `perturbation_unit="single"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multi_pert_metrics.py
import numpy as np, pandas as pd, anndata as ad, pytest, warnings
from perttf.utils.pert_metrics import expand_perturbation_rows, group_moments_from_anndata
from perttf.utils.split_check import check_train_valid_split

def test_expand_single_unit():
    v = np.arange(6).reshape(3, 2); c = np.array(["x", "x", "x"]); p = np.array(["A+B", "WT", "C"])
    ev, ec, ep = expand_perturbation_rows(v, c, p, "+", "single")
    assert ep.tolist() == ["A", "B", "WT", "C"] and ev.shape == (4, 2) and ev[1].tolist() == [0, 1]
    assert expand_perturbation_rows(v, c, p, "+", "set")[2].tolist() == ["A+B", "WT", "C"]

def test_group_moments_single_unit_counts_cell_in_each_group():
    n = 4
    a = ad.AnnData(X=np.ones((n, 3), dtype=np.float32), obs=pd.DataFrame({"celltype": ["x"] * n, "genotype": ["A+B"] * n}))
    g = group_moments_from_anndata(a, np.arange(n), perturbation_unit="single", pert_delimiter="+").finalize()
    assert set(g) == {("x", "A"), ("x", "B")} and g[("x", "A")].n == n

def _adata():
    labels = ["WT"] * 40 + ["A+B"] * 40 + ["A"] * 40 + ["B+C"] * 2
    obs = pd.DataFrame({"celltype": ["x"] * len(labels), "genotype": labels})
    return ad.AnnData(X=np.zeros((len(labels), 2), dtype=np.float32), obs=obs)

def test_split_check_unseen_combination_and_warning():
    a = _adata(); idx = np.arange(a.n_obs)
    train = np.concatenate([idx[:30], idx[80:110]]); valid = np.concatenate([idx[30:40], idx[40:80], idx[120:122]])
    with pytest.warns(UserWarning, match="single"):
        r = check_train_valid_split(a, train, valid, mode="pert", pert_delimiter="+", perturbation_unit="set", min_cells=30)
    assert ("x", "A+B") in r["combination_types"]["unseen_combination"]

def test_split_check_single_unit_uses_components():
    a = _adata(); idx = np.arange(a.n_obs)
    train = np.concatenate([idx[:30], idx[80:110]]); valid = np.concatenate([idx[30:40], idx[40:80]])
    r = check_train_valid_split(a, train, valid, mode="pert", pert_delimiter="+", perturbation_unit="single", min_cells=30)
    assert ("x", "A") in r["combination_types"]["seen_combination"] and ("x", "B") in r["combination_types"]["unseen_perturbation"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multi_pert_metrics.py -v`
Expected: FAIL with ImportError / TypeError

- [ ] **Step 3: Implement in `perttf/utils/pert_metrics.py` and `perttf/utils/split_check.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add perttf/utils/pert_metrics.py perttf/utils/split_check.py tests/test_multi_pert_metrics.py
git commit -m "feat: per-set or per-single evaluation unit for perturbation metrics and split checks"
```

---

### Task 7: Training and evaluation loop wiring, config defaults

**Files:**
- Modify: `perttf/model/train_function.py` (CCE block lines 155-174; classifier losses lines 252-257; evaluation loss line ~568; condition names line ~588; `perturbation_embeddings` gather line ~528; `eval_testdata` row filter line ~727 and outputs lines ~812-816; `wrapper_train` real-groups call line ~888; `produce_training_datasets` split-check call in `train_data_gen.py` line ~78)
- Modify: `perttf/model/train_data_gen.py` (pass `perturbation_unit`, `pert_delimiter` to `check_train_valid_split`)
- Modify: `perttf/model/config_gen.py` (defaults) and `perttf/model/hf.py` `_init_default_train_config_` (line ~815)
- Test: `tests/test_tutorial_smoke.py` (append the multi-mode section below; nothing above it changes)

**Interfaces:**
- Consumes: Task 4 loss functions, Task 6 `expand_perturbation_rows`, Task 1 `labels_in_mapping`, `canonicalize_perturbation_columns`; Task 0 tutorial helpers and fixtures.
- Produces: config defaults `pert_delimiter=None`, `pert_aggregation="sum"`, `perturbation_unit="set"`, `pert_pred_threshold=0.5`, `pert_bce_pos_weight=None`, `max_perts_per_cell=None` in both `generate_config` and `_init_default_train_config_`. `eval_testdata` in multi mode writes `obsm["X_pert_pred_probs"]` as sigmoid probabilities and `obs["predicted_genotype"]` as the delimiter-joined top classes above `pert_pred_threshold` (at most `max_n_pert`, fallback argmax); skips `obs["genotype_id"]` in multi mode.

- [ ] **Step 1: Wire the training loop**

In `train()`: replace both `criterion_pert(...)` calls with `perturbation_classification_loss(logits, labels, config.get('pert_bce_pos_weight'))`; in the CCE block, when `perturbation_labels.dim() == 2`, build `pert_mask = perturbation_overlap_mask(cat, cat, model.n_pert)` over `torch.cat([perturbation_labels, perturbation_labels_next])` (for `'cell_geno'` multiply it element-wise by the cell-type equality mask over `torch.cat([celltype_labels, celltype_labels_next])`) and pass `pert_mask=` to `cce_loss`; legacy path unchanged.

- [ ] **Step 2: Wire `_run_evaluation_batches`**

Use `perturbation_classification_loss` for `loss_pert`; take `perturbations = np.asarray(batch_data["pert_name_next"])` when the key exists, else `labels_to_names(...)`; before `moments.update`, apply `expand_perturbation_rows(normalized, contexts, perturbations, cfg pert_delimiter, cfg perturbation_unit, control)`; for the `perturbation_embeddings` gather, when labels are 2-D use `perturbation_embeddings[perturbation_labels_next.clamp_min(0)]` (3-D, reduced inside the model by Task 3).

- [ ] **Step 3: Wire `eval_testdata`, `wrapper_train`, `produce_training_datasets`**

Row filter: `adata_t[labels_in_mapping(adata_t.obs["genotype_next"], genotype_to_index, delimiter, control)]` with a logged count of dropped rows; canonicalise columns first in multi mode. Multi-mode outputs as in Interfaces. Pass `perturbation_unit` and `pert_delimiter` from config to `group_moments_from_anndata` (wrapper_train) and `check_train_valid_split` (train_data_gen).

- [ ] **Step 4: Add config defaults in `config_gen.generate_config` and `hf._init_default_train_config_`**

- [ ] **Step 5: Add the multi-mode tutorial tests (training half)**

Edit the two Task 0 helpers so they forward the aggregation choice (both calls gain one kwarg; legacy configs resolve to `"sum"`): in `build_model` add `pert_aggregation=config.get("pert_aggregation", "sum")` to the `PerturbationTFModel(...)` call, and in `to_hf_checkpoint` add the same kwarg to the `HFPerturbationTFModel(...)` call. Then append to `tests/test_tutorial_smoke.py`:

```python
# --- multi-perturbation (high-MOI) variants of the tutorials ---------------------------------
# regime "set": designed combinations, evaluated per set; regime "single": random 1-6 guides
# per cell, evaluated per single perturbation (spec §1.1).

MULTI_GENES = [f"P{i}" for i in range(8)]
SET_LABELS = ["WT", "A", "B", "C", "A+B", "B+C", "A+B+C"]


def make_multi_adata(regime, seed=0):
    rng = np.random.default_rng(seed)
    n_per_label = CELLS_PER_GROUP * len(CELLTYPES)
    if regime == "set":
        genotype = np.repeat(SET_LABELS, n_per_label)
    else:
        sets = ["+".join(sorted(rng.choice(MULTI_GENES, rng.integers(1, 7), replace=False)))
                for _ in range(6 * n_per_label)]
        genotype = np.array(["WT"] * n_per_label + sets)
    celltype = np.tile(np.repeat(CELLTYPES, CELLS_PER_GROUP), genotype.size // n_per_label)
    singles = sorted({g for label in genotype for g in label.split("+") if g != "WT"})
    base = rng.gamma(2.0, 1.0, size=N_GENES)
    rate = np.tile(base, (genotype.size, 1))
    for i, g in enumerate(singles):
        has_g = np.array([g in label.split("+") for label in genotype])
        rate[has_g, i * 5:(i + 1) * 5] *= 3.0  # additive per-guide signatures
    rate[celltype == "T2", 50:55] *= 2.5
    counts = rng.poisson(rate).astype(np.float32)
    normed = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    adata = anndata.AnnData(
        X=normed,
        obs=pd.DataFrame({"genotype": genotype, "celltype": celltype},
                         index=[f"cell_{i}" for i in range(genotype.size)]),
        var=pd.DataFrame(index=[f"G{i}" for i in range(N_GENES)]),
    )
    adata.layers["X_binned"] = normed.copy()
    adata.var["highly_variable"] = np.arange(N_GENES) < 40
    return adata, singles


def split_by_control(adata, seed=0, frac=0.2):
    """Random split that keeps control and perturbed cells of every cell type in both halves."""
    rng = np.random.default_rng(seed)
    valid = []
    key = adata.obs["celltype"].astype(str) + "|" + (adata.obs["genotype"] == "WT").astype(str)
    for _, idx in adata.obs.groupby(key, observed=True).indices.items():
        valid.extend(rng.choice(idx, max(5, int(frac * len(idx))), replace=False))
    valid = np.sort(np.array(valid))
    return np.setdiff1d(np.arange(adata.n_obs), valid), valid


def multi_hyperparameters(regime):
    return pert_hyperparameters(
        pert_delimiter="+", perturbation_unit=regime,
        pert_aggregation="sqrt" if regime == "single" else "sum",
        pairing_config={"target_sampling": "cell"} if regime == "single" else {},
    )


def train_multi_model(regime, save_dir):
    adata, singles = make_multi_adata(regime)
    config, _ = generate_config(multi_hyperparameters(regime), wandb_mode="disabled")
    train_idx, valid_idx = split_by_control(adata)
    data_gen = produce_training_datasets(adata, config, next_cell_pred="pert",
                                         train_indices=train_idx, valid_indices=valid_idx)
    model = build_model(data_gen, config)
    best = wrapper_train(model, config, data_gen, save_dir=save_dir, device=DEVICE, eval_adata_dict={})
    return adata, singles, best, config, data_gen


@pytest.fixture(scope="module")
def multi_checkpoint(tmp_path_factory):
    """Regime "set" training run reused by the multi-mode inference tests."""
    save_dir = tmp_path_factory.mktemp("multi_run")
    adata, singles, best, config, data_gen = train_multi_model("set", save_dir)
    hf_dir = tmp_path_factory.mktemp("multi_hf")
    to_hf_checkpoint(best, config, data_gen, hf_dir)
    return {"adata": adata, "singles": singles, "model": best, "config": config,
            "data_gen": data_gen, "save_dir": save_dir, "hf_dir": hf_dir}


@pytest.mark.parametrize("regime", ["set", "single"])
def test_multi_pert_training_tutorial(regime, tmp_path):
    adata, singles, best, config, data_gen = train_multi_model(regime, tmp_path)
    assert set(data_gen["genotype_to_index"]) == set(singles) | {"WT"}
    assert data_gen["n_perturb"] == len(singles) + 1
    counts = data_gen["split_check"]["combination_type_counts"]
    assert "unseen_combination" in counts
    if regime == "single":
        assert counts["unseen_combination"] > 0  # random sets rarely repeat across the split
    assert (tmp_path / "best_model.pt").exists()

    adata_eva = eval_testdata(best, data_gen["adata_sorted"], data_gen["gene_ids"],
                              train_data_dict=data_gen, config=config, device=DEVICE)
    probs = adata_eva.obsm["X_pert_pred_probs"]
    assert probs.shape == (adata_eva.n_obs, data_gen["n_perturb"])
    assert probs.min() >= 0.0 and probs.max() <= 1.0  # sigmoid, not softmax, in multi mode
    for label in adata_eva.obs["predicted_genotype"]:
        assert set(label.split("+")) <= set(data_gen["genotype_to_index"])
    assert "genotype_id" not in adata_eva.obs


def test_legacy_mode_warns_on_plus_labels():
    adata, _ = make_multi_adata("set")
    config, _ = generate_config(pert_hyperparameters(), wandb_mode="disabled")  # pert_delimiter=None
    train_idx, valid_idx = split_by_control(adata)
    with pytest.warns(UserWarning, match=r"\+"):
        data_gen = produce_training_datasets(adata, config, next_cell_pred="pert",
                                             train_indices=train_idx, valid_indices=valid_idx)
    assert "A+B" in data_gen["genotype_to_index"]  # legacy: the combination is its own class
```

Run: `pytest tests/test_tutorial_smoke.py -v`
Expected: the 7 Task 0 tests PASS unchanged; the 3 new tests PASS.

- [ ] **Step 6: Run CI tests and commit**

Run: `pytest tests/ -v` → all PASS

```bash
git add perttf/model/train_function.py perttf/model/train_data_gen.py perttf/model/config_gen.py perttf/model/hf.py tests/test_tutorial_smoke.py
git commit -m "feat: train and evaluate with perturbation sets; config defaults for multi mode"
```

---

### Task 8: Inference entry points and virtual screen

**Files:**
- Modify: `perttf/model/hf.py` (`predict_perturbations` validation lines 299-347, query embeddings lines 388-410; `from_pretrained` records `pert_delimiter`; LoRA `finetune_perturbation` delimiter passthrough ~line 1066)
- Modify: `perttf/utils/virtual_screen.py` (lines 35-47)
- Test: `tests/test_multi_pert_inference.py`; `tests/test_tutorial_smoke.py` (append the inference half of the multi-mode section)

**Interfaces:**
- Consumes: Task 1 helpers, Task 3 model; Task 7's `multi_checkpoint` fixture and `make_multi_adata`.
- Produces:
  - `HFPerturbationTFModel.predict_perturbations(..., pert_delimiter: Optional[str] = "__from_config__")`: resolved delimiter = argument if given, else `self.training_config.get('pert_delimiter')`; if both the config and the argument are non-`None` and differ → `ValueError`; `validate_perturbation_labels(obs genotype ∪ genotype_next, delimiter, control, existing_mapping=self.genotype_to_index)`; membership checks use `labels_in_mapping`; `query_sources` path builds `perturbation_embeddings` rows for every *component* and the gather happens in Task 7's loop. Result `uns["perttf_prediction"]["pert_delimiter"]` records the resolved value.
  - `virtual_screen_src2dest`: candidate filter uses `labels_in_mapping(screen_genes, genotype_to_index, delimiter, control)` with `delimiter = running_parameters.get('pert_delimiter')` falling back to `model.training_config.get('pert_delimiter')`.
  - `lora.label_mappings` call in `finetune_perturbation` passes `delimiter=config.get('pert_delimiter')`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_multi_pert_inference.py
import numpy as np, pandas as pd, anndata as ad, pytest, torch
from perttf.model.hf import HFPerturbationTFModel
from perttf.utils.custom_tokenizer import SimpleVocab

GENES = [f"g{i}" for i in range(1, 8)]

def model(mapping, **cfg):
    vocab = SimpleVocab.from_dict({"<pad>": 0, "<cls>": 1, **{g: i + 2 for i, g in enumerate(GENES)}})
    return HFPerturbationTFModel(vocab=vocab, genotype_to_index=mapping, cell_type_to_index={"c": 0},
                                 d_model=16, nhead=2, d_hid=16, nlayers=1, n_ps=0, do_mvc=True, training_config=cfg).eval()

def adata(next_labels):
    n = len(next_labels)
    a = ad.AnnData(X=np.ones((n, len(GENES)), dtype=np.float32),
                   obs=pd.DataFrame({"celltype": ["c"] * n, "genotype": ["WT"] * n, "genotype_next": next_labels}),
                   var=pd.DataFrame(index=GENES))
    a.layers["X_binned"] = a.X.copy()
    return a

def test_legacy_checkpoint_with_combo_key_rejected():
    m = model({"WT": 0, "A+B": 1})
    with pytest.raises(ValueError, match="A\\+B"):
        m.predict_perturbations(adata(["WT"]), pert_delimiter="+")

def test_delimiter_mismatch_rejected():
    m = model({"WT": 0, "A": 1}, pert_delimiter="+")
    with pytest.raises(ValueError):
        m.predict_perturbations(adata(["WT"]), pert_delimiter="|")

def test_unknown_component_raises_before_compute():
    m = model({"WT": 0, "A": 1})
    with pytest.raises(ValueError, match="B"):
        m.predict_perturbations(adata(["A+B"]), pert_delimiter="+")

def test_set_inference_runs_and_records_delimiter():
    m = model({"WT": 0, "A": 1, "B": 2}, pert_delimiter="+")
    r = m.predict_perturbations(adata(["A+B", "B", "WT"]), prediction_mode="mean", device="cpu")
    assert r.shape == (3, len(GENES)) and r.uns["perttf_prediction"]["pert_delimiter"] == "+"
    assert r.obs["genotype_next"].tolist() == ["A+B", "B", "WT"]
```

Append to `tests/test_tutorial_smoke.py` (the multi-mode counterparts of the inference, LoRA and screen tutorials):

```python
def test_multi_predict_perturbations_tutorial(multi_checkpoint):
    model = HFPerturbationTFModel.from_pretrained(str(multi_checkpoint["hf_dir"]))
    assert model.training_config.pert_delimiter == "+"
    assert model.training_config.pert_aggregation == "sum"
    adata = multi_checkpoint["adata"]
    query = adata[adata.obs["genotype"] == "WT"].copy()
    query.obs["genotype_next"] = "A+C"  # a combination never seen in training, of known singles
    result = model.predict_perturbations(query, prediction_mode="mean", device=DEVICE)
    assert result.obs_names.equals(query.obs_names)
    assert result.obsm["mvc_next_expr"].shape == query.shape
    assert result.uns["perttf_prediction"]["pert_delimiter"] == "+"
    with pytest.raises(ValueError):
        model.predict_perturbations(query, pert_delimiter="|", device=DEVICE)
    bad = query.copy()
    bad.obs["genotype_next"] = "A+Z"
    with pytest.raises(ValueError, match="Z"):
        model.predict_perturbations(bad, device=DEVICE)


def test_multi_virtual_screen_tutorial(multi_checkpoint):
    from perttf.utils.virtual_screen import virtual_screen_src2dest

    adata, model, config = multi_checkpoint["adata"], multi_checkpoint["model"], multi_checkpoint["config"]
    running_parameters = torch.load(multi_checkpoint["save_dir"] / "running_parameters.pt", weights_only=False)
    vocab = torch.load(multi_checkpoint["save_dir"] / "vocab.pt", weights_only=False)
    is_t1 = adata.obs["celltype"] == "T1"
    adata_src = adata[is_t1 & (adata.obs["genotype"] == "WT")].copy()
    adata_des = adata[is_t1 & (adata.obs["genotype"] == "A+B+C")].copy()
    for a in (adata_src, adata_des):
        a.obs["genotype_next"] = a.obs["genotype"].astype("category")
    results = virtual_screen_src2dest(
        adata_src, adata_des, model, vocab, config, running_parameters,
        screen_genes=["A+B", "C", "A+Z"], device=DEVICE,
        n_expands_per_epoch=1, n_epoch=1, wt_pred_next_label="A+B+C",
    )
    assert set(results["sim_ranked"].index) == {"A+B", "C"}  # "A+Z" filtered: Z is unknown


def test_multi_lora_pert_tutorial(multi_checkpoint, tmp_path):
    pytest.importorskip("peft")
    model = HFPerturbationTFModel.from_pretrained(str(multi_checkpoint["hf_dir"]))
    adata = multi_checkpoint["adata"]
    is_t1 = (adata.obs["celltype"] == "T1").to_numpy()
    adapter_dir = tmp_path / "multi_pert_adapter"
    peft_model = model.run_lora_pert_train(
        adata, np.where(~is_t1)[0], np.where(is_t1)[0], epochs=1, batch_size=32,
        lora_config=model.build_lora_config(r=4, lora_alpha=8, lora_dropout=0.0),
        save_dir=str(adapter_dir), seed=0, device=DEVICE,
    )
    assert (adapter_dir / "lora_heads.json").exists()
    assert set(peft_model.get_base_model().genotype_to_index) == set(multi_checkpoint["singles"]) | {"WT"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_multi_pert_inference.py tests/test_tutorial_smoke.py -v`
Expected: `test_multi_pert_inference.py` FAIL (`TypeError: unexpected keyword 'pert_delimiter'`); the three new tutorial tests FAIL; every earlier tutorial test PASS.

- [ ] **Step 3: Implement in `perttf/model/hf.py` and `perttf/utils/virtual_screen.py`**

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add perttf/model/hf.py perttf/utils/virtual_screen.py tests/test_multi_pert_inference.py tests/test_tutorial_smoke.py
git commit -m "feat: multi-perturbation inference with delimiter consistency checks"
```

---

### Task 9: Documentation

**Files:**
- Create: `demos/tutorials/MULTI_PERTURBATION.md`
- Modify: `README.md` (tutorial list), `docs/index.md` (toctree), `docs/guides/training_options.md` (new config keys)

- [ ] **Step 1: Write `demos/tutorials/MULTI_PERTURBATION.md`**

Sections, in this order: label convention (`"A+B"`, control handling, canonical form, the legacy warning when `+` appears without `pert_delimiter`); enabling (`pert_delimiter="+"` in the training config or `predict_perturbations`); regime A vs regime B with the recommended settings table — regime B: `perturbation_unit="single"`, `pairing_config={"target_sampling": "cell"}`, `pert_aggregation="sqrt"` or `"mean"`; regime A: defaults; reading outputs (`X_pert_pred_probs` sigmoid, `predicted_genotype` thresholded set, `pert_pred_threshold`); using an existing single-perturbation checkpoint zero-shot on combinations of known genes; what raises (`A+B` keys in a legacy mapping, delimiter mismatch, unknown components, `max_perts_per_cell`).

- [ ] **Step 2: Add links in `README.md` and `docs/index.md`; add the six config keys with defaults to `docs/guides/training_options.md`**

- [ ] **Step 3: Build docs locally if the docs extra is installed**

Run: `pip install .[docs] && sphinx-build -b html docs docs/_build` (skip if unavailable)
Expected: no warnings about the new page.

- [ ] **Step 4: Commit**

```bash
git add demos/tutorials/MULTI_PERTURBATION.md README.md docs/index.md docs/guides/training_options.md
git commit -m "docs: multi-perturbation (high-MOI) tutorial and config reference"
```
