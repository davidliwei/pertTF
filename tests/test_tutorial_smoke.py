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
