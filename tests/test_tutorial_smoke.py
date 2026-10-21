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
        pert_aggregation=config.get("pert_aggregation", "sum"),
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
        ps_names=data_gen["ps_names"], pert_aggregation=config.get("pert_aggregation", "sum"),
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
               lambda *args, **kwargs: DataLoader(*args, **{**kwargs, "num_workers": 0, "pin_memory": False}))
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


def multi_hyperparameters(regime, **overrides):
    hp = dict(
        pert_delimiter="+", perturbation_unit=regime,
        pert_aggregation="sqrt" if regime == "single" else "sum",
        pairing_config={"target_sampling": "cell"} if regime == "single" else {},
    )
    hp.update(overrides)
    return pert_hyperparameters(**hp)


def train_multi_model(regime, save_dir, **overrides):
    adata, singles = make_multi_adata(regime)
    config, _ = generate_config(multi_hyperparameters(regime, **overrides), wandb_mode="disabled")
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
    if regime == "set":
        assert "unseen_combination" in counts
    assert (tmp_path / "best_model.pt").exists()

    adata_eva = eval_testdata(best, data_gen["adata_sorted"], data_gen["gene_ids"],
                              train_data_dict=data_gen, config=config, device=DEVICE)
    probs = adata_eva.obsm["X_pert_pred_probs"]
    assert probs.shape == (adata_eva.n_obs, data_gen["n_perturb"])
    assert probs.min() >= 0.0 and probs.max() <= 1.0  # sigmoid, not softmax, in multi mode
    assert not np.allclose(probs.sum(1), 1.0)
    for label in adata_eva.obs["predicted_genotype"]:
        assert set(label.split("+")) <= set(data_gen["genotype_to_index"])
        assert label == "WT" or "WT" not in label.split("+")  # control dropped inside combinations
    assert "genotype_id" not in adata_eva.obs


def test_multi_pert_prediction_cap_is_model_side(multi_checkpoint):
    # A control-only query whose labels are all one set: the cap on predicted set size must come
    # from training (max_n_pert recorded in the config), not from the query's own labels. The
    # 1-epoch toy model never puts two classes above 0.5 on WT cells, so threshold 0 makes every
    # class pass and the predicted sets show the cap (top-3 classes, the control dropped).
    data_gen = multi_checkpoint["data_gen"]
    assert data_gen["max_n_pert"] == 3 and multi_checkpoint["config"].max_n_pert == 3
    adata = multi_checkpoint["adata"]
    query = adata[adata.obs["genotype"] == "WT"].copy()
    query.obs["genotype_next"] = "A"  # the query's own labels alone would give a cap of 1
    mapping = {k: data_gen[k] for k in ("genotype_to_index", "cell_type_to_index", "vocab")}
    config = OmegaConf.create(multi_checkpoint["config"].as_dict())
    config.pert_pred_threshold = 0.0

    def predicted_sets(cfg):
        res = eval_testdata(multi_checkpoint["model"], query, None, train_data_dict=mapping,
                            config=cfg, device=DEVICE)
        sets = [label.split("+") for label in res.obs["predicted_genotype"]]
        assert all(set(s) <= set(data_gen["genotype_to_index"]) and "WT" not in s for s in sets)
        return sets

    sets = predicted_sets(config)  # cap from the saved training config: 3
    assert all(2 <= len(s) <= 3 for s in sets)
    config.max_perts_per_cell = 2  # an explicit config cap takes precedence
    assert all(1 <= len(s) <= 2 for s in predicted_sets(config))


def test_legacy_mode_warns_on_plus_labels():
    adata, _ = make_multi_adata("set")
    config, _ = generate_config(pert_hyperparameters(), wandb_mode="disabled")  # pert_delimiter=None
    train_idx, valid_idx = split_by_control(adata)
    with pytest.warns(UserWarning, match=r"\+"):
        data_gen = produce_training_datasets(adata, config, next_cell_pred="pert",
                                             train_indices=train_idx, valid_indices=valid_idx)
    assert "A+B" in data_gen["genotype_to_index"]  # legacy: the combination is its own class


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


def test_multi_virtual_screen_rejects_config_delimiter_mismatch(multi_checkpoint):
    from perttf.utils.virtual_screen import virtual_screen_src2dest

    adata, model = multi_checkpoint["adata"], multi_checkpoint["model"]
    running_parameters = torch.load(multi_checkpoint["save_dir"] / "running_parameters.pt", weights_only=False)
    vocab = torch.load(multi_checkpoint["save_dir"] / "vocab.pt", weights_only=False)
    config = OmegaConf.create(multi_checkpoint["config"].as_dict())
    config.pert_delimiter = None  # running_parameters say "+": eval_testdata would silently drop "A+B" rows
    is_t1 = adata.obs["celltype"] == "T1"
    adata_src = adata[is_t1 & (adata.obs["genotype"] == "WT")].copy()
    adata_des = adata[is_t1 & (adata.obs["genotype"] == "A+B+C")].copy()
    for a in (adata_src, adata_des):
        a.obs["genotype_next"] = a.obs["genotype"].astype("category")
    with pytest.raises(ValueError, match="pert_delimiter"):
        virtual_screen_src2dest(
            adata_src, adata_des, model, vocab, config, running_parameters,
            screen_genes=["A+B", "C"], device=DEVICE,
            n_expands_per_epoch=1, n_epoch=1, wt_pred_next_label="A+B+C",
        )


def test_multi_split_check_sees_canonical_labels():
    # Training writes the combination "B+A", validation "A+B" and a "WT+WT" control: the split check
    # must run on canonical labels, so the validation set is a seen combination and "WT+WT" is a control.
    adata, _ = make_multi_adata("set")
    train_idx, valid_idx = split_by_control(adata)
    genotype = adata.obs["genotype"].to_numpy().astype(object)
    is_train = np.isin(np.arange(adata.n_obs), train_idx)
    genotype[(genotype == "A+B") & is_train] = "B+A"
    wt_valid = np.where((genotype == "WT") & ~is_train)[0]
    genotype[wt_valid[::2]] = "WT+WT"
    adata.obs["genotype"] = genotype
    config, _ = generate_config(multi_hyperparameters("set"), wandb_mode="disabled")
    data_gen = produce_training_datasets(adata, config, next_cell_pred="pert",
                                         train_indices=train_idx, valid_indices=valid_idx)
    types = data_gen["split_check"]["combination_types"]
    assert ("T1", "A+B") in types["seen_combination"] and ("T2", "A+B") in types["seen_combination"]
    groups = [g for gs in types.values() for g in gs]
    assert not any("WT+WT" in str(g) or "B+A" in str(g) for g in groups)


def test_wrapper_train_rejects_pert_aggregation_mismatch(tmp_path):
    adata, _ = make_multi_adata("set")
    config, _ = generate_config(multi_hyperparameters("set", pert_aggregation="sqrt"), wandb_mode="disabled")
    train_idx, valid_idx = split_by_control(adata)
    data_gen = produce_training_datasets(adata, config, next_cell_pred="pert",
                                         train_indices=train_idx, valid_indices=valid_idx)
    vocab = data_gen["vocab"]
    model = PerturbationTFModel(  # no pert_aggregation kwarg: the model sums
        data_gen["n_perturb"], 3, len(data_gen["ps_names"]), len(vocab),
        config.layer_size, config.nhead, config.layer_size, config.nlayers,
        vocab=vocab, dropout=config.dropout, pad_token=config.pad_token, pad_value=config.pad_value,
        do_mvc=config.GEPC, num_batch_labels=data_gen["num_batch_types"], n_input_bins=config.n_bins,
        use_fast_transformer=False, n_cls=data_gen["n_cls"], nlayers_cls=3,
    )
    with pytest.raises(ValueError, match="sum.*sqrt"):
        wrapper_train(model, config, data_gen, save_dir=tmp_path, device=DEVICE, eval_adata_dict={})
    assert not (tmp_path / "running_parameters.pt").exists()


def test_lora_pert_delimiter_override_on_legacy_checkpoint(pert_checkpoint, tmp_path):
    pytest.importorskip("peft")
    model = HFPerturbationTFModel.from_pretrained(str(pert_checkpoint["hf_dir"]))
    assert model.training_config.get("pert_delimiter") is None
    assert set(model.genotype_to_index) == {"WT", "A", "B", "C"}
    adata, _ = make_multi_adata("set")
    is_t1 = (adata.obs["celltype"] == "T1").to_numpy()
    adapter_dir = tmp_path / "legacy_to_multi_adapter"
    peft_model = model.run_lora_pert_train(
        adata, np.where(~is_t1)[0], np.where(is_t1)[0], epochs=1, batch_size=32,
        lora_config=model.build_lora_config(r=4, lora_alpha=8, lora_dropout=0.0),
        save_dir=str(adapter_dir), seed=0, device=DEVICE, pert_delimiter="+",
    )
    assert hasattr(peft_model, "get_base_model")
    assert (adapter_dir / "lora_heads.json").exists()
    assert json.loads((adapter_dir / "lora_heads.json").read_text())["pert_delimiter"] == "+"


def test_lora_rejects_pert_delimiter_mismatch(multi_checkpoint):
    pytest.importorskip("peft")
    model = HFPerturbationTFModel.from_pretrained(str(multi_checkpoint["hf_dir"]))
    adata = multi_checkpoint["adata"]
    is_t1 = (adata.obs["celltype"] == "T1").to_numpy()
    with pytest.raises(ValueError, match="pert_delimiter"):
        model.run_lora_pert_train(adata, np.where(~is_t1)[0], np.where(is_t1)[0], epochs=1,
                                  device=DEVICE, pert_delimiter="|")
    with pytest.raises(ValueError, match="pert_delimiter"):
        model.run_lora_cls_train(adata, "celltype", "classification", epochs=1,
                                 device=DEVICE, pert_delimiter="|")


@pytest.mark.parametrize("cce_mode", ["cell+geno", "cell_geno", "genotype"])
def test_multi_cce_training(cce_mode, tmp_path):
    overrides = dict(CCE=True, cce_mode=cce_mode)
    if cce_mode == "cell+geno":
        overrides["perturbation_input"] = True
    adata, singles, best, config, data_gen = train_multi_model("set", tmp_path, **overrides)
    assert config.CCE and config.cce_mode == cce_mode
    adata_eva = eval_testdata(best, data_gen["adata_sorted"], data_gen["gene_ids"],
                              train_data_dict=data_gen, config=config, device=DEVICE)
    probs = adata_eva.obsm["X_pert_pred_probs"]
    assert probs.shape == (adata_eva.n_obs, data_gen["n_perturb"]) and np.isfinite(probs).all()


def test_from_adata_run_train_multi(tmp_path):
    """The public from_adata(...).run_train(...) path with set labels: the dictionary is built from single
    perturbations and the run records the delimiter (Codex review finding on PR 71)."""
    adata, singles = make_multi_adata("set")
    hp = multi_hyperparameters("set")
    model = HFPerturbationTFModel.from_adata(adata, hp)
    assert set(model.genotype_to_index) == set(singles) | {"WT"}
    train_idx, valid_idx = split_by_control(adata)
    model.run_train(adata, train_config=hp, train_indices=train_idx, valid_indices=valid_idx,
                    save_dir=str(tmp_path / "run_train"), device=DEVICE)
    assert model.training_config.pert_delimiter == "+"
    assert model.training_config.max_n_pert == 3
    reloaded = HFPerturbationTFModel.from_pretrained(str(tmp_path / "run_train"))
    assert set(reloaded.genotype_to_index) == set(singles) | {"WT"}
    query = adata[adata.obs["genotype"] == "WT"].copy()
    query.obs["genotype_next"] = "A+C"
    result = reloaded.predict_perturbations(query, prediction_mode="mean", device=DEVICE)
    assert result.obsm["mvc_next_expr"].shape == query.shape
