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
    with pytest.warns(UserWarning):
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
    # target sampling may draw the control (identity pair), so check over repeated draws
    draws = [ds[0] for _ in range(60)]
    assert {d["pert_name_next"] for d in draws} == {"WT", "A"}
    assert all(d["perturbation_labels"].tolist()[0] == ds.genotype_to_index["WT"] for d in draws)

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

def test_legacy_prediction_only_missing_genotype_maps_to_zero():
    a = tiny(["WT", "A", "WT"])
    a.obs["genotype"] = pd.Series(["WT", np.nan, "A"], index=a.obs.index, dtype=object)
    ds = PertTFDataset(a, genotype_to_index={"WT": 0, "A": 1}, prediction_only=True)
    assert [ds[i]["perturbation_labels"] for i in range(3)] == [0, 0, 1]

def test_multi_prediction_only_unknown_source_component_is_padded():
    a = tiny(["C+A", "WT"]); a.obs["genotype_next"] = ["A", "B"]
    ds = PertTFDataset(a, genotype_to_index={"WT": 0, "A": 1, "B": 2}, prediction_only=True, pert_delimiter="+")
    assert ds[0]["perturbation_labels"].tolist() == [1, -1]
    assert ds[1]["perturbation_labels"].tolist() == [0, -1]
    b = tiny(["C+D", "WT"]); b.obs["genotype_next"] = ["A", "B"]
    ds = PertTFDataset(b, genotype_to_index={"WT": 0, "A": 1, "B": 2}, prediction_only=True, pert_delimiter="+")
    assert ds[0]["perturbation_labels"].tolist() == [-1, -1]

def test_multi_unknown_next_component_still_raises():
    a = tiny(["WT", "WT"]); a.obs["genotype_next"] = ["A+Z", "WT"]
    ds = PertTFDataset(a, genotype_to_index={"WT": 0, "A": 1}, prediction_only=True, pert_delimiter="+")
    with pytest.raises(ValueError, match="Z"):
        ds[0]


def test_max_n_pert_counts_only_rows_the_dataset_can_draw():
    # Validation may hold wider sets than training; the training dataset's width (recorded as the
    # inference cap) must come from its own rows, not from the whole AnnData (Codex review on PR 71).
    a = tiny(["WT", "A", "A+B+C", "WT"])
    train = PertTFDataset(a, indices=np.array([0, 1, 3]), next_cell_pred="identity", pert_delimiter="+")
    assert train.max_n_pert == 1
    valid = PertTFDataset(a, indices=np.array([2]), next_cell_pred="identity", pert_delimiter="+")
    assert valid.max_n_pert == 3
    assert valid[0]["perturbation_labels"].tolist() == [1, 2, 3]
    everything = PertTFDataset(a, next_cell_pred="identity", pert_delimiter="+")
    assert everything.max_n_pert == 3


def test_training_cap_ignores_prepopulated_genotype_next():
    # Training targets come from the pairing, not from genotype_next, so a pre-populated genotype_next
    # column must not widen the training dataset's recorded cap (Codex review on PR 71).
    a = tiny(["WT", "WT", "A", "B"])
    a.obs["genotype_next"] = "A+B+C"
    train = PertTFDataset(a, next_cell_pred="pert", pert_delimiter="+")
    assert train.max_n_pert == 1
    predict = PertTFDataset(a, next_cell_pred="pert", pert_delimiter="+", prediction_only=True,
                            genotype_to_index={"WT": 0, "A": 1, "B": 2, "C": 3})
    assert predict.max_n_pert == 3
    assert predict[0]["perturbation_labels_next"].tolist() == [1, 2, 3]
