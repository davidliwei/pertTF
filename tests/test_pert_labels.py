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
