import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from perttf.model.composition_change_analysis import calculate_lonESS_score, lochness_by_perturbation


def _old_lonESS(adata, overall_fraction_dict=None, nn_name=None, n_neighbors=30, target_genotype=None, delta=0.0001):
    """The pre-vectorisation implementation (whole-label matching), kept as the reference."""
    if overall_fraction_dict is None:
        overall_fraction_dict = adata.obs['genotype'].value_counts(normalize=True).to_dict()
    nn_key = 'distances' if nn_name is None else f'{nn_name}_distances'
    out = []
    for cell_id in adata.obs_names:
        cell_index = adata.obs_names.get_loc(cell_id)
        cell_genotype = target_genotype if target_genotype is not None else adata.obs.loc[cell_id, 'genotype']
        indices = adata.obsp[nn_key][cell_index, :].nonzero()[1]
        hits = sum(adata.obs.loc[adata.obs.index[indices], 'genotype'] == cell_genotype)
        overall = overall_fraction_dict[cell_genotype]
        if overall == 0:
            overall = 0.0001
        score = hits / n_neighbors / overall - 1
        if delta > 0:
            score += np.random.normal(0, delta)
        out.append(score)
    return out


def _single_label_adata(n=240, k=15, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["WT", "A", "B", "C"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    X = rng.normal(size=(n, 10)) + (labels == "A")[:, None] * 1.5
    a = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame({"genotype": labels}, index=[f"c{i}" for i in range(n)]))
    sc.pp.neighbors(a, n_neighbors=k, use_rep="X", key_added="nn")
    return a


@pytest.mark.parametrize("target", [None, "A", "C"])
def test_matches_original_on_single_label_data(target):
    a = _single_label_adata()
    np.random.seed(1)
    ref = _old_lonESS(a, nn_name="nn", n_neighbors=15, target_genotype=target)
    np.random.seed(1)
    new = calculate_lonESS_score(a, nn_name="nn", n_neighbors=15, target_genotype=target)
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-12)


def test_matches_original_with_supplied_fractions():
    a = _single_label_adata()
    fr = {"WT": 0.5, "A": 0.25, "B": 0.0, "C": 0.25}
    ref = _old_lonESS(a, overall_fraction_dict=fr, nn_name="nn", n_neighbors=15, delta=0)
    new = calculate_lonESS_score(a, overall_fraction_dict=fr, nn_name="nn", n_neighbors=15, delta=0)
    np.testing.assert_allclose(new, ref, rtol=0, atol=1e-12)


def _multi_adata():
    labels = ["A+B", "A", "B", "WT", "A+C", "C", "B+C", "WT"]
    a = ad.AnnData(X=np.zeros((8, 2), dtype=np.float32), obs=pd.DataFrame({"genotype": labels}, index=[f"c{i}" for i in range(8)]))
    rng = np.random.default_rng(3)
    D = np.zeros((8, 8))
    for i in range(8):                                   # 3 neighbours per cell, self excluded
        nb = rng.choice([j for j in range(8) if j != i], 3, replace=False)
        D[i, nb] = rng.uniform(0.5, 1.5, 3)
    a.obsp["distances"] = sparse.csr_matrix(D)
    return a, D > 0


def _carries(label, g):
    return g in label.split("+")


def test_multi_label_target_uses_membership():
    a, nb = _multi_adata()
    labels = list(a.obs.genotype)
    frac = np.mean([_carries(x, "A") for x in labels])
    expect = [sum(_carries(labels[j], "A") for j in np.where(nb[i])[0]) / 3 / frac - 1 for i in range(8)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = calculate_lonESS_score(a, n_neighbors=3, target_genotype="A", delta=0, pert_delimiter="+")
    np.testing.assert_allclose(got, expect)


def test_multi_label_own_score_is_mean_over_components():
    a, nb = _multi_adata()
    labels = list(a.obs.genotype)
    def comp_score(i, g):
        frac = np.mean([_carries(x, g) for x in labels])
        return sum(_carries(labels[j], g) for j in np.where(nb[i])[0]) / 3 / frac - 1
    expect = [np.mean([comp_score(i, g) for g in labels[i].split("+")]) for i in range(8)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = calculate_lonESS_score(a, n_neighbors=3, delta=0, pert_delimiter="+")
    np.testing.assert_allclose(got, expect)


def test_errors_for_absent_or_set_target():
    a, _ = _multi_adata()
    with pytest.raises(ValueError, match="not carried by any cell"):
        calculate_lonESS_score(a, n_neighbors=3, target_genotype="Z", pert_delimiter="+")
    with pytest.raises(ValueError, match="multi-perturbation labels need pert_delimiter"):
        calculate_lonESS_score(a, n_neighbors=3, target_genotype="A+Z")   # no delimiter: whole label "A+Z" absent
    with pytest.raises(ValueError, match="single perturbation"):
        calculate_lonESS_score(a, n_neighbors=3, target_genotype="A+B", pert_delimiter="+")


def test_low_count_warning():
    a = _single_label_adata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # 15 x 0.3 = 4.5 expected: no warning
        calculate_lonESS_score(a, nn_name="nn", n_neighbors=15, target_genotype="A")
    with pytest.warns(UserWarning, match="counting noise"):
        calculate_lonESS_score(a, nn_name="nn", n_neighbors=15, target_genotype="C", min_expected_neighbors=3.0)   # 15 x 0.14 = 2.1


def test_lochness_by_perturbation_counts():
    a, nb = _multi_adata()
    labels = list(a.obs.genotype)
    t = lochness_by_perturbation(a, pert_delimiter="+")
    for g in ["A", "B", "C", "WT"]:
        car = [i for i, x in enumerate(labels) if _carries(x, g)]
        obs = sum(_carries(labels[j], g) for i in car for j in np.where(nb[i])[0])
        m = len(car)
        assert t.loc[g, "n_carriers"] == m and t.loc[g, "observed"] == obs
        assert t.loc[g, "expected"] == pytest.approx(m * 3 * (m - 1) / 7)
        assert t.loc[g, "expected_neighbors_per_cell"] == pytest.approx(3 * (m - 1) / 7)


def test_lochness_by_perturbation_expected_matches_permutation_mean():
    rng = np.random.default_rng(5)
    n = 12
    labels = ["A", "A+B", "B", "WT", "A", "C", "B+C", "WT", "A+C", "WT", "B", "C"]
    D = np.zeros((n, n))
    for i in range(n):                                   # variable degree (1-6), self excluded
        nb = rng.choice([j for j in range(n) if j != i], int(rng.integers(1, 7)), replace=False)
        D[i, nb] = 1.0
    t = lochness_by_perturbation(_labelled(labels, D), pert_delimiter="+")
    perm_obs = []
    for _ in range(3000):
        perm = list(rng.permutation(labels))
        perm_obs.append(lochness_by_perturbation(_labelled(perm, D), pert_delimiter="+").observed)
    perm_mean = pd.concat(perm_obs, axis=1).mean(axis=1)
    for g in ["A", "B", "C", "WT"]:
        assert t.loc[g, "expected"] == pytest.approx(perm_mean[g], rel=0.05)


def _labelled(labels, D):
    a = ad.AnnData(X=np.zeros((len(labels), 2), dtype=np.float32),
                   obs=pd.DataFrame({"genotype": labels}, index=[f"c{i}" for i in range(len(labels))]))
    a.obsp["distances"] = sparse.csr_matrix(D)
    return a


def test_lochness_by_perturbation_null_centred():
    rng = np.random.default_rng(0)
    n = 3000
    labels = ["+".join(rng.choice([f"g{i}" for i in range(40)], 3, replace=False)) for _ in range(n)]
    a = ad.AnnData(X=rng.normal(size=(n, 8)).astype(np.float32), obs=pd.DataFrame({"genotype": labels}, index=[f"c{i}" for i in range(n)]))
    sc.pp.neighbors(a, n_neighbors=15, use_rep="X")     # stores 14 neighbours per cell
    t = lochness_by_perturbation(a, pert_delimiter="+")
    assert abs(t.enrichment.mean()) < 0.03              # random labels: no co-localisation (k-based expectation: -1/15)


def test_nonzero_self_edges_are_ignored():
    a, nb = _multi_adata()
    D = nb * 1.0
    with_self = _labelled(list(a.obs.genotype), D + 0.7 * np.eye(len(D)))   # self stored with a nonzero distance
    without = _labelled(list(a.obs.genotype), D)
    pd.testing.assert_frame_equal(lochness_by_perturbation(with_self, pert_delimiter="+"),
                                  lochness_by_perturbation(without, pert_delimiter="+"))
    for target in [None, "A"]:
        np.testing.assert_allclose(
            calculate_lonESS_score(with_self, n_neighbors=3, target_genotype=target, delta=0, pert_delimiter="+"),
            calculate_lonESS_score(without, n_neighbors=3, target_genotype=target, delta=0, pert_delimiter="+"),
            rtol=0, atol=1e-12)
