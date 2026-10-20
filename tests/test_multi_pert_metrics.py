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

def _adata(extra_b=0):
    labels = ["WT"] * 40 + ["A+B"] * 40 + ["A"] * 40 + ["B+C"] * 2 + ["B"] * extra_b
    obs = pd.DataFrame({"celltype": ["x"] * len(labels), "genotype": labels})
    return ad.AnnData(X=np.zeros((len(labels), 2), dtype=np.float32), obs=obs)

def test_split_check_unseen_combination_and_warning():
    a = _adata(extra_b=20); idx = np.arange(a.n_obs)
    train = np.concatenate([idx[:30], idx[80:110], idx[122:142]]); valid = np.concatenate([idx[30:40], idx[40:80], idx[120:122]])
    with pytest.warns(UserWarning, match="single"):
        r = check_train_valid_split(a, train, valid, mode="pert", pert_delimiter="+", perturbation_unit="set", min_cells=30)
    assert ("x", "A+B") in r["combination_types"]["unseen_combination"]
    assert ("x", "B+C") in r["combination_types"]["unseen_perturbation"]

def test_split_check_single_unit_uses_components():
    a = _adata(); idx = np.arange(a.n_obs)
    train = np.concatenate([idx[:30], idx[80:110]]); valid = np.concatenate([idx[30:40], idx[40:80]])
    r = check_train_valid_split(a, train, valid, mode="pert", pert_delimiter="+", perturbation_unit="single", min_cells=30)
    assert ("x", "A") in r["combination_types"]["seen_combination"] and ("x", "B") in r["combination_types"]["unseen_perturbation"]
