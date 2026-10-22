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

def _naive_moments(chunks):
    out = {}
    for v, c, p in chunks:
        for key in {(str(x), str(y)) for x, y in zip(c, p)}:
            m = np.array([(str(x), str(y)) == key for x, y in zip(c, p)])
            n, s, q = out.get(key, (0, 0.0, 0.0))
            out[key] = (n + m.sum(), s + v[m].sum(0), q + np.square(v[m]).sum(0))
    return out

def test_group_moments_update_matches_per_key_sums_across_chunks():
    from perttf.utils.pert_metrics import GroupMoments
    rng = np.random.default_rng(0)
    chunks = [(rng.normal(size=(50, 4)), rng.choice(["x", "y"], 50), rng.choice(["A", "B", "WT", "C"], 50)) for _ in range(3)]
    chunks.append((np.zeros((0, 4)), np.array([], dtype=str), np.array([], dtype=str)))
    gm = GroupMoments()
    for v, c, p in chunks:
        gm.update(v, c, p)
    ref = _naive_moments(chunks)
    assert set(gm._moments) == set(ref)
    for k, (n, s, q) in ref.items():
        assert gm._moments[k][0] == n
        np.testing.assert_allclose(gm._moments[k][1], s); np.testing.assert_allclose(gm._moments[k][2], q)

def test_group_moments_update_many_groups():
    from perttf.utils.pert_metrics import GroupMoments
    n = 20000
    p = np.array([f"g{i}" for i in range(n)])          # ~one group per row, as at high MOI in single mode
    gm = GroupMoments(); gm.update(np.ones((n, 3)), np.array(["x"] * n), p)
    g = gm.finalize()
    assert len(g) == n and g[("x", "g7")].n == 1

def _highmoi_adata():
    labels = ["WT"] * 20 + [f"A+B+G{i}" for i in range(20)] + ["A+C"] * 5 + ["A+D"]
    obs = pd.DataFrame({"celltype": ["x"] * len(labels), "genotype": labels})
    return ad.AnnData(X=np.zeros((len(labels), 2), dtype=np.float32), obs=obs)

def test_split_check_lochness_single_unit_compares_components():
    a = _highmoi_adata(); idx = np.arange(a.n_obs)
    train = np.concatenate([idx[:15], idx[20:30], idx[40:45]]); valid = np.concatenate([idx[15:20], idx[30:40], idx[45:46]])
    with pytest.raises(ValueError, match="absent from training"):        # set labels G30.. are unique per cell
        check_train_valid_split(a, train, valid, mode="lochness")
    with pytest.warns(UserWarning, match="components absent from training"):
        r = check_train_valid_split(a, train, valid, mode="lochness", pert_delimiter="+", perturbation_unit="single")
    assert {"G10", "D"} <= set(r["column_coverage"]["genotype"]["valid_only"])
    assert "A" not in r["column_coverage"]["genotype"]["valid_only"]

def test_split_check_lochness_set_unit_keeps_whole_label_check():
    a = _highmoi_adata(); idx = np.arange(a.n_obs)
    train = np.concatenate([idx[:15], idx[20:30], idx[40:45]]); valid = np.concatenate([idx[15:20], idx[30:40], idx[45:46]])
    with pytest.raises(ValueError, match="absent from training"):
        check_train_valid_split(a, train, valid, mode="lochness", pert_delimiter="+", perturbation_unit="set")
