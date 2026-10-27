from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from scipy import sparse

from scorpus import concat, from_h5ad, load_corpus
from scorpus.pairing import Sampler, cross, make_groups
from perttf.model.corpus_adapter import GeneTokens, PertTFAdapterConfig, PertTFPairedBatchLoader, default_pair_sampler
from perttf.utils.custom_tokenizer import weighted_sample

LABELS = {"perturb_label": "perturbation", "cell_context": "celltype", "batch_id": "batch"}
GENES = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]


def _standalone(tmp_path: Path, dataset: dict):
    """One dataset converted from a small h5ad (which gives each cell its scorpus_total_counts).
    Cells are (perturbation, expressed local genes, counts). With "hvg" = (highly_variable,
    highly_variable_rank), scanpy's HVG columns are saved as ex_var and the corpus reloaded with them."""
    name, genes, cells = dataset["name"], dataset["genes"], dataset["cells"]
    counts = np.zeros((len(cells), len(genes)), dtype=np.int32)
    for row, (_, expressed, values) in enumerate(cells):
        counts[row, expressed] = values
    # Categorical labels: anndata stores a string column as categorical only when it repeats values,
    # and concat cannot stack one member's strings onto another's categoricals.
    obs = pd.DataFrame({
        "cell_context": pd.Categorical(["T_cell"] * len(cells)),
        "perturb_label": pd.Categorical([cell[0] for cell in cells]),
        "batch_id": pd.Categorical(dataset.get("batch", [f"{name}_batch"] * len(cells))),
    }, index=[f"{name}_{i}" for i in range(len(cells))])
    source = tmp_path / f"{name}.h5ad"
    ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(source)
    corpus = from_h5ad(source, tmp_path / name)
    if "hvg" not in dataset:
        return corpus
    corpus.var["highly_variable"], corpus.var["highly_variable_rank"] = dataset["hvg"]
    corpus.save_ex_var()
    return load_corpus(tmp_path / name, ex_var=True)


def _corpus(tmp_path: Path, datasets: list[dict]):
    """One standalone corpus, or a concat view of several with genes matched by name."""
    members = {dataset["name"]: _standalone(tmp_path, dataset) for dataset in datasets}
    if len(members) == 1:
        return members[datasets[0]["name"]]
    return concat(
        members, obs_columns={column: {name: column for name in members} for column in LABELS},
        feature_key={name: "index" for name in members},
    )


def _small(tmp_path):
    """Rows 0: WT, 1: KO_TP53, 2: KO_STAT1 of one dataset with genes A-D; totals 6, 5, 10."""
    return _corpus(tmp_path, [{"name": "ds0", "genes": GENES, "cells": [
        ("WT", [0, 2], [5, 1]), ("KO_TP53", [1, 3], [3, 2]), ("KO_STAT1", [0, 1], [4, 6]),
    ]}])


def _mixed(tmp_path):
    """Two datasets with different panels: ds0 (genes A, B) rows 0-1, totals 5, 4; ds1 (genes B, C)
    rows 2-3, totals 2, 8."""
    return _corpus(tmp_path, [
        {"name": "ds0", "genes": ["GENE_A", "GENE_B"], "cells": [("WT", [0], [5]), ("KO_TP53", [1], [4])]},
        {"name": "ds1", "genes": ["GENE_B", "GENE_C"], "cells": [("WT", [0], [2]), ("KO_STAT1", [1], [8])]},
    ])


def _loader(corpus, rows=None, *, sampler=None, cover="source", perturbed_to_self=True, context_columns=(),
            tokens=None, labels=None, seq_len=4, batch_size=8, seed=0, num_workers=0, multiprocessing_context=None,
            label_fields=LABELS, device="cpu", build_in_workers=False, batches_per_read=1, **config):
    rows = np.arange(len(corpus.metadata_index)) if rows is None else np.asarray(rows)
    sampler = sampler or default_pair_sampler(
        corpus.metadata_index.df[rows], context_columns=context_columns, perturbation_column="perturb_label",
        control_labels=("WT",), perturbed_to_self=perturbed_to_self,
    )
    return PertTFPairedBatchLoader(
        corpus, sampler=sampler, rows=rows, config=PertTFAdapterConfig(label_fields=label_fields, **{"mask_ratio": 0.0, **config}),
        tokens=tokens or GeneTokens.from_corpus(corpus), seq_len=seq_len, batch_size=batch_size, cover=cover,
        labels=labels, seed=seed, num_workers=num_workers, multiprocessing_context=multiprocessing_context,
        device=device, build_in_workers=build_in_workers, batches_per_read=batches_per_read,
    )


# Batches built in the main process (on the loader's device) and on the CPU in the workers.
BUILDS = pytest.mark.parametrize("in_workers", [False, True], ids=["device_build", "worker_build"])


def test_gene_tokens(tmp_path):
    corpus = _small(tmp_path)
    new = GeneTokens.from_corpus(corpus)
    assert list(new.stoi) == ["<pad>", "<cls>", "<unk>", "<eos>", *GENES]
    assert new.token_of_gene.tolist() == [4, 5, 6, 7] and new.tokenizable.all()
    assert (new.pad_id, new.cls_id) == (0, 1)

    pretrained = GeneTokens.from_corpus(corpus, {"<pad>": 0, "<cls>": 1, "GENE_A": 17, "C_alias": 23},
                                        feature_id_to_token={"GENE_A": "GENE_A", "GENE_C": "C_alias"})
    assert pretrained.token_of_gene.tolist() == [17, 0, 23, 0]
    assert pretrained.tokenizable.tolist() == [True, False, True, False]


@BUILDS
def test_values_are_log1p_counts_at_the_source_genes(tmp_path, in_workers):
    corpus = _small(tmp_path)
    loader = _loader(corpus, normalization_scale=1.5, build_in_workers=in_workers)
    batch = loader.build_batch(np.array([0, 1]), np.array([1, 1]), seed=5)

    dense = np.array([[5, 0, 1, 0], [0, 3, 0, 2]], dtype=np.float32)
    size_factor = (np.array([6.0, 5.0]) / 6).astype(np.float32)             # totals over the median total 6
    expected = np.log1p(dense / size_factor[:, None] * 1.5)
    assert torch.equal(batch["gene_ids"], batch["next_gene_ids"])
    assert (batch["gene_ids"][:, 0] == 1).all() and (batch["target_values"][:, 0] == -3).all()
    genes = batch["gene_ids"][:, 1:] - 4                                    # all 4 genes, in sampled order
    assert sorted(genes[0].tolist()) == [0, 1, 2, 3]
    for row, target_row in ((0, 1), (1, 1)):
        np.testing.assert_allclose(batch["target_values"][row, 1:], expected[row, genes[row]], rtol=1e-6)
        np.testing.assert_allclose(batch["target_values_next"][row, 1:], expected[target_row, genes[row]], rtol=1e-6)
    torch.testing.assert_close(batch["sf"], torch.tensor([[1.5], [1.5 * 6 / 5]]))
    torch.testing.assert_close(batch["sf_next"], torch.tensor([[1.5 * 6 / 5], [1.5 * 6 / 5]]))
    assert torch.equal(batch["values"], batch["target_values"])            # mask_ratio 0
    assert batch["index"].tolist() == [0, 1] and batch["next_index"].tolist() == [1, 1]
    assert batch["perturbation_labels"].tolist() == [0, 1] and batch["perturbation_labels_next"].tolist() == [1, 1]
    assert torch.count_nonzero(batch["ps"]) == 0 and batch["ps"].shape == (2, 1)


@BUILDS
def test_masking_spares_cls_and_targets(tmp_path, in_workers):
    loader = _loader(_small(tmp_path), mask_ratio=1.0, build_in_workers=in_workers)
    batch = loader.build_batch(np.array([0]), np.array([1]), seed=3)
    assert batch["values"][0].tolist() == [-3, -1, -1, -1, -1]
    assert -1 not in batch["target_values"][0, 1:].tolist()


@BUILDS
def test_genes_without_a_token_are_never_sampled(tmp_path, in_workers):
    corpus = _small(tmp_path)
    tokens = GeneTokens.from_corpus(corpus, {"<pad>": 0, "<cls>": 1, "GENE_A": 17, "GENE_B": 23})
    loader = _loader(corpus, tokens=tokens, seq_len=2, build_in_workers=in_workers)
    for seed in range(20):
        batch = loader.build_batch(np.array([0, 1]), np.array([0, 1]), seed)
        assert set(batch["gene_ids"][:, 1:].flatten().tolist()) <= {17, 23}


@BUILDS
def test_full_expression_on_mixed_panels(tmp_path, in_workers):
    corpus = _mixed(tmp_path)
    loader = _loader(corpus, seq_len=2, build_in_workers=in_workers)
    batch = loader.build_batch(np.array([0, 2]), np.array([1, 3]), seed=29)

    column = {gene: 1 + corpus.feature_registry.global_feature_ids.index(gene) for gene in ("GENE_A", "GENE_B", "GENE_C")}
    assert batch["full_gene_ids"].tolist() == [[1, 4, 5, 6]] * 2              # CLS, then every corpus gene's token

    def row(values):
        out = [-3.0, 0.0, 0.0, 0.0]
        for gene, value in values.items():
            out[column[gene]] = float(np.float32(value))
        return out

    # Size factors are over each dataset's own median total: ds0 4.5, ds1 5.
    np.testing.assert_allclose(batch["full_expr"][0], row({"GENE_A": np.log1p(5 / (5 / 4.5))}), rtol=1e-6)
    np.testing.assert_allclose(batch["full_expr_next"][0], row({"GENE_B": np.log1p(4 / (4 / 4.5))}), rtol=1e-6)
    np.testing.assert_allclose(batch["full_expr"][1], row({"GENE_B": np.log1p(2 / (2 / 5))}), rtol=1e-6)
    np.testing.assert_allclose(batch["full_expr_next"][1], row({"GENE_C": np.log1p(8 / (8 / 5))}), rtol=1e-6)
    measured = {0: {"GENE_A", "GENE_B"}, 1: {"GENE_B", "GENE_C"}}
    for i, dataset in enumerate((0, 1)):
        expected_mask = [True] + [False] * 3
        for gene in measured[dataset]:
            expected_mask[column[gene]] = True
        assert batch["full_expr_mask"][i].tolist() == expected_mask
        assert batch["full_expr_next_mask"][i].tolist() == expected_mask


@BUILDS
def test_sampling_modes(tmp_path, in_workers):
    """Two datasets with genes A-D and saved scanpy HVGs: ds0 A, B (A ranked first); ds1 C, D (D first)."""
    no = np.nan
    corpus = _corpus(tmp_path, [
        {"name": "ds0", "genes": GENES, "hvg": ([True, True, False, False], [0.0, 1.0, no, no]),
         "cells": [("WT", [0, 2], [5, 1]), ("KO", [1, 3], [3, 2])]},
        {"name": "ds1", "genes": GENES, "hvg": ([False, False, True, True], [no, no, 1.0, 0.0]),
         "cells": [("WT", [0, 1], [2, 4]), ("KO", [2, 3], [6, 7])]},
    ])

    def drawn(seq_len, **config):
        loader = _loader(corpus, seq_len=seq_len, build_in_workers=in_workers, **config)
        cells = np.array([0, 2])                                            # ds0 WT, ds1 WT
        return [(loader.build_batch(cells, cells, seed)["gene_ids"][:, 1:] - 4).numpy() for seed in range(50)]

    # hvg: non_hvg_size other genes, then the HVGs in the remaining places.
    for genes in drawn(2, sampling_mode="hvg", non_hvg_size=1):
        assert np.isin(genes[0], [0, 1]).sum() == 1 and np.isin(genes[1], [2, 3]).sum() == 1
    for genes in drawn(2, sampling_mode="hvg", non_hvg_size=1, hvg_top_k=1):
        assert 0 in genes[0] and 3 in genes[1]
    # expressed: round(0.5 * 2) = 1 expressed gene (ds0 WT: A, C; ds1 WT: A, B), then 1 unexpressed.
    for genes in drawn(2, sampling_mode="expressed", nonzero_prop=0.5):
        assert np.isin(genes[0], [0, 2]).sum() == 1 and np.isin(genes[1], [0, 1]).sum() == 1
    assert set(np.concatenate([genes[0] for genes in drawn(1, sampling_mode="uniform")]).tolist()) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="sampling_mode"):
        _loader(corpus, sampling_mode="HVG")
    with pytest.raises(ValueError, match="non_hvg_size"):
        _loader(corpus, sampling_mode="hvg", seq_len=2)                  # default non_hvg_size 1000


@BUILDS
@pytest.mark.parametrize("config", [
    {"sampling_mode": "uniform"},
    {"sampling_mode": "expressed"},
    {"sampling_mode": "expressed", "nonzero_prop": 0.5, "fix_nonzero_prop": True},
    {"sampling_mode": "hvg", "non_hvg_size": 3, "hvg_top_k": 6},
], ids=["uniform", "expressed", "expressed_fixed", "hvg"])
def test_quotas_match_pertTF(tmp_path, in_workers, config):
    """Per cell, as many genes from each group as pertTF's weighted_sample takes for the same cell."""
    rng = np.random.default_rng(0)
    n_genes, seq_len = 12, 8
    # At most 8 expressed genes: with more, weighted_sample raises for lack of unexpressed genes.
    cells = []
    for i in range(30):
        expressed = np.sort(rng.choice(n_genes, rng.integers(0, 9), replace=False))
        cells.append(("WT" if i % 3 == 0 else "KO", expressed.tolist(), (expressed + 1).tolist()))
    ranks = rng.permutation(n_genes) + 1
    # 8 saved HVGs (scanpy ranks from 0), of which hvg_top_k takes the 6 highest ranked.
    hvg = (ranks <= 8, np.where(ranks <= 8, ranks - 1.0, np.nan))
    corpus = _corpus(tmp_path, [{"name": "ds0", "genes": [f"G{i}" for i in range(n_genes)], "hvg": hvg,
                                 "cells": cells}])
    loader = _loader(corpus, seq_len=seq_len, build_in_workers=in_workers, **config)
    rows = np.arange(len(cells))
    drawn = loader.build_batch(rows, rows, seed=3)["gene_ids"][:, 1:].numpy()
    mode = config["sampling_mode"]
    for row, (_, expressed, counts) in enumerate(cells):
        values = np.zeros(n_genes)
        values[expressed] = counts
        group = values > 0 if mode == "expressed" else ranks <= 6 if mode == "hvg" else np.ones(n_genes, dtype=bool)
        genes = drawn[row][drawn[row] != 0] - 4                             # pad token 0; gene g is token 4 + g
        assert len(set(genes.tolist())) == len(genes)
        index = weighted_sample(values, seq_len, rng=rng, mode={"uniform": "simple"}.get(mode, mode),
                                non_zero_proportion=config.get("nonzero_prop", 0.7),
                                fixed_ratio=config.get("fix_nonzero_prop", False),
                                hvg_nonhvg=(np.flatnonzero(group), np.flatnonzero(~group)),
                                non_hvg_size=config.get("non_hvg_size", 1000))
        assert (group[genes].sum(), (~group[genes]).sum()) == (group[index].sum(), (~group[index]).sum())


def test_device_and_worker_builds_agree(tmp_path):
    """Everything but the random draws is the same; pairs across panels read the target where measured."""
    corpus = _mixed(tmp_path)
    cells, partners = np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0])
    device = _loader(corpus, seq_len=2, mask_ratio=0.5).build_batch(cells, partners, seed=4)
    workers = _loader(corpus, seq_len=2, mask_ratio=0.5, build_in_workers=True).build_batch(cells, partners, seed=4)
    assert device.keys() == workers.keys()
    for key in ("index", "next_index", "sf", "sf_next", "ps", "ps_next", "full_gene_ids", "full_expr", "full_expr_next",
                "full_expr_mask", "full_expr_next_mask", "perturbation_labels", "perturbation_labels_next",
                "batch_labels", "celltype_labels_next"):
        torch.testing.assert_close(workers[key], device[key])
    for batch in (device, workers):
        column = batch["gene_ids"][:, 1:] - 3                               # token 4 + gene -> full_expr column 1 + gene
        assert column.shape == (4, 2)                                       # each cell's whole 2-gene pool
        torch.testing.assert_close(batch["target_values"][:, 1:], batch["full_expr"].gather(1, column))
        measured_next = batch["full_expr_next_mask"].gather(1, column)
        expected_next = torch.where(measured_next, batch["full_expr_next"].gather(1, column), torch.tensor(-2.0))
        torch.testing.assert_close(batch["target_values_next"][:, 1:], expected_next)
        assert ((batch["values"] == -1).sum(dim=1) == 1).all()             # int(2 * 0.5) per row


@BUILDS
def test_masking_replaces_a_fixed_number_per_row(tmp_path, in_workers):
    loader = _loader(_small(tmp_path), mask_ratio=0.5, build_in_workers=in_workers)
    for seed in range(10):
        batch = loader.build_batch(np.array([0, 1, 2]), np.array([0, 1, 2]), seed)
        assert ((batch["values"] == -1).sum(dim=1) == 2).all()             # int(4 * 0.5)
        kept = batch["values"] != -1
        assert torch.equal(batch["values"][kept], batch["target_values"][kept])


@BUILDS
def test_labels(tmp_path, in_workers):
    corpus = _mixed(tmp_path)
    loader = _loader(corpus, seq_len=1, build_in_workers=in_workers, label_fields={**LABELS, "dataset_id": "dataset"},
                     labels={"perturbation": {"KO_STAT1": 0, "KO_TP53": 1, "WT": 2}})
    batch = loader.build_batch(np.array([0]), np.array([3]), seed=1)
    assert batch["dataset_labels"].tolist() == [0] and batch["dataset_labels_next"].tolist() == [1]
    assert batch["perturbation_labels"].tolist() == [2] and batch["perturbation_labels_next"].tolist() == [0]
    assert loader.label_to_index_by_name["batch"] == {"ds0_batch": 0, "ds1_batch": 1}


def test_missing_label_raises(tmp_path):
    corpus = _corpus(tmp_path, [{"name": "ds0", "genes": GENES, "batch": ["b0", None],
                                 "cells": [("WT", [0], [1]), ("KO", [1], [2])]}])
    with pytest.raises(ValueError, match="batch_id"):
        _loader(corpus)


@BUILDS
def test_size_factors_from_stored_totals(tmp_path, in_workers):
    """A cell's size factor is its scorpus_total_counts over the median positive total of its dataset,
    over all the dataset's cells, not only the loader's rows; a cell without counts gets 1."""
    corpus = _corpus(tmp_path, [
        {"name": "ds0", "genes": GENES,                                     # totals 6, 0, 2, 12: median 6
         "cells": [("WT", [0, 1], [2, 4]), ("KO", [], []), ("KO", [2], [2]), ("WT", [3], [12])]},
        {"name": "ds1", "genes": GENES, "cells": [("WT", [0], [3]), ("KO", [1], [5])]},  # median 4
    ])
    loader = _loader(corpus, rows=[0, 1, 2, 4, 5], build_in_workers=in_workers)   # leaves out the cell with 12
    cells = np.arange(5)
    batch = loader.build_batch(cells, cells, seed=0)
    size_factor = np.array([6 / 6, 1.0, 2 / 6, 3 / 4, 5 / 4], dtype=np.float32)
    torch.testing.assert_close(batch["sf"], torch.from_numpy(1 / size_factor).reshape(-1, 1))


def test_corpus_without_total_counts_raises(tmp_path):
    corpus = _small(tmp_path)
    corpus.obs.drop(columns="scorpus_total_counts", inplace=True)          # as converted before scorpus stored totals
    with pytest.raises(ValueError, match="scorpus_total_counts"):
        _loader(corpus)


@BUILDS
def test_default_training_pairs(tmp_path, in_workers):
    """Every cell once as a source; controls go to a perturbed cell of their dataset, perturbed cells to themselves."""
    corpus = _mixed(tmp_path)
    loader = _loader(corpus, batch_size=4, seq_len=1, context_columns=("dataset_id",), build_in_workers=in_workers)
    for epoch in range(5):
        loader.set_epoch(epoch)
        (batch,) = list(loader)
        assert dict(zip(batch["index"].tolist(), batch["next_index"].tolist())) == {0: 1, 1: 1, 2: 3, 3: 3}


def test_default_pairs_without_partners_are_self_pairs(tmp_path):
    corpus = _small(tmp_path)
    for row in (0, 1):                                                      # a lone control, a lone perturbed cell
        (batch,) = list(_loader(corpus, rows=[row], seq_len=3))
        assert batch["index"].tolist() == batch["next_index"].tolist() == [row]
        assert torch.equal(batch["target_values"], batch["target_values_next"])


def test_default_validation_pairs(tmp_path):
    """Every cell once as a target; perturbed targets get the control, the control target itself."""
    corpus = _small(tmp_path)
    (batch,) = list(_loader(corpus, cover="target", perturbed_to_self=False, seq_len=3))
    assert batch["index"].tolist() == [0, 0, 0] and sorted(batch["next_index"].tolist()) == [0, 1, 2]
    with pytest.raises(ValueError, match="no control"):
        _loader(corpus, rows=[1, 2], cover="target", perturbed_to_self=False)


def test_custom_pair_table(tmp_path):
    """Perturbed cells -> controls, written directly with scorpus.pairing."""
    corpus = _small(tmp_path)
    rows = np.array([0, 1])
    groups = make_groups(corpus.metadata_index.df[rows], by=["perturb_label"])
    is_ctrl = pl.col("perturb_label") == "WT"
    (batch,) = list(_loader(corpus, rows=rows, sampler=Sampler(groups, cross(groups, source=~is_ctrl, target=is_ctrl)), seq_len=3))
    assert batch["index"].tolist() == [1] and batch["next_index"].tolist() == [0]


def test_epochs_repeat_and_reorder(tmp_path):
    loader = _loader(_mixed(tmp_path), batch_size=1, seq_len=1)
    order = {}
    for epoch in (0, 0, 1):
        loader.set_epoch(epoch)
        order.setdefault(epoch, []).append([int(batch["index"][0]) for batch in loader])
    assert order[0][0] == order[0][1] and order[1][0] != order[0][0]
    assert len(loader) == 4 and len(loader.dataset) == 4


@BUILDS
def test_spawn_workers_and_several_batches_per_read_give_the_same_batches(tmp_path, in_workers):
    corpus = _mixed(tmp_path)
    options = dict(batch_size=1, seq_len=2, mask_ratio=0.5, build_in_workers=in_workers)
    single = list(_loader(corpus, **options))
    grouped = list(_loader(corpus, batches_per_read=3, **options))
    spawned = list(_loader(corpus, batches_per_read=2, num_workers=2, multiprocessing_context="spawn", **options))
    assert len(single) == len(grouped) == len(spawned) == 4
    for a, b, c in zip(single, grouped, spawned):
        assert a.keys() == b.keys() == c.keys()
        assert all(torch.equal(a[key], b[key]) and torch.equal(a[key], c[key]) for key in a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_batches(tmp_path):
    corpus = _mixed(tmp_path)
    cells, partners = np.array([0, 2]), np.array([1, 3])
    cpu = _loader(corpus, seq_len=1).build_batch(cells, partners, seed=4)
    for in_workers in (False, True):
        gpu = _loader(corpus, seq_len=1, device="cuda", build_in_workers=in_workers).build_batch(cells, partners, seed=4)
        assert gpu.keys() == cpu.keys() and all(value.device.type == "cuda" for value in gpu.values())
        for key in ("index", "next_index", "sf", "sf_next", "full_gene_ids", "full_expr", "full_expr_next",
                    "full_expr_mask", "full_expr_next_mask", "perturbation_labels", "perturbation_labels_next"):
            torch.testing.assert_close(gpu[key].cpu(), cpu[key])
        # GPU random numbers may pick other genes; the values must be those of the picked genes.
        column = gpu["gene_ids"][:, 1:] - 3                                 # token 4 + gene -> full_expr column 1 + gene
        torch.testing.assert_close(gpu["target_values"][:, 1:], gpu["full_expr"].gather(1, column))
        torch.testing.assert_close(gpu["target_values_next"][:, 1:], gpu["full_expr_next"].gather(1, column))
