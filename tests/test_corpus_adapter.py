from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
import yaml

from scorpus.loaders import load_corpus
from scorpus.pairing import Sampler, cross, make_groups
from perttf.model.corpus_adapter import GeneTokens, PertTFAdapterConfig, PertTFPairedBatchLoader, default_pair_sampler

LABELS = {"perturb_label": "perturbation", "cell_context": "celltype", "batch_id": "batch"}
GENES = ["GENE_A", "GENE_B", "GENE_C", "GENE_D"]


def _corpus(tmp_path: Path, datasets: list[dict]):
    """A small aggregate Lance corpus. Each dataset: genes, cells as (perturbation, size factor,
    expressed local genes, counts), and optional HVG ranks (the 2 highest are the default HVGs)."""
    root = tmp_path / "corpus"
    entries, rows, start = [], [], 0
    for index, dataset in enumerate(datasets):
        name, genes, cells = dataset["name"], dataset["genes"], dataset["cells"]
        entries.append({
            "dataset_id": name, "join_mode": "create_new" if index == 0 else "append_routed",
            "dataset_index": index, "cell_count": len(cells), "global_start": start, "global_end": start + len(cells),
        })
        start += len(cells)
        meta = root / "meta" / name
        (meta / "canonical_meta").mkdir(parents=True)
        pl.DataFrame({
            "cell_id": [f"{name}_{i}" for i in range(len(cells))],
            "dataset_id": [name] * len(cells),
            "size_factor": [cell[1] for cell in cells],
            "cell_context": ["T_cell"] * len(cells),
            "perturb_label": [cell[0] for cell in cells],
            "batch_id": dataset.get("batch", [f"{name}_batch"] * len(cells)),
        }).write_parquet(meta / "canonical_meta" / "canonical-obs.parquet")
        pl.DataFrame({
            "origin_index": np.arange(len(genes), dtype=np.int32), "gene_id": genes,
            "canonical_gene_id": genes, "global_id": np.arange(len(genes), dtype=np.int32),
        }).write_parquet(meta / "canonical_meta" / "canonical-var.parquet")
        ranks = np.asarray(dataset.get("hvg_ranks", range(1, len(genes) + 1)), dtype=np.int32)
        pl.DataFrame({"origin_index": np.arange(len(genes), dtype=np.int32), "hvg_rank": ranks,
                      "selected_at_default_n_hvg": ranks <= 2}).write_parquet(meta / "hvg.parquet")
        rows += [(cell[2], cell[3]) for cell in cells]
    with open(root / "corpus-index.yaml", "w") as handle:
        yaml.safe_dump({"kind": "corpus-index", "contract_version": "0.3.0", "corpus_id": "test",
                        "global_metadata": {"backend": "lance", "topology": "aggregate"}, "datasets": entries}, handle)
    (root / "matrix").mkdir()
    lance.write_dataset(pa.table({
        "expressed_gene_indices": pa.array([genes for genes, _ in rows], type=pa.list_(pa.int32())),
        "expression_counts": pa.array([counts for _, counts in rows], type=pa.list_(pa.int32())),
    }), str(root / "matrix" / "aggregated-cells.lance"))
    return load_corpus(str(root))


def _small(tmp_path):
    """Rows 0: WT, 1: KO_TP53, 2: KO_STAT1 of one dataset with genes A-D."""
    return _corpus(tmp_path, [{"name": "ds0", "genes": GENES, "cells": [
        ("WT", 1.0, [0, 2], [5, 1]), ("KO_TP53", 1.2, [1, 3], [3, 2]), ("KO_STAT1", 1.1, [0, 1], [4, 6]),
    ]}])


def _mixed(tmp_path):
    """Two datasets with different panels: ds0 (genes A, B) rows 0-1, ds1 (genes B, C) rows 2-3."""
    return _corpus(tmp_path, [
        {"name": "ds0", "genes": ["GENE_A", "GENE_B"], "cells": [("WT", 1.0, [0], [5]), ("KO_TP53", 1.2, [1], [4])]},
        {"name": "ds1", "genes": ["GENE_B", "GENE_C"], "cells": [("WT", 0.9, [0], [2]), ("KO_STAT1", 1.1, [1], [7])]},
    ])


def _loader(corpus, rows=None, *, sampler=None, cover="source", perturbed_to_self=True, context_columns=(),
            tokens=None, labels=None, seq_len=4, batch_size=8, seed=0, num_workers=0, multiprocessing_context=None,
            label_fields=LABELS, device="cpu", **config):
    rows = np.arange(len(corpus.metadata_index)) if rows is None else np.asarray(rows)
    sampler = sampler or default_pair_sampler(
        corpus.metadata_index.df[rows], context_columns=context_columns, perturbation_column="perturb_label",
        control_labels=("WT",), perturbed_to_self=perturbed_to_self,
    )
    return PertTFPairedBatchLoader(
        corpus, sampler=sampler, rows=rows, config=PertTFAdapterConfig(label_fields=label_fields, **{"mask_ratio": 0.0, **config}),
        tokens=tokens or GeneTokens.from_corpus(corpus), seq_len=seq_len, batch_size=batch_size, cover=cover,
        labels=labels, seed=seed, num_workers=num_workers, multiprocessing_context=multiprocessing_context,
        device=device,
    )


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


def test_values_are_log1p_counts_at_the_source_genes(tmp_path):
    corpus = _small(tmp_path)
    loader = _loader(corpus, normalization_scale=1.5)
    batch = loader.build_batch(np.array([0, 1]), np.array([1, 1]), seed=5)

    dense = np.array([[5, 0, 1, 0], [0, 3, 0, 2]], dtype=np.float32)
    size_factor = np.array([1.0, 1.2], dtype=np.float32)
    expected = np.log1p(dense / size_factor[:, None] * 1.5)
    assert torch.equal(batch["gene_ids"], batch["next_gene_ids"])
    assert (batch["gene_ids"][:, 0] == 1).all() and (batch["target_values"][:, 0] == -3).all()
    genes = batch["gene_ids"][:, 1:] - 4                                    # all 4 genes, in sampled order
    assert sorted(genes[0].tolist()) == [0, 1, 2, 3]
    for row, target_row in ((0, 1), (1, 1)):
        np.testing.assert_allclose(batch["target_values"][row, 1:], expected[row, genes[row]], rtol=1e-6)
        np.testing.assert_allclose(batch["target_values_next"][row, 1:], expected[target_row, genes[row]], rtol=1e-6)
    torch.testing.assert_close(batch["sf"], torch.tensor([[1.5], [1.5 / 1.2]]))
    torch.testing.assert_close(batch["sf_next"], torch.tensor([[1.5 / 1.2], [1.5 / 1.2]]))
    assert torch.equal(batch["values"], batch["target_values"])            # mask_ratio 0
    assert batch["index"].tolist() == [0, 1] and batch["next_index"].tolist() == [1, 1]
    assert batch["perturbation_labels"].tolist() == [0, 1] and batch["perturbation_labels_next"].tolist() == [1, 1]
    assert torch.count_nonzero(batch["ps"]) == 0 and batch["ps"].shape == (2, 1)


def test_masking_spares_cls_and_targets(tmp_path):
    loader = _loader(_small(tmp_path), mask_ratio=1.0)
    batch = loader.build_batch(np.array([0]), np.array([1]), seed=3)
    assert batch["values"][0].tolist() == [-3, -1, -1, -1, -1]
    assert -1 not in batch["target_values"][0, 1:].tolist()


def test_genes_without_a_token_are_never_sampled(tmp_path):
    corpus = _small(tmp_path)
    tokens = GeneTokens.from_corpus(corpus, {"<pad>": 0, "<cls>": 1, "GENE_A": 17, "GENE_B": 23})
    loader = _loader(corpus, tokens=tokens, seq_len=2)
    for seed in range(20):
        batch = loader.build_batch(np.array([0, 1]), np.array([0, 1]), seed)
        assert set(batch["gene_ids"][:, 1:].flatten().tolist()) <= {17, 23}


def test_full_expression_on_mixed_panels(tmp_path):
    corpus = _mixed(tmp_path)
    loader = _loader(corpus, seq_len=2)
    batch = loader.build_batch(np.array([0, 2]), np.array([1, 3]), seed=29)

    column = {gene: 1 + corpus.feature_registry.global_feature_ids.index(gene) for gene in ("GENE_A", "GENE_B", "GENE_C")}
    assert batch["full_gene_ids"].tolist() == [[1, 4, 5, 6]] * 2              # CLS, then every corpus gene's token

    def row(values):
        out = [-3.0, 0.0, 0.0, 0.0]
        for gene, value in values.items():
            out[column[gene]] = float(np.float32(value))
        return out

    np.testing.assert_allclose(batch["full_expr"][0], row({"GENE_A": np.log1p(5 / 1.0)}), rtol=1e-6)
    np.testing.assert_allclose(batch["full_expr_next"][0], row({"GENE_B": np.log1p(4 / 1.2)}), rtol=1e-6)
    np.testing.assert_allclose(batch["full_expr"][1], row({"GENE_B": np.log1p(2 / 0.9)}), rtol=1e-6)
    np.testing.assert_allclose(batch["full_expr_next"][1], row({"GENE_C": np.log1p(7 / 1.1)}), rtol=1e-6)
    measured = {0: {"GENE_A", "GENE_B"}, 1: {"GENE_B", "GENE_C"}}
    for i, dataset in enumerate((0, 1)):
        expected_mask = [True] + [False] * 3
        for gene in measured[dataset]:
            expected_mask[column[gene]] = True
        assert batch["full_expr_mask"][i].tolist() == expected_mask
        assert batch["full_expr_next_mask"][i].tolist() == expected_mask


def test_sampling_modes(tmp_path):
    """Two datasets with genes A-D, opposite HVG ranks; one gene drawn per cell."""
    corpus = _corpus(tmp_path, [
        {"name": "ds0", "genes": GENES, "hvg_ranks": [1, 2, 3, 4],
         "cells": [("WT", 1.0, [0, 2], [5, 1]), ("KO", 1.1, [1, 3], [3, 2])]},
        {"name": "ds1", "genes": GENES, "hvg_ranks": [4, 3, 2, 1],
         "cells": [("WT", 1.0, [0, 1], [2, 4]), ("KO", 1.1, [2, 3], [6, 7])]},
    ])

    def drawn(**config):
        loader = _loader(corpus, seq_len=1, **config)
        cells = np.array([0, 2])                                            # ds0 WT, ds1 WT
        return np.array([loader.build_batch(cells, cells, seed)["gene_ids"][:, 1].tolist() for seed in range(100)]) - 4

    top1 = drawn(sampling_mode="hvg", hvg_weight=1000.0, hvg_top_k=1)
    assert (top1[:, 0] == 0).mean() > 0.95 and (top1[:, 1] == 3).mean() > 0.95
    default_hvg = drawn(sampling_mode="hvg", hvg_weight=1000.0)            # each dataset's 2 default HVGs
    assert np.isin(default_hvg[:, 0], [0, 1]).mean() > 0.95 and np.isin(default_hvg[:, 1], [2, 3]).mean() > 0.95
    expressed = drawn(sampling_mode="expressed", expressed_weight=1000.0)
    assert np.isin(expressed[:, 0], [0, 2]).mean() > 0.95 and np.isin(expressed[:, 1], [0, 1]).mean() > 0.95
    assert set(drawn(sampling_mode="uniform")[:, 0].tolist()) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="sampling_mode"):
        _loader(corpus, sampling_mode="HVG")


def test_labels(tmp_path):
    corpus = _mixed(tmp_path)
    loader = _loader(corpus, seq_len=1, label_fields={**LABELS, "dataset_id": "dataset"},
                     labels={"perturbation": {"KO_STAT1": 0, "KO_TP53": 1, "WT": 2}})
    batch = loader.build_batch(np.array([0]), np.array([3]), seed=1)
    assert batch["dataset_labels"].tolist() == [0] and batch["dataset_labels_next"].tolist() == [1]
    assert batch["perturbation_labels"].tolist() == [2] and batch["perturbation_labels_next"].tolist() == [0]
    assert loader.label_to_index_by_name["batch"] == {"ds0_batch": 0, "ds1_batch": 1}


def test_missing_label_raises(tmp_path):
    corpus = _corpus(tmp_path, [{"name": "ds0", "genes": GENES, "batch": ["b0", None],
                                 "cells": [("WT", 1.0, [0], [1]), ("KO", 1.0, [1], [2])]}])
    with pytest.raises(ValueError, match="batch_id"):
        _loader(corpus)


def test_default_training_pairs(tmp_path):
    """Every cell once as a source; controls go to a perturbed cell of their dataset, perturbed cells to themselves."""
    corpus = _mixed(tmp_path)
    loader = _loader(corpus, batch_size=4, seq_len=1, context_columns=("dataset_id",))
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


def test_spawn_workers_give_the_same_batches(tmp_path):
    corpus = _mixed(tmp_path)
    single = list(_loader(corpus, batch_size=2, seq_len=1, mask_ratio=0.5))
    spawned = list(_loader(corpus, batch_size=2, seq_len=1, mask_ratio=0.5, num_workers=2, multiprocessing_context="spawn"))
    assert len(single) == len(spawned) == 2
    for a, b in zip(single, spawned):
        assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_batches(tmp_path):
    corpus = _mixed(tmp_path)
    cells, partners = np.array([0, 2]), np.array([1, 3])
    cpu = _loader(corpus, seq_len=1).build_batch(cells, partners, seed=4)
    gpu = _loader(corpus, seq_len=1, device="cuda").build_batch(cells, partners, seed=4)
    assert gpu.keys() == cpu.keys() and all(value.device.type == "cuda" for value in gpu.values())
    for key in ("index", "next_index", "sf", "sf_next", "full_gene_ids", "full_expr", "full_expr_next",
                "full_expr_mask", "full_expr_next_mask", "perturbation_labels", "perturbation_labels_next"):
        torch.testing.assert_close(gpu[key].cpu(), cpu[key])
    # GPU random numbers may pick other genes; the values must be those of the picked genes.
    column = gpu["gene_ids"][:, 1:] - 3                                     # token 4 + gene -> full_expr column 1 + gene
    torch.testing.assert_close(gpu["target_values"][:, 1:], gpu["full_expr"].gather(1, column))
    torch.testing.assert_close(gpu["target_values_next"][:, 1:], gpu["full_expr_next"].gather(1, column))
