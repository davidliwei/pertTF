from dataclasses import replace
from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from scorpus.loaders import load_corpus
from perttf.model.corpus_adapter import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    PertTFPairedBatchBuilder,
    PertTFCorpusAdapter,
    PerturbationPairSampler,
)
from perttf.model.corpus_adapter import (
    _collate_perttf_raw_pair_batch,
    _PertTFPairExpressionDataset,
    _PertTFPairReadBatchSampler,
    _read_expression_raw_batch,
    _prepare_perttf_metadata,
)


def _write_canonical_obs(path: Path) -> None:
    frame = pl.DataFrame(
        {
            "cell_id": ["ds0_wt", "ds0_ko"],
            "dataset_id": ["ds0", "ds0"],
            "size_factor": [1.0, 1.2],
            "cell_context": ["T_cell", "T_cell"],
            "perturb_label": ["WT", "KO_TP53"],
            "batch_id": ["batch_0", "batch_1"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_canonical_var(path: Path, gene_names: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": np.arange(len(gene_names), dtype=np.int32),
            "gene_id": [f"ENSG_{gene_name}" for gene_name in gene_names],
            "canonical_gene_id": gene_names,
            "global_id": np.arange(len(gene_names), dtype=np.int32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_hvg_ranking(path: Path, gene_names: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "origin_index": np.arange(len(gene_names), dtype=np.int32),
            "feature_id": gene_names,
            "mean_log1p_expr": np.linspace(0.1, 1.0, len(gene_names), dtype=np.float32),
            "variance_log1p_expr": np.linspace(1.0, 2.0, len(gene_names), dtype=np.float32),
            "dispersion_log": np.linspace(0.5, 1.5, len(gene_names), dtype=np.float32),
            "dispersion_norm": np.linspace(1.5, 0.5, len(gene_names), dtype=np.float32),
            "hvg_rank": np.arange(1, len(gene_names) + 1, dtype=np.int32),
            "selected_at_default_n_hvg": np.arange(len(gene_names), dtype=np.int32) < 2,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(path))


def _write_aggregate_lance(
    path: Path,
    *,
    rows: list[tuple[list[int], list[int]]] | None = None,
) -> None:
    resolved_rows = rows or [([0, 2], [5, 1]), ([1, 3], [3, 2])]
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [genes for genes, _ in resolved_rows],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [counts for _, counts in resolved_rows],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def _build_small_pair_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-paired-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-paired-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [
            {
                "dataset_id": "ds0",
                "join_mode": "create_new",
                "dataset_index": 0,
                "cell_count": 2,
                "global_start": 0,
                "global_end": 2,
            }
        ],
    }
    corpus_root.mkdir(parents=True, exist_ok=True)
    with open(corpus_root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(index_doc, handle, sort_keys=False)

    dataset_root = corpus_root / "meta" / "ds0"
    _write_canonical_obs(dataset_root / "canonical_meta" / "canonical-obs.parquet")
    _write_canonical_var(
        dataset_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
    )
    _write_hvg_ranking(dataset_root / "hvg.parquet", ["GENE_A", "GENE_B", "GENE_C", "GENE_D"])
    _write_aggregate_lance(corpus_root / "matrix" / "aggregated-cells.lance")
    return corpus_root


def _build_three_row_pair_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-three-row-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-three-row-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [
            {
                "dataset_id": "ds0",
                "join_mode": "create_new",
                "dataset_index": 0,
                "cell_count": 3,
                "global_start": 0,
                "global_end": 3,
            }
        ],
    }
    corpus_root.mkdir(parents=True, exist_ok=True)
    with open(corpus_root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(index_doc, handle, sort_keys=False)

    dataset_root = corpus_root / "meta" / "ds0"
    frame = pl.DataFrame(
        {
            "cell_id": ["ds0_wt", "ds0_ko0", "ds0_ko1"],
            "dataset_id": ["ds0", "ds0", "ds0"],
            "size_factor": [1.0, 1.1, 1.2],
            "cell_context": ["T_cell", "T_cell", "T_cell"],
            "perturb_label": ["WT", "KO_TP53", "KO_STAT1"],
            "batch_id": ["batch_0", "batch_1", "batch_2"],
        }
    )
    (dataset_root / "canonical_meta").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(dataset_root / "canonical_meta" / "canonical-obs.parquet"))
    _write_canonical_var(
        dataset_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
    )
    _write_hvg_ranking(dataset_root / "hvg.parquet", ["GENE_A", "GENE_B", "GENE_C", "GENE_D"])
    _write_aggregate_lance(
        corpus_root / "matrix" / "aggregated-cells.lance",
        rows=[([0, 2], [5, 1]), ([1, 3], [3, 2]), ([0, 1], [4, 6])],
    )
    return corpus_root


def _build_mixed_union_pair_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-union-pair-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-union-pair-corpus",
        "global_metadata": {"backend": "lance", "topology": "aggregate"},
        "datasets": [
            {
                "dataset_id": "ds0",
                "join_mode": "create_new",
                "dataset_index": 0,
                "cell_count": 2,
                "global_start": 0,
                "global_end": 2,
            },
            {
                "dataset_id": "ds1",
                "join_mode": "append_routed",
                "dataset_index": 1,
                "cell_count": 2,
                "global_start": 2,
                "global_end": 4,
            },
        ],
    }
    corpus_root.mkdir(parents=True, exist_ok=True)
    with open(corpus_root / "corpus-index.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(index_doc, handle, sort_keys=False)

    ds0_root = corpus_root / "meta" / "ds0"
    _write_canonical_obs(ds0_root / "canonical_meta" / "canonical-obs.parquet")
    _write_canonical_var(
        ds0_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_B"],
    )
    _write_hvg_ranking(ds0_root / "hvg.parquet", ["GENE_A", "GENE_B"])

    ds1_root = corpus_root / "meta" / "ds1"
    frame = pl.DataFrame(
        {
            "cell_id": ["ds1_wt", "ds1_ko"],
            "dataset_id": ["ds1", "ds1"],
            "size_factor": [0.9, 1.1],
            "cell_context": ["T_cell", "T_cell"],
            "perturb_label": ["WT", "KO_STAT1"],
            "batch_id": ["batch_2", "batch_3"],
        }
    )
    (ds1_root / "canonical_meta").mkdir(parents=True, exist_ok=True)
    frame.write_parquet(str(ds1_root / "canonical_meta" / "canonical-obs.parquet"))
    _write_canonical_var(
        ds1_root / "canonical_meta" / "canonical-var.parquet",
        ["GENE_B", "GENE_C"],
    )
    _write_hvg_ranking(ds1_root / "hvg.parquet", ["GENE_B", "GENE_C"])

    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [[0], [1], [0], [1]],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [[5], [4], [2], [7]],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    matrix_path = corpus_root / "matrix" / "aggregated-cells.lance"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(matrix_path), mode="overwrite")
    return corpus_root


def _dense_row(row_index: int) -> np.ndarray:
    if row_index == 0:
        return np.asarray([5.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if row_index == 1:
        return np.asarray([0.0, 3.0, 0.0, 2.0], dtype=np.float32)
    raise AssertionError(f"unexpected row_index {row_index}")


def test_paired_batch_builder_reconstructs_target_values_at_source_sampled_genes(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=7,
    ).pair_source_positions([0, 1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=5,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
    )

    expected_keys = {
        "gene_ids",
        "next_gene_ids",
        "values",
        "target_values",
        "target_values_next",
        "ps",
        "ps_next",
        "sf",
        "sf_next",
        "index",
        "next_index",
    }
    expected_keys.update({f"{label_name}_labels" for label_name in config.label_names})
    expected_keys.update({f"{label_name}_labels_next" for label_name in config.label_names})
    assert expected_keys.issubset(batch.keys())
    assert batch["gene_ids"].shape == (2, 4)
    assert torch.equal(batch["gene_ids"], batch["next_gene_ids"])
    assert torch.equal(batch["values"], batch["target_values"])
    assert batch["sf"].shape == (2, 1)
    assert batch["sf_next"].shape == (2, 1)
    torch.testing.assert_close(
        batch["sf"],
        torch.tensor([[1.0], [1.0 / 1.2]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        batch["sf_next"],
        torch.tensor([[1.0 / 1.2], [1.0 / 1.2]], dtype=torch.float32),
    )
    assert batch["perturbation_labels_next"].tolist() == [1, 0]
    assert batch["batch_labels_next"].tolist() == [1, 1]
    assert batch["next_index"].tolist() == [1, 1]
    assert torch.count_nonzero(batch["ps"]) == 0
    assert torch.count_nonzero(batch["ps_next"]) == 0

    pad_token_id = builder.adapter.to_simple_vocab_stoi()[config.pad_token]
    cls_token_id = builder.adapter.to_simple_vocab_stoi()[config.cls_token]
    offset = builder.adapter.special_token_offset
    assert torch.all(batch["gene_ids"][:, 0] == cls_token_id)
    assert torch.all(batch["target_values"][:, 0] == float(config.cls_value))
    assert torch.all(batch["target_values_next"][:, 0] == float(config.cls_value))

    for batch_row, source_index in enumerate(pair_batch.source_indices.tolist()):
        source_dense = _dense_row(source_index)
        target_dense = _dense_row(int(pair_batch.target_indices[batch_row]))
        for col in range(1, batch["gene_ids"].shape[1]):
            token_id = int(batch["gene_ids"][batch_row, col])
            if token_id == pad_token_id:
                assert batch["target_values"][batch_row, col].item() == float(config.pad_value)
                assert batch["target_values_next"][batch_row, col].item() == float(config.pad_value)
                continue
            global_id = token_id - offset
            assert batch["target_values"][batch_row, col].item() == source_dense[global_id]
            assert batch["target_values_next"][batch_row, col].item() == target_dense[global_id]


def test_paired_batch_builder_lognorm_uses_scaled_reciprocal_size_factor(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(
        control_labels=("WT",),
        mask_ratio=0.0,
        normalize_expression="log1p",
        normalization_scale=1.5,
    )
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=1,
        config=config,
        seed=7,
        drop_last=False,
    ).pair_source_positions([1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=4, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        sampled_gene_ids=torch.arange(4).unsqueeze(0),
    )

    expected = torch.log1p(torch.tensor(_dense_row(1)) / 1.2 * 1.5)
    torch.testing.assert_close(batch["target_values"][0, 1:], expected)
    torch.testing.assert_close(batch["target_values_next"][0, 1:], expected)
    torch.testing.assert_close(batch["sf"], torch.tensor([[1.5 / 1.2]]))
    torch.testing.assert_close(batch["sf_next"], torch.tensor([[1.5 / 1.2]]))


def test_paired_batch_builder_emits_extra_configured_label_tensors(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(
        control_labels=("WT",),
        mask_ratio=0.0,
        label_fields={
            "perturb_label": "perturbation",
            "cell_context": "celltype",
            "batch_id": "batch",
            "dataset_id": "dataset",
        },
    )
    corpus = load_corpus(str(_build_mixed_union_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=1,
        config=config,
        source_indices=[0],
        target_candidate_indices=[3],
        seed=7,
        drop_last=False,
    ).pair_source_positions([0], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=5,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
    )

    for label_name in config.label_names:
        assert f"{label_name}_labels" in batch
        assert f"{label_name}_labels_next" in batch
    assert batch["dataset_labels"].tolist() == [0]
    assert batch["dataset_labels_next"].tolist() == [1]


def test_paired_batch_builder_accepts_precomputed_sampled_gene_ids_and_preserves_padding(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=1.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=1,
        config=config,
        seed=3,
    ).pair_source_positions([0], seed=13)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=17,
        sampled_gene_ids=torch.tensor([[0, 3, -1]], dtype=torch.long),
    )

    pad_token_id = builder.adapter.to_simple_vocab_stoi()[config.pad_token]
    cls_token_id = builder.adapter.to_simple_vocab_stoi()[config.cls_token]
    assert batch["gene_ids"].shape == (1, 4)
    assert batch["gene_ids"][0].tolist() == [cls_token_id, 4, 7, pad_token_id]
    assert batch["values"][0].tolist() == [
        float(config.cls_value),
        float(config.mask_value),
        float(config.mask_value),
        float(config.pad_value),
    ]
    assert batch["target_values"][0].tolist() == [
        float(config.cls_value),
        5.0,
        0.0,
        float(config.pad_value),
    ]
    assert batch["target_values_next"][0].tolist() == [
        float(config.cls_value),
        0.0,
        2.0,
        float(config.pad_value),
    ]


def test_paired_batch_builder_uses_custom_tokenizer_and_excludes_missing_genes(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(
        corpus,
        config,
        tokenizer_stoi={
            "<pad>": 0,
            "<cls>": 1,
            "<unk>": 2,
            "<eos>": 3,
            "GENE_A": 17,
            "GENE_B": 23,
        },
    )
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=1,
        config=config,
        seed=3,
    ).pair_source_positions([0], seed=13)
    builder = PertTFPairedBatchBuilder(
        corpus,
        seq_len=2,
        config=config,
        adapter=adapter,
        missing_token_policy="exclude",
    )

    batch = builder.build_paired_batch(pair_batch, seed=17, sampling_mode="uniform")

    assert batch["gene_ids"].shape == (1, 3)
    assert batch["gene_ids"][0, 0].item() == 1
    assert set(batch["gene_ids"][0, 1:].tolist()).issubset({17, 23})
    assert 0 not in batch["gene_ids"][0, 1:].tolist()


def test_paired_batch_builder_emits_union_full_expression_masks_for_mixed_datasets(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(
        control_labels=("WT",),
        mask_ratio=0.0,
        include_full_expr=True,
    )
    corpus = load_corpus(str(_build_mixed_union_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=19,
    ).pair_source_positions([0, 2], seed=23)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=2, config=config)

    batch = builder.build_paired_batch(
        pair_batch,
        seed=29,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=1,
    )

    cls_token_id = builder.adapter.to_simple_vocab_stoi()[config.cls_token]
    assert batch["full_gene_ids"].shape == (2, 4)
    assert batch["full_gene_ids"][0].tolist() == [cls_token_id, 4, 5, 6]
    assert torch.equal(batch["full_gene_ids"][0], batch["full_gene_ids"][1])

    assert batch["full_expr"].shape == (2, 4)
    assert batch["full_expr_next"].shape == (2, 4)
    assert batch["full_expr_mask"].dtype == torch.bool
    assert batch["full_expr_next_mask"].dtype == torch.bool

    assert batch["index"].tolist() == [0, 2]
    assert batch["next_index"].tolist() == [1, 3]

    assert batch["full_expr"][0].tolist() == [float(config.cls_value), 5.0, 0.0, 0.0]
    assert batch["full_expr_mask"][0].tolist() == [True, True, True, False]
    assert batch["full_expr_next"][0].tolist() == [float(config.cls_value), 0.0, 4.0, 0.0]
    assert batch["full_expr_next_mask"][0].tolist() == [True, True, True, False]

    assert batch["full_expr"][1].tolist() == [float(config.cls_value), 0.0, 2.0, 0.0]
    assert batch["full_expr_mask"][1].tolist() == [True, False, True, True]
    assert batch["full_expr_next"][1].tolist() == [float(config.cls_value), 0.0, 0.0, 7.0]
    assert batch["full_expr_next_mask"][1].tolist() == [True, False, True, True]


def test_build_from_raw_pair_batch_matches_direct_builder_path(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(
        control_labels=("WT",),
        mask_ratio=0.0,
        include_full_expr=True,
    )
    corpus = load_corpus(str(_build_mixed_union_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=19,
    ).pair_source_positions([0, 2], seed=23)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=2, config=config)

    source_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.source_indices)
    target_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.target_indices)

    direct_output = builder.build_paired_batch(
        pair_batch,
        seed=29,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=1,
    )
    raw_batch_output = builder.build_from_raw_pair_batch(
        pair_batch,
        source_raw,
        target_raw,
        seed=29,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=1,
    )

    assert direct_output.keys() == raw_batch_output.keys()
    for key in direct_output:
        torch.testing.assert_close(direct_output[key], raw_batch_output[key])


def test_build_from_raw_pair_batch_rejects_batch_size_mismatch(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=7,
    ).pair_source_positions([0, 1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    source_raw = _read_expression_raw_batch(corpus.expression_reader, [int(pair_batch.source_indices[0])])
    target_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.target_indices)

    with pytest.raises(ValueError, match="source_raw batch_size"):
        builder.build_from_raw_pair_batch(pair_batch, source_raw, target_raw)


def test_build_from_raw_pair_batch_rejects_row_order_mismatch(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=7,
    ).pair_source_positions([0, 1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    source_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.source_indices[::-1])
    target_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.target_indices)

    with pytest.raises(ValueError, match="source_raw global_row_index"):
        builder.build_from_raw_pair_batch(pair_batch, source_raw, target_raw)


def test_build_from_raw_pair_batch_rejects_target_perturbation_shape_mismatch(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=7,
    ).pair_source_positions([0, 1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    invalid_pair_batch = replace(
        pair_batch,
        target_perturbation_ids=np.asarray([0], dtype=np.int64),
    )
    source_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.source_indices)
    target_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.target_indices)

    with pytest.raises(ValueError, match="target_perturbation_ids"):
        builder.build_from_raw_pair_batch(invalid_pair_batch, source_raw, target_raw)


def _build_pair_read_loader(
    corpus,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
    multiprocessing_context: str | None,
):
    pair_sampler = _build_pair_sampler(
        corpus,
        batch_size=batch_size,
        config=PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0),
        seed=seed,
        source_indices=[0, 1],
    )
    request_sampler = _PertTFPairReadBatchSampler(pair_sampler)
    dataset = _PertTFPairExpressionDataset(
        corpus.expression_reader,
        total_rows=len(corpus.metadata_index),
    )
    loader_kwargs = {
        "batch_sampler": request_sampler,
        "collate_fn": _collate_perttf_raw_pair_batch,
        "num_workers": num_workers,
    }
    if multiprocessing_context is not None:
        loader_kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(dataset, **loader_kwargs)


def _build_pair_sampler(
    corpus,
    *,
    batch_size: int,
    config: PertTFAdapterConfig,
    row_indices=None,
    source_indices=None,
    target_candidate_indices=None,
    seed: int = 0,
    drop_last: bool = True,
    perturbed_target_policy: str = "self_to_control_label",
) -> PerturbationPairSampler:
    prepared = _prepare_perttf_metadata(
        corpus.metadata_index,
        config=config,
        row_indices=row_indices,
        source_indices=source_indices,
        target_candidate_indices=target_candidate_indices,
    )
    return PerturbationPairSampler(
        prepared.frame,
        batch_size=batch_size,
        perturbation_column=f"{config.perturbation_label}_id",
        control_perturbation_ids=prepared.control_label_ids,
        pairing_group_columns=tuple(f"{label_name}_id" for label_name in config.pairing_group_labels),
        source_positions=prepared.row_selection.source_positions,
        target_candidate_positions=prepared.row_selection.target_candidate_positions,
        global_positions="global_row_index",
        seed=seed,
        drop_last=drop_last,
        perturbed_target_policy=perturbed_target_policy,
    )


def _assert_pair_read_batch_matches_builder(
    corpus,
    batch: dict[str, object],
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)
    request = batch["request"]
    pair_batch = request.pair_batch
    source_raw = batch["source_raw"]
    target_raw = batch["target_raw"]

    expected_source_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.source_indices)
    expected_target_raw = _read_expression_raw_batch(corpus.expression_reader, pair_batch.target_indices)

    assert source_raw["batch_size"] == expected_source_raw["batch_size"]
    assert target_raw["batch_size"] == expected_target_raw["batch_size"]
    torch.testing.assert_close(
        source_raw["global_row_index"],
        torch.as_tensor(expected_source_raw["global_row_index"], dtype=torch.long),
    )
    assert "dataset_index" not in source_raw
    torch.testing.assert_close(
        source_raw["row_offsets"],
        torch.as_tensor(expected_source_raw["row_offsets"], dtype=torch.long),
    )
    torch.testing.assert_close(
        source_raw["expressed_gene_indices"],
        torch.as_tensor(expected_source_raw["expressed_gene_indices"], dtype=torch.long),
    )
    torch.testing.assert_close(
        source_raw["expression_counts"],
        torch.as_tensor(expected_source_raw["expression_counts"], dtype=torch.float32),
    )
    assert "size_factor" not in source_raw
    torch.testing.assert_close(
        target_raw["global_row_index"],
        torch.as_tensor(expected_target_raw["global_row_index"], dtype=torch.long),
    )
    assert "dataset_index" not in target_raw
    torch.testing.assert_close(
        target_raw["row_offsets"],
        torch.as_tensor(expected_target_raw["row_offsets"], dtype=torch.long),
    )
    torch.testing.assert_close(
        target_raw["expressed_gene_indices"],
        torch.as_tensor(expected_target_raw["expressed_gene_indices"], dtype=torch.long),
    )
    torch.testing.assert_close(
        target_raw["expression_counts"],
        torch.as_tensor(expected_target_raw["expression_counts"], dtype=torch.float32),
    )
    assert "size_factor" not in target_raw

    split_batch = builder.build_from_raw_pair_batch(
        pair_batch,
        source_raw,
        target_raw,
        seed=request.seed,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
    )
    reference_batch = builder.build_paired_batch(
        pair_batch,
        seed=request.seed,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
    )
    assert split_batch.keys() == reference_batch.keys()
    for key in split_batch:
        torch.testing.assert_close(split_batch[key], reference_batch[key])


def _build_public_pair_loader(
    corpus,
    *,
    batch_size: int,
    seq_len: int,
    seed: int,
    num_workers: int,
    multiprocessing_context: str | None,
    drop_last: bool = True,
    perturbed_target_policy: str = "self_to_control_label",
    pairing_mode: str = "source_driven",
    config: PertTFAdapterConfig | None = None,
    adapter: PertTFCorpusAdapter | None = None,
    row_indices=None,
    source_indices=None,
    target_candidate_indices=None,
):
    resolved_config = config or PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    return PertTFPairedBatchLoader(
        corpus,
        batch_size=batch_size,
        seq_len=seq_len,
        config=resolved_config,
        adapter=adapter,
        row_indices=row_indices,
        seed=seed,
        drop_last=drop_last,
        perturbed_target_policy=perturbed_target_policy,
        pairing_mode=pairing_mode,
        source_indices=source_indices,
        target_candidate_indices=target_candidate_indices,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
    )


def _assert_public_loader_matches_builder(loader: PertTFPairedBatchLoader) -> None:
    expected_request = next(iter(loader._request_sampler))[0]
    expected_pair_batch = expected_request.pair_batch
    expected_batch = loader._builder.build_paired_batch(
        expected_pair_batch,
        seed=expected_request.seed,
        sampling_mode="hvg",
        hvg_weight=4.0,
        hvg_top_k=2,
    )

    batches = list(loader)

    assert len(batches) == 1
    actual_batch = batches[0]
    assert "request" not in actual_batch
    assert "source_raw" not in actual_batch
    assert "target_raw" not in actual_batch
    assert actual_batch.keys() == expected_batch.keys()
    for key in actual_batch:
        torch.testing.assert_close(actual_batch[key], expected_batch[key])


def test_pair_read_dataset_state_stays_worker_light(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    builder = PertTFPairedBatchBuilder(
        corpus,
        seq_len=3,
        config=PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0),
    )
    dataset = _PertTFPairExpressionDataset(
        corpus.expression_reader,
        total_rows=len(corpus.metadata_index),
    )

    assert set(dataset.__dict__) == {"_reader", "_total_rows"}
    assert dataset.__dict__["_reader"] is corpus.expression_reader
    assert dataset.__dict__["_total_rows"] == len(corpus.metadata_index)
    assert all(value is not corpus.metadata_index for value in dataset.__dict__.values())
    assert not any(torch.is_tensor(value) and value.is_cuda for value in dataset.__dict__.values())
    assert not any(isinstance(value, torch.device) for value in dataset.__dict__.values())
    assert not any(isinstance(value, torch.nn.Module) for value in dataset.__dict__.values())
    assert not any(
        isinstance(value, PertTFPairedBatchBuilder)
        for value in dataset.__dict__.values()
    )
    assert all(value is not builder.adapter for value in dataset.__dict__.values())


def test_pair_read_batch_sampler_tracks_epoch_and_batch_identity(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    sampler = _build_pair_sampler(
        corpus,
        batch_size=1,
        config=PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0),
        seed=13,
        source_indices=[0, 1],
        drop_last=False,
    )
    request_sampler = _PertTFPairReadBatchSampler(sampler)

    request_sampler.set_epoch(0)
    first_epoch_requests = [batch[0] for batch in request_sampler]
    request_sampler.set_epoch(0)
    repeat_epoch_requests = [batch[0] for batch in request_sampler]
    request_sampler.set_epoch(2)
    second_epoch_requests = [batch[0] for batch in request_sampler]

    assert [request.batch_index for request in first_epoch_requests] == [0, 1]
    assert [request.batch_index for request in second_epoch_requests] == [0, 1]
    assert [request.epoch for request in first_epoch_requests] == [0, 0]
    assert [request.epoch for request in second_epoch_requests] == [2, 2]
    assert [request.seed for request in first_epoch_requests] == [
        request.seed for request in repeat_epoch_requests
    ]
    assert [request.seed for request in second_epoch_requests] != [
        request.seed for request in first_epoch_requests
    ]
    assert [
        tuple(request.pair_batch.source_indices.tolist())
        for request in first_epoch_requests
    ] == [
        tuple(request.pair_batch.source_indices.tolist())
        for request in repeat_epoch_requests
    ]
    assert [
        tuple(request.pair_batch.target_indices.tolist())
        for request in first_epoch_requests
    ] == [
        tuple(request.pair_batch.target_indices.tolist())
        for request in repeat_epoch_requests
    ]


def test_pair_read_dataloader_supports_single_process(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_pair_read_loader(
        corpus,
        batch_size=2,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
    )

    batches = list(loader)

    assert len(batches) == 1
    _assert_pair_read_batch_matches_builder(corpus, batches[0])


def test_public_paired_batch_loader_yields_final_batches_single_process(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=2,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
    )

    assert len(loader) == 1
    assert loader.adapter is loader._builder.adapter
    assert loader.config is loader._builder.config
    assert loader.pair_sampler.batch_size == 2

    _assert_public_loader_matches_builder(loader)


def test_public_paired_batch_loader_omits_full_expression_fields_by_default(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=2,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
    )

    batch = next(iter(loader))

    assert "full_gene_ids" not in batch
    assert "full_expr" not in batch
    assert "full_expr_mask" not in batch
    assert "full_expr_next" not in batch
    assert "full_expr_next_mask" not in batch


@pytest.mark.parametrize(
    ("num_workers", "multiprocessing_context"),
    [(0, None), (2, "spawn")],
)
def test_public_paired_batch_loader_preserves_full_expression_fields(
    tmp_path: Path,
    num_workers: int,
    multiprocessing_context: str | None,
) -> None:
    corpus = load_corpus(str(_build_mixed_union_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=2,
        seq_len=2,
        seed=19,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing_context,
        source_indices=np.asarray([0, 2], dtype=np.int64),
        target_candidate_indices=np.asarray([1, 3], dtype=np.int64),
        config=PertTFAdapterConfig(
            control_labels=("WT",),
            mask_ratio=0.0,
            include_full_expr=True,
        ),
    )

    batch = next(iter(loader))

    assert {"full_gene_ids", "full_expr", "full_expr_mask"}.issubset(batch)
    assert {"full_expr_next", "full_expr_next_mask"}.issubset(batch)
    _assert_public_loader_matches_builder(loader)


def test_public_paired_batch_loader_default_null_policy_drops_null_perturbation(
    tmp_path: Path,
) -> None:
    corpus_path = _build_small_pair_corpus(tmp_path)
    obs_path = corpus_path / "meta" / "ds0" / "canonical_meta" / "canonical-obs.parquet"
    frame = pl.read_parquet(obs_path)
    column = "perturb_label"
    values = frame[column].to_list()
    values[0] = None
    frame = frame.with_columns(pl.Series(column, values))
    frame.write_parquet(obs_path)

    corpus = load_corpus(str(corpus_path))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        config=PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0),
        row_indices=np.asarray([0, 1], dtype=np.int64),
    )

    batch = next(iter(loader))

    assert batch["index"].tolist() == [1]
    assert batch["next_index"].tolist() == [1]
    assert loader.effective_label_row_indices.tolist() == [1]
    assert loader.effective_source_indices.tolist() == [1]
    assert loader.effective_target_candidate_indices.tolist() == [1]


@pytest.mark.parametrize(
    ("column", "label_name"),
    [
        ("cell_context", "celltype"),
        ("perturb_label", "perturbation"),
        ("batch_id", "batch"),
    ],
)
def test_public_paired_batch_loader_custom_drop_null_policy_filters_selected_pool(
    tmp_path: Path,
    column: str,
    label_name: str,
) -> None:
    corpus_path = _build_small_pair_corpus(tmp_path)
    obs_path = corpus_path / "meta" / "ds0" / "canonical_meta" / "canonical-obs.parquet"
    frame = pl.read_parquet(obs_path)
    values = frame[column].to_list()
    values[0] = None
    frame = frame.with_columns(pl.Series(column, values))
    frame.write_parquet(obs_path)

    corpus = load_corpus(str(corpus_path))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        config=PertTFAdapterConfig(
            control_labels=("WT",),
            mask_ratio=0.0,
            drop_null_labels=(label_name,),
        ),
        row_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert loader.effective_label_row_indices.tolist() == [1]

def test_public_paired_batch_loader_selected_row_pool_ignores_null_rows_outside_subset(
    tmp_path: Path,
) -> None:
    corpus_path = _build_small_pair_corpus(tmp_path)
    obs_path = corpus_path / "meta" / "ds0" / "canonical_meta" / "canonical-obs.parquet"
    frame = pl.read_parquet(obs_path).with_columns(
        pl.Series("perturb_label", [None, "KO_TP53"]),
    )
    frame.write_parquet(obs_path)

    corpus = load_corpus(str(corpus_path))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        row_indices=np.asarray([1], dtype=np.int64),
    )

    batch = next(iter(loader))

    assert batch["index"].tolist() == [1]
    assert batch["next_index"].tolist() == [1]
    assert loader.effective_label_row_indices.tolist() == [1]
    assert loader.effective_source_indices.tolist() == [1]
    assert loader.effective_target_candidate_indices.tolist() == [1]


def test_paired_batch_builder_requires_size_factor_metadata(tmp_path: Path) -> None:
    corpus_path = _build_small_pair_corpus(tmp_path)
    obs_path = corpus_path / "meta" / "ds0" / "canonical_meta" / "canonical-obs.parquet"
    pl.read_parquet(obs_path).drop("size_factor").write_parquet(obs_path)

    corpus = load_corpus(str(corpus_path))
    config = PertTFAdapterConfig(control_labels=("WT",), mask_ratio=0.0)
    pair_batch = _build_pair_sampler(
        corpus,
        batch_size=2,
        config=config,
        seed=7,
    ).pair_source_positions([0, 1], seed=11)
    builder = PertTFPairedBatchBuilder(corpus, seq_len=3, config=config)

    with pytest.raises(RuntimeError, match="size_factor"):
        builder.build_paired_batch(pair_batch, seed=5)


def test_public_paired_batch_loader_supports_self_to_self_labels(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        perturbed_target_policy="self_to_self_label",
        source_indices=np.asarray([1], dtype=np.int64),
    )

    batch = next(iter(loader))

    assert batch["index"].tolist() == [1]
    assert batch["next_index"].tolist() == [1]
    assert batch["perturbation_labels"].tolist() == [1]
    assert batch["perturbation_labels_next"].tolist() == [1]
    torch.testing.assert_close(batch["target_values"], batch["target_values_next"])


def test_public_paired_batch_loader_self_pairs_control_without_treated_pool(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        perturbed_target_policy="self_to_self_label",
        row_indices=np.asarray([0], dtype=np.int64),
    )

    batch = next(iter(loader))

    assert batch["index"].tolist() == [0]
    assert batch["next_index"].tolist() == [0]
    assert batch["perturbation_labels"].tolist() == [0]
    assert batch["perturbation_labels_next"].tolist() == [0]
    torch.testing.assert_close(batch["target_values"], batch["target_values_next"])


def test_public_paired_batch_loader_explicit_pools_intersect_row_indices(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_three_row_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        row_indices=np.asarray([0, 1], dtype=np.int64),
        source_indices=np.asarray([1, 2], dtype=np.int64),
        target_candidate_indices=np.asarray([0, 2], dtype=np.int64),
    )

    batch = next(iter(loader))

    assert batch["index"].tolist() == [1]
    assert batch["next_index"].tolist() == [0]
    assert loader.row_indices.tolist() == [0, 1]
    assert loader.effective_label_row_indices.tolist() == [0, 1]
    assert loader.effective_source_indices.tolist() == [1]
    assert loader.effective_target_candidate_indices.tolist() == [0]


def test_public_paired_batch_loader_supports_target_driven_pairing(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_three_row_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=2,
        seq_len=3,
        seed=7,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
        pairing_mode="target_driven",
        source_indices=np.asarray([0], dtype=np.int64),
        target_candidate_indices=np.asarray([1, 2], dtype=np.int64),
    )

    batch = next(iter(loader))

    assert loader.effective_pair_count == 2
    assert batch["index"].tolist() == [0, 0]
    assert sorted(batch["next_index"].tolist()) == [1, 2]
    assert batch["perturbation_labels"].tolist() == [0, 0]
    assert sorted(batch["perturbation_labels_next"].tolist()) == [1, 2]


def test_pair_read_dataloader_supports_spawn_workers(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_pair_read_loader(
        corpus,
        batch_size=2,
        seed=7,
        num_workers=2,
        multiprocessing_context="spawn",
    )

    batches = list(loader)

    assert len(batches) == 1
    _assert_pair_read_batch_matches_builder(corpus, batches[0])


def test_public_paired_batch_loader_does_not_force_spawn_context(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_small_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=2,
        seq_len=3,
        seed=7,
        num_workers=2,
        multiprocessing_context=None,
    )

    assert isinstance(loader._data_loader.dataset, _PertTFPairExpressionDataset)
    assert isinstance(loader._data_loader.batch_sampler, _PertTFPairReadBatchSampler)
    assert "multiprocessing_context" not in loader._loader_kwargs


def test_public_paired_batch_loader_set_epoch_is_repeatable_and_reorders(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_mixed_union_pair_corpus(tmp_path)))
    loader = _build_public_pair_loader(
        corpus,
        batch_size=1,
        seq_len=2,
        seed=0,
        num_workers=0,
        multiprocessing_context=None,
        drop_last=False,
    )

    loader.set_epoch(0)
    epoch0_first = [int(batch["index"][0]) for batch in loader]
    loader.set_epoch(0)
    epoch0_second = [int(batch["index"][0]) for batch in loader]
    loader.set_epoch(1)
    epoch1 = [int(batch["index"][0]) for batch in loader]

    assert epoch0_first == epoch0_second
    assert epoch1 != epoch0_first
