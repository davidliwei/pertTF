from pathlib import Path

import lance
import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import yaml

import perttf.model.corpus_adapter as loader_exports
import perttf.model.corpus_adapter as adapter_exports
from perturb_data_lab.loaders import FeatureRegistry, MetadataIndex, load_corpus
from perttf.model.corpus_adapter import (
    PertTFAdapterConfig,
    PertTFPairedBatchBuilder,
    PertTFPairedBatchLoader,
    PertTFCorpusAdapter,
    PerturbationPairBatch,
    PerturbationPairSampler,
)


def _build_feature_registry() -> FeatureRegistry:
    return FeatureRegistry(
        {
            "ds0": pl.DataFrame(
                {
                    "origin_index": [0, 1],
                    "canonical_gene_id": ["GENE_B", "GENE_A"],
                }
            ),
            "ds1": pl.DataFrame(
                {
                    "origin_index": [0, 1],
                    "canonical_gene_id": ["GENE_A", "GENE_C"],
                }
            ),
        },
        dataset_order=["ds0", "ds1"],
    )


def _build_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(4, dtype=np.int64),
                "cell_id": [f"cell_{idx}" for idx in range(4)],
                "dataset_id": ["ds0", "ds0", "ds1", "ds1"],
                "dataset_index": np.asarray([0, 0, 1, 1], dtype=np.int32),
                "local_row_index": np.asarray([0, 1, 0, 1], dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1, 0.9, 1.2], dtype=np.float32),
                "cell_context": ["T_cell", "B_cell", "T_cell", "NK_cell"],
                "perturb_label": ["KO_TP53", "KO_TP53", "KO_GATA1", "KO_GATA1"],
                "batch_id": ["batch_b", "batch_a", "batch_b", "batch_c"],
            }
        )
    )


def test_perttf_config_defaults_expose_generic_label_fields() -> None:
    config = PertTFAdapterConfig()

    assert config.label_fields == {
        "perturb_label": "perturbation",
        "cell_context": "celltype",
        "batch_id": "batch",
    }
    assert config.metadata_columns == ("perturb_label", "cell_context", "batch_id")
    assert config.label_names == ("perturbation", "celltype", "batch")
    assert config.perturbation_label == "perturbation"
    assert config.pairing_group_labels == ()
    assert config.label_columns_by_name == {
        "perturbation": "perturb_label",
        "celltype": "cell_context",
        "batch": "batch_id",
    }
    assert config.resolved_drop_null_labels == ("perturbation",)
    assert not hasattr(config, "perturbation_column")
    assert not hasattr(config, "cell_context_column")
    assert not hasattr(config, "batch_column")


def test_perttf_config_accepts_explicit_dataset_pairing_and_null_policy() -> None:
    config = PertTFAdapterConfig(
        label_fields={
            "perturb_label": "perturbation",
            "cell_context": "celltype",
            "batch_id": "batch",
            "dataset_id": "dataset",
        },
        pairing_group_labels=("dataset",),
    )

    assert config.label_columns_by_name["dataset"] == "dataset_id"
    assert config.pairing_group_labels == ("dataset",)
    assert config.resolved_drop_null_labels == ("perturbation",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "label_fields": {
                    1: "perturbation",
                    "1": "celltype",
                    "batch_id": "batch",
                }
            },
            "label_fields metadata column must be a non-empty string",
        ),
        (
            {
                "label_fields": {
                    "perturb_label": 1,
                    "cell_context": "1",
                    "batch_id": "batch",
                },
                "perturbation_label": "1",
            },
            "label_fields label name must be a non-empty string",
        ),
        (
            {
                "label_fields": {
                    "perturb_label": "perturbation",
                    "cell_context": "celltype",
                    "batch_id": "batch",
                },
                "perturbation_label": "dataset",
            },
            "perturbation_label must name one configured label field",
        ),
        (
            {
                "pairing_group_labels": ("dataset",),
            },
            "pairing_group_labels must name configured label fields",
        ),
        ({"drop_null_labels": ("dataset",)}, "drop_null_labels must name configured label fields"),
    ],
)
def test_perttf_config_invalid_generic_label_settings_fail_early(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PertTFAdapterConfig(**kwargs)


def _write_canonical_obs(path: Path, *, dataset_id: str, global_start: int) -> None:
    frame = pl.DataFrame(
        {
            "cell_id": [f"{dataset_id}_cell_0", f"{dataset_id}_cell_1"],
            "dataset_id": [dataset_id, dataset_id],
            "size_factor": [1.0, 1.2],
            "cell_context": ["T_cell", "B_cell"],
            "perturb_label": ["KO_TP53", "KO_GATA1"],
            "batch_id": [f"batch_{global_start}", f"batch_{global_start + 1}"],
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


def _write_aggregate_lance(path: Path) -> None:
    table = pa.table(
        {
            "expressed_gene_indices": pa.array(
                [[0, 1], [1], [0], [0, 1]],
                type=pa.list_(pa.int32()),
            ),
            "expression_counts": pa.array(
                [[5, 3], [2], [4], [1, 6]],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(table, str(path), mode="overwrite")


def _build_small_corpus(tmp_path: Path) -> Path:
    corpus_root = tmp_path / "perttf-adapter-corpus"
    index_doc = {
        "kind": "corpus-index",
        "contract_version": "0.3.0",
        "corpus_id": "perttf-adapter-corpus",
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

    _write_canonical_obs(
        corpus_root / "meta" / "ds0" / "canonical_meta" / "canonical-obs.parquet",
        dataset_id="ds0",
        global_start=0,
    )
    _write_canonical_obs(
        corpus_root / "meta" / "ds1" / "canonical_meta" / "canonical-obs.parquet",
        dataset_id="ds1",
        global_start=2,
    )
    _write_canonical_var(
        corpus_root / "meta" / "ds0" / "canonical_meta" / "canonical-var.parquet",
        ["GENE_B", "GENE_A"],
    )
    _write_canonical_var(
        corpus_root / "meta" / "ds1" / "canonical_meta" / "canonical-var.parquet",
        ["GENE_A", "GENE_C"],
    )
    _write_aggregate_lance(corpus_root / "matrix" / "aggregated-cells.lance")
    return corpus_root


def test_corpus_adapter_preserves_feature_registry_order_with_reserved_offset(
    tmp_path: Path,
) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus)

    assert corpus.feature_registry.global_feature_ids == ("GENE_B", "GENE_A", "GENE_C")
    assert adapter.special_token_offset == 4
    assert adapter.tokens_in_order == (
        "<pad>",
        "<cls>",
        "<unk>",
        "<eos>",
        "GENE_B",
        "GENE_A",
        "GENE_C",
    )
    np.testing.assert_array_equal(
        adapter.gene_token_ids,
        np.asarray([4, 5, 6], dtype=np.int64),
    )
    assert adapter.token_id_for_global_id(0) == 4
    assert adapter.token_id_for_global_id(1) == 5
    assert adapter.feature_id_for_token_id(6) == "GENE_C"


def test_corpus_adapter_emits_simple_vocab_compatible_mapping(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus)
    stoi = adapter.to_simple_vocab_stoi()

    assert adapter.stoi == stoi
    assert list(stoi.keys()) == list(adapter.tokens_in_order)
    assert list(stoi.values()) == list(range(len(stoi)))


def test_corpus_adapter_accepts_custom_gene_tokenizer(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(
        corpus,
        tokenizer_stoi={
            "<pad>": 0,
            "<cls>": 1,
            "<unk>": 2,
            "<eos>": 3,
            "GENE_A": 101,
            "GENE_C": 303,
        },
    )

    np.testing.assert_array_equal(adapter.gene_token_ids, [0, 101, 303])
    np.testing.assert_array_equal(
        adapter.gene_token_mapper.tokenizable_by_global_id,
        [False, True, True],
    )
    assert adapter.token_id_for_global_id(1) == 101
    assert adapter.global_id_for_token_id(101) == 1


def test_corpus_adapter_builds_deterministic_maps_and_control_slots(
    tmp_path: Path,
) -> None:
    config = PertTFAdapterConfig(control_labels=("WT", "CTRL"))
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus, config)

    assert adapter.labels_by_name == {
        "perturbation": ("WT", "CTRL", "KO_TP53", "KO_GATA1"),
        "celltype": ("T_cell", "B_cell"),
        "batch": ("batch_0", "batch_1", "batch_2", "batch_3"),
    }
    assert adapter.label_to_index_by_name["celltype"] == {"T_cell": 0, "B_cell": 1}
    assert adapter.control_label_ids == (0, 1)


def test_corpus_adapter_unknown_labels_raise_key_error(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(corpus)

    with pytest.raises(KeyError, match="missing_context"):
        adapter.encode_label("celltype", "missing_context")
    with pytest.raises(KeyError, match="missing_batch"):
        adapter.encode_label("batch", "missing_batch")
    with pytest.raises(KeyError, match="missing_field"):
        adapter.encode_label("missing_field", "anything")


def test_corpus_adapter_builds_from_loaded_small_corpus(tmp_path: Path) -> None:
    corpus = load_corpus(str(_build_small_corpus(tmp_path)))
    adapter = PertTFCorpusAdapter.from_corpus(
        corpus,
        PertTFAdapterConfig(
            control_labels=("WT",),
            label_fields={
                "perturb_label": "perturbation",
                "cell_context": "celltype",
                "batch_id": "batch",
                "dataset_id": "dataset",
            },
            pairing_group_labels=("dataset",),
        ),
    )

    reference = adapter.to_reference_dict()
    assert adapter.config.label_columns_by_name["dataset"] == "dataset_id"
    assert reference["genes"] == ["GENE_B", "GENE_A", "GENE_C"]
    np.testing.assert_array_equal(
        reference["gene_token_ids"],
        np.asarray([4, 5, 6], dtype=np.int64),
    )
    assert reference["simple_vocab_stoi"]["GENE_B"] == 4
    assert reference["labels_by_name"]["dataset"] == ("ds0", "ds1")
    assert reference["label_to_index_by_name"]["celltype"] == {
        "T_cell": 0,
        "B_cell": 1,
    }
    assert reference["label_to_index_by_name"]["perturbation"] == {
        "WT": 0,
        "KO_TP53": 1,
        "KO_GATA1": 2,
    }
    assert adapter.encode_label("dataset", "ds1") == 1


def test_removed_mapping_helpers_are_no_longer_public_exports() -> None:
    for name in ("CategoricalLabelMap", "PertTFLabelAdapter", "PertTFVocabAdapter"):
        assert name not in loader_exports.__all__
        assert name not in adapter_exports.__all__
        assert not hasattr(loader_exports, name)
        assert not hasattr(adapter_exports, name)


def test_retained_perttf_classes_still_import_from_public_modules() -> None:
    expected = {
        "PertTFAdapterConfig": PertTFAdapterConfig,
        "PertTFCorpusAdapter": PertTFCorpusAdapter,
        "PerturbationPairBatch": PerturbationPairBatch,
        "PerturbationPairSampler": PerturbationPairSampler,
        "PertTFPairedBatchBuilder": PertTFPairedBatchBuilder,
        "PertTFPairedBatchLoader": PertTFPairedBatchLoader,
    }

    for name, expected_object in expected.items():
        assert name in loader_exports.__all__
        assert name in adapter_exports.__all__
        assert getattr(loader_exports, name) is expected_object
        assert getattr(adapter_exports, name) is expected_object
