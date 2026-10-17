"""pertTF vocabulary, default pairing and tensor assembly for on-disk corpora.

Which cells are paired is described with ``scorpus.pairing``; this module turns those
pairs into pertTF batches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from scorpus.loaders.expression import ExpressionBatch
from scorpus.loaders.gene_token_mapper import GeneTokenMapper
from scorpus.loaders.index import MetadataIndex, _normalize_candidate_row_indices
from scorpus.loaders.sparse_batch import SparseBatchProcessor
from scorpus.pairing import Sampler, combine, cross, make_groups

if TYPE_CHECKING:
    from scorpus.loaders.corpus_loader import Corpus

__all__ = [
    "PertTFAdapterConfig",
    "PertTFPairedBatchLoader",
    "PertTFPairedBatchBuilder",
    "PerturbationPairBatch",
    "PertTFCorpusAdapter",
    "default_pair_sampler",
]


_DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<cls>", "<unk>", "<eos>")
_DEFAULT_LABEL_FIELD_ITEMS: tuple[tuple[str, str], ...] = (
    ("perturb_label", "perturbation"),
    ("cell_context", "celltype"),
    ("batch_id", "batch"),
)


def _check_nonempty_string(value: Any, *, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_unique_strings(values: Sequence[Any], *, field_name: str) -> None:
    for value in values:
        _check_nonempty_string(value, field_name=field_name)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must be unique")


def _ordered_unique(labels: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(labels))


def _derive_perttf_stream_seed(
    base_seed: int,
    *,
    epoch: int,
    batch_index: int,
    stream_id: int,
) -> int:
    seed_sequence = np.random.SeedSequence(
        [
            int(base_seed),
            int(epoch),
            int(batch_index),
            int(stream_id),
        ]
    )
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def _expression_batch_to_raw_dict(batch: ExpressionBatch) -> dict[str, Any]:
    return {
        "batch_size": batch.batch_size,
        "global_row_index": batch.global_row_index,
        "row_offsets": batch.row_offsets,
        "expressed_gene_indices": batch.expressed_gene_indices,
        "expression_counts": batch.expression_counts,
    }


def _read_expression_raw_batch(expression_reader: Any, indices: Sequence[int] | np.ndarray) -> dict[str, Any]:
    return _expression_batch_to_raw_dict(
        expression_reader.read_expression_flat(np.asarray(indices, dtype=np.int64).tolist())
    )


@dataclass(frozen=True)
class PertTFAdapterConfig:
    """Configuration for the retained pertTF wrapper surface."""

    label_fields: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_LABEL_FIELD_ITEMS)
    )
    perturbation_label: str = "perturbation"
    drop_null_labels: tuple[str, ...] | None = None
    null_label_value: str = "<null>"
    control_labels: tuple[str, ...] = ("WT",)
    special_tokens: tuple[str, ...] = _DEFAULT_SPECIAL_TOKENS
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    unk_token: str = "<unk>"
    eos_token: str = "<eos>"
    pad_value: int = -2
    mask_value: int = -1
    cls_value: int = -3
    append_cls: bool = True
    mask_ratio: float = 0.15
    ps_width: int = 1
    include_full_expr: bool = False
    normalize_expression: str = "none"
    normalization_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.label_fields:
            raise ValueError("label_fields must contain at least one mapping")
        for column, label_name in self.label_fields.items():
            _check_nonempty_string(column, field_name="label_fields metadata column")
            _check_nonempty_string(label_name, field_name="label_fields label name")
        label_names = set(self.label_fields.values())
        if len(label_names) != len(self.label_fields):
            raise ValueError("label_fields label names must be unique")
        _check_nonempty_string(self.perturbation_label, field_name="perturbation_label")
        if self.perturbation_label not in label_names:
            raise ValueError(
                "perturbation_label must name one configured label field"
            )

        drop_labels = set(self.drop_null_labels or ())
        if self.drop_null_labels is not None:
            _check_unique_strings(self.drop_null_labels, field_name="drop_null_labels")
        if drop_labels - label_names:
            raise ValueError("drop_null_labels must name configured label fields")

        _check_nonempty_string(self.null_label_value, field_name="null_label_value")
        _check_unique_strings(self.control_labels, field_name="control_labels")
        _check_unique_strings(self.special_tokens, field_name="special_tokens")
        for token_name, token_value in (
            ("pad_token", self.pad_token),
            ("cls_token", self.cls_token),
            ("unk_token", self.unk_token),
            ("eos_token", self.eos_token),
        ):
            if token_value not in self.special_tokens:
                raise ValueError(
                    f"{token_name}={token_value!r} must appear in special_tokens"
                )
        if not 0.0 <= float(self.mask_ratio) <= 1.0:
            raise ValueError("mask_ratio must be between 0 and 1")
        if int(self.ps_width) <= 0:
            raise ValueError("ps_width must be positive")
        if self.normalize_expression not in {"none", "log1p"}:
            raise ValueError(
                f"normalize_expression must be 'none' or 'log1p', "
                f"got {self.normalize_expression!r}"
            )
        if not np.isfinite(self.normalization_scale) or self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be finite and positive")

    @property
    def metadata_columns(self) -> tuple[str, ...]:
        return tuple(self.label_fields.keys())

    @property
    def label_names(self) -> tuple[str, ...]:
        return tuple(self.label_fields.values())

    @property
    def label_columns_by_name(self) -> dict[str, str]:
        return {label_name: column for column, label_name in self.label_fields.items()}

    @property
    def resolved_drop_null_labels(self) -> tuple[str, ...]:
        return tuple(self.drop_null_labels or (self.perturbation_label,))


@dataclass(frozen=True)
class _PertTFPreparedMetadata:
    row_indices: np.ndarray
    frame: pl.DataFrame
    global_to_local: dict[int, int]
    labels_by_name: dict[str, tuple[str, ...]]
    label_to_index_by_name: dict[str, dict[str, int]]
    control_label_ids: tuple[int, ...]


def _intersect_preserving_order(
    requested_indices: np.ndarray,
    allowed_indices: np.ndarray,
) -> np.ndarray:
    if requested_indices.size == 0 or allowed_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    return requested_indices[
        np.isin(requested_indices, allowed_indices, assume_unique=True)
    ].astype(np.int64, copy=False)


def _prepare_perttf_metadata(
    metadata_index: MetadataIndex,
    *,
    config: PertTFAdapterConfig,
    row_indices: Sequence[int] | np.ndarray | None,
    labels_by_name: dict[str, tuple[str, ...]] | None = None,
    label_to_index_by_name: dict[str, dict[str, int]] | None = None,
) -> _PertTFPreparedMetadata:
    candidate_row_indices = _normalize_candidate_row_indices(metadata_index, row_indices)
    if candidate_row_indices is None:
        candidate_row_indices = np.arange(len(metadata_index), dtype=np.int64)
    else:
        candidate_row_indices = candidate_row_indices.copy()
    required_columns = list(config.metadata_columns)
    if "dataset_index" not in required_columns:
        required_columns.append("dataset_index")
    if "size_factor" in metadata_index.df.columns:
        required_columns.append("size_factor")
    missing_columns = [
        column for column in required_columns if column not in metadata_index.df.columns
    ]
    if missing_columns:
        missing_list = ", ".join(repr(column) for column in missing_columns)
        raise ValueError(f"metadata_index is missing required column(s): {missing_list}")
    selected_metadata = metadata_index.take(candidate_row_indices, columns=required_columns)
    if candidate_row_indices.size != 0:
        keep_mask = np.ones(candidate_row_indices.shape, dtype=bool)
        drop_labels = set(config.resolved_drop_null_labels)
        for column in config.metadata_columns:
            label_name = config.label_fields[column]
            if label_name not in drop_labels:
                continue
            column_null_mask = np.asarray(
                [value is None for value in selected_metadata[column]],
                dtype=bool,
            )
            keep_mask &= ~column_null_mask
        candidate_row_indices = candidate_row_indices[keep_mask]
        selected_metadata = {
            column: np.asarray(values, dtype=object)[keep_mask].tolist()
            for column, values in selected_metadata.items()
        }
    global_to_local = {
        int(global_row_index): local_position
        for local_position, global_row_index in enumerate(candidate_row_indices.tolist())
    }
    if candidate_row_indices.size == 0:
        raise ValueError("pertTF metadata row pool is empty")

    resolved_labels_by_name: dict[str, tuple[str, ...]] = {}
    resolved_label_to_index_by_name: dict[str, dict[str, int]] = {}
    frame_columns: dict[str, Any] = {
        "global_row_index": candidate_row_indices.copy(),
        "dataset_index": np.asarray(selected_metadata["dataset_index"], dtype=np.int32),
    }
    preset_labels_by_name = labels_by_name or {}
    preset_label_to_index_by_name = label_to_index_by_name or {}

    for column, label_name in config.label_fields.items():
        resolved_values = tuple(
            config.null_label_value if value is None else value
            for value in selected_metadata[column]
        )
        for value in resolved_values:
            _check_nonempty_string(value, field_name=f"metadata column {column!r} label")
        label_vocab = preset_labels_by_name.get(label_name)
        label_to_index = preset_label_to_index_by_name.get(label_name)
        if label_vocab is None:
            prepend_labels = (
                config.control_labels if label_name == config.perturbation_label else ()
            )
            label_vocab = _ordered_unique(
                [
                    *prepend_labels,
                    *resolved_values,
                ]
            )
        else:
            label_vocab = tuple(label_vocab)
            _check_unique_strings(label_vocab, field_name=f"label field {label_name!r}")
        if not label_vocab:
            raise ValueError(f"label field {label_name!r} must contain at least one label")
        if label_to_index is None:
            label_to_index = {label: idx for idx, label in enumerate(label_vocab)}
        else:
            label_to_index = {
                label: int(index)
                for label, index in label_to_index.items()
            }
        label_ids = np.asarray([label_to_index[value] for value in resolved_values], dtype=np.int64)
        resolved_labels_by_name[label_name] = label_vocab
        resolved_label_to_index_by_name[label_name] = label_to_index
        frame_columns[label_name] = list(resolved_values)
        frame_columns[f"{label_name}_id"] = label_ids

    if "size_factor" in selected_metadata:
        size_factor_values = np.asarray(selected_metadata["size_factor"], dtype=np.float32)
        if size_factor_values.size != 0 and not np.isnan(size_factor_values).all():
            frame_columns["size_factor"] = size_factor_values

    control_label_ids = tuple(
        resolved_label_to_index_by_name[config.perturbation_label][label]
        for label in config.control_labels
    )

    return _PertTFPreparedMetadata(
        row_indices=candidate_row_indices,
        frame=pl.DataFrame(frame_columns),
        global_to_local=global_to_local,
        labels_by_name=resolved_labels_by_name,
        label_to_index_by_name=resolved_label_to_index_by_name,
        control_label_ids=control_label_ids,
    )


def _positions_for_global_rows(
    prepared_metadata: _PertTFPreparedMetadata,
    global_indices: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(global_indices, dtype=np.int64)
    return np.asarray(
        [prepared_metadata.global_to_local[int(global_idx)] for global_idx in normalized.tolist()],
        dtype=np.int64,
    )


def _take_prepared_column(
    prepared_metadata: _PertTFPreparedMetadata,
    global_indices: np.ndarray,
    column: str,
    *,
    dtype: Any,
) -> np.ndarray:
    positions = _positions_for_global_rows(prepared_metadata, global_indices)
    return np.asarray(prepared_metadata.frame[column], dtype=dtype)[positions]


@dataclass(frozen=True)
class PerturbationPairBatch:
    """Corpus rows of the source and target cells of one pertTF batch."""

    source_indices: np.ndarray
    target_indices: np.ndarray


def default_pair_sampler(
    metadata: pl.DataFrame,
    *,
    context_columns: Sequence[str],
    perturbation_column: str,
    control_labels: Sequence[str],
    perturbed_to_self: bool,
) -> Sampler:
    """pertTF's default pairs over the rows of ``metadata``, as a ``scorpus.pairing`` sampler.

    Controls are paired with a random cell of a perturbation in the same context. A cell with
    no such partner is paired with itself: controls whose context has no perturbation and, with
    ``perturbed_to_self`` (training, covering sources), every perturbed cell. Without it
    (validation, covering targets) control targets are paired with themselves and every
    perturbed target needs a control in its context. Rows with a missing perturbation are never
    paired.
    """
    context = list(context_columns)
    same_group = [*context, perturbation_column]
    groups = make_groups(metadata.select(same_group), by=same_group)
    is_ctrl = pl.col(perturbation_column).is_in(list(control_labels))
    is_perturbed = pl.col(perturbation_column).is_not_null() & ~is_ctrl

    ctrl_to_pert = cross(groups, on=context, source=is_ctrl, target=is_perturbed)
    ctrl_self = cross(groups, on=same_group, source=is_ctrl)
    if perturbed_to_self:
        ctrl_self = ctrl_self.filter(~pl.col("source_group").is_in(ctrl_to_pert["source_group"].to_list()))
        self_pairs = combine(ctrl_self, cross(groups, on=same_group, source=is_perturbed))
    else:
        # Covering targets, a perturbed target without a control would silently never be used.
        unmatched = groups.table.filter(
            is_perturbed & ~pl.col("group").is_in(ctrl_to_pert["target_group"].to_list())
        )
        if unmatched.height:
            raise ValueError(
                f"{unmatched.height} perturbed group(s) have no control in their context, e.g. "
                f"{unmatched.select(same_group).rows()[:3]}"
            )
        self_pairs = ctrl_self
    pairs = combine(
        ctrl_to_pert.with_columns(link=pl.lit("random")),
        self_pairs.with_columns(link=pl.lit("self")),
    )
    return Sampler(groups, pairs, links="link")


@dataclass(frozen=True)
class _PertTFPairReadRequest:
    pair_batch: PerturbationPairBatch
    batch_index: int
    seed: int
    epoch: int


class _PertTFPairReadBatchSampler:
    """One read request per batch of cell pairs, from one epoch of a ``scorpus.pairing`` sampler."""

    def __init__(
        self,
        sampler: Sampler,
        rows: np.ndarray,
        *,
        cover: str,
        batch_size: int,
        seed: int,
        drop_last: bool,
    ) -> None:
        self._sampler = sampler
        self._rows = rows
        self._cover = cover
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._drop_last = bool(drop_last)
        self.epoch = 0
        self._batches: list[PerturbationPairBatch] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = None

    def _epoch_batches(self) -> list[PerturbationPairBatch]:
        if self._batches is None:
            source_rows, target_rows = self._sampler.epoch_rows(self.epoch, seed=self._seed, cover=self._cover)
            # Sampler rows are positions in the table it was built from; map them to corpus rows.
            source_rows = self._rows[source_rows]
            target_rows = self._rows[target_rows]
            size = self._batch_size
            stop = len(source_rows) - size + 1 if self._drop_last else len(source_rows)
            self._batches = [
                PerturbationPairBatch(source_rows[start:start + size], target_rows[start:start + size])
                for start in range(0, stop, size)
            ]
        return self._batches

    def __len__(self) -> int:
        return len(self._epoch_batches())

    def __iter__(self):
        for batch_index, pair_batch in enumerate(self._epoch_batches()):
            yield [
                _PertTFPairReadRequest(
                    pair_batch=pair_batch,
                    batch_index=batch_index,
                    seed=_derive_perttf_stream_seed(
                        self._seed, epoch=self.epoch, batch_index=batch_index, stream_id=1,
                    ),
                    epoch=self.epoch,
                )
            ]


class _PertTFPairExpressionDataset:
    def __init__(
        self,
        expression_reader: Any,
        *,
        total_rows: int,
    ) -> None:
        self._reader = expression_reader
        self._total_rows = int(total_rows)

    def __len__(self) -> int:
        return self._total_rows

    def __getitems__(
        self,
        requests: Sequence[_PertTFPairReadRequest],
    ) -> list[dict[str, Any]]:
        if len(requests) != 1:
            raise ValueError(
                "paired read dataset expected exactly one pre-batched request"
            )
        request = requests[0]
        pair_batch = request.pair_batch
        return [
            {
                "request": request,
                "source_raw": self._read_raw_batch(
                    pair_batch.source_indices,
                ),
                "target_raw": self._read_raw_batch(
                    pair_batch.target_indices,
                ),
            }
        ]

    def _read_raw_batch(
        self,
        indices: np.ndarray,
    ) -> dict[str, Any]:
        return _read_expression_raw_batch(self._reader, indices)


def _collate_expression_like_raw_batch(raw_batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batch_size": raw_batch["batch_size"],
        "global_row_index": torch.as_tensor(
            raw_batch["global_row_index"],
            dtype=torch.long,
        ),
        "row_offsets": torch.as_tensor(raw_batch["row_offsets"], dtype=torch.long),
        "expressed_gene_indices": torch.as_tensor(
            raw_batch["expressed_gene_indices"],
            dtype=torch.long,
        ),
        "expression_counts": torch.as_tensor(
            raw_batch["expression_counts"],
            dtype=torch.float32,
        ),
    }
    if "dataset_index" in raw_batch:
        result["dataset_index"] = torch.as_tensor(
            raw_batch["dataset_index"],
            dtype=torch.long,
        )
    if "size_factor" in raw_batch:
        result["size_factor"] = torch.as_tensor(
            raw_batch["size_factor"],
            dtype=torch.float32,
        )
    return result


def _collate_perttf_raw_pair_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError("paired raw pair collate expected exactly one item")
    batch = items[0]
    return {
        "request": batch["request"],
        "source_raw": _collate_expression_like_raw_batch(batch["source_raw"]),
        "target_raw": _collate_expression_like_raw_batch(batch["target_raw"]),
    }


def _perttf_loader_kwargs(
    *,
    num_workers: int,
    multiprocessing_context: str | None,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> dict[str, Any]:
    workers = int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers == 0:
        if multiprocessing_context is not None:
            raise ValueError("multiprocessing_context requires num_workers > 0")
        if persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")
        if prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        return kwargs

    kwargs["persistent_workers"] = persistent_workers
    if multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = multiprocessing_context
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


class PertTFPairedBatchBuilder:
    """Build pertTF-style paired source/target batches from corpus rows."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        seq_len: int,
        config: PertTFAdapterConfig | None = None,
        adapter: "PertTFCorpusAdapter" | None = None,
        prepared_metadata: _PertTFPreparedMetadata | None = None,
        missing_token_policy: str = "exclude",
        device: torch.device | str | None = "cpu",
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        resolved_adapter = adapter or PertTFCorpusAdapter.from_corpus(corpus, config)
        self.corpus = corpus
        self.adapter = resolved_adapter
        self.config = resolved_adapter.config
        self.seq_len = int(seq_len)
        self.device = torch.device("cpu" if device is None else device)
        assert missing_token_policy in {
            "exclude",
            "pad",
        }, "missing_token_policy must be one of 'exclude' or 'pad'"
        self.missing_token_policy = missing_token_policy
        self._gene_token_mapper = self.adapter.gene_token_mapper
        self._pipeline = SparseBatchProcessor(
            corpus.feature_registry,
            seq_len=self.seq_len,
            sampling_gene_mask=(
                self._gene_token_mapper.tokenizable_by_global_id
                if self.missing_token_policy == "exclude"
                and not bool(self._gene_token_mapper.tokenizable_by_global_id.all())
                else None
            ),
        )
        self._pad_token_id = int(self._gene_token_mapper.pad_token_id)
        if self.config.append_cls and self._gene_token_mapper.cls_token_id is None:
            raise ValueError("append_cls=True requires a cls token in the gene token mapper")
        self._cls_token_id = self._gene_token_mapper.cls_token_id
        self._prepared_metadata = prepared_metadata or _prepare_perttf_metadata(
            corpus.metadata_index,
            config=self.config,
            row_indices=None,
            labels_by_name=self.adapter.labels_by_name,
            label_to_index_by_name=self.adapter.label_to_index_by_name,
        )

    def build_paired_batch(
        self,
        pair_batch: PerturbationPairBatch,
        *,
        seed: int | None = None,
        sampled_gene_ids: torch.Tensor | None = None,
        sampling_mode: str = "hvg",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        hvg_top_k: int | None = None,
    ) -> dict[str, torch.Tensor]:
        source_raw = _read_expression_raw_batch(
            self.corpus.expression_reader,
            pair_batch.source_indices,
        )
        target_raw = _read_expression_raw_batch(
            self.corpus.expression_reader,
            pair_batch.target_indices,
        )

        return self.build_from_raw_pair_batch(
            pair_batch,
            source_raw,
            target_raw,
            seed=seed,
            sampled_gene_ids=sampled_gene_ids,
            sampling_mode=sampling_mode,
            expressed_weight=expressed_weight,
            hvg_weight=hvg_weight,
            hvg_top_k=hvg_top_k,
        )

    def build_from_raw_pair_batch(
        self,
        pair_batch: PerturbationPairBatch,
        source_raw: dict[str, Any],
        target_raw: dict[str, Any],
        *,
        seed: int | None = None,
        sampled_gene_ids: torch.Tensor | None = None,
        sampling_mode: str = "hvg",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        hvg_top_k: int | None = None,
    ) -> dict[str, torch.Tensor]:
        source_raw = self._with_metadata_columns(source_raw)
        target_raw = self._with_metadata_columns(target_raw)
        self._check_raw_pair_batch(pair_batch, source_raw, target_raw)

        source_processed = self._pipeline.process_batch(
            source_raw,
            device=self.device,
            generator=self._torch_generator(seed),
            sampled_gene_ids=sampled_gene_ids,
            sampling_mode=sampling_mode,
            expressed_weight=expressed_weight,
            hvg_weight=hvg_weight,
            hvg_top_k=hvg_top_k,
        )
        target_processed = self._pipeline.process_batch(
            target_raw,
            device=self.device,
            sampled_gene_ids=source_processed["sampled_gene_ids"],
            sampling_mode=sampling_mode,
            expressed_weight=expressed_weight,
            hvg_weight=hvg_weight,
            hvg_top_k=hvg_top_k,
        )

        source_sampled_gene_ids = source_processed["sampled_gene_ids"]
        target_sampled_gene_ids = target_processed["sampled_gene_ids"]
        source_valid_mask = source_processed["valid_mask"]
        target_valid_mask = target_processed["valid_mask"]
        aligned_target_gene_ids = torch.where(
            target_valid_mask,
            target_sampled_gene_ids,
            source_sampled_gene_ids,
        )
        if not torch.equal(source_sampled_gene_ids, aligned_target_gene_ids):
            raise RuntimeError(
                "target reconstruction changed valid source-sampled gene IDs"
            )

        source_processed = self._normalize_sampled_counts(source_processed)
        target_processed = self._normalize_sampled_counts(target_processed)

        gene_ids, source_token_valid_mask = self._to_token_ids(
            source_sampled_gene_ids,
            source_valid_mask,
        )
        source_values = self._to_value_tensor(
            source_processed["sampled_counts"],
            source_token_valid_mask,
        )
        target_values_next = self._to_value_tensor(
            target_processed["sampled_counts"],
            target_valid_mask & source_token_valid_mask,
        )

        batch_size = int(source_processed["batch_size"])
        masked_values = self._mask_values(
            source_values,
            generator=self._torch_generator(None if seed is None else int(seed) + 1),
        )

        batch = {
            "gene_ids": gene_ids,
            "next_gene_ids": gene_ids.clone(),
            "values": masked_values,
            "target_values": source_values,
            "target_values_next": target_values_next,
            "sf": self._size_factor_tensor(source_processed, batch_size),
            "sf_next": self._size_factor_tensor(target_processed, batch_size),
            "index": torch.as_tensor(
                pair_batch.source_indices,
                dtype=torch.long,
                device=self.device,
            ),
            "next_index": torch.as_tensor(
                pair_batch.target_indices,
                dtype=torch.long,
                device=self.device,
            ),
            "ps": torch.zeros(
                (batch_size, int(self.config.ps_width)),
                dtype=torch.float32,
                device=self.device,
            ),
            "ps_next": torch.zeros(
                (batch_size, int(self.config.ps_width)),
                dtype=torch.float32,
                device=self.device,
            ),
        }
        for label_name in self.config.label_names:
            source_label_ids = self._label_ids_for_rows(
                label_name,
                pair_batch.source_indices,
            )
            target_label_ids = self._label_ids_for_rows(
                label_name,
                pair_batch.target_indices,
            )
            batch[f"{label_name}_labels"] = torch.as_tensor(
                source_label_ids,
                dtype=torch.long,
                device=self.device,
            )
            batch[f"{label_name}_labels_next"] = torch.as_tensor(
                target_label_ids,
                dtype=torch.long,
                device=self.device,
            )
        if self.config.include_full_expr:
            full_expr, full_expr_mask = self._build_full_expression_tensor(source_raw)
            full_expr_next, full_expr_next_mask = self._build_full_expression_tensor(
                target_raw
            )
            batch["full_gene_ids"] = self._full_gene_ids(batch_size)
            batch["full_expr"] = full_expr
            batch["full_expr_next"] = full_expr_next
            batch["full_expr_mask"] = full_expr_mask
            batch["full_expr_next_mask"] = full_expr_next_mask
        return batch

    __call__ = build_paired_batch

    def _with_metadata_columns(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        global_indices = self._raw_batch_numpy(raw_batch, "global_row_index", dtype=np.int64)
        resolved = dict(raw_batch)
        resolved["dataset_index"] = _take_prepared_column(
            self._prepared_metadata,
            global_indices,
            "dataset_index",
            dtype=np.int32,
        )
        if "size_factor" in self._prepared_metadata.frame.columns:
            resolved["size_factor"] = _take_prepared_column(
                self._prepared_metadata,
                global_indices,
                "size_factor",
                dtype=np.float32,
            )
        return resolved

    def _label_ids_for_rows(self, label_name: str, global_indices: np.ndarray) -> np.ndarray:
        return _take_prepared_column(
            self._prepared_metadata,
            global_indices,
            f"{label_name}_id",
            dtype=np.int64,
        )

    def _check_raw_pair_batch(
        self,
        pair_batch: PerturbationPairBatch,
        source_raw: dict[str, Any],
        target_raw: dict[str, Any],
    ) -> None:
        if pair_batch.source_indices.shape != pair_batch.target_indices.shape:
            raise ValueError("pair_batch source and target indices must have matching shapes")
        for name, raw_batch, expected_indices in (
            ("source_raw", source_raw, pair_batch.source_indices),
            ("target_raw", target_raw, pair_batch.target_indices),
        ):
            batch_size = int(raw_batch.get("batch_size", -1))
            if batch_size != int(expected_indices.shape[0]):
                raise ValueError(
                    f"{name} batch_size {batch_size} does not match paired batch size "
                    f"{int(expected_indices.shape[0])}"
                )
            raw_indices = self._raw_batch_numpy(raw_batch, "global_row_index", dtype=np.int64)
            if raw_indices.shape != expected_indices.shape or not np.array_equal(
                raw_indices,
                expected_indices,
            ):
                raise ValueError(f"{name} global_row_index does not match pair_batch ordering")

    def _normalize_sampled_counts(
        self, processed: dict[str, Any]
    ) -> dict[str, Any]:
        if self.config.normalize_expression == "none":
            return processed
        if self.config.normalize_expression == "log1p":
            sf = processed["size_factor"]
            counts = processed["sampled_counts"]
            valid = processed["valid_mask"]
            processed["sampled_counts"] = torch.where(
                valid,
                torch.log1p(
                    counts / sf.unsqueeze(1) * float(self.config.normalization_scale)
                ),
                counts,
            )
            return processed
        raise ValueError(
            f"unsupported normalize_expression: {self.config.normalize_expression!r}"
        )

    def _raw_batch_numpy(
        self,
        raw_batch: dict[str, Any],
        key: str,
        *,
        dtype: Any,
    ) -> np.ndarray:
        if key not in raw_batch:
            raise KeyError(f"raw batch is missing required key {key!r}")
        values = raw_batch[key]
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(values, dtype=dtype)

    def _torch_generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        if self.device.type == "cuda":
            generator = torch.Generator(device=self.device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator

    def _to_token_ids(
        self,
        sampled_gene_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, token_valid_mask = self._gene_token_mapper.encode_global_ids(
            sampled_gene_ids.to(dtype=torch.long),
            valid_mask,
        )
        return self._prepend_cls_gene_ids(token_ids), token_valid_mask

    def _to_value_tensor(
        self,
        sampled_counts: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = torch.where(
            valid_mask,
            sampled_counts.to(dtype=torch.float32),
            torch.full_like(
                sampled_counts,
                float(self.config.pad_value),
                dtype=torch.float32,
            ),
        )
        return self._prepend_cls_values(values)

    def _prepend_cls_gene_ids(self, gene_ids: torch.Tensor) -> torch.Tensor:
        if not self.config.append_cls:
            return gene_ids
        cls_column = torch.full(
            (gene_ids.shape[0], 1),
            int(self._cls_token_id),
            dtype=torch.long,
            device=gene_ids.device,
        )
        return torch.cat([cls_column, gene_ids], dim=1)

    def _prepend_cls_values(self, values: torch.Tensor) -> torch.Tensor:
        if not self.config.append_cls:
            return values
        cls_column = torch.full(
            (values.shape[0], 1),
            float(self.config.cls_value),
            dtype=torch.float32,
            device=values.device,
        )
        return torch.cat([cls_column, values], dim=1)

    def _prepend_cls_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.config.append_cls:
            return mask
        cls_column = torch.ones(
            (mask.shape[0], 1),
            dtype=torch.bool,
            device=mask.device,
        )
        return torch.cat([cls_column, mask], dim=1)

    def _full_gene_ids(self, batch_size: int) -> torch.Tensor:
        full_gene_ids = torch.as_tensor(
            self._gene_token_mapper.global_to_token_id,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        full_gene_ids = self._prepend_cls_gene_ids(full_gene_ids)
        return full_gene_ids.expand(batch_size, -1).clone()

    def _build_full_expression_tensor(
        self,
        raw_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(raw_batch["batch_size"])
        global_vocab = self.corpus.feature_registry.global_vocab_size
        full_expr = torch.zeros(
            (batch_size, global_vocab),
            dtype=torch.float32,
            device=self.device,
        )
        row_offsets = np.asarray(raw_batch["row_offsets"], dtype=np.int64)
        local_gene_ids = np.asarray(raw_batch["expressed_gene_indices"], dtype=np.int64)
        counts = np.asarray(raw_batch["expression_counts"], dtype=np.float32)
        dataset_indices = np.asarray(raw_batch["dataset_index"], dtype=np.int64)

        local_to_global = self.corpus.feature_registry.local_to_global_map
        dataset_has_gene = self.corpus.feature_registry.dataset_has_gene
        tokenizable = self._gene_token_mapper.tokenizable_by_global_id
        full_mask = torch.as_tensor(
            dataset_has_gene[dataset_indices] & tokenizable,
            dtype=torch.bool,
            device=self.device,
        )

        for row_idx in range(batch_size):
            start = int(row_offsets[row_idx])
            stop = int(row_offsets[row_idx + 1])
            if start >= stop:
                continue
            dataset_index = int(dataset_indices[row_idx])
            row_local_gene_ids = local_gene_ids[start:stop]
            row_global_gene_ids = local_to_global[dataset_index, row_local_gene_ids]
            if np.any(row_global_gene_ids < 0):
                raise RuntimeError(
                    "raw batch contained unmapped local gene IDs during full_expr build"
                )
            full_expr[
                row_idx,
                torch.as_tensor(
                    row_global_gene_ids,
                    dtype=torch.long,
                    device=self.device,
                ),
            ] = torch.as_tensor(
                counts[start:stop],
                dtype=torch.float32,
                device=self.device,
            )

        if self.config.normalize_expression == "log1p" and "size_factor" in raw_batch:
            sf = torch.as_tensor(
                raw_batch["size_factor"],
                dtype=torch.float32,
                device=self.device,
            )
            full_expr = torch.log1p(
                full_expr / sf.unsqueeze(1) * float(self.config.normalization_scale)
            )

        return self._prepend_cls_values(full_expr), self._prepend_cls_mask(full_mask)

    def _mask_values(
        self,
        values: torch.Tensor,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.config.mask_ratio <= 0.0:
            return values.clone()
        masked = values.clone()
        eligible = (
            (masked != float(self.config.pad_value))
            & (masked != float(self.config.cls_value))
        )
        scores = torch.rand(
            masked.shape,
            device=masked.device,
            generator=generator,
        )
        mask = (scores < float(self.config.mask_ratio)) & eligible
        masked[mask] = float(self.config.mask_value)
        return masked

    def _size_factor_tensor(
        self,
        processed_batch: dict[str, Any],
        batch_size: int,
    ) -> torch.Tensor:
        raw_size_factor = processed_batch.get("size_factor")
        if raw_size_factor is None:
            raise RuntimeError("pertTF batches require size_factor metadata")
        size_factor = torch.as_tensor(
            raw_size_factor,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)
        if size_factor.shape[0] != batch_size:
            raise RuntimeError("size_factor length does not match batch size")
        # PDL stores the normalization denominator; pertTF expects its multiplier.
        return size_factor.reciprocal() * float(self.config.normalization_scale)


@dataclass(frozen=True)
class PertTFCorpusAdapter:
    """Bundle pertTF vocab and label mappings for one loaded corpus."""

    config: PertTFAdapterConfig
    special_tokens: tuple[str, ...]
    feature_ids_by_global_id: tuple[str, ...]
    tokens_in_order: tuple[str, ...]
    stoi: dict[str, int]
    special_token_offset: int
    gene_token_ids: np.ndarray
    gene_token_mapper: GeneTokenMapper
    labels_by_name: dict[str, tuple[str, ...]]
    label_to_index_by_name: dict[str, dict[str, int]]
    control_label_ids: tuple[int, ...]

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        config: PertTFAdapterConfig | None = None,
        row_indices: Sequence[int] | np.ndarray | None = None,
        prepared_metadata: _PertTFPreparedMetadata | None = None,
        tokenizer_stoi: Mapping[str, int] | None = None,
        gene_token_mapper: GeneTokenMapper | None = None,
        feature_id_to_token: Mapping[str, str] | None = None,
        label_mappings: Mapping[str, Mapping[str, int]] | None = None,
    ) -> "PertTFCorpusAdapter":
        resolved = config or PertTFAdapterConfig()
        if prepared_metadata is not None and label_mappings is not None:
            raise ValueError("Pass prepared_metadata or label_mappings, not both")
        ordered_labels = {}
        for name, mapping in (label_mappings or {}).items():
            if name not in resolved.label_names:
                raise ValueError(f"Unknown label mapping: {name}")
            if sorted(mapping.values()) != list(range(len(mapping))):
                raise ValueError(f"{name}: label IDs must be contiguous from zero")
            ordered_labels[name] = tuple(sorted(mapping, key=mapping.get))
        if gene_token_mapper is not None and tokenizer_stoi is not None:
            raise ValueError("pass either gene_token_mapper or tokenizer_stoi, not both")
        if gene_token_mapper is None:
            if tokenizer_stoi is None:
                gene_token_mapper = GeneTokenMapper.from_feature_registry(
                    corpus.feature_registry,
                    special_tokens=resolved.special_tokens,
                    pad_token=resolved.pad_token,
                    cls_token=resolved.cls_token,
                    unk_token=resolved.unk_token,
                )
            else:
                gene_token_mapper = GeneTokenMapper.from_tokenizer_stoi(
                    corpus.feature_registry,
                    tokenizer_stoi,
                    pad_token=resolved.pad_token,
                    cls_token=resolved.cls_token,
                    unk_token=resolved.unk_token,
                    feature_id_to_token=feature_id_to_token,
                )
        gene_token_mapper.check_feature_registry(corpus.feature_registry)
        if prepared_metadata is None:
            prepared_metadata = _prepare_perttf_metadata(
                corpus.metadata_index,
                config=resolved,
                row_indices=row_indices,
                labels_by_name=ordered_labels,
                label_to_index_by_name=label_mappings,
            )
        labels_by_name = {
            label_name: tuple(labels)
            for label_name, labels in prepared_metadata.labels_by_name.items()
        }
        label_to_index_by_name = {
            label_name: dict(label_to_index)
            for label_name, label_to_index in prepared_metadata.label_to_index_by_name.items()
        }
        return cls(
            config=resolved,
            special_tokens=tuple(resolved.special_tokens),
            feature_ids_by_global_id=gene_token_mapper.feature_ids_by_global_id,
            tokens_in_order=gene_token_mapper.tokens_in_order,
            stoi=dict(gene_token_mapper.stoi),
            special_token_offset=len(resolved.special_tokens),
            gene_token_ids=gene_token_mapper.global_to_token_id.copy(),
            gene_token_mapper=gene_token_mapper,
            labels_by_name=labels_by_name,
            label_to_index_by_name=label_to_index_by_name,
            control_label_ids=prepared_metadata.control_label_ids,
        )

    def token_id_for_global_id(self, global_id: int) -> int:
        if global_id < 0 or global_id >= len(self.feature_ids_by_global_id):
            raise IndexError(f"global_id {global_id} out of range")
        return int(self.gene_token_mapper.global_to_token_id[int(global_id)])

    def global_id_for_token_id(self, token_id: int) -> int | None:
        special_ids = {
            int(self.stoi[token])
            for token in self.special_tokens
            if token in self.stoi
        }
        if int(token_id) in special_ids:
            return None
        matches = np.flatnonzero(
            (self.gene_token_mapper.global_to_token_id == int(token_id))
            & self.gene_token_mapper.tokenizable_by_global_id
        )
        if matches.size == 0:
            raise IndexError(f"token_id {token_id} out of range")
        if matches.size > 1:
            raise ValueError(f"token_id {token_id} maps to multiple global gene IDs")
        return int(matches[0])

    def feature_id_for_token_id(self, token_id: int) -> str:
        global_id = self.global_id_for_token_id(token_id)
        if global_id is None:
            raise KeyError(f"token_id {token_id} is reserved for a special token")
        return self.feature_ids_by_global_id[global_id]

    def to_simple_vocab_stoi(self) -> dict[str, int]:
        return dict(self.stoi)

    def to_simple_vocab_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_simple_vocab_stoi(), indent=2, sort_keys=False),
            encoding="utf-8",
        )

    def encode_label(self, name: str, label: Any) -> int:
        _check_nonempty_string(name, field_name="label_name")
        _check_nonempty_string(label, field_name=f"label field {name!r} value")
        return self.label_to_index_by_name[name][label]

    def encode_labels(self, name: str, labels: Sequence[Any]) -> np.ndarray:
        _check_nonempty_string(name, field_name="label_name")
        label_to_index = self.label_to_index_by_name[name]
        for label in labels:
            _check_nonempty_string(label, field_name=f"label field {name!r} value")
        return np.asarray([label_to_index[label] for label in labels], dtype=np.int64)

    def to_reference_dict(self) -> dict[str, Any]:
        return {
            "genes": list(self.feature_ids_by_global_id),
            "gene_token_ids": self.gene_token_ids.copy(),
            "tokenizable_by_global_id": self.gene_token_mapper.tokenizable_by_global_id.copy(),
            "simple_vocab_stoi": self.to_simple_vocab_stoi(),
            "labels_by_name": {
                label_name: tuple(labels)
                for label_name, labels in self.labels_by_name.items()
            },
            "label_to_index_by_name": {
                label_name: dict(label_to_index)
                for label_name, label_to_index in self.label_to_index_by_name.items()
            },
            "control_label_ids": self.control_label_ids,
        }


class PertTFPairedBatchLoader:
    """Slim public iterable that yields final pertTF paired batches.

    ``sampler`` (``scorpus.pairing``) decides which cells are paired. Its rows are positions in
    ``rows``, the corpus rows of the table its groups were built from (see
    ``default_pair_sampler``). Each epoch uses every cell of the covered side once
    (``cover="source"`` for training, ``"target"`` for validation), and the cell pairs are cut
    into batches of ``batch_size``. The target's perturbation label is the target cell's own.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        sampler: Sampler,
        rows: Sequence[int] | np.ndarray,
        batch_size: int,
        seq_len: int,
        cover: str = "source",
        config: PertTFAdapterConfig | None = None,
        adapter: PertTFCorpusAdapter | None = None,
        seed: int = 0,
        drop_last: bool = True,
        sampling_mode: str = "hvg",
        expressed_weight: float = 3.0,
        hvg_weight: float = 3.0,
        hvg_top_k: int | None = None,
        missing_token_policy: str = "exclude",
        num_workers: int = 0,
        multiprocessing_context: str | None = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
        device: torch.device | str | None = "cpu",
    ) -> None:
        self.corpus = corpus
        self.sampler = sampler
        self.rows = np.asarray(rows, dtype=np.int64).copy()
        resolved_config = adapter.config if adapter is not None else (config or PertTFAdapterConfig())
        prepared_metadata = _prepare_perttf_metadata(
            corpus.metadata_index,
            config=resolved_config,
            row_indices=self.rows,
            labels_by_name=None if adapter is None else adapter.labels_by_name,
            label_to_index_by_name=None if adapter is None else adapter.label_to_index_by_name,
        )
        resolved_adapter = adapter
        if resolved_adapter is None:
            resolved_adapter = PertTFCorpusAdapter.from_corpus(
                corpus,
                resolved_config,
                prepared_metadata=prepared_metadata,
            )
        self._builder = PertTFPairedBatchBuilder(
            corpus,
            seq_len=seq_len,
            config=resolved_config,
            adapter=resolved_adapter,
            prepared_metadata=prepared_metadata,
            missing_token_policy=missing_token_policy,
            device=device,
        )
        self.adapter = self._builder.adapter
        self.config = self._builder.config
        self.effective_label_row_indices = prepared_metadata.row_indices.copy()
        self._request_sampler = _PertTFPairReadBatchSampler(
            sampler,
            self.rows,
            cover=cover,
            batch_size=batch_size,
            seed=seed,
            drop_last=drop_last,
        )
        self._dataset = _PertTFPairExpressionDataset(
            corpus.expression_reader,
            total_rows=len(corpus.metadata_index),
        )
        self._loader_kwargs = _perttf_loader_kwargs(
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        self._sampling_mode = sampling_mode
        self._expressed_weight = float(expressed_weight)
        self._hvg_weight = float(hvg_weight)
        self._hvg_top_k = hvg_top_k
        self._data_loader = DataLoader(
            self._dataset,
            batch_sampler=self._request_sampler,
            collate_fn=_collate_perttf_raw_pair_batch,
            **self._loader_kwargs,
        )

    def __len__(self) -> int:
        return len(self._request_sampler)

    @property
    def dataset(self):
        """Evaluation needs a prediction_only flag, not an AnnData dataset."""
        return self._dataset

    @property
    def epoch(self) -> int:
        return self._request_sampler.epoch

    def set_epoch(self, epoch: int) -> None:
        self._request_sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for raw_pair_batch in self._data_loader:
            request = raw_pair_batch["request"]
            yield self._builder.build_from_raw_pair_batch(
                request.pair_batch,
                raw_pair_batch["source_raw"],
                raw_pair_batch["target_raw"],
                seed=request.seed,
                sampling_mode=self._sampling_mode,
                expressed_weight=self._expressed_weight,
                hvg_weight=self._hvg_weight,
                hvg_top_k=self._hvg_top_k,
            )
