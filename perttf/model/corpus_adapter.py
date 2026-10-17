"""pertTF batches from on-disk scorpus corpora.

Which cells are paired is described with ``scorpus.pairing``. This module turns the pairs into
pertTF tensors: gene tokens, sampled genes, log1p expression, masking, CLS, full expression and
labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from scorpus.loaders import ExpressionBatch
from scorpus.pairing import Sampler, combine, cross, make_groups

if TYPE_CHECKING:
    from scorpus import Corpus

__all__ = [
    "GeneTokens",
    "PertTFAdapterConfig",
    "PertTFPairedBatchLoader",
    "default_pair_sampler",
]

SPECIAL_TOKENS = ("<pad>", "<cls>", "<unk>", "<eos>")
SAMPLING_MODES = ("uniform", "expressed", "hvg")


@dataclass(frozen=True)
class GeneTokens:
    """pertTF token IDs of the corpus genes (the corpus gene axis is the global gene ID)."""

    stoi: dict[str, int]
    token_of_gene: np.ndarray  # global gene ID -> token ID; the pad token for genes without one
    tokenizable: np.ndarray  # global gene ID -> the gene has a token
    pad_id: int
    cls_id: int

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        tokenizer_stoi: Mapping[str, int] | None = None,
        feature_id_to_token: Mapping[str, str] | None = None,
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
    ) -> "GeneTokens":
        """A new vocabulary (special tokens, then the corpus genes), or the genes' IDs in an
        existing tokenizer such as a pretrained checkpoint's; genes it lacks get no token."""
        genes = corpus.feature_registry.global_feature_ids
        if tokenizer_stoi is None:
            stoi = {token: token_id for token_id, token in enumerate((*SPECIAL_TOKENS, *genes))}
        else:
            stoi = {token: int(token_id) for token, token_id in tokenizer_stoi.items()}
        names = genes if feature_id_to_token is None else [feature_id_to_token.get(gene) for gene in genes]
        return cls(
            stoi=stoi,
            token_of_gene=np.array([stoi.get(name, stoi[pad_token]) for name in names], dtype=np.int64),
            tokenizable=np.array([name in stoi for name in names], dtype=bool),
            pad_id=stoi[pad_token],
            cls_id=stoi[cls_token],
        )


@dataclass(frozen=True)
class PertTFAdapterConfig:
    """How corpus cells become pertTF tensors."""

    label_fields: dict[str, str]  # metadata column -> label name, e.g. {"perturb_label": "perturbation"}
    normalization_scale: float = 1.0  # values are log1p(count / size_factor * normalization_scale)
    mask_ratio: float = 0.15
    pad_value: float = -2
    mask_value: float = -1
    cls_value: float = -3
    # Genes of a cell are drawn from those its dataset measured (and that have a token):
    # "uniform", "expressed" (expressed genes get 1 + expressed_weight) or "hvg" (HVGs get
    # 1 + hvg_weight; the dataset's default HVGs, or its hvg_top_k highest ranked).
    sampling_mode: str = "uniform"
    expressed_weight: float = 3.0
    hvg_weight: float = 3.0
    hvg_top_k: int | None = None


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


class _PairReads:
    """DataLoader dataset: reads one batch's source and target expression, in worker processes."""

    def __init__(self, reader, n_cells: int) -> None:
        self._reader = reader
        self._n_cells = n_cells

    def __len__(self) -> int:
        return self._n_cells

    def __getitem__(self, item: tuple[int, np.ndarray, np.ndarray]) -> tuple[int, ExpressionBatch, ExpressionBatch]:
        batch_index, source_rows, target_rows = item
        read = self._reader.read_expression_flat
        return batch_index, read(source_rows.tolist()), read(target_rows.tolist())


class _EpochReads:
    """DataLoader sampler: (batch index, source rows, target rows) of the loader's current epoch."""

    def __init__(self, loader: "PertTFPairedBatchLoader") -> None:
        self._loader = loader

    def __iter__(self):
        rows = self._loader.rows
        for i, (source, target) in enumerate(self._loader._epoch_batches()):
            yield i, rows[source], rows[target]

    def __len__(self) -> int:
        return len(self._loader._epoch_batches())


def _unchanged(item):
    return item


def _with_cls(values: torch.Tensor, cls_value) -> torch.Tensor:
    cls_column = torch.full((len(values), 1), cls_value, dtype=values.dtype, device=values.device)
    return torch.cat([cls_column, values], dim=1)


class PertTFPairedBatchLoader:
    """pertTF batches of (source, target) cell pairs.

    ``sampler`` (``scorpus.pairing``) decides which cells are paired. Its rows are positions in
    ``rows``, the corpus rows of the metadata table it was built from (see
    ``default_pair_sampler``). Each epoch uses every cell of the covered side once
    (``cover="source"`` for training, ``"target"`` for validation) and cuts the pairs into
    batches of ``batch_size``. Worker processes read expression; the tensors are built in the
    main process, on ``device`` (a GPU is much faster for gene sampling and full expression).

    ``labels`` gives {label: ID} for some or all label names of ``config.label_fields``; the
    others are numbered in order of first appearance in ``rows``. The target's labels are the
    target cell's own.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        sampler: Sampler,
        rows: Sequence[int] | np.ndarray,
        config: PertTFAdapterConfig,
        tokens: GeneTokens,
        seq_len: int,
        batch_size: int,
        cover: str = "source",
        labels: Mapping[str, Mapping[str, int]] | None = None,
        seed: int = 0,
        drop_last: bool = False,
        num_workers: int = 0,
        multiprocessing_context: str | None = None,
        persistent_workers: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        if config.sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, got {config.sampling_mode!r}")
        self.corpus, self.sampler, self.config, self.tokens = corpus, sampler, config, tokens
        self.rows = np.asarray(rows, dtype=np.int64)
        self.seq_len, self.batch_size, self.cover = seq_len, batch_size, cover
        self.seed, self.drop_last = seed, drop_last
        self.device = torch.device(device)
        self.epoch = 0
        self._batches = None
        self.dataset = _PairReads(corpus.expression_reader, len(self.rows))
        # One DataLoader for all epochs, so persistent workers are reused; its sampler follows set_epoch.
        self._data_loader = DataLoader(
            self.dataset, sampler=_EpochReads(self), batch_size=None, collate_fn=_unchanged,
            num_workers=num_workers, multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
        )

        # Per-cell metadata, indexed by position in rows.
        metadata = corpus.metadata_index.df[self.rows]
        self.label_to_index_by_name: dict[str, dict[str, int]] = {}
        self._label_ids: dict[str, np.ndarray] = {}
        for column, name in config.label_fields.items():
            values = metadata[column]
            # A missing label would become a wrong class ID without any error.
            if values.null_count():
                raise ValueError(f"{values.null_count()} rows have no {column!r}; leave them out of rows")
            mapping = dict((labels or {}).get(name) or {
                label: index for index, label in enumerate(values.unique(maintain_order=True).to_list())
            })
            self.label_to_index_by_name[name] = mapping
            self._label_ids[name] = values.replace_strict(mapping, return_dtype=pl.Int32).to_numpy()
        self._dataset_index = metadata["dataset_index"].to_numpy()
        self._size_factor = metadata["size_factor"].to_numpy().astype(np.float32)

        registry = corpus.feature_registry
        on_device = self._on_device
        self._local_to_global = on_device(registry.local_to_global_map.astype(np.int64))
        self._n_genes = registry.global_vocab_size
        self._has_gene = on_device(registry.dataset_has_gene)
        self._tokenizable = on_device(tokens.tokenizable)
        # Genes without a token are never sampled.
        self._sampling_pool = self._has_gene & self._tokenizable
        if config.hvg_top_k is None:
            self._hvg = on_device(registry.hvg_mask)
        else:
            ranks = registry.hvg_rank_matrix
            self._hvg = on_device((ranks > 0) & (ranks <= config.hvg_top_k))
        self._token_of_gene = on_device(tokens.token_of_gene)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = None

    def _epoch_batches(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._batches is None:
            source, target = self.sampler.epoch_rows(self.epoch, seed=self.seed, cover=self.cover)
            size = self.batch_size
            stop = len(source) - size + 1 if self.drop_last else len(source)
            self._batches = [(source[i:i + size], target[i:i + size]) for i in range(0, stop, size)]
        return self._batches

    def __len__(self) -> int:
        return len(self._epoch_batches())

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        batches = self._epoch_batches()
        for i, source_expr, target_expr in self._data_loader:
            yield self._build(*batches[i], source_expr, target_expr, self._batch_seed(i))

    def build_batch(self, source: np.ndarray, target: np.ndarray, seed: int) -> dict[str, torch.Tensor]:
        """One batch for the cells at positions ``source`` and ``target`` of ``rows``."""
        read = self.corpus.expression_reader.read_expression_flat
        return self._build(source, target, read(self.rows[source].tolist()), read(self.rows[target].tolist()), seed)

    def _batch_seed(self, batch_index: int) -> int:
        return int(np.random.SeedSequence([self.seed, self.epoch, batch_index, 1]).generate_state(1, dtype=np.uint64)[0])

    def _build(self, source, target, source_expr, target_expr, seed: int) -> dict[str, torch.Tensor]:
        config, on_device = self.config, self._on_device
        dataset, dataset_next = on_device(self._dataset_index[source]), on_device(self._dataset_index[target])
        counts, counts_next = self._dense_counts(source_expr, dataset), self._dense_counts(target_expr, dataset_next)
        measured, measured_next = self._has_gene[dataset], self._has_gene[dataset_next]
        size_factor, size_factor_next = on_device(self._size_factor[source]), on_device(self._size_factor[target])

        # Genes are sampled for the source cell (only measured genes with a token; -1 where it has
        # too few) and the target is read at the same genes, where its dataset measured them.
        genes = self._sample_genes(counts, dataset, seed)
        safe_genes = genes.clamp(min=0)
        valid = genes >= 0
        valid_next = valid & measured_next.gather(1, safe_genes)
        gene_ids = torch.where(valid, self._token_of_gene[safe_genes], self.tokens.pad_id)
        values = torch.where(valid, self._log1p(counts.gather(1, safe_genes), size_factor), config.pad_value)
        values_next = torch.where(
            valid_next, self._log1p(counts_next.gather(1, safe_genes), size_factor_next), config.pad_value,
        )
        gene_ids = _with_cls(gene_ids, self.tokens.cls_id)
        values = _with_cls(values, config.cls_value)

        n_cells = len(source)
        batch = {
            "gene_ids": gene_ids,
            "next_gene_ids": gene_ids.clone(),
            "values": self._mask(values, seed + 1),
            "target_values": values,
            "target_values_next": _with_cls(values_next, config.cls_value),
            # pertTF's sf is the normalization multiplier; the corpus stores the size factor.
            "sf": size_factor.reshape(-1, 1).reciprocal() * config.normalization_scale,
            "sf_next": size_factor_next.reshape(-1, 1).reciprocal() * config.normalization_scale,
            "index": on_device(self.rows[source]),
            "next_index": on_device(self.rows[target]),
            "ps": torch.zeros((n_cells, 1), device=self.device),
            "ps_next": torch.zeros((n_cells, 1), device=self.device),
            "full_gene_ids": _with_cls(self._token_of_gene.unsqueeze(0), self.tokens.cls_id).expand(n_cells, -1).clone(),
            "full_expr": _with_cls(self._log1p(counts, size_factor), config.cls_value),
            "full_expr_next": _with_cls(self._log1p(counts_next, size_factor_next), config.cls_value),
            # Unmeasured genes and genes without a token stay out of the full-expression losses.
            "full_expr_mask": _with_cls(measured & self._tokenizable, True),
            "full_expr_next_mask": _with_cls(measured_next & self._tokenizable, True),
        }
        for name, ids in self._label_ids.items():
            batch[f"{name}_labels"] = torch.as_tensor(ids[source], dtype=torch.long, device=self.device)
            batch[f"{name}_labels_next"] = torch.as_tensor(ids[target], dtype=torch.long, device=self.device)
        return batch

    def _dense_counts(self, expr: ExpressionBatch, dataset: torch.Tensor) -> torch.Tensor:
        """Counts on the corpus gene axis (cells x genes), filled on the device from the sparse batch."""
        cell = self._on_device(np.repeat(np.arange(len(dataset)), np.diff(expr.row_offsets)))
        gene = self._local_to_global[dataset[cell], self._on_device(expr.expressed_gene_indices.astype(np.int64))]
        counts = torch.zeros((len(dataset), self._n_genes), device=self.device)
        counts[cell, gene] = self._on_device(expr.expression_counts.astype(np.float32))
        return counts

    def _on_device(self, array) -> torch.Tensor:
        return torch.as_tensor(array, device=self.device)

    def _log1p(self, counts: torch.Tensor, size_factor: torch.Tensor) -> torch.Tensor:
        return torch.log1p(counts / size_factor.unsqueeze(1) * self.config.normalization_scale)

    def _sample_genes(self, counts: torch.Tensor, dataset: torch.Tensor, seed: int) -> torch.Tensor:
        """``seq_len`` distinct genes per cell, by sampling weight; -1 where a cell has too few."""
        config = self.config
        pool = self._sampling_pool[dataset]
        weight = pool.float()
        if config.sampling_mode == "expressed":
            weight += (pool & (counts > 0)).float() * config.expressed_weight
        elif config.sampling_mode == "hvg":
            weight += (pool & self._hvg[dataset]).float() * config.hvg_weight
        total = weight.sum(dim=1, keepdim=True)
        probs = weight / total.clamp_min(1.0)
        drawable = probs.clone()
        drawable[total[:, 0] == 0, 0] = 1.0  # a cell with no gene to draw gets only -1
        generator = torch.Generator(device=self.device).manual_seed(seed)
        genes = torch.multinomial(drawable, self.seq_len, replacement=False, generator=generator)
        return torch.where(probs.gather(1, genes) > 0, genes, -1)

    def _mask(self, values: torch.Tensor, seed: int) -> torch.Tensor:
        """Replace a ``mask_ratio`` share of the expression values (not pad or CLS) by ``mask_value``."""
        config = self.config
        maskable = (values != config.pad_value) & (values != config.cls_value)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        scores = torch.rand(values.shape, device=self.device, generator=generator)
        masked = values.clone()
        masked[(scores < config.mask_ratio) & maskable] = config.mask_value
        return masked
