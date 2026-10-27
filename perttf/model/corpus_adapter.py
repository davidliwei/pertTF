"""pertTF batches from on-disk scorpus corpora.

Which cells are paired is described with ``scorpus.pairing``. This module turns the pairs into
pertTF tensors: gene tokens, sampled genes, log1p expression, masking, CLS, full expression and
labels. A batch is built either in the main process on the loader's device, or on the CPU in the
DataLoader workers (``build_in_workers``); both draw genes with the quotas of pertTF's collator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from scorpus.loaders import ExpressionBatch, FederatedLanceReader, LanceDatasetEntry
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
    # Genes of a cell are drawn from its pool (the genes its dataset measured that have a token),
    # with the quotas of pertTF's collator (custom_tokenizer.weighted_sample), uniformly within each
    # group; a cell whose pool fits in seq_len gets its whole pool:
    #   "uniform":   seq_len genes.
    #   "expressed": round(nonzero_prop * seq_len) of the cell's expressed genes (all if fewer), the
    #                rest of seq_len from its unexpressed genes; with fix_nonzero_prop at most
    #                round(k * (1 - p) / p) of those, for k expressed genes drawn.
    #   "hvg":       non_hvg_size of the dataset's other genes, and its HVGs in the remaining
    #                seq_len - non_hvg_size places (a uniform subset if it has more). HVGs are the
    #                dataset's default ones, or its hvg_top_k highest ranked. pertTF's config sets
    #                max_seq_len to n_hvg + non_hvg_size + 1 for this mode.
    sampling_mode: str = "uniform"
    nonzero_prop: float = 0.7
    fix_nonzero_prop: bool = False
    non_hvg_size: int = 1000
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


def _worker_reader(corpus: Corpus):
    """The reader given to worker processes. A concatenated corpus's own reader holds every member
    corpus, obs included, which each spawned worker would unpickle; this one holds table paths and
    row ranges only."""
    if not corpus.members:
        return corpus.expression_reader
    entries = []
    for entry in corpus.dataset_entries:
        table = corpus.members[entry.dataset_id].dataset_entries[0]
        entries.append(LanceDatasetEntry(entry.dataset_id, entry.global_start, entry.global_end, table.lance_path,
                                         table.expression_layout, table.counts_dtype))
    return FederatedLanceReader(entries)


def _csr_rows(csr, positions: np.ndarray):
    """Rows ``positions`` of a CSR (offsets, genes, counts), as a new CSR."""
    offsets, genes, counts = csr
    starts = offsets[positions]
    lengths = offsets[positions + 1] - starts
    new_offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = np.repeat(starts - new_offsets[:-1], lengths) + np.arange(new_offsets[-1])
    return new_offsets, genes[flat], counts[flat]


def _read_batches(reader, items: list[dict], builder) -> list[tuple]:
    """One read for the cells of ``items`` (batches; the reader reads each distinct cell once and
    returns the rows in the given order); then per batch (index, source expression, target
    expression), or (index, the batch built on the CPU by ``builder``)."""
    rows = np.concatenate([np.concatenate([item["source_rows"], item["target_rows"]]) for item in items])
    read = reader.read_expression_flat(rows)
    csr = (read.row_offsets, read.expressed_gene_indices, read.expression_counts)
    out, start = [], 0
    for item in items:
        n = len(item["source_rows"])
        source, target = np.arange(start, start + n), np.arange(start + n, start + 2 * n)
        start += 2 * n
        if builder is None:
            out.append((item["index"], ExpressionBatch(n, item["source_rows"], *_csr_rows(csr, source)),
                        ExpressionBatch(n, item["target_rows"], *_csr_rows(csr, target))))
        else:
            cells = _csr_rows(csr, np.concatenate([source, target]))
            out.append((item["index"], builder.build(item, cells, np.arange(n), np.arange(n, 2 * n))))
    return out


class _PairReads:
    """DataLoader dataset, in worker processes: one read for several consecutive batches."""

    def __init__(self, reader, n_cells: int, builder=None) -> None:
        self._reader, self._n_cells, self._builder = reader, n_cells, builder

    def __len__(self) -> int:
        return self._n_cells

    def __getitem__(self, items: list[dict]) -> list[tuple]:
        return _read_batches(self._reader, items, self._builder)


class _EpochReads:
    """DataLoader sampler: the loader's current epoch, ``batches_per_read`` consecutive batches per item."""

    def __init__(self, loader: "PertTFPairedBatchLoader", batches_per_read: int) -> None:
        self._loader, self._batches_per_read = loader, batches_per_read

    def __iter__(self):
        loader = self._loader
        batches = loader._epoch_batches()
        for first in range(0, len(batches), self._batches_per_read):
            yield [loader._item(i, *batches[i], loader._batch_seed(i))
                   for i in range(first, min(first + self._batches_per_read, len(batches)))]

    def __len__(self) -> int:
        return -(-len(self._loader._epoch_batches()) // self._batches_per_read)


def _unchanged(item):
    return item


def _with_cls(values: torch.Tensor, cls_value) -> torch.Tensor:
    cls_column = torch.full((len(values), 1), cls_value, dtype=values.dtype, device=values.device)
    return torch.cat([cls_column, values], dim=1)


def _gene_quotas(config: PertTFAdapterConfig, n_pool: torch.Tensor, n_group: torch.Tensor, seq_len: int):
    """Genes per cell from its group (its whole pool, its expressed genes or its HVGs) and from the rest
    of its pool, as in pertTF's custom_tokenizer.weighted_sample."""
    n_rest = n_pool - n_group
    if config.sampling_mode == "uniform":
        k_group, k_rest = n_group.clamp(max=seq_len), torch.zeros_like(n_group)
    elif config.sampling_mode == "expressed":
        p = config.nonzero_prop
        k_group = n_group.clamp(max=round(seq_len * p))
        k_rest = seq_len - k_group
        if config.fix_nonzero_prop:
            k_rest = torch.minimum(k_rest, torch.round(k_group.double() * (1 - p) / p).long())
        # pertTF's rng.choice raises when a cell has too few unexpressed genes; here it gets all of them.
        k_rest = torch.minimum(k_rest, n_rest)
    else:
        k_rest = n_rest.clamp(max=config.non_hvg_size)
        k_group = torch.minimum(n_group, seq_len - k_rest)
    whole = n_pool <= seq_len  # nothing to sample: pertTF keeps every gene
    return torch.where(whole, n_group, k_group), torch.where(whole, n_rest, k_rest)


def _smallest_keys(keys: torch.Tensor, members: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Per row, the ``k[i]`` member columns with the smallest keys, then -1. With uniform random keys
    every k-subset of the members is equally likely."""
    width = int(k.max())
    columns = torch.topk(torch.where(members, keys, 2.0), width, dim=1, largest=False).indices
    return torch.where(torch.arange(width, device=keys.device) < k.unsqueeze(1), columns, -1)


def _uniform_subsets(n: np.ndarray, k: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per row i, ``k[i]`` distinct values of range(n[i]), uniformly, in random order; then -1."""
    width = int(k.max())
    out = np.full((len(n), width), -1, dtype=np.int64)
    small = (k > 0) & (n < 4 * k)
    for i in np.flatnonzero(small):
        out[i, :k[i]] = rng.permutation(n[i])[:k[i]]
    big = np.flatnonzero((k > 0) & ~small)
    if len(big):
        # Draw with replacement, keep each distinct value once, then a random k of those: by
        # symmetry every k-subset of range(n) is equally likely.
        n_draws = width + width // 4 + 64
        draws = np.sort((rng.random((len(big), n_draws)) * n[big, None]).astype(np.int64), axis=1)
        keys = rng.random(draws.shape)
        keys[:, 1:][draws[:, 1:] == draws[:, :-1]] = 2.0
        pick = np.argsort(keys, axis=1)[:, :width]
        wanted = np.arange(width) < k[big, None]
        # Fewer distinct draws than k would silently repeat genes.
        assert (np.take_along_axis(keys, pick, axis=1)[wanted] < 2.0).all(), "too few distinct draws; raise n_draws"
        out[big] = np.where(wanted, np.take_along_axis(draws, pick, axis=1), -1)
    return out


def _padded_lists(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row of ``mask`` its True columns, padded with -1; and their number."""
    counts = mask.sum(axis=1)
    lists = np.full((len(mask), max(1, int(counts.max()))), -1, dtype=np.int64)
    for row in range(len(mask)):
        lists[row, :counts[row]] = np.flatnonzero(mask[row])
    return lists, counts


class _CpuBatchBuilder:
    """A batch built with numpy in a DataLoader worker: the loader's tensors, except that the full gene
    axis stays sparse (global gene IDs and log1p values of every stored count) for ``_finish``."""

    def __init__(self, local_to_global, pool, hvg, token_of_gene, pad_id, cls_id, config, seq_len) -> None:
        n_datasets, n_genes = pool.shape
        self.local_to_global = local_to_global.astype(np.int64)
        self.global_to_local = np.full((n_datasets, n_genes), -1, dtype=np.int64)
        for d in range(n_datasets):
            local = np.flatnonzero(self.local_to_global[d] >= 0)
            self.global_to_local[d, self.local_to_global[d, local]] = local
        self.n_pool = pool.sum(axis=1)
        if config.sampling_mode == "expressed":
            # The pool in dataset-local gene order, so a cell's stored genes have sorted pool positions.
            self.local_pool = np.full((n_datasets, max(1, int(self.n_pool.max()))), -1, dtype=np.int64)
            self.pool_position = np.full(local_to_global.shape, -1, dtype=np.int64)
            for d in range(n_datasets):
                local = np.flatnonzero(self.local_to_global[d] >= 0)
                in_pool = local[pool[d, self.local_to_global[d, local]]]
                self.local_pool[d, :len(in_pool)] = self.local_to_global[d, in_pool]
                self.pool_position[d, in_pool] = np.arange(len(in_pool))
        else:
            group = pool if config.sampling_mode == "uniform" else pool & hvg
            self.group_lists, self.n_group = _padded_lists(group)
            self.rest_lists, self.n_rest = _padded_lists(pool & ~group)
        self.n_local = local_to_global.shape[1]
        self.token_of_gene, self.pad_id, self.cls_id = token_of_gene, pad_id, cls_id
        self.config, self.seq_len = config, seq_len

    def build(self, item: dict, csr, source: np.ndarray, target: np.ndarray) -> dict[str, torch.Tensor]:
        """``source`` / ``target``: positions of each pair's cells among the rows of ``csr``."""
        config = self.config
        rng = np.random.default_rng(item["seed"])
        dataset, dataset_next = item["dataset"], item["dataset_next"]
        size_factor, size_factor_next = item["size_factor"][:, None], item["size_factor_next"][:, None]

        genes = self._sample_genes(dataset, csr, source, rng)
        valid = genes >= 0
        safe = genes.clip(min=0)
        local = np.where(valid, self.global_to_local[dataset[:, None], safe], -1)
        # The target is read at the same genes, where its dataset measured them.
        local_next = np.where(valid, self.global_to_local[dataset_next[:, None], safe], -1)
        values = np.where(valid, self._log1p(self._lookup(csr, source, local), size_factor), config.pad_value)
        values_next = np.where(local_next >= 0, self._log1p(self._lookup(csr, target, local_next), size_factor_next),
                               config.pad_value)
        gene_ids = _cls_first(np.where(valid, self.token_of_gene[safe], self.pad_id), self.cls_id)
        values = _cls_first(values.astype(np.float32), config.cls_value)

        out = {
            "gene_ids": gene_ids, "values": self._mask(values, rng), "target_values": values,
            "target_values_next": _cls_first(values_next.astype(np.float32), config.cls_value),
            "dataset": dataset, "dataset_next": dataset_next,
            "size_factor": item["size_factor"], "size_factor_next": item["size_factor_next"],
            "index": item["source_rows"], "next_index": item["target_rows"],
        }
        for side, rows, cell_dataset, factor in (("full", source, dataset, size_factor),
                                                 ("full_next", target, dataset_next, size_factor_next)):
            offsets, row_genes, row_counts = _csr_rows(csr, rows)
            cell = np.repeat(np.arange(len(rows)), np.diff(offsets))
            out[f"{side}_offsets"] = offsets
            out[f"{side}_genes"] = self.local_to_global[cell_dataset[cell], row_genes]
            out[f"{side}_values"] = self._log1p(row_counts, factor[cell, 0])
        for name in item["labels"]:
            out[f"{name}_labels"] = item["labels"][name]
            out[f"{name}_labels_next"] = item["labels_next"][name]
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in out.items()}

    def _log1p(self, counts, size_factor):
        return np.log1p(counts.astype(np.float32) / size_factor * np.float32(self.config.normalization_scale))

    def _quotas(self, n_pool, n_group):
        quotas = _gene_quotas(self.config, torch.from_numpy(n_pool), torch.from_numpy(n_group), self.seq_len)
        return tuple(quota.numpy() for quota in quotas)

    def _sample_genes(self, dataset, csr, source, rng) -> np.ndarray:
        """Global gene IDs per source cell, with -1 at the end of rows shorter than the longest."""
        if self.config.sampling_mode == "expressed":
            genes = self._sample_expressed(dataset, csr, source, rng)
        else:
            k_group, k_rest = self._quotas(self.n_pool[dataset], self.n_group[dataset])
            parts = []
            for lists, n, k in ((self.group_lists, self.n_group[dataset], k_group),
                                (self.rest_lists, self.n_rest[dataset], k_rest)):
                positions = _uniform_subsets(n, k, rng)
                parts.append(np.where(positions >= 0, lists[dataset[:, None], positions.clip(min=0)], -1))
            genes = np.concatenate(parts, axis=1)
        genes = np.take_along_axis(genes, np.argsort(genes < 0, axis=1, kind="stable"), axis=1)
        return genes[:, :(genes >= 0).sum(axis=1).max()]

    def _sample_expressed(self, dataset, csr, source, rng) -> np.ndarray:
        """Expressed genes are the cell's stored genes with a count above 0 that are in its pool."""
        n_cells = len(dataset)
        offsets, genes, counts = _csr_rows(csr, source)
        cell = np.repeat(np.arange(n_cells), np.diff(offsets))
        position = self.pool_position[dataset[cell], genes]
        keep = (position >= 0) & (counts > 0)
        cell, position = cell[keep], position[keep]
        stride = int(self.n_pool.max()) + 1
        key = cell * stride + position
        if not (np.diff(key) > 0).all():  # genes within a row not sorted in the table
            key = np.sort(key)
            cell, position = key // stride, key % stride
        n_expressed = np.bincount(cell, minlength=n_cells)
        start = np.concatenate([[0], np.cumsum(n_expressed)[:-1]])
        n_pool = self.n_pool[dataset]
        k_expressed, k_unexpressed = self._quotas(n_pool, n_expressed)

        picked = _uniform_subsets(n_expressed, k_expressed, rng)
        expressed = np.append(position, -1)[np.where(picked >= 0, start[:, None] + picked, len(position))]
        # The j-th unexpressed pool position is j + #{i: e_i - i <= j}, for the cell's sorted expressed
        # positions e; so a uniform k-subset of range(n_unexpressed) is one of the unexpressed genes.
        j = _uniform_subsets(n_pool - n_expressed, k_unexpressed, rng)
        shifted = cell * stride + position - (np.arange(len(position)) - start[cell])
        below = np.searchsorted(shifted, np.arange(n_cells)[:, None] * stride + j, side="right") - start[:, None]
        positions = np.concatenate([expressed, np.where(j >= 0, j + below, -1)], axis=1)
        return np.where(positions >= 0, self.local_pool[dataset[:, None], positions.clip(min=0)], -1)

    def _lookup(self, csr, rows, local_genes):
        """Counts of CSR rows ``rows`` at dataset-local genes (B x L); 0 where not stored or the gene is -1."""
        offsets, genes, counts = csr
        key = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets)) * self.n_local + genes
        if not (np.diff(key) > 0).all():  # genes within a row not sorted in the table
            order = np.argsort(key, kind="stable")
            key, counts = key[order], counts[order]
        query = rows[:, None] * self.n_local + local_genes.clip(min=0)
        at = np.searchsorted(key, query).clip(max=len(key) - 1)
        found = (key[at] == query) & (local_genes >= 0)
        return np.where(found, counts[at], 0)

    def _mask(self, values, rng):
        """Exactly int(n * mask_ratio) of each row's values (not pad or CLS) set to mask_value."""
        config = self.config
        maskable = (values != config.pad_value) & (values != config.cls_value)
        n_mask = (maskable.sum(axis=1) * config.mask_ratio).astype(np.int64)
        order = np.argsort(np.where(maskable, rng.random(values.shape), 2.0), axis=1)
        chosen = np.zeros(values.shape, dtype=bool)
        np.put_along_axis(chosen, order, np.arange(values.shape[1]) < n_mask[:, None], axis=1)
        masked = values.copy()
        masked[chosen] = config.mask_value
        return masked


def _cls_first(values: np.ndarray, cls_value) -> np.ndarray:
    return np.concatenate([np.full((len(values), 1), cls_value, dtype=values.dtype), values], axis=1)


class PertTFPairedBatchLoader:
    """pertTF batches of (source, target) cell pairs.

    ``sampler`` (``scorpus.pairing``) decides which cells are paired. Its rows are positions in
    ``rows``, the corpus rows of the metadata table it was built from (see
    ``default_pair_sampler``). Each epoch uses every cell of the covered side once
    (``cover="source"`` for training, ``"target"`` for validation) and cuts the pairs into
    batches of ``batch_size``. Worker processes read expression, ``batches_per_read`` consecutive
    batches at a time (each distinct cell once; small batches are otherwise limited by the cost of
    each read). The tensors are built in the main process on ``device``, or with
    ``build_in_workers`` on the CPU in the workers, leaving only the full gene axis to be filled on
    ``device``. Both draw genes with pertTF's quotas (see ``PertTFAdapterConfig``); for the same
    seed they draw different genes.

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
        build_in_workers: bool = False,
        batches_per_read: int = 1,
    ) -> None:
        if config.sampling_mode not in SAMPLING_MODES:
            raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, got {config.sampling_mode!r}")
        # pertTF's weighted_sample has the same limit; its config sets seq_len to n_hvg + non_hvg_size.
        if config.sampling_mode == "hvg" and config.non_hvg_size > seq_len:
            raise ValueError(f"non_hvg_size ({config.non_hvg_size}) is larger than seq_len ({seq_len})")
        self.corpus, self.sampler, self.config, self.tokens = corpus, sampler, config, tokens
        self.rows = np.asarray(rows, dtype=np.int64)
        self.seq_len, self.batch_size, self.cover = seq_len, batch_size, cover
        self.seed, self.drop_last = seed, drop_last
        self.device = torch.device(device)
        self.epoch = 0
        self._batches = None

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
        pool = registry.dataset_has_gene & tokens.tokenizable
        self._sampling_pool = on_device(pool)
        if config.hvg_top_k is None:
            hvg = registry.hvg_mask
        else:
            ranks = registry.hvg_rank_matrix
            hvg = (ranks > 0) & (ranks <= config.hvg_top_k)
        self._hvg = on_device(hvg)
        self._token_of_gene = on_device(tokens.token_of_gene)

        self._builder = _CpuBatchBuilder(
            registry.local_to_global_map, pool, hvg, tokens.token_of_gene, tokens.pad_id, tokens.cls_id, config, seq_len,
        ) if build_in_workers else None
        self.dataset = _PairReads(_worker_reader(corpus), len(self.rows), self._builder)
        # One DataLoader for all epochs, so persistent workers are reused; its sampler follows set_epoch.
        self._data_loader = DataLoader(
            self.dataset, sampler=_EpochReads(self, batches_per_read), batch_size=None, collate_fn=_unchanged,
            num_workers=num_workers, multiprocessing_context=multiprocessing_context,
            persistent_workers=persistent_workers,
        )

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
        for read in self._data_loader:
            for i, *expression in read:
                if self._builder is None:
                    yield self._build(*batches[i], *expression, self._batch_seed(i))
                else:
                    yield self._finish(*expression)

    def build_batch(self, source: np.ndarray, target: np.ndarray, seed: int) -> dict[str, torch.Tensor]:
        """One batch for the cells at positions ``source`` and ``target`` of ``rows``."""
        ((_, *expression),) = _read_batches(
            self.corpus.expression_reader, [self._item(-1, source, target, seed)], self._builder,
        )
        return self._build(source, target, *expression, seed) if self._builder is None else self._finish(*expression)

    def _batch_seed(self, batch_index: int) -> int:
        return int(np.random.SeedSequence([self.seed, self.epoch, batch_index, 1]).generate_state(1, dtype=np.uint64)[0])

    def _item(self, index: int, source: np.ndarray, target: np.ndarray, seed: int) -> dict:
        """What a worker needs for one batch: its cells' corpus rows and, to build it, their metadata."""
        return {
            "index": index, "seed": seed, "source_rows": self.rows[source], "target_rows": self.rows[target],
            "dataset": self._dataset_index[source], "dataset_next": self._dataset_index[target],
            "size_factor": self._size_factor[source], "size_factor_next": self._size_factor[target],
            "labels": {name: ids[source] for name, ids in self._label_ids.items()},
            "labels_next": {name: ids[target] for name, ids in self._label_ids.items()},
        }

    def _finish(self, built: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """A batch built in a worker, moved to the device, with its full gene axis filled there."""
        config = self.config
        cpu = {key: value.to(self.device) for key, value in built.items()}
        n_cells = len(cpu["gene_ids"])
        batch = {key: cpu[key] for key in ("gene_ids", "values", "target_values", "target_values_next", "index", "next_index")}
        batch["next_gene_ids"] = cpu["gene_ids"].clone()
        batch["sf"] = cpu["size_factor"].reshape(-1, 1).reciprocal() * config.normalization_scale
        batch["sf_next"] = cpu["size_factor_next"].reshape(-1, 1).reciprocal() * config.normalization_scale
        batch["ps"] = torch.zeros((n_cells, 1), device=self.device)
        batch["ps_next"] = torch.zeros((n_cells, 1), device=self.device)
        batch["full_gene_ids"] = _with_cls(self._token_of_gene.unsqueeze(0), self.tokens.cls_id).expand(n_cells, -1).clone()
        for side, dataset_key, key in (("full", "dataset", "full_expr"), ("full_next", "dataset_next", "full_expr_next")):
            cell = torch.repeat_interleave(torch.arange(n_cells, device=self.device), cpu[f"{side}_offsets"].diff())
            dense = torch.zeros((n_cells, self._n_genes), device=self.device)
            dense[cell, cpu[f"{side}_genes"]] = cpu[f"{side}_values"]
            batch[key] = _with_cls(dense, config.cls_value)
            batch[f"{key}_mask"] = _with_cls(self._has_gene[cpu[dataset_key]] & self._tokenizable, True)
        for name in self._label_ids:
            batch[f"{name}_labels"] = cpu[f"{name}_labels"].long()
            batch[f"{name}_labels_next"] = cpu[f"{name}_labels_next"].long()
        return batch

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
        """Genes per cell with pertTF's quotas; -1 at the end of rows shorter than the longest."""
        mode = self.config.sampling_mode
        pool = self._sampling_pool[dataset]
        group = pool if mode == "uniform" else pool & (counts > 0) if mode == "expressed" else pool & self._hvg[dataset]
        k_group, k_rest = _gene_quotas(self.config, pool.sum(dim=1), group.sum(dim=1), self.seq_len)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        keys = torch.rand(pool.shape, device=self.device, generator=generator)
        genes = torch.cat([_smallest_keys(keys, group, k_group), _smallest_keys(keys, pool & ~group, k_rest)], dim=1)
        genes = genes.gather(1, torch.sort((genes < 0).to(torch.int8), dim=1, stable=True).indices)
        return genes[:, :int((genes >= 0).sum(dim=1).max())]

    def _mask(self, values: torch.Tensor, seed: int) -> torch.Tensor:
        """Replace exactly int(n * mask_ratio) of each row's values (not pad or CLS) by ``mask_value``,
        as pertTF's random_mask_value does."""
        config = self.config
        maskable = (values != config.pad_value) & (values != config.cls_value)
        n_mask = (maskable.sum(dim=1).double() * config.mask_ratio).long()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        keys = torch.where(maskable, torch.rand(values.shape, device=self.device, generator=generator), 2.0)
        chosen = torch.zeros_like(maskable)
        chosen.scatter_(1, keys.argsort(dim=1), torch.arange(values.shape[1], device=self.device) < n_mask.unsqueeze(1))
        masked = values.clone()
        masked[chosen] = config.mask_value
        return masked
