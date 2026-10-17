#!/usr/bin/env python3
"""Run a bounded pertTF public-loader smoke on the Marson/Xorion aggregate corpus.

This script loads an existing ``scorpus`` corpus, verifies per-dataset
``hvg.parquet`` discovery, exercises the
public ``PertTFPairedBatchLoader`` path until it finds a bounded mixed-dataset
batch, and writes lightweight smoke summaries.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scorpus.loaders import load_corpus
from perttf.model.corpus_adapter import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    PertTFCorpusAdapter,
    default_pair_sampler,
)
from scorpus.materializers.paths import resolve_corpus_paths


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_now()}] {message}", flush=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected mapping YAML at {path}")
    return loaded


def _max_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


@dataclass(frozen=True)
class DatasetEntry:
    dataset_id: str
    dataset_index: int
    global_start: int
    global_end: int


@dataclass(frozen=True)
class ObservedLoaderBatch:
    batch: dict[str, torch.Tensor]
    batch_index: int
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_dataset_indices: np.ndarray
    target_dataset_indices: np.ndarray
    source_dataset_ids: tuple[str, ...]
    target_dataset_ids: tuple[str, ...]
    source_celltype_labels: tuple[str, ...]
    target_celltype_labels: tuple[str, ...]
    source_perturbation_labels: tuple[str, ...]
    target_perturbation_labels: tuple[str, ...]


def _load_dataset_entries(corpus_root: Path) -> list[DatasetEntry]:
    index_doc = _read_yaml(corpus_root / "corpus-index.yaml")
    entries: list[DatasetEntry] = []
    for item in index_doc.get("datasets", []):
        dataset_id = str(item["dataset_id"])
        entries.append(
            DatasetEntry(
                dataset_id=dataset_id,
                dataset_index=int(item["dataset_index"]),
                global_start=int(item["global_start"]),
                global_end=int(item["global_end"]),
            )
        )
    if not entries:
        raise ValueError(f"no datasets found in {corpus_root / 'corpus-index.yaml'}")
    return entries


def _dataset_ids_from_indices(corpus, dataset_indices: np.ndarray) -> tuple[str, ...]:
    dataset_ids = corpus.feature_registry.dataset_ids
    return tuple(str(dataset_ids[int(idx)]) for idx in dataset_indices.tolist())


def _observe_loader_batch(
    corpus,
    config: PertTFAdapterConfig,
    batch_index: int,
    batch: dict[str, torch.Tensor],
) -> ObservedLoaderBatch:
    source_indices = np.asarray(batch["index"].detach().cpu().tolist(), dtype=np.int64)
    target_indices = np.asarray(batch["next_index"].detach().cpu().tolist(), dtype=np.int64)
    label_columns = config.label_columns_by_name
    celltype_column = label_columns["celltype"]
    perturbation_column = label_columns[config.perturbation_label]
    metadata_columns = ["dataset_index", celltype_column, perturbation_column]
    source_meta = corpus.take_metadata(source_indices.tolist(), columns=metadata_columns)
    target_meta = corpus.take_metadata(target_indices.tolist(), columns=metadata_columns)
    source_dataset_indices = np.asarray(source_meta["dataset_index"], dtype=np.int32)
    target_dataset_indices = np.asarray(target_meta["dataset_index"], dtype=np.int32)
    return ObservedLoaderBatch(
        batch=batch,
        batch_index=batch_index,
        source_indices=source_indices,
        target_indices=target_indices,
        source_dataset_indices=source_dataset_indices,
        target_dataset_indices=target_dataset_indices,
        source_dataset_ids=_dataset_ids_from_indices(corpus, source_dataset_indices),
        target_dataset_ids=_dataset_ids_from_indices(corpus, target_dataset_indices),
        source_celltype_labels=tuple(
            str(value) for value in source_meta[celltype_column]
        ),
        target_celltype_labels=tuple(
            str(value) for value in target_meta[celltype_column]
        ),
        source_perturbation_labels=tuple(
            str(value) for value in source_meta[perturbation_column]
        ),
        target_perturbation_labels=tuple(
            str(value) for value in target_meta[perturbation_column]
        ),
    )


def _select_public_loader_batch(
    corpus,
    loader: PertTFPairedBatchLoader,
    *,
    requested_dataset_ids: tuple[str, ...],
    max_batches: int,
) -> ObservedLoaderBatch:
    requested = tuple(dict.fromkeys(requested_dataset_ids))
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        observed = _observe_loader_batch(corpus, loader.config, batch_index, batch)
        unique_dataset_ids = tuple(dict.fromkeys(observed.source_dataset_ids))
        _log(
            "Observed public loader batch "
            f"{batch_index}: source datasets {', '.join(unique_dataset_ids)}"
        )
        if len(set(unique_dataset_ids)) < 2:
            continue
        if requested and not set(requested).issubset(set(unique_dataset_ids)):
            continue
        return observed
    requested_note = (
        f" containing requested dataset ids {list(requested)}"
        if requested
        else ""
    )
    raise RuntimeError(
        f"failed to find a mixed-dataset public-loader batch{requested_note} "
        f"within {max_batches} batches"
    )


def _row_global_counts(raw_batch: dict[str, Any], row_pos: int, local_to_global: np.ndarray) -> dict[int, float]:
    row_offsets = np.asarray(raw_batch["row_offsets"], dtype=np.int64)
    start = int(row_offsets[row_pos])
    end = int(row_offsets[row_pos + 1])
    local_ids = np.asarray(raw_batch["expressed_gene_indices"][start:end], dtype=np.int64)
    counts = np.asarray(raw_batch["expression_counts"][start:end], dtype=np.float32)
    mapped = local_to_global[local_ids]
    return {
        int(global_id): float(count)
        for global_id, count in zip(mapped.tolist(), counts.tolist())
        if int(global_id) >= 0
    }


def _read_expression_raw_batch(corpus, indices: list[int]) -> dict[str, Any]:
    batch = corpus.expression_reader.read_expression_flat(indices)
    return {
        "batch_size": batch.batch_size,
        "global_row_index": batch.global_row_index,
        "row_offsets": batch.row_offsets,
        "expressed_gene_indices": batch.expressed_gene_indices,
        "expression_counts": batch.expression_counts,
    }


def _verify_sampled_alignment(
    corpus,
    adapter: PertTFCorpusAdapter,
    observed: ObservedLoaderBatch,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source_raw = _read_expression_raw_batch(corpus, observed.source_indices.tolist())
    target_raw = _read_expression_raw_batch(corpus, observed.target_indices.tolist())
    pad_token_id = adapter.to_simple_vocab_stoi()[adapter.config.pad_token]
    cls_offset = 1 if adapter.config.append_cls else 0
    token_offset = adapter.special_token_offset
    local_to_global = corpus.feature_registry.local_to_global_map
    checked_positions = 0
    mismatches: list[str] = []

    for row_pos in range(len(observed.source_indices)):
        source_counts = _row_global_counts(
            source_raw,
            row_pos,
            local_to_global[int(observed.source_dataset_indices[row_pos])],
        )
        target_counts = _row_global_counts(
            target_raw,
            row_pos,
            local_to_global[int(observed.target_dataset_indices[row_pos])],
        )
        for col in range(cls_offset, batch["gene_ids"].shape[1]):
            token_id = int(batch["gene_ids"][row_pos, col])
            if token_id == pad_token_id:
                continue
            global_id = token_id - token_offset
            expected_source = source_counts.get(global_id, 0.0)
            expected_target = target_counts.get(global_id, 0.0)
            actual_source = float(batch["target_values"][row_pos, col].item())
            actual_target = float(batch["target_values_next"][row_pos, col].item())
            checked_positions += 1
            if not np.isclose(actual_source, expected_source):
                mismatches.append(
                    f"source row={row_pos} col={col} global_id={global_id} expected={expected_source} actual={actual_source}"
                )
            if not np.isclose(actual_target, expected_target):
                mismatches.append(
                    f"target row={row_pos} col={col} global_id={global_id} expected={expected_target} actual={actual_target}"
                )

    return {
        "gene_ids_match": bool(torch.equal(batch["gene_ids"], batch["next_gene_ids"])),
        "checked_positions": checked_positions,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "passed": bool(torch.equal(batch["gene_ids"], batch["next_gene_ids"]) and not mismatches),
    }


def _verify_full_expression_masks(
    corpus,
    adapter: PertTFCorpusAdapter,
    observed: ObservedLoaderBatch,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    cls_offset = 1 if adapter.config.append_cls else 0
    source_expected = torch.as_tensor(
        corpus.feature_registry.dataset_has_gene[observed.source_dataset_indices],
        dtype=torch.bool,
    )
    target_expected = torch.as_tensor(
        corpus.feature_registry.dataset_has_gene[observed.target_dataset_indices],
        dtype=torch.bool,
    )
    source_mask = batch["full_expr_mask"][:, cls_offset:].cpu()
    target_mask = batch["full_expr_next_mask"][:, cls_offset:].cpu()
    source_expr = batch["full_expr"][:, cls_offset:].cpu()
    target_expr = batch["full_expr_next"][:, cls_offset:].cpu()
    source_present_zero = ((source_mask) & (source_expr == 0)).sum(dim=1).tolist()
    source_absent_zero = ((~source_mask) & (source_expr == 0)).sum(dim=1).tolist()
    target_present_zero = ((target_mask) & (target_expr == 0)).sum(dim=1).tolist()
    target_absent_zero = ((~target_mask) & (target_expr == 0)).sum(dim=1).tolist()

    return {
        "full_gene_ids_shape": list(batch["full_gene_ids"].shape),
        "full_expr_shape": list(batch["full_expr"].shape),
        "full_expr_next_shape": list(batch["full_expr_next"].shape),
        "source_mask_matches_dataset_presence": bool(torch.equal(source_mask, source_expected)),
        "target_mask_matches_dataset_presence": bool(torch.equal(target_mask, target_expected)),
        "source_present_zero_counts": [int(value) for value in source_present_zero],
        "source_absent_zero_counts": [int(value) for value in source_absent_zero],
        "target_present_zero_counts": [int(value) for value in target_present_zero],
        "target_absent_zero_counts": [int(value) for value in target_absent_zero],
        "passed": bool(
            torch.equal(source_mask, source_expected)
            and torch.equal(target_mask, target_expected)
            and any(int(value) > 0 for value in source_present_zero)
            and any(int(value) > 0 for value in source_absent_zero)
            and any(int(value) > 0 for value in target_present_zero)
            and any(int(value) > 0 for value in target_absent_zero)
        ),
    }


def _dataset_hvg_summary(corpus, entries: list[DatasetEntry]) -> list[dict[str, Any]]:
    feature_registry = corpus.feature_registry
    rank_matrix = feature_registry.hvg_rank_matrix
    has_gene = feature_registry.dataset_has_gene
    summary: list[dict[str, Any]] = []
    for entry in entries:
        hvg_path = resolve_corpus_paths(
            corpus.topology,
            corpus.corpus_root,
            entry.dataset_id,
        ).meta_root / "hvg.parquet"
        summary.append(
            {
                "dataset_id": entry.dataset_id,
                "dataset_index": entry.dataset_index,
                "hvg_ranking_path": str(hvg_path),
                "hvg_ranking_exists": hvg_path.exists(),
                "feature_count": int(has_gene[entry.dataset_index].sum()),
                "ranked_feature_count": int((rank_matrix[entry.dataset_index] > 0).sum()),
            }
        )
    return summary


def _write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    dataset_lines = "\n".join(
        f"- `{item['dataset_id']}`: ranked `{item['ranked_feature_count']}` / feature `{item['feature_count']}`, hvg path exists `{item['hvg_ranking_exists']}`"
        for item in summary["hvg_discovery"]
    )
    selected_lines = "\n".join(
        f"- `{item['dataset_id']}` row `{item['source_index']}` perturb `{item['source_perturbation_label']}`"
        for item in summary["batch"]["selected_sources"]
    )
    path.write_text(
        "\n".join(
            [
                "# Marson/Xorion pertTF Public Loader Smoke",
                "",
                f"- `Timestamp`: `{summary['timestamp']}`",
                f"- `Corpus Root`: `{summary['corpus_root']}`",
                f"- `Datasets Covered`: `{summary['dataset_count']}`",
                f"- `Global Vocab Size`: `{summary['global_vocab_size']}`",
                f"- `Batch Size`: `{summary['batch']['paired_rows_sampled']}`",
                f"- `HVG Top K`: `{summary['batch']['hvg_top_k']}`",
                f"- `Public Loader Batch Index`: `{summary['batch']['public_loader_batch_index']}`",
                f"- `Public Loader Workers`: `{summary['loader']['num_workers']}` ({summary['loader']['multiprocessing_context']})",
                f"- `Effective Label Rows`: `{summary['loader']['effective_label_row_count']}` / `{summary['loader']['total_row_count']}`",
                f"- `Pair Groups`: `{summary['loader']['pair_group_count']}`",
                f"- `Allowed Group Pairs`: `{summary['loader']['allowed_pair_count']}`",
                "",
                "## HVG Discovery",
                "",
                dataset_lines,
                "",
                "## Selected Source Rows",
                "",
                selected_lines,
                "",
                "## Checks",
                "",
                f"- Same-dataset check: `{summary['batch']['same_dataset_check']}` (required `{summary['batch']['same_dataset_check_required']}`)",
                f"- Same-celltype check: `{summary['batch']['same_celltype_check']}` (required `{summary['batch']['same_celltype_check_required']}`)",
                f"- Sampled source/target gene alignment: `{summary['alignment']['passed']}` ({summary['alignment']['checked_positions']} checked positions)",
                f"- Union full-expression masks: `{summary['full_expression_masks']['passed']}`",
                f"- Worker dataset stayed expression-only: `{summary['loader']['expression_only_state_passed']}`",
                "",
                "## Runtime Notes",
                "",
                f"- `load_corpus_seconds`: `{summary['timing_seconds']['load_corpus']:.2f}`",
                f"- `loader_setup_seconds`: `{summary['timing_seconds']['build_loader']:.2f}`",
                f"- `batch_search_seconds`: `{summary['timing_seconds']['find_batch']:.2f}`",
                f"- `max_rss_mib`: `{summary['runtime_notes']['max_rss_mib']:.1f}`",
                "",
                "## Limitations",
                "",
                "- This smoke imports the pertTF-owned adapter but does not modify either checkout.",
                "- The smoke validates the public `PertTFPairedBatchLoader` path on bounded batches, not full pertTF training or throughput.",
                "- `full_expr_mask` / `full_expr_next_mask` are emitted and validated here; the corpus training route consumes them in the pertTF loss path.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 7 Marson/Xorion pertTF public-loader smoke.",
    )
    parser.add_argument("--corpus-root", required=True, help="Aggregate corpus root.")
    parser.add_argument("--output-dir", required=True, help="Directory for smoke outputs.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sampled sequence length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of paired rows to validate.")
    parser.add_argument("--hvg-top-k", type=int, default=2000, help="Runtime HVG top-k threshold.")
    parser.add_argument("--hvg-weight", type=float, default=4.0, help="Runtime HVG bonus weight.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic RNG seed.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Public-loader DataLoader worker count.",
    )
    parser.add_argument(
        "--multiprocessing-context",
        default="spawn",
        help="Optional multiprocessing context for the public loader.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=32,
        help="Maximum public-loader batches to scan before failing to find a mixed batch.",
    )
    parser.add_argument(
        "--control-label",
        action="append",
        default=["WT"],
        help="Control perturbation label to reserve in the adapter (repeatable).",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Optional dataset IDs that must appear in the selected mixed batch (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.monotonic()
    corpus_root = Path(args.corpus_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command_snapshot = {
        "timestamp": _utc_now(),
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "hvg_top_k": int(args.hvg_top_k),
        "hvg_weight": float(args.hvg_weight),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "multiprocessing_context": (
            None if args.multiprocessing_context in ("", "none", "None") else str(args.multiprocessing_context)
        ),
        "max_batches": int(args.max_batches),
        "control_labels": list(dict.fromkeys(str(value) for value in args.control_label)),
        "requested_dataset_ids": list(dict.fromkeys(str(value) for value in args.dataset_id)),
    }
    (output_dir / "phase-07-smoke-command.json").write_text(
        json.dumps(command_snapshot, indent=2),
        encoding="utf-8",
    )

    _log(f"Loading corpus from {corpus_root}")
    load_start = time.monotonic()
    corpus = load_corpus(corpus_root)
    load_seconds = time.monotonic() - load_start
    _log(
        "Loaded corpus with "
        f"{len(corpus.feature_registry.dataset_ids)} datasets, "
        f"global vocab {corpus.feature_registry.global_vocab_size}, "
        f"max RSS {_max_rss_mib():.1f} MiB"
    )

    entries = _load_dataset_entries(corpus_root)
    hvg_summary = _dataset_hvg_summary(corpus, entries)
    missing_hvg = [item["dataset_id"] for item in hvg_summary if not item["hvg_ranking_exists"]]
    incomplete_hvg = [
        item["dataset_id"]
        for item in hvg_summary
        if item["ranked_feature_count"] != item["feature_count"]
    ]
    if missing_hvg:
        raise RuntimeError(f"missing hvg.parquet outputs for datasets: {missing_hvg}")
    if incomplete_hvg:
        raise RuntimeError(
            "HVG rank discovery did not cover every feature for datasets: "
            f"{incomplete_hvg}"
        )
    _log("Verified hvg.parquet discovery for all datasets")

    loader_start = time.monotonic()
    config = PertTFAdapterConfig(
        control_labels=tuple(dict.fromkeys(str(value) for value in args.control_label)),
        include_full_expr=True,
        label_fields={
            "perturb_label": "perturbation",
            "cell_context": "celltype",
            "batch_id": "batch",
            "dataset_id": "dataset",
        },
        mask_ratio=0.0,
    )
    # pertTF's default training pairs within the same dataset and cell context.
    pairing_context_columns = ("dataset_index", "cell_context")
    sampler = default_pair_sampler(
        corpus.metadata_index.df,
        context_columns=pairing_context_columns,
        perturbation_column="perturb_label",
        control_labels=config.control_labels,
        perturbed_to_self=True,
    )
    multiprocessing_context = command_snapshot["multiprocessing_context"]
    loader = PertTFPairedBatchLoader(
        corpus,
        sampler=sampler,
        rows=np.arange(len(corpus.metadata_index), dtype=np.int64),
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        config=config,
        seed=int(args.seed),
        sampling_mode="hvg",
        hvg_weight=float(args.hvg_weight),
        hvg_top_k=int(args.hvg_top_k),
        num_workers=int(args.num_workers),
        multiprocessing_context=multiprocessing_context,
    )
    if not loader.effective_label_row_indices.size:
        raise RuntimeError("public pertTF loader resolved no usable rows")
    loader_seconds = time.monotonic() - loader_start
    _log(
        "Built public pertTF loader in "
        f"{loader_seconds:.2f}s with "
        f"{len(loader.effective_label_row_indices)} effective label rows, "
        f"{len(sampler.groups.rows)} pair groups, "
        f"{sampler.n_bundles} allowed group pairs"
    )

    dataset_state = dict(loader._data_loader.dataset.__dict__)
    expression_only_state_passed = (
        set(dataset_state) == {"_reader", "_total_rows"}
        and all(value is not corpus.metadata_index for value in dataset_state.values())
        and not any(isinstance(value, PertTFCorpusAdapter) for value in dataset_state.values())
    )
    requested_dataset_ids = tuple(dict.fromkeys(str(value) for value in args.dataset_id))
    batch_start = time.monotonic()
    observed = _select_public_loader_batch(
        corpus,
        loader,
        requested_dataset_ids=requested_dataset_ids,
        max_batches=int(args.max_batches),
    )
    batch_seconds = time.monotonic() - batch_start
    batch = observed.batch
    _log(
        "Selected public loader batch "
        f"{observed.batch_index} with source datasets "
        f"{', '.join(dict.fromkeys(observed.source_dataset_ids))}"
    )

    same_dataset = bool(
        np.array_equal(observed.source_dataset_indices, observed.target_dataset_indices)
    )
    same_celltype = bool(
        np.array_equal(
            np.asarray(observed.source_celltype_labels, dtype=object),
            np.asarray(observed.target_celltype_labels, dtype=object),
        )
    )
    mixed_dataset = bool(len(set(observed.source_dataset_ids)) > 1)
    require_same_dataset = "dataset_index" in pairing_context_columns
    require_same_celltype = "cell_context" in pairing_context_columns
    alignment = _verify_sampled_alignment(corpus, loader.adapter, observed, batch)
    full_expr_masks = _verify_full_expression_masks(corpus, loader.adapter, observed, batch)

    summary = {
        "timestamp": _utc_now(),
        "corpus_root": str(corpus_root),
        "dataset_count": len(entries),
        "global_vocab_size": int(corpus.feature_registry.global_vocab_size),
        "hvg_discovery": hvg_summary,
        "loader": {
            "num_workers": int(args.num_workers),
            "multiprocessing_context": (
                str(multiprocessing_context) if multiprocessing_context is not None else "none"
            ),
            "expression_only_state_passed": bool(expression_only_state_passed),
            "dataset_state_keys": sorted(dataset_state.keys()),
            "pairing_context_columns": list(pairing_context_columns),
            "total_row_count": int(len(corpus.metadata_index)),
            "effective_label_row_count": int(len(loader.effective_label_row_indices)),
            "pair_group_count": int(len(sampler.groups.rows)),
            "allowed_pair_count": int(sampler.n_bundles),
        },
        "batch": {
            "hvg_top_k": int(args.hvg_top_k),
            "hvg_weight": float(args.hvg_weight),
            "seq_len": int(args.seq_len),
            "paired_rows_sampled": int(len(observed.source_indices)),
            "mixed_dataset_batch": mixed_dataset,
            "public_loader_batch_index": int(observed.batch_index),
            "requested_dataset_ids": list(requested_dataset_ids),
            "requested_dataset_ids_satisfied": bool(
                set(requested_dataset_ids).issubset(set(observed.source_dataset_ids))
            ),
            "same_dataset_check": same_dataset,
            "same_dataset_check_required": require_same_dataset,
            "same_celltype_check": same_celltype,
            "same_celltype_check_required": require_same_celltype,
            "selected_sources": [
                {
                    "dataset_id": dataset_id,
                    "source_index": int(source_index),
                    "source_perturbation_label": perturbation_label,
                }
                for dataset_id, source_index, perturbation_label in zip(
                    observed.source_dataset_ids,
                    observed.source_indices.tolist(),
                    observed.source_perturbation_labels,
                )
            ],
            "source_indices": [int(value) for value in observed.source_indices.tolist()],
            "target_indices": [int(value) for value in observed.target_indices.tolist()],
            "source_dataset_ids": list(observed.source_dataset_ids),
            "target_dataset_ids": list(observed.target_dataset_ids),
            "source_celltype_labels": list(observed.source_celltype_labels),
            "target_celltype_labels": list(observed.target_celltype_labels),
            "source_perturbation_labels": list(observed.source_perturbation_labels),
            "target_perturbation_labels": list(observed.target_perturbation_labels),
            "batch_field_shapes": {
                key: list(value.shape)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            },
        },
        "alignment": alignment,
        "full_expression_masks": full_expr_masks,
        "runtime_notes": {
            "max_rss_mib": _max_rss_mib(),
            "device": str(loader._builder.device),
            "control_labels": list(config.control_labels),
        },
        "timing_seconds": {
            "load_corpus": load_seconds,
            "build_loader": loader_seconds,
            "find_batch": batch_seconds,
            "total": time.monotonic() - start_time,
        },
    }

    summary_path = output_dir / "phase-07-smoke-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_summary(output_dir / "phase-07-smoke-summary.md", summary)
    _log(f"Wrote smoke summaries to {summary_path}")

    if require_same_dataset and not same_dataset:
        raise RuntimeError("same-dataset pairing invariant failed")
    if require_same_celltype and not same_celltype:
        raise RuntimeError("same-celltype pairing invariant failed")
    if not mixed_dataset:
        raise RuntimeError("mixed-dataset smoke batch was not achieved")
    if not alignment["passed"]:
        raise RuntimeError("sampled source/target gene alignment validation failed")
    if not full_expr_masks["passed"]:
        raise RuntimeError("full-expression mask validation failed")

    _log("Smoke completed successfully")


if __name__ == "__main__":
    main()
