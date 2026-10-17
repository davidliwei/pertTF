"""Prepare pertTF training and validation directly from sparse on-disk corpora."""

from __future__ import annotations

from dataclasses import asdict

import anndata as ad
import numpy as np

from scorpus.loaders import dense_counts

from .corpus_adapter import GeneTokens, PertTFAdapterConfig, PertTFPairedBatchLoader, default_pair_sampler
from .pert_emb import build_perturbation_mapping
from ..utils.custom_tokenizer import SimpleVocab
from ..utils.split_check import check_train_valid_split
from ..utils.pert_metrics import GroupMoments, normalize_expression


def corpus_reference_groups(corpus, rows, *, context_column, perturbation_column, gene_mask, target_sum=10000.0, chunk_rows=1024):
    """Stream reference moments over the same common gene axis used in evaluation."""
    selected_genes = np.flatnonzero(gene_mask)
    moments = GroupMoments()
    rows = np.asarray(rows, dtype=np.int64)
    for start in range(0, len(rows), chunk_rows):
        selected = rows[start:start + chunk_rows]
        batch = corpus.expression_reader.read_expression_flat(selected.tolist())
        metadata = corpus.take_metadata(selected, columns=[context_column, perturbation_column])
        counts, _ = dense_counts(corpus, batch, genes=selected_genes)
        expression = counts.astype(np.float64)
        moments.update(
            normalize_expression(expression, target_sum=target_sum, input_scale="counts"),
            metadata[context_column], metadata[perturbation_column],
        )
    return moments.finalize()


def produce_corpus_datasets(
    corpus,
    config,
    *,
    train_indices,
    valid_indices,
    perturbation_column: str,
    celltype_column: str,
    batch_column: str,
    control_label: str,
    pert_sources: dict | None = None,
    genotype_to_index: dict | None = None,
    cell_type_to_index: dict | None = None,
    batch_to_index: dict | None = None,
    tokenizer_stoi: dict | None = None,
    feature_id_to_token: dict | None = None,
    pairing_columns: tuple[str, ...] = (),
    normalization_scale: float = 1.0,
    num_workers: int = 0,
):
    """Return the data_gen dictionary consumed by wrapper_train.

    This counts-based route uses continuous log1p expression and pertTF's default
    pairs (``default_pair_sampler``) within the same dataset and celltype: training
    uses every training cell once as a source, validation every validation cell
    once as a target. Additional pairing columns can constrain donor/condition.
    Train and valid indices each include their available controls; only controls
    may overlap.
    The one expanded perturbation mapping is shared by both loaders and must be
    passed unchanged to the model/UnifiedPertEncoder. Supply tokenizer_stoi when
    using a pretrained expression vocabulary.

    Perturbation metrics use the tokenizable intersection of the validation
    datasets' feature panels. Full-expression losses use availability masks.
    """
    if config.get("next_cell_pred_type", "pert") != "pert" or not config.get("GEPC", False):
        raise ValueError("Corpus perturbation training requires next_cell_pred_type='pert' and GEPC=True")
    if config.get("input_emb_style", "continuous") != "continuous":
        raise ValueError("Corpus log1p expression requires input_emb_style='continuous'")
    if config.get("ps_weight", 0) != 0 or config.get("pred_lochness_next", 0):
        raise ValueError("Corpus training does not supply perturbation-score targets; disable PS objectives")
    if not corpus.features_aligned:
        raise ValueError("Align corpus features explicitly with concat(feature_key=...) before training")
    if any(kind == "non_count" for kind in corpus.expression_kinds.values()):
        raise ValueError("This training route requires counts; a no_count corpus needs explicit preprocessing first")
    label_fields = {perturbation_column: "perturbation", celltype_column: "celltype", batch_column: "batch"}
    if len(label_fields) != 3:
        raise ValueError("Perturbation, celltype, and batch columns must be distinct")
    context_columns = list(dict.fromkeys(("dataset_id", celltype_column, *pairing_columns)))
    frame = corpus.metadata_index.df.select(list(dict.fromkeys(["dataset_index", *label_fields, *context_columns]))).to_pandas()
    selected_rows = np.union1d(train_indices, valid_indices)
    if frame.iloc[selected_rows][list(dict.fromkeys([*label_fields, *context_columns]))].isna().any().any():
        raise ValueError("Selected training/validation metadata contains missing labels")
    split_report = check_train_valid_split(
        ad.AnnData(obs=frame), train_indices, valid_indices, "pert",
        control_value=control_label, context_col=celltype_column,
        perturbation_col=perturbation_column, min_cells=config.get("perturbation_metric_min_cells", 30),
    )
    sources = pert_sources or {}
    mapping = genotype_to_index if genotype_to_index is not None else build_perturbation_mapping(
        frame.iloc[selected_rows][perturbation_column].tolist(), sources, control_label=control_label,
    )
    source_ids = {gene for genes, _ in sources.values() for gene in genes}
    if source_ids.difference(mapping):
        raise ValueError("genotype_to_index must include external source IDs")
    adapter_config = PertTFAdapterConfig(
        label_fields=label_fields, normalization_scale=normalization_scale,
        mask_ratio=config.get("mask_ratio", 0.15),
        pad_value=config.get("pad_value", -2), mask_value=config.get("mask_value", -1),
        cls_value=config.get("cls_value", -3),
        sampling_mode=config.get("sampling_mode", "uniform"),
    )
    # Both loaders share one ID per label; labels not supplied are numbered over the selected rows.
    selected_frame = frame.iloc[selected_rows]
    labels = {
        "perturbation": mapping,
        "celltype": cell_type_to_index or {label: i for i, label in enumerate(dict.fromkeys(selected_frame[celltype_column]))},
        "batch": batch_to_index or {label: i for i, label in enumerate(dict.fromkeys(selected_frame[batch_column]))},
    }
    # A pretrained vocabulary need not have every corpus gene; genes without a token are left out.
    tokens = GeneTokens.from_corpus(
        corpus, tokenizer_stoi, feature_id_to_token, pad_token=config.get("pad_token", "<pad>"),
    )
    used_datasets = selected_frame["dataset_index"].unique()
    coverage = corpus.feature_registry.dataset_has_gene[used_datasets] & tokens.tokenizable
    if not coverage.any(axis=1).all():
        raise ValueError("Every selected dataset must have at least one tokenizable gene")
    options = dict(
        config=adapter_config, tokens=tokens, labels=labels, batch_size=config.get("batch_size", 32),
        seq_len=config.get("max_seq_len", 1024) - 1, seed=config.get("seed", 0),
        num_workers=num_workers, multiprocessing_context="spawn" if num_workers else None,
    )
    metadata = corpus.metadata_index.df
    pairing = dict(context_columns=context_columns, perturbation_column=perturbation_column, control_labels=(control_label,))
    train_rows = np.asarray(train_indices, dtype=np.int64)
    train_loader = PertTFPairedBatchLoader(
        corpus, rows=train_rows, cover="source",
        sampler=default_pair_sampler(metadata[train_rows], perturbed_to_self=True, **pairing), **options,
    )
    valid_rows = np.asarray(valid_indices, dtype=np.int64)
    valid_loader = PertTFPairedBatchLoader(
        corpus, rows=valid_rows, cover="target",
        sampler=default_pair_sampler(metadata[valid_rows], perturbed_to_self=False, **pairing), **options,
    )
    datasets = frame.iloc[valid_rows]["dataset_index"].unique()
    metric_gene_mask = corpus.feature_registry.dataset_has_gene[datasets].all(axis=0)
    metric_gene_mask &= tokens.tokenizable
    if not metric_gene_mask.any():
        raise ValueError("Validation datasets have no shared tokenizable genes")
    valid_loader.metric_gene_mask = metric_gene_mask
    reference_groups = corpus_reference_groups(
        corpus, valid_rows, context_column=celltype_column,
        perturbation_column=perturbation_column, gene_mask=metric_gene_mask,
        target_sum=config.get("perturbation_metric_target_sum", 10000.0),
    )
    genes = np.asarray(corpus.feature_registry.global_feature_ids)
    return {
        "train_loader": train_loader, "valid_loader": valid_loader,
        "vocab": SimpleVocab.from_dict(tokens.stoi),
        "genes": genes, "gene_ids": tokens.token_of_gene,
        "genotype_to_index": dict(mapping), "cell_type_to_index": labels["celltype"],
        "num_batch_types": len(labels["batch"]), "ps_names": [],
        "perturbation_validation": True, "control_label": control_label,
        "perturbation_reference_groups": reference_groups,
        "metric_gene_ids": genes[metric_gene_mask],
        "split_report": split_report,
        "corpus_config": {
            "adapter": asdict(adapter_config),
            "batch_to_index": labels["batch"],
            "expression_kinds": dict(corpus.expression_kinds),
            "metric_gene_ids": genes[metric_gene_mask].tolist(),
        },
    }
