"""Prepare pertTF training and validation directly from sparse on-disk corpora."""

from __future__ import annotations

from dataclasses import asdict

import anndata as ad
import numpy as np

from scorpus.loaders import GeneTokenMapper

from .corpus_adapter import PertTFAdapterConfig, PertTFCorpusAdapter, PertTFPairedBatchLoader
from .pert_emb import build_perturbation_mapping
from ..utils.custom_tokenizer import SimpleVocab
from ..utils.split_check import check_train_valid_split
from ..utils.pert_metrics import GroupMoments, normalize_expression


def corpus_reference_groups(corpus, rows, *, context_column, perturbation_column, gene_mask, target_sum=10000.0, chunk_rows=1024):
    """Stream reference moments over the same common gene axis used in evaluation."""
    selected_genes = np.flatnonzero(gene_mask)
    global_to_output = np.full(corpus.feature_registry.global_vocab_size, -1, dtype=np.int64)
    global_to_output[selected_genes] = np.arange(len(selected_genes))
    moments = GroupMoments()
    rows = np.asarray(rows, dtype=np.int64)
    for start in range(0, len(rows), chunk_rows):
        selected = rows[start:start + chunk_rows]
        batch = corpus.expression_reader.read_expression_flat(selected.tolist())
        metadata = corpus.take_metadata(selected, columns=["dataset_index", context_column, perturbation_column])
        expression = np.zeros((len(selected), len(selected_genes)), dtype=np.float64)
        for i, dataset_index in enumerate(metadata["dataset_index"]):
            global_ids = corpus.feature_registry.local_to_global_map[dataset_index, batch.row_gene_indices(i)]
            positions = global_to_output[global_ids]
            keep = positions >= 0
            expression[i, positions[keep]] = batch.row_counts(i)[keep]
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

    This counts-based route uses continuous log1p expression, same-dataset and
    same-celltype pairs, source-driven training and target-driven validation.
    Additional pairing columns can constrain donor/condition. Train and valid
    indices each include their available controls; only controls may overlap.
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
    frame = corpus.metadata_index.df.to_pandas()
    label_fields = {perturbation_column: "perturbation", celltype_column: "celltype", batch_column: "batch"}
    if len(label_fields) != 3:
        raise ValueError("Perturbation, celltype, and batch columns must be distinct")
    groups = []
    for column in dict.fromkeys(("dataset_id", celltype_column, *pairing_columns)):
        if column not in label_fields:
            label_fields[column] = f"pair_{len(groups)}"
        groups.append(label_fields[column])
    selected_rows = np.union1d(train_indices, valid_indices)
    if frame.iloc[selected_rows][list(label_fields)].isna().any().any():
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
        label_fields=label_fields, control_labels=(control_label,),
        pairing_group_labels=tuple(groups), include_full_expr=True,
        normalize_expression="log1p", normalization_scale=normalization_scale,
        mask_ratio=config.get("mask_ratio", 0.15),
        pad_token=config.get("pad_token", "<pad>"),
        pad_value=config.get("pad_value", -2), mask_value=config.get("mask_value", -1),
        cls_value=config.get("cls_value", -3),
    )
    label_mappings = {"perturbation": mapping}
    if cell_type_to_index is not None:
        label_mappings["celltype"] = cell_type_to_index
    if batch_to_index is not None:
        label_mappings["batch"] = batch_to_index
    gene_mapper = None
    if tokenizer_stoi is not None:
        # pertTF/scGPT vocabularies need not have UNK: this route excludes
        # untokenizable genes instead of representing them with an UNK token.
        gene_mapper = GeneTokenMapper.from_tokenizer_stoi(
            corpus.feature_registry, tokenizer_stoi,
            pad_token=adapter_config.pad_token, cls_token=adapter_config.cls_token,
            feature_id_to_token=feature_id_to_token,
        )
    elif feature_id_to_token is not None:
        raise ValueError("feature_id_to_token requires tokenizer_stoi")
    adapter = PertTFCorpusAdapter.from_corpus(
        corpus, adapter_config, row_indices=selected_rows,
        gene_token_mapper=gene_mapper,
        label_mappings=label_mappings,
    )
    used_datasets = frame.iloc[selected_rows]["dataset_index"].unique()
    coverage = corpus.feature_registry.dataset_has_gene[used_datasets] & adapter.gene_token_mapper.tokenizable_by_global_id
    if not coverage.any(axis=1).all():
        raise ValueError("Every selected dataset must have at least one tokenizable gene")
    options = dict(
        adapter=adapter, batch_size=config.get("batch_size", 32),
        seq_len=config.get("max_seq_len", 1024) - 1,
        seed=config.get("seed", 0), drop_last=False,
        sampling_mode=config.get("sampling_mode", "uniform"),
        num_workers=num_workers, multiprocessing_context="spawn" if num_workers else None,
    )
    if options["seq_len"] < 1:
        raise ValueError("max_seq_len must allow at least one gene and CLS")
    train_loader = PertTFPairedBatchLoader(
        corpus, row_indices=train_indices, perturbed_target_policy="self_to_self_label", **options,
    )
    valid_rows = np.asarray(valid_indices, dtype=np.int64)
    controls = valid_rows[frame.iloc[valid_rows][perturbation_column].to_numpy() == control_label]
    valid_loader = PertTFPairedBatchLoader(
        corpus, row_indices=valid_rows, source_indices=controls, target_candidate_indices=valid_rows,
        pairing_mode="target_driven", **options,
    )
    datasets = frame.iloc[valid_rows]["dataset_index"].unique()
    metric_gene_mask = corpus.feature_registry.dataset_has_gene[datasets].all(axis=0)
    metric_gene_mask &= adapter.gene_token_mapper.tokenizable_by_global_id
    if not metric_gene_mask.any():
        raise ValueError("Validation datasets have no shared tokenizable genes")
    valid_loader.metric_gene_mask = metric_gene_mask
    reference_groups = corpus_reference_groups(
        corpus, valid_rows, context_column=celltype_column,
        perturbation_column=perturbation_column, gene_mask=metric_gene_mask,
        target_sum=config.get("perturbation_metric_target_sum", 10000.0),
    )
    return {
        "train_loader": train_loader, "valid_loader": valid_loader,
        "vocab": SimpleVocab.from_dict(adapter.to_simple_vocab_stoi()),
        "genes": np.asarray(adapter.feature_ids_by_global_id), "gene_ids": adapter.gene_token_ids,
        "genotype_to_index": dict(mapping), "cell_type_to_index": adapter.label_to_index_by_name["celltype"],
        "num_batch_types": len(adapter.labels_by_name["batch"]), "ps_names": [],
        "perturbation_validation": True, "control_label": control_label,
        "perturbation_reference_groups": reference_groups,
        "metric_gene_ids": np.asarray(adapter.feature_ids_by_global_id)[metric_gene_mask],
        "split_report": split_report,
        "corpus_config": {
            "adapter": asdict(adapter_config),
            "batch_to_index": adapter.label_to_index_by_name["batch"],
            "expression_kinds": dict(corpus.expression_kinds),
            "metric_gene_ids": np.asarray(adapter.feature_ids_by_global_id)[metric_gene_mask].tolist(),
        },
    }
