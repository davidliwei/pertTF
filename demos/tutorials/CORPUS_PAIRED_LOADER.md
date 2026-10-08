# pertTF Paired Corpus Loader

`PertTFPairedBatchLoader` builds pertTF-style source/target paired batches from a
canonicalized `scorpus` corpus. It is the recommended public path when a
training loop needs pertTF-compatible tensors. The adapter is owned by pertTF;
scorpus supplies the model-independent corpus interface.

## Requirements

- The corpus may be directly converted or use the curated canonical workflow.
- `load_corpus(corpus_root)` must succeed.
- The canonical obs metadata must contain the columns used in `label_fields`.
- `size_factor` must be available in the loaded metadata index.

## Basic Usage

```python
from scorpus import load_corpus
from perttf.model.corpus_adapter import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
)

corpus = load_corpus("/path/to/corpus")

config = PertTFAdapterConfig(
    label_fields={
        "perturb_label": "perturbation",
        "cell_context": "celltype",
        "batch_id": "batch",
        "dataset_index": "dataset",
    },
    perturbation_label="perturbation",
    control_labels=("ctrl",),
    pairing_group_labels=("dataset", "celltype"),
    normalize_expression="log1p",
    normalization_scale=1.0,
)

loader = PertTFPairedBatchLoader(
    corpus,
    batch_size=8,
    seq_len=1024,
    config=config,
    sampling_mode="hvg",
    hvg_top_k=2000,
    num_workers=0,
)

batch = next(iter(loader))
```

## Label Configuration

`label_fields` maps corpus metadata columns to pertTF label names.

Example:

```python
label_fields={
    "perturb_label": "perturbation",
    "cell_context": "celltype",
    "batch_id": "batch",
}
```

This produces batch keys such as:

- `perturbation_labels`
- `perturbation_labels_next`
- `celltype_labels`
- `celltype_labels_next`
- `batch_labels`
- `batch_labels_next`

`perturbation_label` names the configured label that drives source/target
perturbation pairing. It must refer to one of the label names, not the raw
metadata column name.

`control_labels` lists the metadata values treated as controls. Match these to
the values produced by canonicalization, for example `("ctrl",)` or `("WT",)`.

## Pairing Groups

By default, pairing is perturbation-aware but does not force same-dataset or
same-celltype matches.

Use `pairing_group_labels` when pairs must match extra labels:

```python
# Same dataset only
PertTFAdapterConfig(
    label_fields={"perturb_label": "perturbation", "dataset_index": "dataset"},
    perturbation_label="perturbation",
    pairing_group_labels=("dataset",),
)

# Same dataset and same cell type
PertTFAdapterConfig(
    label_fields={
        "perturb_label": "perturbation",
        "dataset_index": "dataset",
        "cell_context": "celltype",
    },
    perturbation_label="perturbation",
    pairing_group_labels=("dataset", "celltype"),
)
```

Every label listed in `pairing_group_labels` must be present in `label_fields`.

## Row Pools

The loader can restrict which rows participate in pairing.

- `row_indices`: restrict both source and target pools to a global row subset
- `source_indices`: restrict only source rows
- `target_candidate_indices`: restrict only possible target rows

All indices are corpus-global row indices.

```python
loader = PertTFPairedBatchLoader(
    corpus,
    batch_size=16,
    seq_len=512,
    config=config,
    source_indices=source_rows,
    target_candidate_indices=target_rows,
)
```

The loader exposes useful effective pools after null-label filtering:

```python
loader.effective_label_row_indices
loader.effective_source_indices
loader.effective_target_candidate_indices
loader.effective_pair_count
```

## Target-Driven Pairing

Use target-driven pairing to visit every target cell once per epoch. Perturbed
targets sample a matched control source with replacement, while control targets
pair to the identical control cell:

```python
loader = PertTFPairedBatchLoader(
    corpus,
    batch_size=16,
    seq_len=512,
    config=config,
    row_indices=train_rows,
    source_indices=control_rows,
    target_candidate_indices=np.concatenate([control_rows, perturbed_rows]),
    pairing_mode="target_driven",
    drop_last=False,
)
```

Target-driven pairing requires control-only sources. Every control target must
also be in the source pool so it can self-pair. Pairing groups configured by
`pairing_group_labels` constrain each control sampled for a perturbed target to
the target's group. Epoch length is determined by the number of target rows;
with `drop_last=True`, the final incomplete batch is omitted. Source-driven
pairing remains the default.

## Perturbed Source Policy

`perturbed_target_policy` controls how a source row that is already perturbed is
paired:

- `"self_to_control_label"` (default): reuse the source expression as the target and emit the control label
- `"self_to_self_label"`: reuse the source expression as the target and emit its own perturbation label
- `"matched_control_cell"`: pair to a control row from the same configured pairing group

With `"self_to_self_label"`, a control source still pairs to a treated target when one exists in its pairing group. If none exists, the control pairs to itself with the control label.

For an idempotent perturbation target, use:

```python
loader = PertTFPairedBatchLoader(
    corpus,
    batch_size=16,
    seq_len=512,
    config=config,
    perturbed_target_policy="self_to_self_label",
)
```

## Null Labels

By default, rows with null values for the perturbation label are dropped during
loader construction.

Use `drop_null_labels` to control which labels trigger dropping:

```python
config = PertTFAdapterConfig(
    label_fields={"perturb_label": "perturbation", "cell_context": "celltype"},
    perturbation_label="perturbation",
    drop_null_labels=("perturbation", "celltype"),
)
```

The names in `drop_null_labels` are label names from `label_fields`.

## Sampling and Tensor Shape

Main sampling knobs:

- `seq_len`: number of sampled gene tokens before optional CLS insertion
- `sampling_mode`: usually `"hvg"`
- `hvg_top_k`: optional limit for HVG-prioritized sampling
- `expressed_weight` and `hvg_weight`: weights for expressed and HVG genes
- `missing_token_policy`: behavior for genes that cannot be tokenized

Major output keys:

- `gene_ids`, `next_gene_ids`: sampled token IDs
- `values`: masked source expression values
- `target_values`: unmasked source values
- `target_values_next`: paired target values aligned to sampled source genes
- with `normalize_expression="log1p"`, expression values are `log1p(count / size_factor * normalization_scale)`
- `sf`, `sf_next`: `normalization_scale / size_factor`, matching pertTF's normalization-multiplier convention
- `index`, `next_index`: source and target global row indices
- `{label}_labels`, `{label}_labels_next`: integer label IDs for configured labels

If `PertTFAdapterConfig(include_full_expr=True)` is used, the loader also emits:

- `full_gene_ids`
- `full_expr`
- `full_expr_next`
- `full_expr_mask`
- `full_expr_next_mask`

## Workers and Epochs

With `num_workers > 0`, pair planning stays in the main process. Worker processes
only read source and target expression rows.

Use `set_epoch(epoch)` for deterministic reshuffling between epochs:

```python
for epoch in range(10):
    loader.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

For Lance-backed corpora, `multiprocessing_context="spawn"` is usually safest
when using workers.

## Common Failures

- Missing metadata column: load the corpus with the needed canonical obs column or fix `label_fields`.
- No valid control rows: make sure `control_labels` matches canonical `perturb_label` values.
- No valid paired target pool: relax `pairing_group_labels` or provide broader target rows.
- Too many rows dropped: inspect nulls in the metadata columns listed in `drop_null_labels`.
- Missing HVG rankings: run `recalc-hvg` or use a sampling mode that does not depend on HVG ranks.
