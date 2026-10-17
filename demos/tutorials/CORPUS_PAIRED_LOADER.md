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

Which cells are paired is described with `scorpus.pairing` (see scorpus
`docs/pairing.md`). `default_pair_sampler` builds pertTF's default pairs; the
loader turns them into batches.

```python
import numpy as np
from scorpus import load_corpus
from perttf.model.corpus_adapter import (
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    default_pair_sampler,
)

corpus = load_corpus("/path/to/corpus")

config = PertTFAdapterConfig(
    label_fields={
        "perturb_label": "perturbation",
        "cell_context": "celltype",
        "batch_id": "batch",
    },
    perturbation_label="perturbation",
    control_labels=("ctrl",),
    normalize_expression="log1p",
    normalization_scale=1.0,
)

rows = np.arange(len(corpus.metadata_index))      # or the training rows
sampler = default_pair_sampler(
    corpus.metadata_index.df[rows],
    context_columns=("dataset_index", "cell_context"),
    perturbation_column="perturb_label",
    control_labels=("ctrl",),
    perturbed_to_self=True,
)
loader = PertTFPairedBatchLoader(
    corpus,
    sampler=sampler,
    rows=rows,
    cover="source",
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

`perturbation_label` names the configured label used for the perturbation
label tensors. It must refer to one of the label names, not the raw metadata
column name. The target's perturbation label is always the target cell's own.

`control_labels` lists the metadata values treated as controls. Match these to
the values produced by canonicalization, for example `("ctrl",)` or `("WT",)`.

## Which Cells Are Paired

`default_pair_sampler(metadata, context_columns=..., perturbation_column=...,
control_labels=..., perturbed_to_self=...)` pairs cells only within the same
values of `context_columns` (metadata columns, for example dataset, cell type or
donor):

- Training (`perturbed_to_self=True`, loader `cover="source"`): every cell is a
  source once per epoch. A control is paired with a random cell of a
  perturbation in its context; a perturbed cell is paired with itself; a control
  whose context has no perturbation is paired with itself.
- Validation (`perturbed_to_self=False`, loader `cover="target"`): every cell is
  a target once per epoch. A perturbed target gets a random control of its
  context; a control target is paired with itself. Every perturbed target needs
  a control in its context, otherwise `default_pair_sampler` raises.

Rows with a missing perturbation are never paired.

The sampler's rows are positions in the metadata table it was built from;
`rows` gives the corpus row of each of them. Build the sampler on
`corpus.metadata_index.df[rows]` and pass the same `rows` to the loader.

## Custom Pairings

Any `scorpus.pairing` sampler works. For example, perturbed cells paired with a
control of the same dataset:

```python
import polars as pl
from scorpus.pairing import Sampler, cross, make_groups

metadata = corpus.metadata_index.df[rows]
groups = make_groups(metadata, by=["dataset_index", "perturb_label"])
is_ctrl = pl.col("perturb_label") == "ctrl"
pairs = cross(groups, on=["dataset_index"], source=~is_ctrl, target=is_ctrl)
loader = PertTFPairedBatchLoader(
    corpus, sampler=Sampler(groups, pairs), rows=rows, batch_size=16, seq_len=512, config=config,
)
```

The loader covers one side per epoch (`cover`), flattens each sample's cell
pairs and cuts them into batches of `batch_size`; with `drop_last=True` the
final incomplete batch is omitted. pertTF needs one source cell per target
cell, so set-level samples must have equal numbers of cells on both sides.

`loader.effective_label_row_indices` lists the rows kept for label lookup after
null-label filtering.

## Null Labels

By default, rows with null values for the perturbation label are dropped from
the loader's label lookup; `default_pair_sampler` never pairs them.

Use `drop_null_labels` to control which labels trigger dropping (a custom pair
table must not pair the dropped rows):

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
- Perturbed groups without a control (validation): relax `context_columns` or include those contexts' controls in `rows`.
- Too many rows dropped: inspect nulls in the metadata columns listed in `drop_null_labels`.
- Missing HVG rankings: run `recalc-hvg` or use a sampling mode that does not depend on HVG ranks.
