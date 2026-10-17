# pertTF Paired Corpus Loader

`PertTFPairedBatchLoader` builds pertTF source/target batches from a `scorpus`
corpus. `scorpus.pairing` decides which cells are paired; the loader reads the
cells and turns them into pertTF tensors: gene tokens, sampled genes, log1p
expression, masking, CLS, full expression and labels. `produce_corpus_datasets`
(see [CORPUS_TRAINING.md](CORPUS_TRAINING.md)) builds both loaders for training;
use the loader directly to inspect batches or to pair cells differently.

## Requirements

- `load_corpus(corpus_root)` must succeed (scorpus >= 0.4.0).
- The metadata must contain the columns in `label_fields`, `dataset_index` and
  `size_factor`, with no missing labels in the loader's rows.

## Basic Usage

```python
import numpy as np
from scorpus import load_corpus
from perttf.model.corpus_adapter import (
    GeneTokens,
    PertTFAdapterConfig,
    PertTFPairedBatchLoader,
    default_pair_sampler,
)

corpus = load_corpus("/path/to/corpus")
rows = np.arange(len(corpus.metadata_index))      # or the training rows

config = PertTFAdapterConfig(
    label_fields={"perturb_label": "perturbation", "cell_context": "celltype", "batch_id": "batch"},
    normalization_scale=1.0,
    sampling_mode="hvg",
    hvg_top_k=2000,
)
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
    config=config,
    tokens=GeneTokens.from_corpus(corpus),
    seq_len=1024,
    batch_size=8,
    cover="source",
    device="cuda",                                 # where batches are built
)

batch = next(iter(loader))
batch = loader.build_batch(np.array([0, 1]), np.array([1, 1]), seed=0)   # chosen cells
```

## Gene Tokens

`GeneTokens.from_corpus(corpus)` makes a new vocabulary: `<pad>`, `<cls>`,
`<unk>`, `<eos>`, then the corpus genes. For a pretrained checkpoint pass its
vocabulary, `GeneTokens.from_corpus(corpus, tokenizer_stoi, feature_id_to_token)`;
corpus genes it lacks get no token and are never sampled or scored.

## Labels

`label_fields` maps metadata columns to pertTF label names and gives the batch
keys `{name}_labels` (source) and `{name}_labels_next` (target), for example
`perturbation_labels` and `celltype_labels_next`. The target's labels are the
target cell's own. Pass `labels={name: {label: id}}` to fix IDs (for example a
checkpoint's `genotype_to_index`); other labels are numbered in order of first
appearance in `rows`. Rows with a missing label raise: leave them out of `rows`.

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
    corpus, sampler=Sampler(groups, pairs), rows=rows, config=config,
    tokens=GeneTokens.from_corpus(corpus), seq_len=512, batch_size=16,
)
```

The loader covers one side per epoch (`cover`), flattens each sample's cell
pairs and cuts them into batches of `batch_size`; with `drop_last=True` the
final incomplete batch is omitted. pertTF needs one source cell per target
cell, so set-level samples must have equal numbers of cells on both sides.

## Sampling and Tensor Shape

`seq_len` genes are drawn per source cell, without replacement, from the genes
its dataset measured that have a token; the target cell is read at the same
genes. `sampling_mode` sets the weights:

- `"uniform"`: every gene the same
- `"expressed"`: expressed genes get `1 + expressed_weight`
- `"hvg"`: HVGs get `1 + hvg_weight`; the dataset's default HVGs, or its
  `hvg_top_k` highest ranked (from `hvg.parquet`)

Batch keys:

- `gene_ids`, `next_gene_ids`: CLS, then the sampled genes' token IDs
- `values`: source values with a `mask_ratio` share replaced by `mask_value`
- `target_values`: unmasked source values
- `target_values_next`: target values at the source's genes
- expression values are `log1p(count / size_factor * normalization_scale)`;
  positions without a gene hold `pad_value`, the CLS position `cls_value`
- `sf`, `sf_next`: `normalization_scale / size_factor`, pertTF's normalization multiplier
- `index`, `next_index`: source and target corpus rows
- `{label}_labels`, `{label}_labels_next`: label IDs
- `full_gene_ids`, `full_expr`, `full_expr_next`: every corpus gene, after CLS
- `full_expr_mask`, `full_expr_next_mask`: genes the cell's dataset measured and
  that have a token; the full-expression losses use only these

## Workers and Epochs

With `num_workers > 0`, worker processes only read source and target
expression; pairing and the tensors stay in the main process and are built on
`device`. Build on the GPU: gene sampling and full expression are much slower
on the CPU. `persistent_workers=True` keeps the workers between epochs. For
Lance-backed corpora, `multiprocessing_context="spawn"` is usually safest.

Use `set_epoch(epoch)` for deterministic reshuffling between epochs:

```python
for epoch in range(10):
    loader.set_epoch(epoch)
    for batch in loader:
        train_step(batch)
```

## Common Failures

- Missing metadata column: load the corpus with the needed column or fix `label_fields`.
- No valid control rows: make sure `control_labels` matches the `perturb_label` values.
- Perturbed groups without a control (validation): relax `context_columns` or include those contexts' controls in `rows`.
- Missing labels: filter those rows out of `rows`.
- No HVG rankings: `"hvg"` then samples uniformly; run `recalc-hvg` for rankings.
