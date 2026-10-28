# pertTF Paired Corpus Loader

`PertTFPairedBatchLoader` builds pertTF source/target batches from a `scorpus`
corpus. `scorpus.pairing` decides which cells are paired; the loader reads the
cells and turns them into pertTF tensors: gene tokens, sampled genes, log1p
expression, masking, CLS, full expression and labels. `produce_corpus_datasets`
(see [CORPUS_TRAINING.md](CORPUS_TRAINING.md)) builds both loaders for training;
use the loader directly to inspect batches or to pair cells differently.

## Requirements

- `load_corpus(corpus_root)` must succeed (scorpus >= 0.6.0).
- The metadata must contain the columns in `label_fields` and `dataset_index`,
  with no missing labels in the loader's rows.
- Every dataset's obs must have `scorpus_total_counts` (written by scorpus
  conversion). A cell's size factor is its total over `target_sum` (default
  10,000) or, with `target_sum=None`, over the median positive total of its
  dataset's cells among the loader's `rows`; 1 for a cell without counts.

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
    target_sum=10000.0,                            # or None: each dataset's median total in rows
    sampling_mode="hvg",                           # each dataset's highly_variable genes (2,000 here)
    non_hvg_size=1000,
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
    seq_len=3000,                                  # hvg: n_hvg + non_hvg_size, as pertTF's config sets
    batch_size=8,
    cover="source",
    device="cuda",                                 # where batches are built (or finished, see below)
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

Genes are drawn per source cell from its pool: the genes its dataset measured
that have a token. The target cell is read at the same genes. `sampling_mode`
uses the quotas of pertTF's collator (`custom_tokenizer.weighted_sample`),
uniformly within each group:

- `"simple"`: `seq_len` genes
- `"expressed"`: `round(nonzero_prop * seq_len)` of the cell's expressed genes
  (all if fewer), the rest from its unexpressed genes; with `fix_nonzero_prop`,
  at most `round(k * (1 - nonzero_prop) / nonzero_prop)` unexpressed genes for
  `k` expressed genes drawn
- `"hvg"`: `non_hvg_size` of the dataset's other genes, and its HVGs in the
  remaining `seq_len - non_hvg_size` places (a uniform subset if it has more).
  HVGs are the genes of the dataset's var column `highly_variable`, whichever
  way it was set (any scanpy flavor, another method, a hand-made list): set
  `corpus.var["highly_variable"]`, save it with `save_ex_var` (or
  `save_ex_var(name)` to keep several HVG sets) and load with
  `load_corpus(root, ex_var=True)` (or `ex_var=name`). No rank is used; the
  number of HVGs is the number of `True` genes. pertTF's config sets `max_seq_len` to
  `n_hvg + non_hvg_size + 1` in this mode; `non_hvg_size` larger than `seq_len`
  is an error.

A cell whose pool fits in `seq_len` gets its whole pool. Rows shorter than the
longest in the batch end in padding, and the batch is only as wide as its
longest row. A cell with too few unexpressed genes for its quota gets all of
them (pertTF's collator raises there).

Batch keys:

- `gene_ids`, `next_gene_ids`: CLS, then the sampled genes' token IDs
- `values`: source values with exactly `int(n * mask_ratio)` of a row's `n`
  genes replaced by `mask_value`, as pertTF's `random_mask_value` does; for a
  pair across gene panels, genes the target's dataset did not measure are not
  among the `n`, so the masked-gene losses never score a target's missing gene
- `target_values`: unmasked source values
- `target_values_next`: target values at the source's genes (`pad_value` where
  the target's dataset did not measure the gene)
- expression values are `log1p(count / size_factor)`;
  positions without a gene hold `pad_value`, the CLS position `cls_value`
- `sf`, `sf_next`: `1 / size_factor`, pertTF's normalization multiplier
- `index`, `next_index`: source and target corpus rows
- `{label}_labels`, `{label}_labels_next`: label IDs
- `full_gene_ids`, `full_expr`, `full_expr_next`: every corpus gene, after CLS
- `full_expr_mask`, `full_expr_next_mask`: genes the cell's dataset measured and
  that have a token; the full-expression losses use only these

## Workers and Epochs

With `num_workers > 0`, worker processes read expression; pairing stays in the
main process. Each read covers `batches_per_read` consecutive batches and reads
every distinct cell once, so a self pair is one read. Small batches are limited
by the fixed cost of each read on network storage: on beacon's NFS, 64-pair
reads gave under 1k pairs/s with 8 workers, 1,024-pair reads 6-10k. Choose
`batches_per_read` so a read covers about 1,000 pairs; batches are the same for
any value.

The tensors are built in one of two places:

- default: in the main process, on `device` (a GPU is much faster than the CPU
  for this). Highest throughput, but it takes GPU time from every training step
  (~37 ms per 1,024 pairs on an RTX 6000 Ada).
- `build_in_workers=True`: on the CPU in the workers; the main process only
  fills the full gene axis on `device` (~12 ms per 1,024 pairs). Each worker
  needs ~0.75-0.95 s of CPU per 1,024 pairs, so throughput scales with workers.

Both draw with the same quotas, but different random numbers pick different
genes for the same seed. `persistent_workers=True` keeps the workers between
epochs. For Lance-backed corpora, `multiprocessing_context="spawn"` is usually
safest. Workers of a concatenated corpus get a reader that holds only table
paths and row ranges, not the member corpora.

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
- No HVGs for a dataset: `"hvg"` then gives its cells only `non_hvg_size` other genes; compute HVGs (e.g. scanpy on `to_anndata_lazy`), save them with `save_ex_var` and load with `ex_var=True`.
- Missing `scorpus_total_counts`: the corpus was converted with scorpus older than 0.5.0; re-convert it.
