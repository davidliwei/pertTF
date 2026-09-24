# Loading perturbation embedding presets

Use published ESM2, GenePT, or GEARS embeddings with the existing perturbation
lookup and feature encoders. This changes data loading, not model architecture.

## Prepare embeddings and the dataset together

```python
from perttf.model.pert_emb import load_pert_embeddings

prepared = load_pert_embeddings("esm2", adata, intersect_type="source")
matrix = prepared["pert_embedding"]
genotype_to_index = prepared["genotype_index_gears"]
adata_subset = prepared["adata_subset"]
```

The historical `genotype_index_gears` return key applies to all embedding types.
Pass the returned AnnData and mapping into `produce_training_datasets()` before
constructing the model. The matrix row order and mapping are constructed
together, rather than aligning against a previously created model mapping.

- `intersect_type="common"` (default): retain source-covered perturbations found
  in AnnData, using the existing sorted intersection.
- `intersect_type="source"`: retain all source perturbations, including those
  with no cells in this dataset, using source order.
- A zero-valued WT row is appended once, and cells outside the resulting mapping
  are excluded. No missing-source random embeddings are introduced.
- `filter_by_human` still accepts the optional gene-list file used by older calls.

For an existing `PertLabelEncoder`, construct the model with
`pert_dim=matrix.shape[1]`, then initialize its lookup weights:

```python
from perttf.model.pert_emb import load_pert_embedding_to_model

model = load_pert_embedding_to_model(model, matrix, requires_grad=False)
```

Use `requires_grad=True` for trainable lookup weights. For the existing
`FeaturePertEncoder`, supply `pert_features=matrix` when constructing either
`PerturbationTFModel` or `HFPerturbationTFModel`; its MLP projects the raw features
into `pert_dim`. Both routes use the same prepared mapping and cell subset.

## Custom embeddings

Load the data yourself and supply a matching gene list and NumPy matrix:

```python
prepared = load_pert_embeddings(
    None,
    adata,
    custom_genes=gene_names,
    custom_embeddings=my_matrix,
    intersect_type="source",
)
```

The IDs must be unique and their count must match the matrix row count. Do not
select a preset alongside custom inputs. File loading, any identifier conversion,
and any concatenation/alignment of custom representations are the caller's work.

## Read raw source tables without preparing AnnData

```python
from perttf.model.pert_emb import load_perturbation_sources

sources = load_perturbation_sources("genept+esm2")
genept_genes, genept_matrix = sources["genept"]
esm2_genes, esm2_matrix = sources["esm2"]
sources["my_source"] = (my_gene_names, my_matrix)
```

This returns independent `(gene_names, numpy_matrix)` tuples. It does not
intersect or concatenate tables, filter cells, append WT, or construct a model.
`custom_sources={"my_source": (my_gene_names, my_matrix)}` can also be passed
directly to this reader; preset/custom names must not collide. All custom inputs
are in memory and cause no downloads.

## Published data and downloading

Presets come from
[`weililab/perturbation-embeddings`](https://huggingface.co/datasets/weililab/perturbation-embeddings)
at pinned revision `5b01881945b81d9250cf600fb07968539ed8292b`:

| Preset | Genes | Dimensions | Matrix size |
|---|---:|---:|---:|
| `esm2` | 19,516 | 5,120 | ~400 MB |
| `genept` | 19,134 | 3,072 | ~235 MB |
| `gears` | 9,853 | 32 | ~1.3 MB |

ESM2 and GenePT are restricted to their respective human gene lists. GEARS comes
from a legacy checkpoint with unknown training provenance; see the dataset card.

HF files are fetched directly into memory, without a persistent file cache or
authentication. Calling the reader again downloads again; reuse the returned
tuples when appropriate. Both readers accept `revision=`. Branches/tags are
resolved once before reading files so a call cannot mix versions. Record the
chosen commit with experiment provenance; the default is exported as
`DEFAULT_EMBEDDING_REVISION`.

## Migrating older calls

- `path1` and `path2` are accepted but **ignored**, with one warning if either is
  supplied. `embed_type` alone selects the HF preset. In particular, a local
  GEARS checkpoint path no longer selects that checkpoint's weights through
  `load_pert_embeddings()`.
- `load_pert_embeddings()` accepts one preset only. `"concat"` and
  `"genept+esm2"` are not accepted here; prepare your concatenation explicitly and
  pass the resulting gene list and matrix as custom inputs.
- `_load_raw_embeddings()` has been removed.
- The older `load_pert_embedding_from_gears()` remains unchanged and still reads
  a local GEARS checkpoint. `load_pert_embedding_to_model()` is unchanged too.
