# Unified perturbation embeddings

For new HF models, `pert_sources` combines fixed external gene features with
trainable embeddings for perturbations missing all selected sources. Existing
models using `PertLabelEncoder` or `pert_features` retain their original behavior.

## Choose sources

```python
from perttf.model.pert_emb import load_perturbation_sources

sources = load_perturbation_sources("genept+esm2")
# Also supported: "gears", "esm2", "genept", or any distinct combination.
```

Names always select files from the public
[`weililab/perturbation-embeddings`](https://huggingface.co/datasets/weililab/perturbation-embeddings)
dataset. The default revision is pinned to
`5b01881945b81d9250cf600fb07968539ed8292b`; `revision=` selects another release.
ESM2 and GenePT are filtered using their respective human gene lists, with
19,516 and 19,134 entries. Public files are downloaded directly into memory,
without a persistent HF file cache or authentication. Each preset load downloads
again: approximately 400 MB for ESM2, 235 MB for GenePT, and 1.3 MB for GEARS
(matrix sizes in decimal units). Keep and reuse the returned tuples to avoid
repeated downloads. No cache directory is required or created by this loader.
If selecting a branch/tag revision, it is resolved once before reading any files.
Record your chosen commit SHA separately with experiment provenance; the default
is also exported as `DEFAULT_EMBEDDING_REVISION`.

Sources stay separate: `"genept+esm2"` means two projection branches, not raw
feature concatenation. Gene sets are combined by union; no training cells are
dropped by this raw-table reader. The legacy preparation workflow below applies
its existing intersection and AnnData-subsetting policies to these same tables.

### Custom sources

```python
sources = load_perturbation_sources(
    "esm2",
    custom_sources={"my_source": (my_gene_names, my_numpy_matrix)},
)
# Or use only custom_sources, without any preset.
```

Every source is an in-memory `(gene_names, numpy_matrix)` tuple. The matrix has
one row per gene and one column per feature. You can also supply these tuples
directly to the HF model, without calling a loader:

```python
sources = {
    "my_source": (["GENE_A", "GENE_B"], feature_matrix),  # shape (2, d)
}
```

Gene IDs must be unique within a source, and their count must match the matrix
row count; these are the table checks. Source names must be distinct (a custom
source cannot overwrite a selected preset). The encoder converts matrices to
float32 and requires source names without dots. Raw values and identifiers are
not normalized or repaired. An ID may occur in several sources. Custom tuples
are already loaded by the caller and do not cause downloads or file reads.

## Existing lookup and feature encoders

`load_pert_embeddings()` prepares the matrix, genotype mapping, and filtered
AnnData together, before dataset/model construction. Select one HF preset, or
supply an already loaded gene list and NumPy matrix:

```python
from perttf.model.pert_emb import load_pert_embeddings, load_pert_embedding_to_model

prepared = load_pert_embeddings("esm2", adata, intersect_type="source")
# Or: load_pert_embeddings(None, adata, custom_genes=genes, custom_embeddings=matrix)
adata_subset = prepared["adata_subset"]
genotype_to_index = prepared["genotype_index_gears"]
features = prepared["pert_embedding"]
```

Pass `adata_subset` and this `genotype_to_index` to the existing dataset
preparation flow, then construct the model with the resulting perturbation count.
The historical `genotype_index_gears` return key is retained for all source types.

- `intersect_type="common"`: use source-covered perturbations observed in AnnData.
- `intersect_type="source"`: retain all supported source perturbations, including
  ones absent from this dataset.
- `embed_type` accepts only `"esm2"`, `"genept"`, or `"gears"`. For a custom
  representation, including concatenation, load and align the vectors yourself
  and supply `custom_genes` plus `custom_embeddings` instead of a preset.
- The optional `filter_by_human` gene-list file retains its existing behavior.
- Add WT with a zero row and subset cells to the resulting mapping. No missing
  perturbations receive random rows, and no existing mapping is imposed on the
  source tables.

For `PertLabelEncoder`, set `pert_dim=features.shape[1]` at construction and
initialize its weights with
`load_pert_embedding_to_model(model, features, requires_grad=False)` (or `True`
for a trainable lookup). For the existing `FeaturePertEncoder`, pass
`pert_features=features` during construction; its single MLP projects the complete
feature matrix into `pert_dim`. Both original model and HF constructors retain
these legacy routes. The feature-encoder demo's manually generated complete
feature matrix also continues to work unchanged.

`path1` and `path2` remain accepted but are ignored: supplying either emits one
deprecation warning. In particular, a local GEARS checkpoint path no longer
selects its weights; `embed_type="gears"` selects the published HF table.
The older `load_pert_embedding_from_gears()` remains available for directly
loading a local GEARS checkpoint. The internal `_load_raw_embeddings()` reader
has been removed from the newer loader.
For custom inputs, pass `embed_type=None` and supply both the gene list and NumPy
matrix; gene IDs must be unique and their count must match the matrix row count.
Custom inputs and a preset are mutually exclusive. This preparation workflow does not introduce the unified encoder's
union/fallback policy into either legacy encoder; multi-source support remains
available separately through `load_perturbation_sources()` for the unified route.

## Construct a unified HF model

Build the expanded mapping before preparing datasets or constructing the model.
It includes the dataset perturbations, control, and the union of selected source
IDs. Dataset names keep their first-occurrence order; external-only IDs are
appended in sorted order. Include all dataset labels needed by your training and
evaluation cohorts when preparing this fixed vocabulary.

```python
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.pert_emb import build_perturbation_mapping
from perttf.model.train_data_gen import produce_training_datasets

genotype_to_index = build_perturbation_mapping(
    adata.obs["genotype"], sources, control_label="WT",
)
data_gen = produce_training_datasets(
    adata, config,
    genotype_to_index=genotype_to_index,
    next_cell_pred="pert",
    train_indices=train_indices,
    valid_indices=valid_indices,
)

model = HFPerturbationTFModel(
    vocab=data_gen["vocab"],
    genotype_to_index=genotype_to_index,
    cell_type_to_index=data_gen["cell_type_to_index"],
    pert_sources=sources,
    control_label="WT",
    d_model=128,
    pert_dim=128,
    nhead=4,
    do_mvc=True,
    n_ps=0,
)
```

The same mapping is used by `PertTFDataset`, the encoder and the perturbation
classifier. `n_pert` is its full size. The encoder does not add IDs: it only
constructs source-row and fallback-row lookup buffers. Source IDs absent from the
supplied mapping are rejected, so build it after selecting your source tables.
There is no second internal genotype mapping.

External-only IDs are classifier classes too. When classification loss is
enabled, they participate in the softmax without positive training examples.
This is an explicit consequence of using one expanded vocabulary for everything.

The `pert_sources` construction option belongs only to `HFPerturbationTFModel`;
the original `PerturbationTFModel` constructor keeps its existing encoder routes.
When injecting source perturbations into transformer
tokens (`perturbation_input=True`), keep `pert_dim == d_model`, as required by the
existing additive input path. Target perturbation conditioning supports a
different `pert_dim`.

The representation is:

1. **Any external coverage:** each available fixed feature vector passes through
   its own `Linear -> ReLU -> Linear` MLP and a non-affine LayerNorm. Average only
   the available branch outputs, then apply a shared final affine LayerNorm.
2. **No external coverage:** use one randomly initialized, trainable vector in
   `pert_dim`, followed by that same final LayerNorm. There are no random
   replacements for individual missing source components.

External tables are buffers. MLPs, fallback vectors, and the final LayerNorm are
parameters. Freezing input vectors does not freeze their projected outputs.
`pert_sources={}` explicitly selects a fallback-only unified encoder.

The control label must be present in the supplied mapping. Its input feature
row is forced to zero in every source (without modifying the caller's tables),
and it receives no random fallback. By default its projected output may be
nonzero due to learned biases. Set `pert_pad_id` to the control index to mask
the final output to zero. With no sources, the control enters the final norm as
a zero vector.

Train the model through the existing forward/training functions. The mapping
and branches are fixed at construction. Dynamic registration, new source
branches, and LoRA/partial-transfer support for this route are not included.
A fallback vector needs training examples to become informative; simply
registering an uncovered held-out ID does not provide zero-shot information.

## Save and restore

```python
model.save_pretrained("my_model")
model = HFPerturbationTFModel.from_pretrained("my_model")
```

Feature buffers, fallback vectors, and projection parameters are in the model
state. Source IDs and gene order are in `running_parameters.pt`.
Reload reconstructs from these artifacts, without fetching external embeddings.
Unified checkpoints require strict loading; replacing mappings or sources is
not a supported reload operation. Use the HF export above for a complete
architecture-bearing checkpoint rather than relying on legacy training-config
exports to describe every constructor setting.

## Inference and temporary perturbations

For a registered perturbation (including one covered only by external tables),
assign its name in `adata.obs["genotype_next"]` and call
`model.predict_perturbations(adata)` as usual. The source-cell `genotype` and
target `genotype_next` use the same saved expanded mapping. `celltype` uses its
saved mapping. Registered targets follow the normal label-based forward path;
direct dataset/evaluation calls can use the same mapping without dropping
external-only perturbations.

For an entirely new target ID, supply feature rows for existing source types:

```python
result = model.predict_perturbations(
    adata,
    prediction_mode="mean",
    query_sources={
        "esm2": (["NEW_PERTURBATION"], query_esm2_matrix),  # shape (1, saved ESM2 input width)
    },
)
```

Use `NEW_PERTURBATION` in the relevant `genotype_next` rows. It must not collide
with any registered training or external-table ID. The same new ID can appear
in several query sources; their available outputs are averaged normally.
The existing MLPs are used; no model entries, fallback vectors, or parameters are
added. Source dimensions, row alignment, and ID collisions are checked. Biological suitability
for the named feature space is the caller's responsibility.

Unknown source types are rejected because no trained projection exists. Missing
IDs without supplied vectors are rejected rather than randomly represented at
inference. Only temporary query IDs need an evaluation-local mapping extension
and explicit representation inputs. Perturbation classifier outputs cover the
saved expanded vocabulary, not these temporary query IDs.

The low-level encoder also exposes `encode_queries(query_sources)` in eval mode,
returning ordered query IDs and their representations. `PerturbationTFModel`
accepts batch-aligned `pert_embeddings_next` for direct target conditioning;
the HF inference method handles this batching automatically.
