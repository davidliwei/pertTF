# Unified perturbation embeddings

Train a perturbation model whose perturbation encoder reads fixed gene embeddings (ESM2, GenePT, STRING, ...).
Perturbations covered by an embedding table can be predicted without training cells; perturbations that no
table covers get a learned vector instead. Several tables can be combined.

For the older single-table encoders (`load_pert_embeddings`, `PertLabelEncoder`, `pert_features`), see
[Loading perturbation embedding presets](EMBEDDING_PRESETS.md).

## 1. Prepare data

The same data as the pretraining tutorial: log-normalized expression in the layer `X_binned`, HVGs in
`adata.var['highly_variable']`, and `adata.obs` columns `genotype` (`WT` for controls) and `celltype`.

```python
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="weililab/pancreatic_18clone", filename="./18clones_seurat.h5ad",
                repo_type="dataset", local_dir="./")
adata = sc.read_h5ad("./18clones_seurat.h5ad")
adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)

# "44_HHEX" -> "HHEX", "46_HHEX_het" -> "HHEXhet", "47_WT" -> "WT"
def clean_genotype(gene_label):
    parts = str(gene_label).split("_", 1)
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.replace("_", "").split("/")[0]

adata = adata[adata.obs["gene"] != "NA"].copy()
adata.obs["genotype"] = adata.obs["gene"].map(clean_genotype)
adata.layers["X_binned"] = adata.X
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
```

## 2. Load embedding tables

```python
from perttf.model.pert_emb import load_perturbation_sources

sources = load_perturbation_sources("esm2+genept")
# {"esm2": (gene_names, matrix), "genept": (gene_names, matrix)}
```

Each table is a `(gene_names, numpy_matrix)` tuple with one row per gene. Presets are `esm2`, `genept`, `gears`,
`string_net`, `string_prot`, `crispr_score_strict` and `crispr_score_mean20` (sizes and origin in
[EMBEDDING_PRESETS.md](EMBEDDING_PRESETS.md)); join them with `+`. Your own tables are added the same way:

```python
sources = load_perturbation_sources("esm2", custom_sources={"my_source": (my_gene_names, my_matrix)})
# or without any preset: sources = {"my_source": (my_gene_names, my_matrix)}
```

## 3. Split perturbations

Validation perturbations pick the best epoch; the test perturbation is never seen in training.

```python
HELD_OUT = ["FOXA1", "OTUD5"]
TEST_PERT = "HHEX"

genotype, celltype = adata.obs["genotype"], adata.obs["celltype"]
is_control = (genotype == "WT").to_numpy()
is_held_out = genotype.isin(HELD_OUT).to_numpy()
is_test = (genotype == TEST_PERT).to_numpy()
# every validation cell type needs both control and perturbed cells
valid_target = is_held_out & celltype.isin(celltype[is_control].unique()).to_numpy()
valid_control = is_control & celltype.isin(celltype[valid_target].unique()).to_numpy()
train_indices = np.where(~is_held_out & ~is_test)[0]
valid_indices = np.where(valid_target | valid_control)[0]
```

## 4. Build and train

`from_adata` builds an untrained model. With `pert_sources` in the config it gets the unified encoder, and its
perturbation dictionary holds the dataset labels, `WT` and every gene of every table. `HHEX` is in ESM2 and GenePT,
so it already has a place in the model although it has no training cells.

```python
from perttf.model.hf import HFPerturbationTFModel

SAVE_DIR = "./pertTF_unified"

config = dict(
    # architecture
    layer_size=32, nlayers=2, nhead=4, dropout=0.2,
    GEPC=True, distribution="nb", explicit_zero_prob=True,
    fast_transformer=True, fast_transformer_backend="flash",
    # input genes: 3000 HVGs plus 1000 other genes sampled per cell
    sampling_mode="hvg", n_hvg=3000, non_hvg_size=1000,
    # perturbation prediction: a control cell plus a target perturbation -> the perturbed cell
    pert_sources=sources,
    next_cell_pred_type="pert",
    next_weight=1.0,          # loss on the predicted perturbed cell; 0 would ignore the perturbation
    this_weight=1.0,          # loss on reconstructing the input cell
    cell_type_classifier=False,
    genotype_classifier=False,
    perturbation_checkpoint_metric="pearson_delta_sample",
    # training
    epochs=20, batch_size=64, lr=1e-3, mask_ratio=0.15, amp=True, seed=42, log_interval=100,
)

model = HFPerturbationTFModel.from_adata(adata, config)
model.run_train(adata, train_indices=train_indices, valid_indices=valid_indices, save_dir=SAVE_DIR)
```

`run_train` restores the best epoch and saves the model with `save_pretrained`. The embedding tables are saved in
the checkpoint, so reloading downloads nothing:

```python
model = HFPerturbationTFModel.from_pretrained(SAVE_DIR, use_fast_transformer=True, fast_transformer_backend="flash")
```

## 5. Predict the unseen test perturbation

Control cells of the cell types that have `HHEX` cells are each asked to predict `HHEX`, then compared with the
real `HHEX` cells.

```python
from perttf.utils.pert_metrics import compute_metrics_from_anndata

query_control = is_control & celltype.isin(celltype[is_test].unique()).to_numpy()
adata_query = adata[query_control].copy()
adata_query.obs["genotype_next"] = TEST_PERT
adata_real = adata[is_test | query_control].copy()

adata_pred = model.predict_perturbations(adata_query, device="cuda")   # predictions in .obsm["mvc_next_expr"]
metrics = compute_metrics_from_anndata(adata_real, adata_pred, real_layer="X_binned",
                                       prediction_scale=model.prediction_scale)
print(pd.Series(metrics["aggregate"], name=TEST_PERT))
```

## 6. Predict a gene missing from the model

A gene in none of the tables is not in the model. Give its embedding at prediction time through an existing
table name; the trained projection of that table is used and the model is not changed.

```python
new_vector = ...   # shape (1, 5120): the ESM2 embedding of the new gene's protein
adata_query.obs["genotype_next"] = "NEW_GENE"
adata_pred_new = model.predict_perturbations(
    adata_query, device="cuda", query_sources={"esm2": (["NEW_GENE"], new_vector)})
```

## Details

- **How tables are combined.** Each table has its own small MLP (`Linear -> ReLU -> Linear`, then LayerNorm).
  A perturbation's embedding is the mean over the tables that contain it, followed by a shared LayerNorm. Tables are
  not concatenated. A perturbation in no table (here `HHEXhet`, for example) gets one learned vector, which is only
  informative if it has training cells.
- **The perturbation dictionary.** Dataset labels (and `WT`) first, then all other table genes in sorted order. The
  data loader, the encoder and the classifier share it. With `genotype_classifier=True` every table gene is a class,
  most without examples.
- **Control.** The control label (`WT`, or `control_label` in the config) has a zero input row in every table; its
  embedding can still become nonzero through the MLP biases. To force it to zero, build the model yourself (below)
  with `pert_pad_id=genotype_to_index["WT"]`.
- **Building the model yourself.** Without `from_adata`, make the dictionary with
  `build_perturbation_mapping(adata.obs["genotype"], sources, control_label="WT")` and pass the same object to
  `produce_training_datasets(..., genotype_to_index=...)` and to
  `HFPerturbationTFModel(genotype_to_index=..., pert_sources=sources, ...)`.
- **Embedding size.** `pert_dim` sets the embedding size (default `d_model`). With `perturbation_input=True` it must
  equal `d_model`.
- **Not supported** on this route: changing tables when reloading, and LoRA fine-tuning of a unified checkpoint.
  (`run_lora_pert_train(pert_mode="new", pert_source=...)` builds its own unified encoder on a pretrained model.)
- **Downloads.** Presets come from
  [`weililab/perturbation-embeddings`](https://huggingface.co/datasets/weililab/perturbation-embeddings) at a pinned
  revision (`DEFAULT_EMBEDDING_REVISION`; `revision=` selects another). They are read into memory without a cache,
  so each call downloads again (ESM2 ~400 MB, GenePT ~235 MB); keep the returned `sources`.
