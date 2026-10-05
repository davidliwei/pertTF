
# LoRA Fine-Tuning Tutorial

You can run this tutorial on Google Colab!  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidliwei/pertTF/blob/main/demos/tutorials/lora_finetuning_tutorial.ipynb)

### Preparation
```python
from huggingface_hub import hf_hub_download
import scanpy as sc
import anndata as ad
import numpy as np
from perttf.model.hf import HFPerturbationTFModel

# Download a demo dataset
hf_hub_download(repo_id="weililab/pancreatic_18clone", filename="./18clones_seurat.h5ad", repo_type='dataset', local_dir='./')
adata = sc.read_h5ad("./18clones_seurat.h5ad")
adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)

# Genotype labels use the pretrained model's names ('WT' for controls, e.g. '46_HHEX_het' -> 'HHEXhet')
def clean_genotype(gene_label):
    if gene_label == 'NA' or str(gene_label) == 'nan':
        return 'WT'
    parts = str(gene_label).split('_', 1)
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.replace('_', '').split('/')[0]

adata.obs['genotype'] = adata.obs['gene'].map(clean_genotype)

# Preprocess: log-normalized expression must be in a layer called 'X_binned'
adata.layers['X_binned'] = adata.X
# Find highly variable genes (model expects a highly_variable column in adata.var)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
```

### Load a pretrained model
```python
model = HFPerturbationTFModel.from_pretrained(
    'weililab/pertTF-tiny',
    use_fast_transformer=True,
    fast_transformer_backend='flash'
)
```

### Configure LoRA
```python
# Use the built-in helper to create a LoRA config with sensible defaults
lora_config = model.build_lora_config(
    r=8,              # rank of the low-rank matrices
    lora_alpha=32,     # scaling factor
    lora_dropout=0.1,  # dropout on LoRA layers
)

# By default, LoRA targets these layers:
#   qkv_proj, out_proj, linear1, linear2, decoder.fc.0, decoder.fc.2
# You can override this by passing target_modules=[...] to build_lora_config()
```

### Fine-tune
`run_lora_cls_train` trains a new head that predicts any cell-level column of `adata.obs` from the cell embedding, together with LoRA adapters on the transformer:
- `cls_col` — the column to predict, e.g. `'genotype'`, `'celltype'` or any other label; the head is new, so labels the pretrained model has never seen (here `CCDC6`) work too
- `cls_task` — `'classification'` or `'regression'` (numeric column)
- `this_weight` — weight of the expression reconstruction losses (masked-gene and GEPC) trained in the same pass; default `1.0`, `0` trains the head only

```python
from sklearn.model_selection import train_test_split

# Explicit 80/20 split, stratified by genotype, so the held-out cells can be used for evaluation
train_idx, valid_idx = train_test_split(np.arange(adata.n_obs), test_size=0.2, stratify=adata.obs['genotype'], random_state=42)

peft_model = model.run_lora_cls_train(
    adata=adata,
    cls_col='genotype',          # adata.obs column to predict
    cls_task='classification',   # or 'regression' for a numeric column
    train_indices=train_idx,
    valid_indices=valid_idx,
    epochs=5,
    batch_size=8,
    lr=1e-3,
    lora_config=lora_config,
    save_dir='my_lora_adapter',  # adapter weights saved here
)
```

The `run_lora_cls_train` method handles:
- Wrapping the base model with PEFT/LoRA (only the adapter weights and the new head are trained)
- Creating train/validation data loaders from your AnnData (a random `train_val_split` when no indices are given); for classification every validation class must also be in training
- Training with best-model checkpointing on the validation loss of the head
- Saving the adapter to `save_dir` (produces `adapter_config.json`, `adapter_model.safetensors` and `lora_heads.json`, which records the head's column, task and labels)

### Additional training options
```python
# For larger datasets or GPU memory constraints:
peft_model = model.run_lora_cls_train(
    adata=adata,
    cls_col='genotype',
    cls_task='classification',
    epochs=10,
    batch_size=16,
    lr=5e-4,
    lora_config=lora_config,
    save_dir='my_lora_adapter',
    amp=True,              # enable automatic mixed precision (fp16)
    seed=42,               # reproducibility seed
)

# Train the head only, without the expression reconstruction losses:
peft_model = model.run_lora_cls_train(adata=adata, cls_col='celltype', cls_task='classification', this_weight=0.0)
```

### Load a saved adapter for inference
```python
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.train_function import eval_testdata
import numpy as np

# Define a reusable evaluation wrapper that works with both base and PEFT models
def eval_wrapper(model, adata_test, expression=False, **kwargs):
    bm = model.get_base_model() if hasattr(model, 'get_base_model') else model
    res = eval_testdata(
        model,
        adata_test,
        None,
        train_data_dict={
            'genotype_to_index': bm.genotype_to_index,
            'vocab': bm.vocab,
            'cell_type_to_index': bm.cell_type_to_index
        },
        config=bm.training_config,
        mvc_full_expr=expression,
        predict_expr=expression,
        **kwargs,  # e.g. sample, sizefactor, sample_seed
    )
    return res
```

#### Classification task
```python
adata_valid = adata[valid_idx].copy()

# Zero-shot baseline: the pretrained heads, without fine-tuning (only genotypes the pretrained head knows)
base_model = HFPerturbationTFModel.from_pretrained('weililab/pertTF-tiny', use_fast_transformer=True, fast_transformer_backend='flash')
base_model.to('cuda')
base_model.eval()
known = adata_valid.obs['genotype'].isin(base_model.genotype_to_index).to_numpy()
adata_zero = eval_wrapper(base_model, adata_valid[known].copy())
adata_zero.obs['predicted_genotype']
adata_zero.obs['predicted_celltype']

# Fine-tuned head: load the base model (HF id or local checkpoint) with the adapter attached;
# stops if the base is not the checkpoint the adapter was trained on
peft_model = HFPerturbationTFModel.load_lora_adapter('weililab/pertTF-tiny', 'my_lora_adapter',
                                                     use_fast_transformer=True, fast_transformer_backend='flash')
peft_model.to('cuda')
peft_model.eval()

# predict_cls embeds the cells with the adapted model and applies the head
adata_valid.obs['predicted_genotype_ft'] = peft_model.predict_cls(adata_valid)
```

Each `load_lora_adapter` call returns an independent model, so several adapters can be loaded side by side. On an adapted model, the `predicted_genotype` / `predicted_celltype` columns of `eval_testdata` still come from the pretrained heads, which were not trained with the adapter; use `predict_cls` for the fine-tuned head.

#### Perturbation prediction (expression)
`run_lora_pert_train` fine-tunes a checkpoint trained for perturbation prediction: a control (`WT`) cell plus a target perturbation is mapped to the expression of a perturbed cell of the same cell type. The train/validation split is required, because it decides what validation measures. Here the perturbations are split three ways, all unseen in training: validation (`FOXA1`, `OTUD5`, picks the best epoch), test (`HHEX`, predicted at the end) and training (all others). Control cells may appear in every set, perturbed cells may not, and every validation cell type needs both.

The best epoch is chosen on **Pearson delta** (`checkpoint_metric='pearson_delta'`, the default): per (cell type, perturbation) group with at least 30 cells, the correlation over all genes between the predicted and observed change from control, averaged over groups. Predictions are sampled from the model's expression distribution and observed expression is read from the training input layer, so no precomputed DE genes are needed. Other options: `'mse_delta'`, `'ttest_de_overlap_at_n'`, `'ttest_de_direction_match'`, `'mvc_next'`.

```python
perturb_model = HFPerturbationTFModel.from_pretrained(
    'weililab/perttf-tiny-perturb-5k-nb',
    use_fast_transformer=True,
    fast_transformer_backend='flash'
)

# Keep genes in the model's vocabulary and perturbations the model knows (perturbation labels index its
# perturbation embedding); drop cells without an assigned guide ('NA'), which are not true controls
shared_genes = adata.var.index.isin(perturb_model.vocab.stoi)
known_pert = adata.obs['genotype'].isin(perturb_model.genotype_to_index)
adata_pert = adata[known_pert & (adata.obs['gene'] != 'NA'), shared_genes].copy()
adata_pert.layers['X_binned'] = adata_pert.X

# the perturb 5k model works on 5K HVGs that were in the training data, thus we want to use them all
if perturb_model.training_config['sampling_mode'] == 'hvg':
    adata_pert.var['highly_variable'] = True

# Unseen-perturbation split: two validation perturbations and one test perturbation, none of them in training.
# Validation also gets the control cells of the validation cell types.
HELD_OUT = ['FOXA1', 'OTUD5']
TEST_PERT = 'HHEX'
genotype, celltype = adata_pert.obs['genotype'], adata_pert.obs['celltype']
is_control = (genotype == 'WT').to_numpy()
is_held_out = genotype.isin(HELD_OUT).to_numpy()
is_test = (genotype == TEST_PERT).to_numpy()
valid_target = is_held_out & celltype.isin(celltype[is_control].unique()).to_numpy()
valid_control = is_control & celltype.isin(celltype[valid_target].unique()).to_numpy()
train_indices = np.where(~is_held_out & ~is_test)[0]
valid_indices = np.where(valid_target | valid_control)[0]

lora_config = perturb_model.build_lora_config(r=32, lora_alpha=32, lora_dropout=0.1)
peft_perturb = perturb_model.run_lora_pert_train(
    adata=adata_pert,
    train_indices=train_indices,
    valid_indices=valid_indices,
    epochs=5,
    batch_size=128,
    lr=1e-3,
    lora_config=lora_config,
    save_dir='lora_perturb_adapter',
    seed=42,
    checkpoint_metric='pearson_delta',  # default
)
```

Test query: control cells of the cell types that have `HHEX` cells each predict `HHEX` (set in the `genotype_next` column). The predictions are compared with the real `HHEX` cells using the same metrics, for the fine-tuned model and for the pretrained model without adapter.

```python
import pandas as pd
from perttf.utils.pert_metrics import compute_metrics_from_anndata, prediction_scale

query_control = is_control & celltype.isin(celltype[is_test].unique()).to_numpy()
adata_query = adata_pert[query_control].copy()
adata_query.obs['genotype_next'] = TEST_PERT
adata_real = adata_pert[is_test | query_control].copy()   # observed test cells and the same controls

pretrained = HFPerturbationTFModel.from_pretrained('weililab/perttf-tiny-perturb-5k-nb', use_fast_transformer=True, fast_transformer_backend='flash')
pretrained.to('cuda')
peft_perturb.eval()
pretrained.eval()

test_metrics = {}
for name, m in (('pretrained', pretrained), ('LoRA fine-tuned', peft_perturb)):
    # sample: draw expression from the predicted distribution; sizefactor: at each control cell's library size
    adata_pred = eval_wrapper(m, adata_query, expression=True, sample=True, sizefactor=True, sample_seed=42)
    bm = m.get_base_model() if hasattr(m, 'get_base_model') else m
    test_metrics[name] = compute_metrics_from_anndata(
        adata_real,
        adata_pred,                       # predicted expression in obsm['mvc_next_expr']
        real_layer='X_binned',            # observed expression: the training input layer
        prediction_scale=prediction_scale(bm.distribution),
    )

# per (cell type, perturbation) group with at least 30 cells, and averaged over groups
pd.concat({name: pd.DataFrame(res['per_group']).T for name, res in test_metrics.items()})
pd.DataFrame({name: res['aggregate'] for name, res in test_metrics.items()}).T
```
