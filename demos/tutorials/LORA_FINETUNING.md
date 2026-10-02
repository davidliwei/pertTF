
# LoRA Fine-Tuning Tutorial

You can run this tutorial on Google Colab!  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cDBYSzVtdGUxT4rTyeq1jPQPun9QQhI5)

### Preparation
```python
from huggingface_hub import hf_hub_download
import scanpy as sc
import anndata as ad
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
What is fine-tuned for classification is chosen with two arguments:
- `ft_cls='genotype'`, `'celltype'` or `['genotype', 'celltype']` — keep training the pretrained classifier head(s). The heads stay frozen and LoRA adapts the transformer that feeds them. Every label must be one the model already knows (an error lists unknown labels); a warning is shown when only part of the model's labels are present.
- `new_cls='<obs column>'` with `new_cls_task='classification'` or `'regression'` — train a new head from scratch on the cell embedding, for any cell-level target, including labels the model has never seen.

With neither (the default), LoRA adapts only the expression objective.

```python
# a) Fine-tune the pretrained heads on cells whose genotype the model knows (CCDC6 is not one of them)
adata_known = adata[adata.obs['genotype'].isin(model.genotype_to_index)].copy()
peft_model = model.run_lora_train(
    adata=adata_known,
    epochs=5,
    batch_size=8,
    lr=1e-3,
    train_val_split=0.2,       # 80/20 train/validation split
    lora_config=lora_config,
    ft_cls=['genotype', 'celltype'],
    save_dir='my_lora_adapter',  # adapter weights saved here
)

# b) Train a new genotype head on all cells, starting from a freshly loaded base model
#    (run_lora_train modifies the model it is called on)
model_new = HFPerturbationTFModel.from_pretrained('weililab/pertTF-tiny', use_fast_transformer=True, fast_transformer_backend='flash')
peft_new = model_new.run_lora_train(
    adata=adata,
    epochs=5,
    batch_size=8,
    lr=1e-3,
    lora_config=lora_config,
    new_cls='genotype',             # adata.obs column to predict
    new_cls_task='classification',  # or 'regression' for a numeric column
    save_dir='my_lora_new_head_adapter',
)
```

The `run_lora_train` method handles:
- Wrapping the base model with PEFT/LoRA (only adapter weights, and any new head, are trained)
- Creating train/validation data loaders from your AnnData
- Training with best-model checkpointing on the validation loss of the chosen heads (expression MSE by default)
- Saving the adapter to `save_dir` (produces `adapter_config.json`, `adapter_model.safetensors` and `lora_heads.json`, which records `ft_cls` and any new head with its labels)

### Additional training options
```python
# For larger datasets or GPU memory constraints:
peft_model = model.run_lora_train(
    adata=adata,
    epochs=10,
    batch_size=16,
    lr=5e-4,
    lora_config=lora_config,
    save_dir='my_lora_adapter',
    amp=True,              # enable automatic mixed precision
    amp_dtype='bf16',      # use bfloat16 (or 'fp16')
    log_interval=100,      # print training loss every N batches
    seed=42,               # reproducibility seed
)
```

### Load a saved adapter for inference
```python
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.train_function import eval_testdata
import numpy as np

# Define a reusable evaluation wrapper that works with both base and PEFT models
def eval_wrapper(model, adata_test, expression=False):
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
        predict_expr=expression
    )
    return res
```

#### Classification task
```python
# 1. Load the same base model used for fine-tuning
base_model = HFPerturbationTFModel.from_pretrained(
    'weililab/pertTF-tiny',
    use_fast_transformer=True,
    fast_transformer_backend='flash'
)
base_model.to('cuda')

# 2. Apply the saved LoRA adapter (rebuilds a new head first when the adapter has one)
peft_model = base_model.load_lora_adapter('my_lora_adapter')
peft_model.eval()

# 3. Run inference — returns predicted genotype and cell type
adata_eva = eval_wrapper(peft_model, adata_known)
adata_eva.obs['predicted_genotype']
adata_eva.obs['predicted_celltype']
```

#### Classification with a new head
```python
base_new = HFPerturbationTFModel.from_pretrained('weililab/pertTF-tiny', use_fast_transformer=True, fast_transformer_backend='flash')
base_new.to('cuda')
peft_new = base_new.load_lora_adapter('my_lora_new_head_adapter')
peft_new.eval()

# the new head predicts from the cell embedding that eval_testdata stores in obsm['X_scGPT']
adata_eva = eval_wrapper(peft_new, adata)
adata_eva.obs['predicted_genotype_new'] = base_new.predict_new_cls(adata_eva.obsm['X_scGPT'])
```

#### Perturbation prediction (expression)
```python
perturb_model = HFPerturbationTFModel.from_pretrained(
    'weililab/pertTF-perturb_5k_mvc_only',
    use_fast_transformer=True,
    fast_transformer_backend='flash'
)

# the perturb 5k model works on 5K HVGs that were in the training data, thus we want to use them all
# evaluation will subset your adata to the 5000 HVGs before inference
if perturb_model.training_config['sampling_mode'] == 'hvg': 
    adata.var.highly_variable = True

# to initiate perturbations set target perturbations using the genotype_next column
# assuming all cells in adata are non-perturbed (perturbed randomly for demo)
adata.obs['genotype_next'] = np.random.choice(['FOXA2', 'PDX1'], adata.shape[0])

# Fine-tune with LoRA
lora_config = perturb_model.build_lora_config(r=8, lora_alpha=32, lora_dropout=0.1)
peft_perturb = perturb_model.run_lora_train(
    adata=adata,
    epochs=5,
    batch_size=8,
    lr=1e-3,
    lora_config=lora_config,
    save_dir='lora_perturb_adapter',
)

# to initiate perturbations set target perturbations using the genotype_next column
# assuming all cells in adata are non-perturbed (perturbed randomly for demo)
adata.obs['genotype_next'] = np.random.choice(['FOXA2', 'PDX1'], adata.shape[0])

# Run with expression=True to get predicted expression values
adata_eva = eval_wrapper(peft_perturb, adata, expression=True)

# corresponding perturbations are found in:
adata_eva.obs['genotype_next']
# perturbed expressions are found:
adata_eva.obsm['mvc_next_expr']
```
