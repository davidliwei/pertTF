
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
```python
peft_model = model.run_lora_train(
    adata=adata,
    epochs=5,
    batch_size=8,
    lr=1e-3,
    train_val_split=0.2,       # 80/20 train/validation split
    lora_config=lora_config,
    save_dir='my_lora_adapter',  # adapter weights saved here
)

# Training prints validation MSE after each epoch and automatically
# restores the best checkpoint (lowest validation MSE) at the end.
```

The `run_lora_train` method handles:
- Wrapping the base model with PEFT/LoRA (only adapter weights are trained)
- Creating train/validation data loaders from your AnnData
- Training with best-model checkpointing based on validation MSE
- Saving the adapter to `save_dir` (produces `adapter_config.json` and `adapter_model.safetensors`)

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
from peft import PeftModel
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

# 2. Apply the saved LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, 'my_lora_adapter')
peft_model.eval()

# 3. Run inference — returns predicted genotype and cell type
adata_eva = eval_wrapper(peft_model, adata)
adata_eva.obs['predicted_genotype']
adata_eva.obs['predicted_celltype']
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
