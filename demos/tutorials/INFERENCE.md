
# Inference Tutorial
### Preparation
```python
# first we load in some packages
from huggingface_hub import hf_hub_download, login
import scanpy as sc
import numpy as np
from pertTF.perttf.model.train_function import eval_testdata
from pertTF.perttf.model.hf import HFPerturbationTFModel
import anndata as ad

# Now download from huggingface a demo dataset
# login(token='YOUR_HF_TOKEN') #try this if download fails
hf_hub_download(repo_id="weililab/pancreatic_18clone", filename="./18clones_seurat.h5ad", repo_type='dataset', local_dir='./')
adata = sc.read_h5ad("./18clones_seurat.h5ad")
adata = ad.AnnData(X=adata.raw.X, obs=adata.obs, var=adata.raw.var)

# preprocess the data 
# log normalized expression data needs to be moved to layer under name 'X_binned'
adata.layers['X_binned'] = adata.X
# find highly variable genes (model expects a highly_variable col in adata.var)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
```

### define simple evaluation wrapper
```python
def eval_wrapper(model, adata_test, expression = False):
    res = eval_testdata(
        model, 
        adata_test, 
        None,
        train_data_dict={
        'genotype_to_index': model.genotype_to_index, 
        'vocab': model.vocab,
        'cell_type_to_index': model.cell_type_to_index
        },
        config = model.training_config, 
        mvc_full_expr = expression, 
        predict_expr = expression)
    return res
```

### classification task
```python
classify_model = HFPerturbationTFModel.from_pretrained('weililab/pertTF-tiny', use_fast_transformer = True, fast_transformer_backend = 'flash')
classify_model.to('cuda')
adata_eva = eval_wrapper(classify_model, adata)
# predicted genotype and celltypes are found here
adata_eva.obs['predicted_genotype']
adata_eva.obs['predicted_celltype']
```

### cell composition scoring
```python
lochness_model = HFPerturbationTFModel.from_pretrained('weililab/pertTF_virtual_screen_lochness', use_fast_transformer = True, fast_transformer_backend = 'flash')
lochness_model.to('cuda')
# under lochness mode, we assign a perturbation (randomly for demo)
adata.obs['genotype_next'] = np.random.choice(['FOXA2', 'PDX1'], adata.shape[0])

# inference
adata_eva = eval_wrapper(lochness_model, adata)

# predicted lochness score / lochness score after perturbation can be seen here
adata_eva.obsm['ps_pred']
adata_eva.obsm['ps_pred_next']
```

### perturbation task
```python
perturb_model = HFPerturbationTFModel.from_pretrained('weililab/pertTF-perturb_5k_mvc_only', use_fast_transformer = True, fast_transformer_backend = 'flash')
perturb_model.to('cuda')

# the perturb 5k model works on 5K HVGs that were in the training data, thus we want to use them all
# evaluation will subset your adata to the 5000 HVGs before inference
if perturb_model.training_config['sampling_mode'] == 'hvg': 
    adata.var.highly_variable = True

# to initiate perturbations set target perturbations using the genotype_next column
# assuming all cells in adata are non-perturbed (perturbed randomly for demo)
adata.obs['genotype_next'] = np.random.choice(['FOXA2', 'PDX1'], adata.shape[0])

# inference
adata_eva = eval_wrapper(perturb_model, adata, expression = True)

# corresponding perturbations are found in:
adata_eva.obs['genotype_next']
# perturbed expressions are found:
adata_eva.obsm['mvc_next_expr']
```
