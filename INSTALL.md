# INSTALLATION OF THE VARIOUS PACKAGES:
Install each package in its own environments is the best way to do it

## Install scGPT (only scGPT)
Installing scGPT can be quite awful, so....lets get to it (DISCLAIMER: the following is based on a bunch of github issues from back in 2023 and 2024, the hardest part was installing flash attention, afterwards it was resolving torchtext)

```bash
mamba (or conda) create -n scGPT_env python=3.10 cudatoolkit=11.7 cudatoolkit-dev 'gxx>=6.0.0,<12.0' cudnn r-base r-devtools
mamba activate scGPT_env
pip install torch==1.13.0 torchvision torchtext orbax==0.1.7 torchdata torchmetrics pyro-ppl lightning
pip install --no-deps scgpt
pip install ipykernel pandas scanpy "scvi-tools<1.0" numba --upgrade "numpy<1.24" scib datasets==2.14.5 transformers==4.33.2 wandb cell-gears==0.0.2 torch_geometric

```
This should resolve all conflicts between torch versions, cuda versions for scGPT, the main issue is that of torchtext being deprecated, other torch versions < 2.3.0 might work 


## Install Newest PertTF version 
**Newest Version of PertTF can use Flash Attention for at least > 2x speed up, works on GPU enabled machines**

This version still requires torchtext to be installed, because scgpt imports it, but it will not throw errors for torch > 2.3.0 version, this allows flash-attention to be utilized. pertTF no longer imports torchtext componenets, thus will work fine

## Clean Install
```bash
mamba create -n pertTF_flashv2_1 cuda-toolkit=12.8 python=3.10 'gxx>=6.0.0,<12.0' cudnn jupyter ipykernel ca-certificates matplotlib -y -c pytorch -c nvidia -c conda-forge
pip install torch==2.6.0 torchvision orbax==0.1.7 torchdata torchmetrics 
pip install --no-deps scgpt 
pip install --no-deps torchtext==0.5.0 # pointless but REQUIRED for scgpt import
pip install pandas scanpy "scvi-tools<1.0" numba --upgrade "numpy<1.24" scib datasets==2.14.5 transformers==4.33.2 wandb cell-gears==0.0.2 torch_geometric pyarrow==15.0.2 sentencepiece

# Install Flash attention v1 with the following (2x speed up) uncommented
# pip install flash-attn==1.0.5 --no-build-isolation

## Install Flash attention v2 (1.5x fasters than v1) with the following (or find a wheel that fits your python, cuda and torch version in github)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 

```

## INSTALL with FLASH ATTENTION v3 Beta (only on H200 architecture, 1.5-2x speed up over flash v2), 
```bash
mamba create -n pertTF_flashv3 cuda-toolkit=12.8 python=3.10 'gxx>=6.0.0,<12.0' cudnn jupyter ipykernel ca-certificates matplotlib -y -c pytorch -c nvidia -c conda-forge
pip install torch==2.8.0 torchvision orbax==0.1.7 torchdata torchmetrics 
pip install --no-deps scgpt 
pip install --no-deps torchtext==0.5.0 # pointless but needed but needed for scgpt import
pip install pandas scanpy "scvi-tools<1.0" numba --upgrade "numpy<1.24" scib datasets==2.14.5 transformers==4.33.2 wandb cell-gears==0.0.2 torch_geometric pyarrow==15.0.2 ninja packages sentencepiece
# To install flash attention v3 (1.5-2x speed up over v2) git required > 30mins and > 400GB RAM with 32 CPUS
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install 
```
