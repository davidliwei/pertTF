
## Install Newest PertTF version 
**Newest Version of PertTF can use Flash Attention for at least > 2x speed up, works on GPU enabled machines**

This version still requires torchtext to be installed, because scgpt imports it, but it will not throw errors for torch > 2.3.0 version, this allows flash-attention to be utilized. pertTF no longer imports torchtext componenets, thus will work fine

### Colab Install
**NOTE for GOOGLE COLAB installation, check python, cuda and torch versions**
 - Tesla T4s can only use flash v1
 - V100s have no flash attn support (hopefully if SDPA layers are integrated into pertTF.py it can bring better performance)
 - A100/L4 can use flash v2

When installing flash v2 directly via `pip install flash-attn=xxx` as that will force compilation which takes forever, always install wheels
currently colab comes with pytorch 2.8 and cuda 12.6
The following might work (probably needs some tweeking)
```
pip install orbax==0.1.7
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install pandas scanpy "scvi-tools<1.0" numba datasets transformers==4.33.2 wandb cell-gears==0.1.2 torch_geometric pyarrow sentencepiece
```

### Clean Install on HPC
```bash
mamba create -n torch_flashv2 python=3.10 cuda-toolkit=12.8 'gxx>=6.0.0,<12.0' cudnn jupyter ipykernel ca-certificates matplotlib -y -c pytorch -c nvidia -c conda-forge
pip install torch==2.6.0 torchvision orbax==0.1.7 torchdata torchmetrics 
pip install pandas scanpy "scvi-tools<1.0" numba --upgrade "numpy<1.24" scib datasets transformers==4.33.2 wandb cell-gears==0.1.2 torch_geometric pyarrow sentencepiece

# Install Flash attention v1 with the following (2x speed up) uncommented
# pip install flash-attn==1.0.5 --no-build-isolation

## Install Flash attention v2 (1.5x fasters than v1) with the following (or find a wheel that fits your python, cuda and torch version in github)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 

```

#### INSTALL with FLASH ATTENTION v3 Beta (only on H200 architecture, 1.5-2x speed up over flash v2), 
```bash
mamba create -n pertTF_flashv3 python=3.10 'gxx>=6.0.0,<12.0' cudnn jupyter ipykernel ca-certificates matplotlib -y -c pytorch -c nvidia -c conda-forge
pip install torch==2.8.0 torchvision orbax==0.1.7 torchdata torchmetrics 
pip install pandas scanpy "scvi-tools<1.0" numba --upgrade "numpy<1.24" scib datasets==2.14.5 transformers==4.33.2 wandb cell-gears==0.0.2 torch_geometric pyarrow==15.0.2 ninja packages sentencepiece
# To install flash attention v3 (1.5-2x speed up over v2) git required > 30mins and > 400GB RAM with 32 CPUS
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install 
```
