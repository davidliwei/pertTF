<img src="assets/img/LOGO.png" alt="" width="800"/>

**pertTF is a transformer model designed for single-cell perturbation predictions.**
-----
# Installation
## Prerequisite environment
pertTF require `torch > 2.3.0` and `cuda > 12.0` 

best way to install is to set up a seperate envrionment with conda or mamba
```bash
# create independent environment (recommonded)
mamba create -n pertTF_env python=3.10 cuda-toolkit=12.8 'gxx>=6.0.0,<12.0' cudnn ca-certificates -y -c pytorch -c nvidia -c conda-forge

# pip install required packages
# it is best to install torch == 2.6.0 to match the flash attention compiled wheel below
# higher versions of torch may present difficulties for installing flash attention 2 
pip install torch==2.6.0 torchvision orbax==0.1.7 torchdata torchmetrics pandas scanpy numba --upgrade "numpy<1.24" datasets transformers==4.33.2 wandb torch_geometric pyarrow sentencepiece huggingface_hub omegaconf
```
flash attention is strongly recommended for training or finetuning

```bash
# flash attention 2 installation
#check ABI true/false first
python -c "import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# install appropraite version (the example below is for ABI=FALSE)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 

# flash attention 3 installation (recommended for torch > 2.6.0 and hopper GPUs)
# To install flash attention v3 (1.5-2x speed up over v2) requires > 30mins, > 400GB RAM, 32 CPUS (aim for more than this)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install 
```
## pertTF installation
You can install and use pertTF in two ways.

The first way, pertTF is avaiable on PyPI. Use the following command to install pertTF:
```bash
pip install -i https://test.pypi.org/simple/ pertTF
```
The second way is suitable for you to run the most recent pertTF source code. First, fork our pertTF GitHub repository:

```bash
git clone https://github.com/davidliwei/pertTF.git
```
Then, in your python code, you can directly use the pertTF package:
```python
import sys
sys.path.insert(0, '/content/pertTF/')
```
-----------------------

## [Inference Tutorial](demos/tutorials/INFERENCE.md)
