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

The first way, pertTF is avaiable on PyPI and testPyPI. Use one of the following command to install pertTF:

```bash
pip install pertTF
```

or 

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

## Training optimizers and learning-rate schedules

Existing training configurations retain Adam and the once-per-epoch StepLR
schedule (`schedule_ratio` is the decay factor). For warmup followed by cosine
decay, add these settings to your training configuration:

```python
optimizer = "muon"          # "adam", "adamw", or "muon"
lr = 0.003                  # peak LR; choose for your experiment
muon_aux_lr_ratio = 0.01     # auxiliary Adam peak LR / Muon peak LR
weight_decay = 0.01         # used by AdamW and Muon; legacy Adam is unchanged
scheduler = "cosine"        # default: "step"
warmup_epochs = 0.5         # fractional epochs allowed; default: 0 (no warmup)
```

Muon requires the optional `muon` package exposing `MuonWithAuxAdam` and
`SingleDeviceMuonWithAuxAdam` (the [Keller Jordan implementation](https://github.com/KellerJordan/Muon)).
Trainable matrix weights use Muon (momentum 0.95); embeddings, biases and other
lower-dimensional parameters use auxiliary Adam (betas 0.9, 0.95). Embeddings
are identified by module type, including shared embedding weights. Frozen
parameters are excluded. Muon does not automatically select the base LR or its
schedule. Both groups receive the same LR multiplier; their ratio stays fixed.

The schedule assumes a fixed-length training loader and one optimizer update
per batch. At setup, `updates_per_epoch = len(train_loader)`,
`total_updates = epochs * updates_per_epoch`, and
`warmup_updates = int(warmup_epochs * updates_per_epoch)`. Positive warmup rises
from 1% to 100% of each group's peak LR, then cosine decays to zero. A warmup
that rounds to zero updates is omitted. Cosine requires
`0 <= warmup_epochs < epochs`; epoch-based StepLR does not use warmup.

Cosine advances after each successful optimizer update, not at epoch end.
AMP-overflow skips do not advance it, so skipped updates or early termination
can leave the schedule short of its planned endpoint. The main training wrapper
records resolved optimizer settings and cosine update counts in its existing
configuration artifacts. W&B logs `train/lr` and, for Muon, `train/lr_muon` and
`train/lr_aux_adam` for the update just attempted.

For direct calls to `create_optimizer_dict`, pass
`steps_per_epoch=len(train_loader)` when selecting cosine. The main wrapper
supplies this automatically. LoRA fine-tuning (`run_lora_cls_train`,
`run_lora_pert_train`) does not read these options from the checkpoint: it always
uses AdamW with the once-per-epoch StepLR (`schedule_ratio`). Separate
DAB/adversarial schedules are unchanged. This extension does not add
optimizer-state checkpoint resumption.

## Tutorials

All these tutorials can run on Google Colab. 

### Using the pertTF model

- [Unified perturbation embeddings](demos/tutorials/PERTURBATION_SOURCES.md): select HF or custom sources, combine available features, and learn fallback embeddings for uncovered perturbations.
- [Inference Tutorial](demos/tutorials/INFERENCE.md) and [Google Colab notebook](https://colab.research.google.com/drive/1KWWvRJJJOj9QXuF5zgddLHkdCx6KJnTa?usp=sharing): use the HuggingFace pertTF model to make inferences
- [LoRA fine tuning tutorial](demos/tutorials/LORA_FINETUNING.md) and [python notebook](demos/tutorials/lora_finetuning_tutorial.ipynb): use LoRA to fine tune the pertTF model

### Virtual screen and modeling composition change

- [Virtual CRISPR screens](demos/tutorials/virtual_pooled_screen.ipynb) and [Google Colab notebook](https://colab.research.google.com/drive/179y3UUTXvCHGpmc7lwrOgc8I3svhnD9Z?usp=sharing): perform virtual pooled CRISPR screens between two cell populations from pertTF model
- [Train pertTF to predict composition change](demos/tutorials/train_pertTF_with__lochNESS.ipynb) and [Google Colab notebook](https://colab.research.google.com/drive/1QiWBKbMOGJwthIqZMG-BYGbengguxQ7A?usp=sharing): calculate lochNESS scores, train pertTF to predict cell compositions from lochNESS scores. In addition, this notebook demonstrates the combination of external gene information (e.g., essential genes) to train the model.
- [Inference of composition changes using CRISPRi-based Perturb-seq](demos/tutorials/Inference_using_Perturbseq.ipynb) and [Google Colab notebook](https://colab.research.google.com/drive/1OiMNI7R1SoicIO9YRDoWFLki1e3pJcHL?usp=sharing): use pertTF (lochNESS-aware during training) to infer composition changes using the predicted lochNESS scores. Include the inference of essential genes and unseen genes (like CTNNB1).
- [virtual Perturb-seq](demos/tutorials/TUTORIAL_50gene_eval.ipynb) and [Google Colab notebook](https://colab.research.google.com/drive/1sOineIfuNHcSr10s84H_QuY9wm4CKvxU?usp=sharing): use pertTF to perform virtual Perturb-seq, and compare with the real CRISPRi-based Perturb-seq.

### Modeling other types of perturbation beyond genetic perturbation

- [Feature-conditioned perturbation encoder (zero-shot)](demos/tutorials/feature_pert_encoder_demo.ipynb) and [Google Colab notebook](https://colab.research.google.com/drive/1Dr8W8WxmKybCe79MJT27vCVO8q7vuAKG?usp=sharing): train pertTF with the `FeaturePertEncoder` (via the `pert_features` option) so it can embed perturbations beyond genetic perturbations, and never seen in training. Using a small LARRY cytokine dataset with one condition held out, the notebook runs a mini training/validation loop and shows that the held-out condition stays at random init under the default `PertLabelEncoder` but becomes informed under the feature-conditioned encoder.

## References

- our [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.03.12.711379v1).
- [HuggingFace website](https://huggingface.co/weililab) hosting models and datasets.
- [Li lab website](https://weililab.org)
