# Train from scorpus corpora

This tutorial goes from h5ad files to a trained pertTF model without loading
the expression into memory: the data stay on disk as scorpus corpora and are
read batch by batch during training. pertTF owns the corpus adapter; scorpus
owns conversion, sparse reads, metadata and combining datasets. The ordinary
pertTF model does not require scorpus.

The corpus route needs scorpus >= 0.6.0, which is not on PyPI. Install it from
GitHub, or put a checkout on PYTHONPATH:

```bash
pip install "scorpus @ git+https://github.com/weili-lab/scorpus.git@v0.6.0"
# or
export PYTHONPATH=/path/to/perttf_corpus_data_adapter:/path/to/scorpus/src
```

Run data preparation and training on a compute node, not a login node.

The steps:

1. Convert each h5ad to a corpus
2. Choose HVGs (for `sampling_mode="hvg"`)
3. Harmonize metadata and combine datasets
4. Split cells into training and validation
5. Configure and build the loaders (pairing and dataloaders)
6. Build the model and train
7. After training

The examples use two datasets: `screen.h5ad`, a perturbation screen with
perturbations in obs `target_gene` (controls `non-targeting`) and sequencing
lanes in `batch`; and `context.h5ad`, control cells of another cell context.

## 1. Convert each h5ad

```python
from scorpus import from_h5ad

from_h5ad("screen.h5ad", "corpora/screen.corpus")
from_h5ad("context.h5ad", "corpora/context.corpus")
```

`X` (or a layer, `layers="counts"`) must hold raw counts. Each corpus stores the
counts on disk, the h5ad's obs and var, and each cell's total count as the obs
column `scorpus_total_counts`, from which pertTF computes size factors. The
output directory must not exist. Nothing is normalized at this step.

## 2. Choose HVGs

Only needed for `sampling_mode="hvg"`. A dataset's HVGs are exactly the `True`
genes of its var column `highly_variable`; choose them with any method on an
AnnData export of the corpus and save the column next to the corpus:

```python
import numpy as np
import scanpy as sc
from scorpus import load_corpus

for name in ("screen", "context"):
    corpus = load_corpus(f"corpora/{name}.corpus")
    adata = corpus.to_anndata()     # raw counts in X; to_anndata_lazy() for large datasets
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=2000)
    corpus.var["highly_variable"] = adata.var["highly_variable"].to_numpy()
    corpus.save_ex_var("hvg2000")   # ex_var_hvg2000.parquet next to the corpus
```

- `seurat_v3` works on raw counts. The `seurat` and `cell_ranger` flavors expect
  log-normalized values: run `adata.X = adata.X.astype(np.float32)`,
  `sc.pp.normalize_total(adata, target_sum=1e4)` and `sc.pp.log1p(adata)` first.
  This normalization only serves to choose HVGs; nothing normalized is stored,
  because pertTF normalizes while building each batch (step 5).
- The export keeps the corpus's gene order, so the column can be copied as is.
  The corpus's own `var.parquet` is never changed; several HVG sets can be saved
  under different names and chosen when loading.
- A dataset without HVGs gets only `non_hvg_size` other genes per cell in
  `"hvg"` mode. If the h5ad's var already has `highly_variable`, conversion keeps
  it and the corpus uses it without this step.

## 3. Harmonize metadata and combine datasets

pertTF needs one perturbation, one cell-type and one batch column across all
datasets, with one control label. Build them per dataset from its own obs, then
combine the datasets into one corpus:

```python
import pandas as pd
from scorpus import concat, load_corpus

CONTROL = "non-targeting"
specs = {   # per dataset: cell type and batch from its obs
    "screen": (lambda o: "H1", lambda o: o["batch"]),    # one cell line
    "context": (lambda o: o["context"], lambda o: "none"),
}
members, obs = {}, {}
for name, (cell_type, batch) in specs.items():
    member = load_corpus(f"corpora/{name}.corpus", ex_var="hvg2000")
    o = member.obs.copy()
    o["perturbation"] = o["target_gene"].astype(str)     # map other control labels to CONTROL here
    o["cell_type"] = pd.Series(cell_type(o), index=o.index).astype(str)
    o["batch"] = name + ":" + pd.Series(batch(o), index=o.index).astype(str)
    members[name], obs[name] = member, o
corpus = concat(
    members, obs=obs,
    obs_columns={column: {name: column for name in members} for column in ("perturbation", "cell_type", "batch")},
    feature_key={name: "index" for name in members},     # genes matched by var_names
)
```

- `feature_key` decides which genes are the same across datasets: `"index"`
  (var_names) or a var column per dataset, e.g. Ensembl IDs. Use an ID every
  dataset shares; genes a dataset did not measure are left out of its cells'
  samples and losses.
- Each dataset keeps its own gene panel, HVGs and size factors.
- Give all three columns the same type in every dataset (here strings).

## 4. Split cells into training and validation

`train_indices` and `valid_indices` are rows of the combined corpus. For
example, hold out 10% of each screen's perturbations with up to 2,000 of its
controls:

```python
import numpy as np

meta = corpus.metadata_index.df.select(["dataset_index", "perturbation"]).to_pandas()
rng = np.random.default_rng(0)
train, valid = [], []
for _, frame in meta.groupby("dataset_index"):
    is_control = frame["perturbation"] == CONTROL
    perturbations = frame.loc[~is_control, "perturbation"].unique()
    if len(perturbations) == 0:      # controls only: training only
        train.append(frame.index.to_numpy())
        continue
    held_out = frame["perturbation"].isin(rng.choice(perturbations, max(1, len(perturbations) // 10), replace=False))
    controls = frame.index[is_control].to_numpy()
    valid.append(np.concatenate([frame.index[held_out].to_numpy(),
                                 rng.choice(controls, min(2000, len(controls)), replace=False)]))
    train.append(frame.index[~held_out].to_numpy())   # the other perturbations and all controls
train_rows, valid_rows = np.sort(np.concatenate(train)), np.sort(np.concatenate(valid))
```

- Controls may be in both splits; perturbed cells may not.
- Every validation context (dataset and cell type, plus any `pairing_columns`)
  needs both controls and perturbed cells, so a dataset of controls only can
  only be used for training.
- Perturbation metrics need at least `perturbation_metric_min_cells` cells
  (default 30) per validation group.
- Rows with a missing perturbation, cell type or batch must be left out.

## 5. Configure and build the loaders

The configuration is the one `wrapper_train` reads, passed as a wandb config.

```python
import torch
import wandb
from perttf.model.corpus_data import produce_corpus_datasets

N_HVG, NON_HVG = 2000, 1000
parameters = dict(
    # Required by the corpus route
    next_cell_pred_type="pert", GEPC=True, input_emb_style="continuous", ps_weight=0.0,
    # Gene sampling, with pertTF's quotas
    sampling_mode="hvg", non_hvg_size=NON_HVG, max_seq_len=N_HVG + NON_HVG + 1,   # + 1 for CLS
    batch_size=32, mask_ratio=0.15, pad_token="<pad>", pad_value=-2, mask_value=-1,
    # Training
    epochs=1, lr=1e-4, schedule_ratio=0.9, amp=True, seed=0, log_interval=50, do_train=True,
    dab_weight=0.0, ADV=False, use_batch_label=False, perturbation_input=False,
    next_weight=1.0, this_weight=1.0, ecs_thres=0.0, CCE=False,
    explicit_zero_prob=False, mvc_masked_train=False,
    distribution="nb",              # negative binomial expression decoder (also passed to the model)
    cell_type_classifier=True, genotype_classifier=True,
    cell_type_classifier_weight=0.1, perturbation_classifier_weight=0.1,
    # Validation metrics
    perturbation_metric_min_cells=30, perturbation_metric_mode="mean",
    perturbation_metric_target_sum=10000.0, dataset_name="screen_and_context",
)
device = torch.device("cuda")
run = wandb.init(mode="disabled", config=parameters)
data_gen = produce_corpus_datasets(
    corpus, run.config, train_indices=train_rows, valid_indices=valid_rows,
    perturbation_column="perturbation", celltype_column="cell_type", batch_column="batch",
    control_label=CONTROL,
    # pert_sources=sources,         # {source_name: (perturbation_ids, features)}
    # pairing_columns=("donor",),   # pair only within these extra columns too
    target_sum=10000.0,             # each cell scaled to 10,000 counts; None: its dataset's median
    num_workers=4,                  # expression is read in worker processes
    batches_per_read=max(1, 1024 // parameters["batch_size"]),   # ~1,000 pairs per read
    device=device,                  # batches are built on the training GPU
    # build_in_workers=True,        # or on the CPU in the workers (frees the GPU)
)
```

- `sampling_mode`: `"uniform"`, `"expressed"` (`nonzero_prop`,
  `fix_nonzero_prop`) or `"hvg"` (`non_hvg_size`), with the quotas of pertTF's
  collator. In `"hvg"` mode a cell gets at most `max_seq_len - 1 - non_hvg_size`
  of its dataset's HVGs (a random subset if it has more), so datasets with
  different numbers of HVGs share one `max_seq_len`.
- Pairing (`default_pair_sampler`, `scorpus.pairing`), within the same dataset
  and cell type plus any `pairing_columns`: training uses every training cell
  once per epoch as a source; a control is paired with a random perturbed cell
  of its context, a perturbed cell with itself (keeping its own label), and a
  control whose context has no perturbation with itself. Validation uses every
  validation cell once as a target: a perturbed target gets a random control of
  its context, a control is paired with itself. Split and missing-control
  checks run here.
- Workers: each read covers `batches_per_read` batches; on network storage a
  read of about 1,000 pairs is efficient. With `num_workers > 0` the workers are
  started with `spawn`, which imports your script again in each worker: put the
  script's code under `if __name__ == "__main__":`. For pairings other than the default,
  use the loader directly: [CORPUS_PAIRED_LOADER.md](CORPUS_PAIRED_LOADER.md).
- IDs: the perturbation mapping is shared by both loaders, the encoder and the
  classifier; source-only IDs stay in it. To fine-tune a checkpoint, pass its
  `genotype_to_index`, `cell_type_to_index`, `batch_to_index` and
  `tokenizer_stoi` instead of letting new ones be made; external feature
  sources must be covered by that perturbation mapping.

## 6. Build the model and train

```python
from pathlib import Path
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.train_function import wrapper_train

model = HFPerturbationTFModel(
    n_pert=len(data_gen["genotype_to_index"]), nlayers_pert=2, n_ps=0,
    d_model=128, nhead=4, d_hid=256, nlayers=2, dropout=0.1,
    vocab=data_gen["vocab"], pad_token="<pad>", pad_value=-2,
    do_mvc=True, do_dab=False, use_batch_labels=False, num_batch_labels=data_gen["num_batch_types"],
    domain_spec_batchnorm=False, n_bins=0, ecs_threshold=0.0, explicit_zero_prob=False,
    use_fast_transformer=False, pre_norm=False,
    n_cls=len(data_gen["cell_type_to_index"]), nlayers_cls=2, input_emb_style="continuous",
    genotype_to_index=data_gen["genotype_to_index"], cell_type_to_index=data_gen["cell_type_to_index"],
    control_label=CONTROL, distribution=run.config["distribution"],
    # pert_sources=sources,
).to(device)
save_dir = Path("runs/screen_and_context")   # a Path
save_dir.mkdir(parents=True, exist_ok=True)
best_model = wrapper_train(model, run.config, data_gen, save_dir=save_dir, device=device)
```

The architecture above is small, for a first run; choose sizes as for any
pertTF model (see the training options guide).

## 7. After training

- `save_dir` holds `best_model.pt`, per-epoch checkpoints, `vocab.pt`, `config.json`
  and `running_parameters.pt` (mappings and the adapter settings under
  `corpus_config`).
- Each epoch, validation logs perturbation metrics on the held-out
  perturbations: `pearson_delta_mean`, `mse_delta_mean`,
  `ttest_de_overlap_at_n_mean`, `ttest_de_direction_match_mean`; the best
  checkpoint is chosen by `mvc_next`. They use the genes that every validation
  dataset measured and that have a token (`data_gen["metric_gene_ids"]`).
- To evaluate a model on the validation loader again:

  ```python
  from perttf.model.train_function import evaluate

  result = evaluate(best_model, data_gen["valid_loader"], run.config, data_gen["vocab"], device=device,
                    cell_type_to_index=data_gen["cell_type_to_index"],
                    genotype_to_index=data_gen["genotype_to_index"], expr_prediction_mode="mean")
  result["losses"], result["predicted_groups"]
  ```

- `predict_perturbations` and `run_lora_train` are AnnData-based. Cells of one
  dataset can be exported with `corpus.to_anndata(global_row_indices=rows)`;
  the model was trained on `log1p(count / total × target_sum)` values.

## How values are normalized

- Input and target values are `log1p(count / size_factor)`, computed while each
  batch is built. `size_factor` is a cell's `scorpus_total_counts` over
  `target_sum` (default 10,000, the same for every dataset) or, with
  `target_sum=None`, over the median positive total of its dataset's cells in
  that loader's rows (training and validation then each use their own median);
  1 for a cell without counts.
- `sf` is the normalization multiplier, `1 / size_factor`. With a count
  distribution (`distribution="nb"` above), the decoder divides its prediction by
  `sf` to get the cell's expected counts, and the loss turns the target values
  back into the cell's counts with `expm1(value) / sf`.
- Corpora stored without counts are rejected by this route.
- Original and recovered counts are recorded separately by the corpus. Recovery
  passing integer checks does not prove original absolute molecular counts.
- Expression vocabularies and perturbation vocabularies are separate. External
  perturbation feature IDs must match metadata labels; combination labels are
  not automatically decomposed into constituent gene embeddings.
- The corpus expression is continuous, not binned. Fine-tuning a model trained
  on binned input requires a deliberate preprocessing decision.

Full-expression losses use only the genes a cell's dataset measured; special
positions do not enter them. The target cell is read at the source cell's
genes; same-dataset pairing ensures they share a panel. Perturbation metrics
normalize predictions and references on the same gene axis; reference moments
are streamed from the corpus. The evaluator groups by cell type and
perturbation; if dataset or donor should define separate evaluation contexts,
build a combined cell-type column.

## Verification

With pertTF and scorpus importable:

```bash
python -m pytest tests/test_corpus_training.py tests/test_corpus_adapter.py -q
```

`test_corpus_training.py` converts two datasets with different gene panels and
metadata names, checks shared IDs and same-dataset control pairs, trains one
epoch through the HF model and `wrapper_train`, saves a checkpoint and runs
validation.
