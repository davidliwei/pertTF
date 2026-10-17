# Train from scorpus corpora

pertTF owns the corpus adapter. scorpus owns conversion, sparse reads,
original metadata, and temporary dataset composition. Installing/importing the
ordinary pertTF model does not require the optional corpus package.

The corpus route needs scorpus >= 0.3.0, which is not on PyPI. Install it from
GitHub, or put a checkout on PYTHONPATH:

```bash
pip install git+https://github.com/weili-lab/scorpus.git
# or
export PYTHONPATH=/path/to/perttf_corpus_data_adapter:/path/to/scorpus/src
```

Use the project's Python environment. Do large data preparation and training on
Slurm, not the login node. The included CPU integration test uses only 12
synthetic cells.

## Prepare data

Convert each source independently and combine its metadata as described in
scorpus's `docs/composable_corpora.md`. This example assumes `corpus`
has collective `perturbation`, `cell_type`, and `batch` columns and explicitly
aligned feature IDs.

```python
from perttf.model.corpus_data import produce_corpus_datasets

# config is the same training configuration used by wrapper_train.
# Required: next_cell_pred_type='pert', GEPC=True,
# input_emb_style='continuous', ps_weight=0, sampling_mode='uniform' (or
# 'expressed'; 'hvg' requires rankings already available in the corpus).
# max_seq_len includes CLS, matching pertTF's existing convention.
data_gen = produce_corpus_datasets(
    corpus,
    config,
    train_indices=train_rows,       # include the training controls
    valid_indices=validation_rows, # include the validation controls
    perturbation_column="perturbation",
    celltype_column="cell_type",
    batch_column="batch",
    control_label="ctrl",
    pert_sources=sources,           # {source_name: (perturbation_ids, features)}
    # pairing_columns=("donor",),  # if a collective donor column is available
    normalization_scale=1.0,
)
```

Pairs come from `default_pair_sampler` (`scorpus.pairing`). Training uses every
training cell once as a source: controls are paired with a random perturbed cell
of their context, perturbed cells with themselves (keeping their own label).
Validation visits every validation cell once as a target: perturbed targets get
a random control of their context, controls are paired with themselves. The
context is the same dataset and cell type, plus any supplied pairing columns.
Controls may overlap splits; perturbed target rows may not. Split checks and
missing-control checks run before training.

The same expanded perturbation mapping is used by both loaders, the encoder,
and classifier. Source-only IDs remain in the mapping, even without positive
training examples. For checkpoint fine-tuning, supply the checkpoint's
`genotype_to_index`, `cell_type_to_index`, `batch_to_index`, and `tokenizer_stoi`
rather than regenerating IDs; external
feature sources must be covered by that perturbation mapping.

## Construct the unified model and train

```python
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.train_function import wrapper_train

model = HFPerturbationTFModel(
    vocab=data_gen["vocab"],
    genotype_to_index=data_gen["genotype_to_index"],
    cell_type_to_index=data_gen["cell_type_to_index"],
    pert_sources=sources,
    control_label="ctrl",
    n_ps=0,
    do_mvc=True,
    input_emb_style="continuous",
    num_batch_labels=data_gen["num_batch_types"],
    # Supply the desired architecture and matching training settings here.
)
best_model = wrapper_train(model, config, data_gen, save_dir=save_dir, device=device)
```

Initialize wandb as in the existing training workflow (disabled mode is fine).
Create `save_dir` before calling the wrapper. The model must be on `device`.

## Expression and evaluation semantics

- Input/target values are `log1p(count / size_factor * normalization_scale)`.
- `sf` is the normalization multiplier, `normalization_scale / size_factor`,
  as expected by pertTF's count-distribution losses.
- Explicitly stored non-count corpora are rejected by this counts-based route.
- Original and recovered counts are recorded separately by the corpus. Recovery
  passing integer checks does not prove original absolute molecular counts.
- Expression vocabularies and perturbation vocabularies are separate. External
  perturbation feature IDs must match metadata labels; combination labels are
  not automatically decomposed into constituent gene embeddings.
- The corpus expression is continuous, not binned. Fine-tuning a model trained
  on binned input requires a deliberate preprocessing decision.

Full-expression training and validation losses honor source/target gene
availability masks. Unmeasured genes and special-token positions do not enter
those losses. Target expression is aligned to source token positions for
sampled-gene training; same-dataset pairing ensures those genes share a panel.

Perturbation metrics use only the tokenizable gene intersection across selected
validation datasets. Predictions and references use the **same ordered axis**,
and normalization is performed on that axis on both sides. Inspect
`data_gen['metric_gene_ids']` to see it. Reference moments are streamed from the
corpus; no full AnnData conversion is needed. The existing evaluator groups by
the supplied cell-type/context column and perturbation label. If dataset/donor
must define distinct evaluation contexts, prepare a composite context column.

Existing high-level `predict_perturbations` and `run_lora_train` remain
AnnData-based; this new route feeds `wrapper_train` directly. Small inference
subsets can still be exported to AnnData. The low-level paired-loader API is
documented in [CORPUS_PAIRED_LOADER.md](CORPUS_PAIRED_LOADER.md).

## Verification

With pertTF and scorpus importable:

```bash
python -m pytest tests/test_corpus_training.py -q
```

This converts two datasets with different gene panels and metadata names,
checks shared expanded IDs and same-dataset control pairs, trains one epoch
through the actual unified HF model and wrapper, saves a checkpoint, and runs
validation. It also checks loss masking for unavailable genes.
