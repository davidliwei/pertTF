# Multi-Perturbation (High-MOI) Tutorial

Some Perturb-seq screens deliver several perturbations to the same cell, either as designed combinations (for example `A+B`) or as random guide sets at high multiplicity of infection (MOI). pertTF can train on, predict and screen such cells when you tell it how a label is split into perturbations. Without a delimiter, nothing changes: every label stays one opaque class.

### Label convention
A cell's perturbation is one string whose components are joined by a delimiter, for example `"A+B"`.

- The control label (`perturbation_control_value`, default `"WT"`) is dropped inside a combination: `"WT+A"` is `"A"`, and `"WT+WT"` is the control `"WT"`.
- Duplicate components collapse, and order does not matter. Labels are rewritten to a canonical form (components sorted and joined by the delimiter), so `"B+A"` becomes `"A+B"`.
- Components must be non-empty and have no leading or trailing whitespace; otherwise a `ValueError` lists the malformed labels.
- The model learns one embedding per single perturbation (plus the control), not one per combination.
- In legacy mode (no delimiter), labels are never split. If any label contains `+`, a `UserWarning` says it is treated as a single legacy label.

### Enable multi-perturbation mode
Set `pert_delimiter` in the training configuration:

```python
from perttf.model.config_gen import generate_config

# A complete minimal hyperparameter dict (the one used by tests/test_tutorial_smoke.py,
# shrunk for CPU). For real runs use the dict from your training tutorial instead.
hyperparameters = dict(
    seed=0, dataset_name="multi_pert_demo", do_train=True,
    GEPC=True, ecs_thres=0.0, ecs_weight=1.0, dab_weight=0.0,
    this_weight=0.0, next_weight=1.0, next_cell_pred_type="pert",
    cell_type_classifier=True, cell_type_classifier_weight=1.0,
    genotype_classifier=True, perturbation_classifier_weight=1.0,
    ps_weight=0.0, perturbation_input=False, CCE=False, mask_ratio=0.0,
    epochs=1, n_bins=0, lr=1e-3, batch_size=32,
    layer_size=16, nlayers=1, nhead=2, dropout=0.0,
    schedule_ratio=0.97, schedule_interval=1, log_interval=1000,
    fast_transformer=False, pre_norm=False, amp=False, do_sample_in_train=False,
    ADV=False, adv_weight=0, adv_E_delay_epochs=0, adv_D_delay_epochs=0, lr_ADV=1e-3,
    DSBN=False, per_seq_batch_sample=False, use_batch_label=False, explicit_zero_prob=False,
    n_hvg=60, mask_value=-1, pad_value=-2, pad_token="<pad>", cls_token="<cls>",
    sampling_mode="simple", non_hvg_size=0, fix_nonzero_prop=True, nonzero_prop=0.0,
    hvg_col="highly_variable", mvc_masked_train=False, distribution=None, use_ot=False,
    save_eval_interval=1, eval_expr_interval=2,
    perturbation_metric_min_cells=5, perturbation_checkpoint_metric="mvc_next",
)
config, _ = generate_config({**hyperparameters, "pert_delimiter": "+", "perturbation_unit": "set"},
                            wandb_mode="disabled")
print(config.pert_delimiter, config.pert_aggregation, config.perturbation_unit)
```

In practice, pass `pert_delimiter="+"` together with your usual hyperparameters (and the regime settings below). The delimiter is stored with the checkpoint, so `predict_perturbations` reads it from the saved training config and you do not need to repeat it. The perturbation head becomes a multi-label head: one independent sigmoid per single perturbation, trained with binary cross-entropy.

| Setting | Default | Meaning |
|---|---|---|
| `pert_delimiter` | `None` | Label delimiter such as `"+"`; `None` is legacy single-label mode |
| `pert_aggregation` | `"sum"` | How component embeddings are combined: `"sum"`, `"mean"` or `"sqrt"` (sum divided by the square root of the set size) |
| `perturbation_unit` | `"set"` | Whether validation metrics treat each full set as one perturbation (`"set"`) or each component separately (`"single"`) |
| `pert_pred_threshold` | `0.5` | Probability above which a perturbation joins the predicted set |
| `pert_bce_pos_weight` | `None` | Optional positive-class weight for the binary cross-entropy |
| `max_perts_per_cell` | `None` | Optional cap on the set size; labels above it raise an error, and it also caps predicted sets |

### Choose a regime
Pick the settings that match your screen:

| Regime | Typical data | Recommended settings |
|---|---|---|
| A: designed combinations | A modest set of intended combinations (`A+B`, `B+C`, ...) | Defaults: `perturbation_unit="set"`, `pert_aggregation="sum"` |
| B: random high-MOI | Roughly 5 to 10 random guides per cell | `perturbation_unit="single"`, `pairing_config={"target_sampling": "cell"}`, `pert_aggregation="sqrt"` or `"mean"` |

In regime B a plain sum of many normalised embeddings grows with the set size and can saturate the downstream MLP, so use `"sqrt"` or `"mean"`. These are the same settings used by the multi-perturbation tests in `tests/test_tutorial_smoke.py`:

```python
# `hyperparameters` is the complete dict from the snippet above (or from your training tutorial)
config, _ = generate_config(
    {**hyperparameters, "pert_delimiter": "+", "perturbation_unit": "single",   # regime B
     "pert_aggregation": "sqrt", "pairing_config": {"target_sampling": "cell"}},
    wandb_mode="disabled")
# then run the usual produce_training_datasets / build_model / wrapper_train steps
```

When you construct `PerturbationTFModel` (or `HFPerturbationTFModel`) by hand, pass the aggregation to the constructor yourself:

```python
model = PerturbationTFModel(..., pert_aggregation=config.get("pert_aggregation", "sum"))
```

The model reads `pert_aggregation` from its constructor, not from the training config, so omitting this kwarg silently trains with `"sum"` whatever the config says.

During `produce_training_datasets`, the split check adds an `unseen_combination` category to `data_gen["split_check"]["combination_type_counts"]`. It appears only when a delimiter is set and `perturbation_unit="set"`: a validation set whose components were all seen in training but never together. It also warns if fewer than half of the validation (context, set) groups have enough cells, in which case consider `perturbation_unit="single"`.

### Read the outputs
`eval_testdata` (and `predict_perturbations`) return an AnnData with:

- `obsm["X_pert_pred_probs"]`: independent sigmoid probabilities, one column per single perturbation. They do not sum to one, unlike the softmax output of legacy mode.
- `obs["predicted_genotype"]`: the predicted set, as a canonical label such as `"A+B"`. A perturbation joins the set when its probability exceeds `pert_pred_threshold`; if none does, the top class is used. The control never appears together with other classes.
- The set is capped by `max_perts_per_cell` if you set it, otherwise by the largest set seen in training (recorded in the config as `max_n_pert`), and if neither is available (for example a legacy checkpoint) by the number of perturbation classes only. It is never capped by the data being predicted.
- No `genotype_id` column is written in multi-perturbation mode.

```python
adata_eva = eval_testdata(model, data_gen["adata_sorted"], data_gen["gene_ids"],
                          train_data_dict=data_gen, config=config, device=device)
probs = adata_eva.obsm["X_pert_pred_probs"]
print(adata_eva.obs["predicted_genotype"].value_counts().head())
```

### Predict a combination of known genes
A model trained with a delimiter can predict combinations it never saw, as long as every component is a known single perturbation. Set the target in `genotype_next` and call `predict_perturbations` as in the [inference tutorial](INFERENCE.md):

```python
from perttf.model.hf import HFPerturbationTFModel

model = HFPerturbationTFModel.from_pretrained("path/to/multi_pert_checkpoint")
query = adata[adata.obs["genotype"] == "WT"].copy()
query.obs["genotype_next"] = "A+C"
result = model.predict_perturbations(query, prediction_mode="mean", device="cuda")
print(result.uns["perttf_prediction"]["pert_delimiter"])  # "+"
```

You can also use an existing single-perturbation checkpoint zero-shot on combinations of its known genes by passing `pert_delimiter="+"` explicitly; the model then combines the known gene embeddings according to its `pert_aggregation`. An explicit delimiter is accepted when the checkpoint has none, but it raises when it contradicts the delimiter stored in the checkpoint. On such a checkpoint, use the expression outputs (`mvc_next_expr` and the other predicted expression) only: in multi mode `X_pert_pred_probs` and `predicted_genotype` come from an independent sigmoid applied to a perturbation head that was trained with a softmax, so they are not calibrated probabilities (and the rows no longer sum to one).

The same label rules apply to `virtual_screen_src2dest` (`screen_genes` may contain sets such as `"A+B"`; entries with an unknown component are filtered out with a printed warning) and to `run_lora_pert_train`: the genotype labels in your data must all be known model labels, where sets are split into components with the config delimiter and control value before the check. `run_lora_pert_train` and `run_lora_cls_train` also take a `pert_delimiter` argument resolved the same way as in `predict_perturbations` (default: the checkpoint's value; an explicit value is accepted when the checkpoint has none, so you can fine-tune a single-perturbation checkpoint on combinations with `pert_delimiter="+"`; a contradicting value raises).

### lochNESS scores for multi-perturbation labels
`calculate_lonESS_score` matches whole labels by default, which finds almost nothing at high MOI (nearly every set label is unique). Pass the delimiter so that a cell "carries" `g` when `g` is one of its components:

```python
from perttf.model.composition_change_analysis import calculate_lonESS_score, lochness_by_perturbation

sc.pp.neighbors(adata, n_neighbors=300, key_added="nn_300")
adata.obs["lonESS_score_A"] = calculate_lonESS_score(
    adata, nn_name="nn_300", n_neighbors=300, target_genotype="A", pert_delimiter="+")
summary = lochness_by_perturbation(adata, nn_name="nn_300", pert_delimiter="+", min_cells=20)
```

- `target_genotype` must be a single component; without `target_genotype`, a cell's score is the mean of its per-component scores.
- **Check the counts before using per-cell scores.** A cell expects `n_neighbors × (fraction of cells carrying g)` carriers among its neighbours. In a genome-wide high-MOI screen that is ~0.1–0.2 even with 300 neighbours, so most per-cell scores sit at −1 and one carrier neighbour moves a cell to ≈ +9. The function warns when the expected count is below `min_expected_neighbors` (default 1). This count depends on the MOI and the number of targets, not on how many cells you have.
- `lochness_by_perturbation` pools the counts over all carriers of each perturbation (observed vs expected under random placement, `enrichment = observed / expected − 1`; the expectation uses the graph's actual neighbour counts, k − 1 per cell for `sc.pp.neighbors(n_neighbors=k)`). Pooling gives far more counts per perturbation than any single cell has, so prefer it at high MOI. Check it against random carrier sets of the same size before reading small enrichments.

### What raises
- A malformed label (empty or whitespace-padded component): `ValueError` listing the labels.
- Keys containing the delimiter in an existing single-label mapping (for example a legacy `"A+B"` class when you pass `pert_delimiter="+"`): `ValueError`. Such a checkpoint treats `A+B` as one class, so retrain with the delimiter or use labels without it.
- Components absent from the model mapping in `predict_perturbations`: `ValueError` naming the unknown labels.
- A `pert_delimiter` that disagrees with the checkpoint (`predict_perturbations`) or with the run's configuration (`virtual_screen_src2dest`): `ValueError`.
- Labels with more components than `max_perts_per_cell`: `ValueError`.
