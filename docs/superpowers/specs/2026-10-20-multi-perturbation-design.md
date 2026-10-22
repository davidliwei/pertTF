# Multi-perturbation (high-MOI Perturb-seq) support for pertTF — design

Date: 2026-10-20
Status: approved direction (2026-10-20), nothing implemented yet. Revised once after an
independent builder check (calls marked **[builder-check call]**) and once after owner
feedback (marked **[owner]**).

## 1. Intent

High-MOI Perturb-seq assigns several guide RNAs to one cell, so one cell can carry
several perturbations at once (e.g. `GENE_A+GENE_B+GENE_C`). pertTF currently models
each cell's perturbation as exactly one categorical label. The goal is to let a cell
carry a *set* of perturbations, in both the source cell (`genotype`) and the target
condition (`genotype_next`), while:

- leaving every existing single-perturbation workflow, checkpoint and tutorial working
  unchanged when the feature is off;
- keeping the code change small and localized;
- making the representation compositional, so a combination never seen in training can
  still be encoded from its constituent single-perturbation embeddings.

### Goals
- Train and predict with cells carrying 1..k perturbations.
- Reuse existing HF checkpoints (trained with single perturbations) for multi-perturbation
  inference and LoRA fine-tuning without re-architecting them.
- Keep `genotype_to_index` a vocabulary of *single* perturbations plus the control label.
- Keep pairing, loss, metrics, split checks and virtual-screen utilities usable in both
  data regimes described in §1.1.

### Non-goals
- Learning gene–gene interaction terms beyond what the downstream MLPs (`PertExpEncoder`,
  decoders) can learn from an additive perturbation embedding.
- Attention-based or order-aware set encoders for the perturbation set (future work, §8).
- Guide-level modelling (several guides for one gene are collapsed to the gene by the
  user's preprocessing, exactly as today).
- Renaming the `obs` columns (`genotype`, `genotype_next`) or changing h5ad conventions.

### Anti-goals
- Any numeric change in outputs of an existing single-perturbation model when the feature
  is off.
- Breaking `from_pretrained(strict=True)` of existing checkpoints.
- Treating every combination as its own class (see §8).

### 1.1 Two data regimes (both required) **[owner]**

The framework must support both of the following, chosen per dataset by configuration. The
right *evaluation and pairing unit* depends on which one the data is:

- **Regime A, designed combinations** (Norman-style): tens to hundreds of distinct sets,
  each with many cells. The natural unit is the *set*: pair, group and score per combination.
- **Regime B, random high-MOI** (true high-MOI Perturb-seq): thousands of distinct sets,
  most with 1–5 cells. Per-set grouping is useless (no group reaches
  `perturbation_metric_min_cells`), so the unit must be the *single perturbation*: a cell
  carrying `{A,B}` contributes to the marginal groups `A` and `B`.

Both are supported through one config key, `perturbation_unit` (`"set"` default for
backward compatibility, `"single"` for regime B); see §3.7. In regime B a cell can carry
**5–7 and up to ~10 perturbations**, so nothing in the design may assume a small set width:
`max_n_pert` is taken from the data, the aggregation must stay well-scaled at k≈10 (§3.4),
and the multi-label classifier must cope with ~10 positives per row (§3.5). When
`perturbation_unit="set"` is used on data where fewer than half of the validation
`(context, set)` groups reach `perturbation_metric_min_cells`, the split check emits a
warning recommending `"single"`.

## 2. Current state (what the code assumes today)

File references are to commit `4d93bfa`. All claims below were verified by the builder check.

| Area | Where | Single-perturbation assumption |
|------|-------|--------------------------------|
| Label vocabulary | `PertTFUniDataManager.set_genotype_index`, `PertTFDataset.__init__` (`pert_data_loader.py`), `build_perturbation_mapping` (`pert_emb.py`), `lora.label_mappings` | mapping built from `obs['genotype'].unique()` strings |
| Dataset item | `PertTFDataset.__getitem__` | emits scalar ints `perturbation_labels`, `perturbation_labels_next` via `genotype_to_index[label_str]` |
| Collator | `PertBatchCollator.__call__` | generic: keys containing `label` are stacked into long tensors; keys containing `name` are passed through as python lists |
| Pairing | `ContextAwarePairing` (`pert_pairing.py`) | compares the raw string to `control_value`; groups target cells by raw string; `condition_label` is the raw string |
| Encoders | `PertLabelEncoder`, `FeaturePertEncoder` (`modules.py`), `UnifiedPertEncoder` (`pert_encoder.py`) | all accept any index shape and return `(*shape, pert_dim)`. `PertLabelEncoder` applies LayerNorm after `nn.Embedding`, so a `padding_idx` row is **not** zero after the norm. `UnifiedPertEncoder` raises on negative indices. |
| Model | `PerturbationTFModel._encode` / `.forward` (`pertTF.py:153,250,297`) | three `self.pert_encoder(...)` calls expecting `(batch, pert_dim)`; `pert_embeddings_next` must be `(batch, pert_dim)` (line 250's result is unused) |
| Classifier | `PerturbationDecoder` + `nn.CrossEntropyLoss` in `train()` and `_run_evaluation_batches()` | softmax over `n_pert` classes, scalar targets |
| Contrastive loss | `train()` with `cce_mode` in `'cell_geno'`, `'genotype'`, `'cell+geno'` (default) | all pass `perturbation_labels(_next)` to `cce_loss` → `SUPCON_loss`, which does `labels.view(-1,1)` and raises on a 2-D tensor |
| Validation metrics | `_run_evaluation_batches` → `labels_to_names(batch['perturbation_labels_next'])`; `GroupMoments` keyed on `(context, perturbation_string)` | scalar label tensor; groups below `perturbation_metric_min_cells` (30) are dropped |
| Split check | `check_train_valid_split` (`split_check.py`) | set algebra on raw strings |
| Inference | `eval_testdata` drops rows with `obs['genotype_next']` not in mapping; `predict_perturbations` raises on any unknown label and requires output rows == input rows; `predicted_genotype = argmax` | raw-string membership |
| Query embeddings | `predict_perturbations` → `perturbation_embeddings[perturbation_labels_next]` | scalar fancy-indexing |
| Virtual screen | `virtual_screen_src2dest` (`virtual_screen.py:36-44`) | filters `screen_genes` against `genotype_to_index.keys()` itself |
| Lochness mode | `PertTFDataset.__getitem__` lochness branch | scalar `pert_label_next` |
| Checkpoints | `HFPerturbationTFModel` | `n_pert = len(genotype_to_index)`; constructor forwards an explicit argument list, so an unknown kwarg is treated as a *training* key and is **not** restored by `from_pretrained` |
| CI | `.github/workflows/ci.yml`, `pyproject.toml` | installs only `scanpy`, `huggingface_hub`, `pytest`, `torch`, `tqdm`; `hf.py` needs `omegaconf`, `train_function.py` needs `wandb`, `lora.py` needs `peft` |

No code parses `+`-joined labels today (the only `+` handling is the embedding-preset
selector in `pert_emb.py`).

## 3. Decisions

### 3.1 Data convention
- Perturbations stay in `obs['genotype']` / `obs['genotype_next']` as strings joined by a
  delimiter, default `"+"`, configurable.
- Opt-in through one config key `pert_delimiter` (default `None`). With `None` nothing is
  split and every code path is the legacy path.
- The control label (`perturbation_control_value`, default `"WT"`) means "no perturbation".
  Inside a combination (`"WT+GENE_A"`) it is dropped; alone it stays the control.
- Order and duplicates are ignored: `"B+A"`, `"A+B"`, `"A+B+A"` all mean `{A,B}`. The
  canonical form is sorted and delimiter-joined.
- **[builder-check call] Canonicalisation happens in place on the two string columns**, by
  `PertTFUniDataManager.__init__`, standalone `PertTFDataset.__init__`, `eval_testdata` and
  `predict_perturbations`, through one helper `canonicalize_perturbation_columns(adata, ...)`
  that logs one line when any label changed. This follows the existing precedent of
  `add_batch_info` mutating `obs`, costs one string per cell, and guarantees that pairing,
  split check and metrics (which read the raw column) see the same condition the encoder sees.

### 3.2 "Multi mode" is a property of the configuration, not of the data
**[builder-check call]** Multi mode means `pert_delimiter is not None`. In multi mode the
dataset always emits padded arrays (width `max_n_pert ≥ 1`, computed per dataset), the
classifier always uses the multi-label loss, and inference always writes the multi-label
outputs (§3.4), regardless of whether the particular AnnData happens to contain any
combinations. This keeps output semantics tied to the model/config a user chose, not to the
input file. (The earlier "use CE when width is 1" idea is dropped.)

### 3.3 Tensor representation
- In multi mode `perturbation_labels` and `perturbation_labels_next` are int arrays of shape
  `(max_n_pert,)` right-padded with the sentinel `-1`. In legacy mode they stay scalars, so
  the collator, training loop and every consumer see what they see today.
- `-1` is the pad because it needs no extra row in any embedding table and no change to
  `n_pert`, so existing checkpoints load strictly. It never reaches an encoder: it is clamped
  to `0` for the lookup and the result is masked to zero.

### 3.4 Perturbation set encoding
One function, used everywhere a perturbation embedding is produced:

```python
def encode_perturbation_set(encoder, labels, aggregation="sum"):
    # labels: (batch,) legacy, or (batch, k) padded with -1
    if labels.dim() == 1:
        return encoder(labels)                    # legacy path, untouched numerics
    valid = labels >= 0                           # (batch, k)
    emb = encoder(labels.clamp_min(0))            # (batch, k, pert_dim)
    emb = emb * valid.unsqueeze(-1)
    out = emb.sum(1)
    if aggregation == "mean":
        out = out / valid.sum(1, keepdim=True).clamp_min(1)
    return out
```

- `pert_aggregation` options: `sum` (default; additive model, as in GEARS), `mean`, and
  `sqrt` (sum divided by `sqrt(k)`, which keeps the expected norm of a sum of k roughly
  uncorrelated unit-scale vectors constant). All three coincide at k=1, and there is no
  post-aggregation LayerNorm, so a width-1 set is numerically identical to the scalar path
  (required test). **[owner]** Because regime B reaches k≈10, the documentation recommends
  `sqrt` or `mean` for random high-MOI data: a plain sum of ten LayerNorm-scaled vectors
  grows to several times a single embedding's norm, which saturates the sigmoid MLP in
  `PertExpEncoder` and, with `perturbation_input=True`, swamps the gene-token embeddings it is
  added to.
- Works unchanged for `PertLabelEncoder`, `FeaturePertEncoder`, `UnifiedPertEncoder`.
- `PerturbationTFModel` gains the kwarg `pert_aggregation` and its three call sites use the
  helper. The `pert_embeddings_next` route (`query_sources`) gathers `(batch, k, pert_dim)` and
  reduces with the same masked sum.
- **[builder-check fix]** `pert_aggregation` must be a *named* parameter of
  `HFPerturbationTFModel.__init__` and be forwarded to the parent, so it is kept in
  `config.json` and restored by `from_pretrained`. Passing it through `**kwargs` would
  silently demote it to a training key.

### 3.5 Classification objective
- `PerturbationDecoder` unchanged.
- Helper `perturbation_classification_loss(logits, labels, pos_weight=None)` in
  `custom_loss.py`: scalar labels → `CrossEntropyLoss`; 2-D labels → `BCEWithLogitsLoss`
  against a scattered multi-hot target. Used at the two sites in `train()` and the one in
  `_run_evaluation_batches()`.
- Inference outputs in multi mode: `X_pert_pred_probs` holds sigmoid probabilities;
  `predicted_genotype` is the delimiter-joined set of classes with probability > 0.5, falling
  back to the argmax class when none passes. **[builder-check call]** The threshold is
  exposed as config `pert_pred_threshold` (default 0.5) and an optional
  `pert_bce_pos_weight` is accepted, because an untuned BCE head with 1–10 positives among
  hundreds of classes may keep every sigmoid below 0.5; users who need a set prediction
  should tune these or read `X_pert_pred_probs` directly. **[owner]** With up to ~10
  positives per row, `predicted_genotype` also reports at most `max_n_pert` classes (the
  highest-probability ones above the threshold), so a poorly calibrated head cannot emit
  absurdly long sets. Legacy mode keeps softmax + argmax.

### 3.6 Contrastive (CCE) loss — all modes
**[builder-check call]** Instead of a per-combination scalar id, multi mode passes a
*positive mask* to `SUPCON_loss` (which already accepts `mask=`): two cells are positives
when their perturbation sets share at least one single (`M @ M.T > 0` on the multi-hot
matrix), combined with the cell-type equality mask for the `cell_geno` / `cell+geno` modes.
`cce_loss` gains an optional `pert_mask` argument; scalar labels keep today's path. This
works in both regimes (§1.1) and removes the need for any `combo_to_index` bookkeeping.

### 3.7 Pairing, split check, metrics
- `ContextAwarePairing` is unchanged: any non-control string is a perturbed cell, and
  `condition_label` is the (canonical) string. For regime B the documentation recommends
  `pairing_config={"target_sampling": "cell"}` (already supported) so that rare sets do not
  dominate training.
- New config key `perturbation_unit`:
  - `"set"` (default, legacy): `GroupMoments` and `check_train_valid_split` key on the
    canonical set string, as today. The split report gains a category `unseen_combination`
    (all singles seen, set unseen).
  - `"single"`: `GroupMoments.update` is called once per single in the cell's set (the row is
    counted in each marginal group); `group_moments_from_anndata` and the split check expand
    sets the same way; `compute_perturbation_metrics` is unchanged because it only sees
    `(context, name)` keys.
- Evaluation gets condition names from a new pass-through key `pert_name_next` (string per
  row, forwarded by the collator because the key contains `name`) instead of
  `labels_to_names`, so it never inverts a 2-D tensor.

### 3.8 Unknown singles at inference
**[builder-check call]** Each entry point keeps its current contract: `predict_perturbations`
raises if any single in any label is unknown (so its "output rows == input rows" guarantee
holds); `eval_testdata` drops rows whose label contains an unknown single and logs how many.
`labels_in_mapping` returns `True` only when every single is known. The zero-shot claim for
existing checkpoints therefore applies to combinations of *known* singles only.

### 3.8a Label validation and delimiter-collision check **[owner]**
A helper `validate_perturbation_labels(labels, delimiter, control, existing_mapping=None)`
runs wherever a mapping is built or a checkpoint is applied to data
(`PertTFUniDataManager.__init__`, standalone `PertTFDataset.__init__`, `build_perturbation_mapping`,
`lora.label_mappings`, `predict_perturbations`, `eval_testdata`):

1. **Legacy mode** (`pert_delimiter=None`): if any label contains `"+"`, warn once that such
   labels are treated as opaque single perturbations and that `pert_delimiter="+"` enables
   set semantics. Nothing else changes, so existing datasets keep working.
2. **Multi mode, malformed labels**: raise on empty components (`"A++B"`, `"+A"`, `"A+"`),
   on a label that is only the delimiter, and on whitespace-only components. Components are
   not stripped silently; the error lists the offending labels.
3. **Multi mode, pre-existing vocabulary**: if an `existing_mapping` (a checkpoint's
   `genotype_to_index`, or a user-supplied mapping) has any key containing the delimiter,
   raise: that model learned combinations as opaque classes and cannot be used with set
   semantics under this delimiter. The message names the keys and suggests either a
   different delimiter or legacy mode.
4. **Multi mode, checkpoint consistency**: `from_pretrained` records `pert_delimiter` from
   `training_config.json`; `predict_perturbations` and LoRA fine-tuning raise if the caller
   passes a different, non-`None` delimiter than the one the model was trained with.
5. **Width cap**: optional config `max_perts_per_cell`; when set, a label with more
   components raises. When unset, `max_n_pert` is whatever the data contains (10 or more is
   expected in regime B) and is logged.

### 3.9 Vocabulary building
New module `perttf/utils/pert_labels.py`:
`split_perturbation_label`, `canonical_label`, `unique_perturbations` (first-occurrence
order of singles, control included), `labels_in_mapping`,
`canonicalize_perturbation_columns`, `labels_to_padded_array`. With `delimiter=None` each is
an identity. Call sites: `PertTFUniDataManager.set_genotype_index`, `PertTFDataset.__init__`
fallback mapping and `__getitem__`, `build_perturbation_mapping`, `lora.label_mappings`,
`predict_perturbations`, `eval_testdata`, `virtual_screen_src2dest`.

### 3.10 Configuration and persistence
- Config keys: `pert_delimiter` (None), `pert_aggregation` ("sum" | "mean" | "sqrt"),
  `perturbation_unit` ("set" | "single"), `pert_pred_threshold` (0.5),
  `pert_bce_pos_weight` (None), `max_perts_per_cell` (None). Added to `generate_config` and
  `_init_default_train_config_`.
- `pert_aggregation` is a model constructor parameter (changes forward numerics) and lives
  in `config.json`. The others are data/training conventions and live in
  `training_config.json`, which `predict_perturbations`, LoRA fine-tuning and
  `eval_testdata` already read; each accepts a per-call override.
- `n_pert`, `genotype_to_index`, `running_parameters.pt`: unchanged.

### 3.11 Lochness, virtual screen, composition analysis
- Lochness branch builds `pert_label_next` with `labels_to_padded_array` so widths stay
  consistent within a dataset.
- **[builder-check fix]** `virtual_screen_src2dest` filters candidates with
  `genotype_to_index.keys()`; it switches to `labels_in_mapping` so `"A+B"` candidates are
  screened. `generate_pert_embeddings` uses single genes and needs no change.
  *(Correction, 2026-10-22:* `calculate_lonESS_score` in `composition_change_analysis.py` did
  need a change: it matched whole labels, so at high MOI it returned ≈ −1 everywhere or raised
  `KeyError`. It now takes `pert_delimiter` (membership matching) and has a per-perturbation
  summary, `lochness_by_perturbation`.)

## 4. File-by-file change list

| File | Change | Size |
|------|--------|------|
| `perttf/utils/pert_labels.py` (new) | parsing, canonicalisation, validation (§3.8a), mapping, padding helpers | ~140 lines |
| `perttf/utils/pert_data_loader.py` | read `pert_delimiter`; canonicalise columns; mapping from singles; `max_n_pert`; emit padded arrays and `pert_name_next` | ~45 lines |
| `perttf/model/pert_encoder.py` | `encode_perturbation_set` | ~20 lines |
| `perttf/model/pertTF.py` | `pert_aggregation` kwarg; three call sites; `pert_embeddings_next` 3-D reduce | ~25 lines |
| `perttf/custom_loss.py` | `perturbation_classification_loss`; `pert_mask` in `cce_loss` | ~35 lines |
| `perttf/model/train_function.py` | loss helper (3 sites); CCE mask; `pert_name_next`; `perturbation_unit` expansion; multi-mode sigmoid outputs; row filter | ~60 lines |
| `perttf/model/hf.py` | named `pert_aggregation` param; config defaults; validation via helpers; `query_sources` gather+reduce | ~35 lines |
| `perttf/utils/pert_metrics.py` | `group_moments_from_anndata` set expansion for `"single"` | ~15 lines |
| `perttf/utils/split_check.py` | `unseen_combination`; `"single"` expansion | ~25 lines |
| `perttf/model/pert_emb.py`, `perttf/model/lora.py`, `perttf/utils/virtual_screen.py`, `perttf/model/config_gen.py` | helper calls / defaults | ~5 lines each |
| `pyproject.toml` | add `omegaconf`, `safetensors`, `wandb`, `peft` to the `test` extra | 1 line |
| `perttf/model/config_gen.py` | drop `wandb.Settings(start_method="fork")` (rejected by wandb ≥ 0.29; a no-op since 0.19) | 1 line |
| `tests/test_tutorial_smoke.py` (new) | §6: tutorial-derived CPU smoke tests, legacy baseline plus multi-mode variants | ~290 + ~150 lines |
| `tests/test_multi_perturbation.py` (new) | §6 unit tests | ~180 lines |
| `demos/tutorials/MULTI_PERTURBATION.md` (new), README/docs index link | user guide incl. regime A/B recommendations | — |

Untouched: `pert_pairing.py`, all encoder classes, `PerturbationDecoder`, `preprocessor.py`,
`custom_tokenizer.py`, checkpoint format.

## 5. Compatibility guarantees

1. `pert_delimiter=None` (default): labels are scalars, every helper is an identity, no tensor
   shape or loss change. Existing tests and tutorials pass unchanged.
2. Existing checkpoints load with `strict=True`: no parameter or buffer is added. A
   single-perturbation checkpoint can run multi-perturbation inference by passing
   `pert_delimiter="+"` to `predict_perturbations`; the embedding sum is a zero-shot additive
   estimate for combinations of known singles.
3. A width-1 padded label gives the same embedding as the scalar label (tested).
4. Unified encoder (`pert_sources`), `query_sources`, control-row zeroing and `pert_pad_id`
   semantics are untouched.

## 6. Tests

CI only has `scanpy`, `huggingface_hub`, `pytest`, `torch`, `tqdm`; the test extra gains
`omegaconf`, `safetensors` (so `hf.py` imports), `wandb` and `peft` (so the tutorial entry points
import). Unit tests exercise the new helpers directly and never import `train_function.py` or
`lora.py`.

### 6.1 Tutorial compatibility tests **[owner]**

The notebooks in `demos/tutorials/` are the executable documentation of the end-to-end contracts
but need downloads and a GPU. `tests/test_tutorial_smoke.py` replays each notebook's library
calls on a 240-cell synthetic Perturb-seq AnnData (4 genotypes × 2 cell types, log-normalised
`X_binned`, `highly_variable`) with a 16-dim, 1-layer model, one epoch, CPU, about 25 s total:

| Notebook | Calls replayed |
|----------|----------------|
| `train_pertTF_with__lochNESS`, `feature_pert_encoder_demo` | `generate_config(wandb_mode="disabled")` → `produce_training_datasets` → `PerturbationTFModel(...)` as the notebooks construct it → `wrapper_train` (saves `vocab.pt`, `running_parameters.pt`, `config.json`, `best_model.pt`) → `eval_testdata` (`X_scGPT`, `X_scGPT_next`, `predicted_genotype`, `predicted_celltype`, `genotype_pred_probs`) |
| perturbation objective (as the `perttf-tiny-perturb-5k-nb` checkpoints) | same pipeline with `next_cell_pred_type="pert"`, explicit split, `perturbation_validation`, `pert_valid_pairs` |
| `TUTORIAL_50gene_eval`, `INFERENCE.md` | `HFPerturbationTFModel` save/`from_pretrained` round trip (strict), `eval_testdata(..., predict_expr=True, mvc_full_expr=True, sizefactor=True, sample=True)`, `compute_metrics_from_anndata`, `predict_perturbations` (shape, `uns["perttf_prediction"]`, unknown-label error) |
| `lora_finetuning_tutorial` | `build_lora_config`, `run_lora_cls_train` → `load_lora_adapter` → `predict_cls`; `run_lora_pert_train` on an unseen-context split → `eval_testdata` through the PEFT model |
| `virtual_pooled_screen` | `virtual_screen_src2dest` (candidate filtering, `sim_ranked` columns, `adata_last_eva`) |

The file is written and committed before any feature code (plan Task 0) and must pass unchanged
after every task; Tasks 7 and 8 append the same scenarios run with `pert_delimiter="+"` for both
regimes (designed sets `"A+B"`, `"A+B+C"` with `perturbation_unit="set"`; random sets of 1–6
guides from 8 genes with `perturbation_unit="single"` and `pert_aggregation="sqrt"`), plus the
legacy-mode warning when `+` labels are used without a delimiter. Writing the baseline exposed two
facts about the unmodified repo: `generate_config` fails on wandb ≥ 0.29 (fixed in Task 0), and
the data manager's hardcoded `num_workers=8` costs ~45 s per training run on macOS (worked around
in the test only; see §7).

- `pert_labels`: split, control dropping, duplicates, ordering, `None` identity,
  canonicalisation, padding; validation: legacy-mode warning on `"+"`, malformed labels
  raise, mapping keys containing the delimiter raise, `max_perts_per_cell` enforced.
- `encode_perturbation_set`: width-1 2-D equals 1-D for all three encoders; pad positions
  contribute zero; `sum` vs `mean`.
- Dataset: tiny AnnData with `"WT"`, `"A"`, `"A+B"`, `"B+A"` gives mapping `{WT,A,B}`,
  `max_n_pert == 2`, padded arrays, canonical `pert_name_next`; legacy mode gives scalars.
- `perturbation_classification_loss`: CE for scalars, BCE multi-hot for 2-D; `cce_loss` with
  `pert_mask`.
- `GroupMoments`/split-check expansion under `perturbation_unit="single"`.
- Forward pass of `PerturbationTFModel` with 2-D labels of width 10 (regime B), including
  `perturbation_input=True` and `pred_lochness_next=True`; `sqrt` aggregation keeps the
  aggregated norm within a constant factor of a single embedding's norm.
- Checkpoint consistency: a model whose mapping contains `"A+B"` raises when used with
  `pert_delimiter="+"`; a model trained with `"+"` raises when inference passes `"|"`.
- `HFPerturbationTFModel` save/`from_pretrained` round trip restores `pert_aggregation="mean"`.

## 7. Left to the builder

- Exact log/warning wording.
- Whether `canonicalize_perturbation_columns` keeps categorical dtype or converts to object.
- Follow-up outside this feature: `PertTFUniDataManager._create_loaders_from_dataset` hardcodes
  `num_workers=8`; making it `config.get("num_workers", 8)` would let CPU users (and the tutorial
  tests, which currently monkeypatch the loader) avoid ~45 s of worker start-up per run on macOS.

## 8. Alternatives considered

- **Combination as its own label**: zero code today, but no compositionality, no zero-shot
  combinations, classifier grows with the number of combinations, and ESM2/GenePT sources
  cannot cover a combination. Documented fallback only.
- **Dedicated pad row in the embedding table**: cleaner than `-1` but changes `n_pert` and
  breaks strict loading of existing checkpoints.
- **Attention/DeepSets set encoder**: learns interactions but adds parameters and breaks
  single-perturbation equivalence. Future `pert_aggregation="attn"`.
- **Multi-hot `obsm` matrix instead of delimited strings**: awkward in h5ad and inconsistent
  with the `genotype_next` workflow.
