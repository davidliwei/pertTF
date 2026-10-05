# Training optimizers and learning-rate schedules

Existing training configurations retain Adam and the once-per-epoch StepLR
schedule (`schedule_ratio` is the decay factor). The new training configuration
settings and their defaults:

| Setting | Default | Meaning |
|---|---|---|
| `optimizer` | `"adam"` | `"adam"`, `"adamw"` or `"muon"` |
| `weight_decay` | `0.01` | AdamW and Muon only |
| `muon_aux_lr_ratio` | `0.01` | Muon only: auxiliary Adam peak LR / Muon peak LR |
| `scheduler` | `"step"` | `"step"` (StepLR once per epoch) or `"cosine"` (per update) |
| `warmup_epochs` | `0.0` | Cosine only; fractional epochs allowed |

For warmup followed by cosine decay, add these settings to your training
configuration:

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
