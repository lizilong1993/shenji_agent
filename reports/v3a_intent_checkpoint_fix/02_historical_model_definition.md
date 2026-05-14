# Phase 2 — Historical Model Definition

## Approach
Searched git history and current source for old intent model definitions.

## Findings

### Current Source (HEAD)
`ai/models.py` contains **both** model architectures side by side:

1. **`IntentionLSTM`** (old architecture, line 7):
   - Uses `self.fc_intent` linear layer
   - Default `num_classes=4` (ATTACK, DEFENSE, RECON, RETREAT)
   - Forward returns `[batch, num_classes]` (no horizon dimension)

2. **`MultiHorizonIntentionModel`** (current architecture, line 62):
   - Uses `self.head` Sequential module (LayerNorm + Linear + ReLU + Linear)
   - Default `hidden_dim=128, horizons=5, num_classes=7`
   - Forward returns `[batch, horizons, num_classes]`

### Historical Snapshots
At commit `bccc61c4` (round005: recover sources), the `models.py` was identical to HEAD — both classes already present.

### Training Script
At commit `bccc61c4`, `training/supervised.py` already imported `MultiHorizonIntentionModel` with `num_classes=7`, **not** `IntentionLSTM`. The training script has always used the current model architecture since the earliest available commit.

### Checkpoint Origin
The model manifest reports source commit `3d3e69a5a...` dated `20260428_235813` — this commit is not in the current repository history. The checkpoint was produced from a **pre-backup-loss codebase** that is not recoverable.

### Output Dimension Mismatch
| Artifact | Output Size | Source |
|----------|------------|--------|
| Checkpoint (`fc_intent.weight`) | 8 | Unknown training config |
| Current `IntentionLSTM` default | 4 | Source code default |
| Current `MultiHorizonIntentionModel` | 7 per horizon | `training/supervised.py` |

## Classification
- **Historical old `fc_intent` class found:** Yes — `IntentionLSTM` still in source.
- **Checkpoint trained with `num_classes=8`:** Confirmed from weight shape.
- **Training config for `num_classes=8`:** Not recoverable (pre-backup commit).
- **Current training uses:** `MultiHorizonIntentionModel(hidden_dim=128, num_classes=7)`.
- **Conclusion:** The checkpoint is a `num_classes=8` variant of `IntentionLSTM`, produced from a lost training run before the current git history began.
