# Intent Checkpoint Compatibility Status

## Current State (2026-05-14)

**Final state:** LEGACY_INTENT_LOADER_READY_FOR_REPLAY

## What was done

| Step | Status |
|------|--------|
| Checkpoint architecture audit | Complete |
| Historical model definition search | Complete |
| Compatibility strategy selection | Strategy B — Legacy intent model loader |
| LegacyIntentModel implementation | Complete (ai/models.py) |
| LegacySituationModel implementation | Complete (ai/models.py) |
| Policy loader auto-detection | Complete (ai/policy.py) |
| Tests | 13/13 pass |
| Full-agent compatibility smoke | PASS (9/9 phases) |

## Checkpoint details

- `ai/weights/intention_lstm.pth` — old `IntentionLSTM` with `num_classes=8`
- Loads into `LegacyIntentModel` which wraps `IntentionLSTM(num_classes=8)`
- Output adapted to `[batch, 1, 7]` (horizon dim + 8→7 truncation)
- 8th class (index 7) is safely discarded as `_DIRECTION_STEPS` has indices 0-6

## Policy diagnostics

- `intent_model_architecture`: `LegacyIntentModel` (reported in `get_diagnostics()`)
- All three models (intent, situation, tactical) load and function

## Blocker resolution

**Original blocker:** "ai/weights/intention_lstm.pth uses old fc_intent architecture; current MultiHorizonIntentionModel expects head module."

**Resolution:** LegacyIntentModel auto-detects old fc_intent checkpoints and loads them via the compatible IntentionLSTM class, adapting the output shape to match policy expectations.

## Full-agent replay readiness

- Policy loads completely ✓
- Agent module imports ✓
- Intent inference works ✓
- Situation scoring works ✓
- Tactical scoring works (both on/off) ✓

**Outcome-level tactical ablation is now executable** at the smoke level, pending SDK-backed replay loop for formal evaluation.
