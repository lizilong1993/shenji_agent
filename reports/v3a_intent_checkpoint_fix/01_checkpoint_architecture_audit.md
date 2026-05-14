# Phase 1 — Checkpoint Architecture Audit

## Timestamp
2026-05-14

## Checkpoint
`ai/weights/intention_lstm.pth`

## Summary

| Question | Answer |
|----------|--------|
| Is checkpoint readable? | Yes — `OrderedDict` with 22 keys, loaded on CPU. |
| Is it old `fc_intent` style? | **Yes** — contains `fc_intent.weight` and `fc_intent.bias`. |
| Is it current `head` style? | No — no `head.*` keys present. |
| Can it be loaded directly into `MultiHorizonIntentionModel`? | No — missing 6 `head.*` keys, has 2 unexpected `fc_intent` keys. |
| Can it be mapped by key renaming only? | No — `fc_intent.weight` shape `[8, 512]` vs `head.3.weight` expected `[35, 512]`. |
| Are tensor shapes compatible with current model? | No. |
| Can it load into old `IntentionLSTM(num_classes=4)`? | No — `fc_intent.weight` shape `[8, 512]` vs expected `[4, 512]`. |
| Can it load into old `IntentionLSTM(num_classes=8)`? | **Yes** — all 22 keys match exactly. |
| Is a legacy class needed? | Yes — `IntentionLSTM(num_classes=8)` can load the checkpoint. |
| Is retraining required? | **TBD** — a legacy loader `IntentionLSTM(num_classes=8)` can load the checkpoint, but the 8-class output vs current 7-class expectation requires an adapter wrapper. |

## Key Findings

1. The checkpoint is a **raw state_dict** (not wrapped in `model_state`).
2. All LSTM and attention keys are identical between old `IntentionLSTM` and current `MultiHorizonIntentionModel`.
3. The only differences are in the output head:
   - Old: `fc_intent.weight` `[8, 512]`, `fc_intent.bias` `[8]`
   - Current: `head.*` Sequential with `[35, 512]` final layer
4. The old model appears to have been trained with **num_classes=8** while the current `IntentionLSTM` (still in source) defaults to `num_classes=4`.

## Shape Comparison

### LSTM Weights (compatible)
All LSTM keys have identical shapes between checkpoint and both model variants.

### Attention Weights (compatible)
All attention keys have identical shapes.

### Output Head (incompatible)
| Key | Checkpoint | Current Model | Old Model (num_classes=4) |
|-----|-----------|--------------|--------------------------|
| `fc_intent.weight` | `[8, 512]` | N/A (no `fc_intent`) | `[4, 512]` |
| `fc_intent.bias` | `[8]` | N/A (no `fc_intent`) | `[4]` |
| `head.0.weight` | N/A (no `head`) | `[512]` | N/A |
| `head.0.bias` | N/A (no `head`) | `[512]` | N/A |
| `head.3.weight` | N/A (no `head.3`) | `[35, 512]` | N/A |
| `head.3.bias` | N/A (no `head.3`) | `[35]` | N/A |

## Conclusion
The checkpoint is structurally a snapshot of `IntentionLSTM` with `num_classes=8`. It cannot be loaded into any currently expected model type without an adapter. A legacy loader wrapping `IntentionLSTM(num_classes=8)` plus an 8→7 output adapter is feasible but introduces semantic uncertainty about the extra output class.
