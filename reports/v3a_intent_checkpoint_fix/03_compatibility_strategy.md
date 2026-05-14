# Phase 3 — Compatibility Strategy

## Strategy Evaluation

| Strategy | Feasible | Safe | Recommended |
|----------|----------|------|-------------|
| **A — Key remapping adapter** | No | N/A | ❌ Shapes differ: `fc_intent.weight [8,512]` cannot map to `head.3.weight [35,512]`. |
| **B — Legacy intent model loader** | Yes | With constraints | ✅ `IntentionLSTM(num_classes=8)` loads checkpoint exactly. Output wrapper adds horizon dim and truncates 8→7 for direction matching. |
| **C — Checkpoint conversion** | Yes | Low confidence | ⚠️ Converting [8,512] → [35,512] requires learned projection or padding. Semantic mapping of 8th class is unknown. |
| **D — Retraining required** | Yes | Safest | ⚠️ Cleanest long-term fix, but leaves outcome ablation blocked until retraining is complete. |

## Selected Strategy: B — Legacy Intent Model Loader

### Rationale
1. The old `IntentionLSTM` class is **still present** in `ai/models.py` — no code archaeology needed.
2. The checkpoint loads **directly** into `IntentionLSTM(input_dim=108, hidden_dim=256, num_classes=8)` — all 22 keys match perfectly.
3. The only gap is output shape adaptation: the old model outputs `[batch, 8]`, the policy expects `[batch, horizons, num_classes]`.
4. A thin wrapper (`LegacyIntentWrapper`) can:
   - Internally instantiate `IntentionLSTM(num_classes=8)`
   - Load checkpoint into it
   - Forward: call old model, slice to 7 classes (directions 0-6), unsqueeze horizon dim
5. The 8th class (index 7) is safely discarded — `_DIRECTION_STEPS` only has indices 0-6.

### Risk
- The 8th class may represent a semantically meaningful output dimension (e.g., an additional intent type). Truncating it may degrade intent prediction quality.
- This does **not** affect safety — the system degrades gracefully (treats unknown class as ignored).

### Fallback
If smoke testing reveals degraded behavior, switch to Strategy D.

## Decision
**Adopt Strategy B** — implement legacy intent model loader with explicit output adaptation.
