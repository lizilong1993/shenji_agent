# V3A-FIX Final Decision

## Decision: LEGACY_INTENT_LOADER_READY_FOR_REPLAY

**Timestamp:** 2026-05-14

## Criteria Verification

| Criterion | Met? | Detail |
|-----------|------|--------|
| Safe adapter implemented | ✅ | `LegacyIntentModel` in `ai/models.py` wraps `IntentionLSTM(num_classes=8)`, loads old checkpoint without modification |
| Restored checkpoint loads | ✅ | `ai/weights/intention_lstm.pth` loads with `strict=True`, output adapted to `[batch, 1, 7]` |
| Situation checkpoint also loads | ✅ | `LegacySituationModel` handles bare `SituationGCN` checkpoint |
| Auto-detection implemented | ✅ | Policy `_load_intent_model` and `_load_situation_model` detect old vs new format |
| Full-agent compatibility smoke passes | ✅ | 9/9 phases: module import, class import, policy config, all three models, diagnostics, intent inference, situation scoring |
| Outcome ablation executable | ✅ | Both tactical_on and tactical_off configure; SDK-backed replay loop is only missing piece |
| Tests pass | ✅ | 87 unit tests + 35 targeted tests all pass |
| compileall passes | ✅ | All modules compile clean |
| No large assets committed | ✅ | Only code, tests, reports, and text evidence |

## What Was Done

1. **Phase 0‑2:** Audited the checkpoint architecture — old `IntentionLSTM(num_classes=8)` confirmed.
2. **Phase 3:** Selected Strategy B (Legacy Intent Model Loader) as safest compatible approach.
3. **Phase 4:** Implemented `LegacyIntentModel` and `LegacySituationModel` in `ai/models.py`; auto-detection in `ai/policy.py`; 13 unit tests.
4. **Phase 5:** Full-agent compatibility smoke — PASS (9/9).
5. **Phase 7:** Outcome-level tactical ablation now executable — both on/off modes work.
6. **Phase 8:** Updated V3 evidence package with intent compatibility status.

## What Was NOT Done

- **Outcome-level tactical contribution** is NOT claimed — formal SDK-backed paired replay evaluation is needed.
- **Checkpoint conversion** was NOT performed — original checkpoint remains untouched.
- **Retraining** was NOT required — legacy loader resolves compatibility.
- **V3B/C/D formal evaluation** was NOT started.

## Recommended Next Goal

**V3B: Formal tactical outcome ablation via SDK replay**

Run paired tactical-on vs tactical-off games using the SDK replay loop:
- Same seed, scenario, opponent
- Multiple seeds (≥10) for statistical confidence
- Metrics: win/loss, rating delta, route empty-rate, tactical action change rate
- Compare individual vs paired statistical tests

If V3B establishes outcome contribution, proceed to V3C (route reliability) and V3D (ablation rigor with sidecar promotion).
