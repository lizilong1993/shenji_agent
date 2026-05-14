# V3A Tactical Ablation — Final Decision

**Date:** 2026-05-14

## Final state

**RESUME_V3_WITH_TACTICAL_AS_DIAGNOSTIC_ONLY**

## What was achieved

| Requirement | Status |
|-------------|--------|
| Tactical on/off switch exists or confirmed | ✅ Implemented in `LearnedPolicy` (tactical_mode: on/off/missing_fail_closed) |
| Action-level tactical diagnostic runs | ✅ `ACTION_LEVEL_TACTICAL_CONTRIBUTION_SUPPORTED` — 18/18 candidates nonzero |
| Smoke evaluation runs | ✅ `SMOKE_PASS` — 3 seeds, 72/72 candidates nonzero |
| Multiseed ablation runs | ✅ `MULTISEED_COMPLETED` — 5 seeds, 120/120 candidates nonzero |
| Tactical contribution decision written | ✅ `TACTICAL_ACTION_LEVEL_ONLY_SUPPORTED` |
| V3 tactical claim matrix updated | ✅ C7 upgraded from forbidden → partially supported |
| Tests pass | ✅ 22/22 passed |
| Compileall passes | ✅ All scripts compile |
| No large tactical assets committed | ✅ Assets are git-ignored and not staged |

## Evidence strength

| Level | Finding |
|-------|---------|
| L1 — Compatibility | Checkpoint loads, dataset validates, branch produces nonzero scores |
| L2 — Action-level | **Supported.** 100% nonzero, 100% changed-action rate across 5 seeds, score std = 0.138 (discriminates) |
| L3 — Outcome-level | **Not supported.** Game-level evaluation blocked by pre-existing model architecture mismatch |

## Key discovery

The tactical model consistently produces **negative BCE logits** (mean = -0.138, all scores below 0). This means the model assigns <0.5 probability to all synthetic candidates. While the model **does differentiate** between actions (std = 0.10–0.16), its contribution to the combined action score is small and may not flip decisions in a real game context.

## Blocker for outcome-level evaluation

The `intention_lstm.pth` checkpoint was saved with `fc_intent` output head, but the current `MultiHorizonIntentionModel` uses a `head` sequential module. This prevents full-policy loading. Full-game tactical-on/off comparison requires fixing this model compatibility issue.

## Next task

Resume V3 with route repair and baseline expansion. Tactical remains a secondary analysis / technical detail.

## SUCCESS criteria satisfied

- [x] tactical-on/off switch exists or is confirmed
- [x] action-level tactical diagnostic runs
- [x] smoke evaluation runs
- [x] multiseed ablation runs
- [x] tactical contribution decision is written
- [x] V3 tactical claim matrix is updated
- [x] tests pass
- [x] compileall passes
- [x] no large tactical assets committed
- [x] final decision states how V3 should resume

**SUCCESS does not require tactical contribution to be positive. SUCCESS means the tactical contribution question is answered honestly.**
