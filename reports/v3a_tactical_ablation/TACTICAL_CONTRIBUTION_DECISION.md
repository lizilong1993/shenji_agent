# Tactical Contribution Decision

**Date:** 2026-05-14

## Decision

**TACTICAL_ACTION_LEVEL_ONLY_SUPPORTED**

The tactical branch changes action selection under diagnostic evaluation, but outcome-level contribution remains unproven.

## Evidence summary

| Level | Status | Evidence |
|-------|--------|----------|
| L1 — Compatibility | ✅ Achieved | Checkpoint loads, dataset validates, branch produces nonzero scores |
| L2 — Action-level | ✅ SUPPORTED | 120/120 candidates nonzero across 5 seeds, changed_action_rate = 1.0, consistent across seeds |
| L3 — Outcome-level | ❌ Not supported | Game-level evaluation blocked by pre-existing model-architecture mismatch; win rate/rating not measurable |

## Action-level evidence details

- **tactical_reached_count:** 24/24 per seed
- **tactical_nonzero_score_count:** 24/24 per seed (100%)
- **tactical_changed_action_count:** 24/24 per seed (100%)
- **changed_action_rate:** 1.0
- **changed_action_legality:** All synthetic candidates structurally legal
- **route_empty_rate:** N/A (synthetic diagnostic)
- **score variance across seeds:** Low (std of seed means = 0.010)

## Why outcome-level is not supported

The full agent policy requires intent and situation models that are incompatible with the current weight files (pre-existing architecture mismatch). Without running the full agent in a game environment, we cannot measure:

- Win rate impact
- Rating improvement
- Promotion metric changes
- Any game-level objective improvement

## Allowed paper claim

> "The tactical branch changes action selection under diagnostic evaluation, but outcome contribution remains limited or unproven."

## Forbidden paper claim

> "The tactical branch improves agent performance."

## Recommendation

Fix the intent/situation model weight compatibility issue, then run full-game tactical-on/off ablation to determine outcome-level contribution.
