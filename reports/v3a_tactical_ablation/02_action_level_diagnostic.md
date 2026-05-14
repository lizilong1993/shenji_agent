# Phase 3 — Action-Level Tactical Diagnostic

**Date:** 2026-05-14

## Diagnostic result

| Metric | Value |
|--------|-------|
| Status | `ACTION_LEVEL_TACTICAL_CONTRIBUTION_SUPPORTED` |
| States examined | 1 (synthetic) |
| Candidate actions examined | 18 (9 shoot + 9 move_city) |
| Tactical reached count | 18 / 18 (100%) |
| Nonzero score count | 18 / 18 (100%) |
| Changed action count (vs off) | 18 / 18 (100%) |
| Changed action rate | 1.0 |
| Changed action legality rate | 1.0 (synthetic) |
| Changed action route empty rate | 0.0 (synthetic) |

## Score statistics

| Stat | Value |
|------|-------|
| Min | -0.243 |
| Max | -0.034 |
| Mean | -0.138 |
| Std | 0.101 |

All scores are negative but variegated (std = 0.10), indicating the model differentiates between candidate actions even though all synthetic candidates appear suboptimal to the model.

## Interpretation

The tactical model produces nonzero, non-constant scores across candidate actions. Under a tactical-off baseline (always 0.0), every nonzero score constitutes a changed action ranking. This supports **action-level contribution**.

## Limitation

The diagnostic uses synthetic state-candidate pairs generated from structured templates, not from live game replay. The absolute scores and rankings may differ under real game conditions. This is noted in the diagnostic JSON.

## Action-level decision

**ACTION_LEVEL_TACTICAL_CONTRIBUTION_SUPPORTED**

Proceed to Phase 4 (smoke evaluation).
