# Tactical Contribution Status

**State:** `TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE`

## Summary

Authenticated paired tactical-on/off evaluation executes successfully and invokes the tactical branch, but outcome-level contribution remains weak or inconclusive under the tested protocol.

The tactical branch produces nonzero, variegated scores in every tactical-on game (351 total nonzero counts across 15 pairs), while tactical-off always produces zeros (0 nonzero counts). Route reliability does not degrade (0% empty rate across all 30 games). However, the outcome delta mean is -1.27 (bootstrap 95% CI [-50.07, 34.82], Cohen's d = -0.015), and win rate is identical (0.533) between modes.

## Key numbers

| Metric | Value |
|--------|-------|
| Paired comparisons | 15 (5 seeds × 3 scenarios) |
| Completion rate | 15/15 (100%) |
| Tactical nonzero scores (ON) | 351 total across all pairs |
| Tactical nonzero scores (OFF) | 0 |
| Score delta mean | -1.27 |
| Score delta std | 85.84 |
| Bootstrap 95% CI | [-50.07, 34.82] |
| Cohen's d | -0.015 |
| Win rate delta | 0.0 |
| Route empty rate (both modes) | 0.0 |

## Current limitation

Tactical is confirmed to execute and affect gameplay, but the net outcome-level effect is zero on average with high variance. No consistent positive outcome benefit is demonstrated.

## Paper claim classification

Tactical is classified as **diagnostic/action-selection contribution only**. Outcome contribution is weak or inconclusive. Include as bounded technical analysis in the paper, not as a main result.

## Previous states

| Round | State | Notes |
|---|---|---|
| V3A | ACTION_LEVEL_ONLY | Action-level diagnostic only, no game-level evaluation |
| V3B | ACTION_LEVEL_ONLY (outcome INCONCLUSIBLE) | Formal paired evaluation completed but auth-blocked |
| V3C | SDK_REPLAY_AUTH_RECOVERED | Auth fixed at smoke level |
| V3D | WEAK_OR_INCONCLUSIVE | Authenticated formal paired evaluation completed |
