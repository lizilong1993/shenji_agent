# V3D Formal Paired Protocol

## Design

Paired tactical-on/off evaluation under authenticated SDK replay.
Each pair runs the same seed/scenario/map twice, differing only in `tactical_mode`.

## Seed list

`[0, 1, 2, 3, 4]` — 5 seeds for statistical power.

## Scenario/map selection

| Scenario ID | Map |
|---|---|
| 1231 | map_123 |
| 1531 | map_53 |
| 1910631192 | map_19 |

These are the 3 default scenarios from the V3B formal ablation infrastructure.

## Paired design

15 paired comparisons = 5 seeds × 3 scenarios.
Each pair runs `tactical_on` then `tactical_off` with identical configuration.

## Identical non-tactical configuration

- Policy profile: `challenger` (both on and off)
- Blue opponent: `baseline` profile
- Timeout steps: 2000
- Seed offset for blue: +1000 (per existing code)
- All other parameters identical between paired runs

## Outcome metric

Primary: `score_red` (red total score)
Secondary: winner (red win/loss/draw)

## Battle progression metrics

- Steps completed
- Duration (seconds)

## Tactical invocation metrics

- `tactical_invocation_count`: total policy evaluation samples from score_means
- `tactical_nonzero_count`: samples with nonzero tactical_score

## Route empty-rate metrics

- `route_empty_count`: number of move_route samples with empty route
- `route_empty_rate`: empty / total move attempts

## Illegal action metrics

Not directly recorded by current infrastructure (policy_stats action_counts may provide proxy).

## Confidence interval method

Bootstrap with 1000 resamples, 95% confidence level, seed=42.

## Stopping rules

- If more than 50% of pairs fail (crash/timeout): stop and report as BLOCKED
- If specific scenario consistently crashes, exclude it and continue
- If the ablation takes > 12 hours wall time: recommend SAFE_STOP for resource reasons

## Claim gates

| Gate | Condition |
|---|---|
| OUTCOME_CONTRIBUTION_SUPPORTED | Positive outcome delta, CI entirely > 0, tactical invoked |
| WEAK_OR_INCONCLUSIVE | Tactical invoked, but delta small/CI crosses zero |
| NOT_SHOWN | Delta zero or negative with no diagnostic benefit |
| BLOCKED | Cannot produce paired outcomes |
