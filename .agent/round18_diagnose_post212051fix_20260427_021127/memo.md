# Diagnosis Memo: Post-212051fix Compare

## Verdict

- `experiments/shenji_agent/league_20260426_082209` did **not** complete.
- It stopped **abnormally** after the last partial checkpoint at `91 / 100`, then was later **superseded by a fresh relaunch** that produced `experiments/shenji_agent/league_20260426_160721`.
- `league_20260426_160721` is **related to the same compare lineage**, but it is **not** a continuation of `082209`; it is a new replay from `0 / 100`.

## What Happened

1. Detached compare prefix `.agent/round18_compare_post_212051fix_20260426_082101*` launched at `2026-04-26T08:21:30.761927Z`.
2. The corresponding run dir `league_20260426_082209` progressed until:
   - `completed_pairs = 91 / 100`
   - `heartbeat_ts = 2026-04-26T15:44:26.627378Z`
   - `current_scenario = 2030111194.json`
   - `current_leg_index = 1`
3. After that point, no terminal artifacts were written:
   - no detached `status.json`
   - no run-dir `participants.json`
   - no `match_results.json`
   - no `rating.json`
   - no `promotion.json`
   - no `exploitability.json`
   - no `summary.md`
4. The logs under the detached prefix are empty, and the watchdog sidecar is stale.
5. `training/selfplay.py` would have written abort sidecars on handled exceptions. Because none exist, the most defensible diagnosis is an external hard stop / lost supervisor path after the last partial snapshot.

## Recoverability

- Recoverable from `082209` itself:
  - only partial metrics from the first `91` completed pairs.
  - partial summary:
    - `overall.point_estimate = 0.5000`
    - `complex.point_estimate = 0.5000`
    - `learned_path_coverage = 0.05177909764603313`
    - `move_path_coverage = 0.04174066272938779`
    - `shoot_path_coverage = 1.0`
- Not recoverable from `082209` itself:
  - its own terminal `100 / 100` metrics.
  - any true terminal sidecars for that original process.

## Relation Of `160721`

- Relevant:
  - same compare command
  - same candidate/incumbent artifact pair
  - same candidate/incumbent commit SHAs
  - byte-identical `map_list.json`
- Exact relation:
  - `.agent/round18_compare_relaunch_20260426_160631.observe.txt` shows a fresh run beginning at `0 / 100`
  - `.agent/round18_compare_relaunch_20260426_160631.status.json` shows normal exit `0`
  - `league_20260426_160721` contains full terminal sidecars
- Why it is not a continuation:
  - the first `91` matches match on core outcome/order fields
  - but policy stats differ from match `0`
  - `082209` recorded non-zero learned coverage, while `160721` recorded zero learned coverage throughout

## Relaunch Judgment

- For `082209` itself, a fresh relaunch **was required** because there is no supported resume path and no terminal sidecars.
- That relaunch already happened at `2026-04-26T16:06:56Z`, producing `league_20260426_160721`.
- Therefore:
  - `082209` cannot be treated as complete.
  - the lineage already has a superseding completed rerun.
  - no additional relaunch is required merely to diagnose the abnormal state.
