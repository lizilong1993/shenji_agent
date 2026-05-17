# Memo: Relaunch Zero Coverage Trustworthiness

## Direct Answer

- `league_20260426_160721` has complete terminal sidecars and exit code `0`.
- Its terminal metrics are:
  - `overall.point_estimate = 0.5000`
  - `overall.ci95 = [0.5000, 0.5000]`
  - `complex.point_estimate = 0.5000`
  - `complex.ci95 = [0.5000, 0.5000]`
  - `learned_path_coverage = 0.0000`
  - `move_path_coverage = 0.0000`
  - `shoot_path_coverage = 0.0000`
  - `promotion.accepted = false`

## Trustworthiness Judgment

- The zero-coverage result is internally consistent with `160721`'s own sidecars, but it is **not trustworthy as the post-212051fix branch result**.
- It most likely reflects a compare runtime action-path issue, not a genuine “candidate and incumbent both had zero learned engagement for the whole window” result.

## Strongest Evidence

1. `160721` says learned models loaded successfully on both sides:
   - candidate `policy_profiles = ['challenger']`
   - opponent `policy_profiles = ['incumbent']`
   - `intent_model_loaded_count = 2`
   - `situation_model_loaded_count = 2`
   - candidate `inference_devices = ['cuda']`
   - correct `loaded_from_dirs`
2. Yet every one of its `100` matches records:
   - candidate `move.total = 0`, `shoot.total = 0`
   - opponent `move.total = 0`, `shoot.total = 0`
   - `trace_samples = []`
3. The interrupted control run `082209` used:
   - the same candidate/incumbent source SHAs
   - the same artifact directories
   - the same `map_list.json`
   - the same first-`91` core match order/outcome fields
4. But `082209` recorded non-zero candidate and opponent policy counts in all completed `91` matches, with partial coverage:
   - `learned_path_coverage = 0.05177909764603313`
   - `move_path_coverage = 0.04174066272938779`
   - `shoot_path_coverage = 1.0`
5. `training/selfplay.py` only aggregates agent diagnostics into match sidecars; `training/evaluate.py` only re-aggregates those match sidecars.
   - So the discrepancy is upstream of `rating.json`, not caused by post-hoc summarization.

## Why This Points To A Path Issue

- The compare process loads participant code from the artifact `source_commit_sha`, not from the current working tree.
- The sidecars say those learned models were loaded.
- The agent setup logic would keep `use_ai_optimization` enabled for `challenger` and `incumbent` profiles unless model loading failed.
- Under the action-generation path, `Move`/`Shoot` diagnostics are produced by `policy.note_action(...)`.
- Seeing loaded models plus `0` total `Move` and `0` total `Shoot` on both sides for the full `100`-pair window is incompatible with the same-lineage `082209` behavior and is not believable as a normal branch-strength measurement.

## Acceptance Judgment

- `league_20260426_160721` should **not** be accepted as the trustworthy terminal branch result for post-212051fix.
- If a valid branch compare result is still required, a **fresh compare relaunch is required**.
- This memo is read-only diagnosis only; no relaunch was performed here.
