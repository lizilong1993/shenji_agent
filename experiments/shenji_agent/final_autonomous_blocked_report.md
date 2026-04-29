# Shenji Agent Final Autonomous Blocked Report

## Final Status
B blocked-with-hard-evidence after Iteration 008.

The Iteration 007 route/normalizer/tactical plumbing fixes remained intact, but Iteration 008 could not produce promotion-quality score behavior. This round proved that historical artifacts can only produce heuristic distance-progress labels, and new objective logging under the current SDK run state collected only zero score-delta setup/deployment actions. The remaining blocker is objective-supervision collection: no current artifact contains non-zero score/capture/damage labels aligned to normal Move/Shoot/Occupy decisions.

## Current State
- Working directory: `/remote-home/lizilong/shenji_agent_fresh`.
- Original repo: `/remote-home/lizilong/shenji_agent`; not synchronized this round.
- Linux HEAD: `789ce7fe43d83016e1bc2d873492822a02ca9421`.
- Reverts: none.
- Windows sync: intentionally not attempted per Iteration 008 instruction.
- Windows target for later manual sync: `C:\Users\lizilong\Desktop\WHUAI\Projects\SituationAwareness\code\miaosuan20260119`.

## Fresh Clone Ahead Commits
- `ecd962e feat: add objective labels and diagnostics`
- `d3680ae chore: publish objective policy manifest`
- `1d7cdf3 fix: correct objective policy manifest sha`
- `e23a98f fix: keep objective diagnostics out of match stats`
- `789ce7f docs: finalize iteration 008 blocked evidence`

## Changed Files
- Source: `ai/policy.py`, `training/selfplay.py`.
- Script: `scripts/build_objective_labels.py`.
- Config: `configs/selfplay_smoke_objective_quality.yaml`.
- Tests: `tests/test_objective_label_builder.py`, `tests/test_objective_policy_integration.py`, `tests/test_objective_progress_diagnostics.py`.
- Manifest: `ai/weights/model_manifest.json`.
- Reports: `iteration_008.md`, `objective_label_source_audit.md`, `objective_label_blocked_analysis.md`, `stalemate_action_diagnosis.md`, this report.
- Artifacts: objective labels, model diagnostics, objective smoke sidecars, interrupted medium evidence.

## Objective Label Audit
- Existing artifacts label output: `experiments/shenji_agent/objective_labels/latest`.
- Labels from old artifacts: `1480`.
- Confidence: `1453 medium`, `27 low`, `0 high`.
- Label kinds: `1453 distance_to_objective_delta`, `27 final_outcome_credit`.
- Action types: `1453 Move`, `27 Shoot`.
- Missing from old artifacts: per-action score delta, occupy success, city capture delta, damage/kill delta, and unit position timeline.

## Objective Policy and Diagnostics
- Added heuristic objective score into Move/Shoot ranking because labels were not sufficient for a neural ObjectiveScorer.
- Manifest records `objective_enabled=true`, `objective_mode=heuristic_policy_score`, label counts, and no neural objective model trained.
- Added sidecars: `objective_progress_diagnostics.json`, `action_outcome_diagnostics.json`, `city_control_timeline.json`.
- Fixed diagnostics storage after an interrupted medium run showed match stats became too large.

## Model Diagnostics
Artifact: `experiments/shenji_agent/model_quality_after_iter008_objective_policy.json`.

- Situation/intent/tactical/normalizer loaded.
- Situation per-node variance mean: `0.0025503166689304635`.
- Remaining warning: `intent_negative_move_bias`.

## Self-Play Evidence
Objective smoke artifact: `experiments/shenji_agent/league_20260429_032914`.

- Scores: all legs `0.0-0.0`.
- Learned/move/shoot coverage: `0.0 / 0.0 / 0.0`.
- Action outcome records: `11520`.
- Action types: `303`, `333`, `GetOn`; no normal Move/Shoot/Occupy.
- Occupy opportunities/successes: `0 / 0`.
- Score delta actions: `0`.
- Damage actions: `0`.
- City value changes: `0`.
- SDK emitted repeated `did not pass authentication`.

Objective medium attempt: `experiments/shenji_agent/league_20260429_033459`.

- Status: interrupted after `3/5` paired matches due diagnostics serialization overhead.
- Completed paired scores: all `0.0-0.0`.
- Promotion: `accepted=false`, failure `compare_interrupted`.
- Same SDK authentication messages and no candidate valid/generated normal action counts in completed pairs.

Formal self-play was not run because smoke did not improve and the medium objective attempt did not complete or show non-zero score behavior.

## Promotion Gate
- A-complete not satisfied.
- No formal `promotion_accepted=true`.
- Current objective smoke promotion was smoke-only and invalid as final promotion evidence.
- Score/Elo/candidate superiority gates remain unproven.

## Verification
- `pytest tests -q`: `111 passed in 51.02s`.

## Remaining Blocker
The project is blocked on collecting score-producing objective supervision. Historical artifacts are insufficient, and current SDK self-play did not expose normal Move/Shoot/Occupy action maps under repeated authentication messages. The next minimal step is to restore normal SDK battle-phase execution, then rerun a short objective smoke with Iteration 008 sidecars enabled and rebuild labels from non-zero score/capture/damage events.

## Git Status
Remaining uncommitted entries are pre-existing/generated or local assets: `__pycache__`, `.tmp_claude/`, `ai/weights/weights/`, `ai/weights_incumbent_rule/`, `datasets`, `land_wargame_sdk/land_wargame_sdk`, and two untracked helper scripts.

## Manual Sync Instructions
Suggested source directory:
`/remote-home/lizilong/shenji_agent_fresh`

Suggested target directory:
`C:\Users\lizilong\Desktop\WHUAI\Projects\SituationAwareness\code\miaosuan20260119`

Sync important files:
- `ai/**/*.py`
- `training/**/*.py`
- `scripts/build_objective_labels.py`
- `configs/selfplay_smoke_objective_quality.yaml`
- `tests/**/*.py`
- `ai/weights/model_manifest.json`
- `experiments/shenji_agent/**/*.md`
- `experiments/shenji_agent/objective_labels/latest/summary.json`
- `experiments/shenji_agent/objective_labels/latest/objective_labels.jsonl`
- `experiments/shenji_agent/objective_labels/after_smoke_20260429_032914/summary.json`
- `experiments/shenji_agent/model_quality_after_iter008_objective_policy.json`
- `experiments/shenji_agent/league_20260429_032914/summary.md`
- `experiments/shenji_agent/league_20260429_032914/rating.json`
- `experiments/shenji_agent/league_20260429_032914/promotion.json`
- `experiments/shenji_agent/league_20260429_032914/participants.json`
- `experiments/shenji_agent/league_20260429_032914/objective_progress_diagnostics.json`
- `experiments/shenji_agent/league_20260429_032914/action_outcome_diagnostics.json`
- `experiments/shenji_agent/league_20260429_032914/city_control_timeline.json`

Do not sync:
- `.git`
- `.tmp_claude`
- `__pycache__`
- `.pytest_cache`
- full `datasets`, unless explicitly needed
- full `land_wargame_sdk`, unless explicitly modified
- full large historical `experiments` tree
- unrelated caches or local SDK/weight directories
