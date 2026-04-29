# Objective Label Blocked Analysis

## Status
Blocked-with-hard-evidence after Iteration 008 attempts.

## What Was Tried
- Audited datasets, self-play artifacts, SDK scenarios, observation schema, training loaders, agent diagnostics, and policy scoring inputs.
- Added `scripts/build_objective_labels.py`.
- Built labels from all historical self-play artifacts into `experiments/shenji_agent/objective_labels/latest`.
- Added per-step `objective_progress_diagnostics.json`, `action_outcome_diagnostics.json`, and `city_control_timeline.json`.
- Added objective-aware heuristic policy scoring and diagnostics in `ai/policy.py`.
- Ran model diagnostics after objective policy integration.
- Ran objective smoke self-play.
- Started objective medium self-play; interrupted after `3/5` paired matches because diagnostics serialization became too expensive, with completed pairs still all `0.0-0.0`.
- Fixed diagnostics storage overhead in `training/selfplay.py` so future runs keep detailed objective records in sidecars rather than match stats.

## Data Source Gaps
- Historical `match_results.json`: final scores and policy route stats only; no per-turn state, city ownership timeline, damage events, or action outcomes.
- Historical summaries/ratings/promotions: gate-level metrics only.
- Historical diagnostics/move diagnostics: no usable per-turn objective files existed before Iteration 008.
- Scenario JSON: has initial city value/owner and operator value/type/position, but no rollout outcomes.
- WG-StateGraph `score.csv`: final human-game score components, but not linked to self-play action choices or per-action deltas.
- WG-StateGraph labels: graph-level binary only.
- Tactical dataset: action-position classification signal, not score/capture/damage reward.
- Intent dataset: trajectories, not score/capture reward.

## Why Score-Producing Targets Still Cannot Be Trained Reliably
- Existing artifact labels: `1480` labels, `1453 medium` distance-progress labels and `27 low` final-outcome labels, `0 high` score/capture/damage labels.
- Existing reward signs: `1453` positive, `0` negative, `27` zero. This is not a balanced action-value dataset and cannot train a robust scorer.
- Objective smoke labels: `13440 high` score-delta labels, all exactly zero. Action types were setup/deployment (`303`, `333`, `GetOn`, `None`), not normal Move/Shoot/Occupy.
- No current run produced non-zero `score_delta`, city capture, damage, or kill labels.
- Current SDK runs printed repeated `did not pass authentication` and did not expose normal valid action maps in objective smoke/partial medium.

## Minimal Viable Next Data Collection
- Restore SDK authentication/normal battle phase so valid Move/Shoot/Occupy actions are exposed.
- Keep Iteration 008 sidecars enabled after the storage fix.
- Run a short smoke until action outcome diagnostics show non-zero valid Move/Occupy/Shoot opportunities.
- Only then generate labels with high-confidence score/capture/damage fields.
- Train an objective scorer only after labels contain both positive and negative/non-progress examples.

## Blocker
The remaining blocker is not route plumbing. It is the absence of usable score-producing objective supervision plus the current SDK/authentication state preventing new high-value objective labels from being collected in smoke self-play.
