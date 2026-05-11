# Shenji Mainline Program

Updated: 2026-04-23
Scope: define the hard-stop criterion for the `intent_lstm + situation_gcn` mainline in this repository.

## What "SOTA" means here

There is no public, protocol-matched leaderboard for the Miaosuan closed-loop land-wargame controller implemented in this repo. Recent wargame papers such as `WGSR-Bench` evaluate large-language-model strategic reasoning, not the online `TrainEnv` control loop used by `ai/agent.py`, so they are useful context but not a hard-stop surface for this project.

This project therefore adopts a SOTA-aligned stop rule by combining the closest public evaluation norms that do match the workload shape:

- `AlphaGo Zero`: promote only when the new policy beats the current best in a gated arena, not when training loss looks good.
- `AlphaStar`: evaluate through a league with exploiters because win rate against a single incumbent is not enough in non-transitive games.
- `SMAC` / `SMACv2` / `SMAC-Hard`: use held-out combat evaluation, emphasize hard scenarios, and reject benchmarks that can be solved by open-loop or overfit-to-opponent behaviour.
- `DeepNash`: robustness and "hard-to-exploit" behaviour matter in large imperfect-information strategy games, not just headline win rate.

This is a project inference from public evidence, not a claim that a published paper already defines an official Miaosuan stop line.

## Source of truth

- Fast development gate: `configs/selfplay_baseline.yaml`
- Hard stop / publishable stop gate: `configs/selfplay_sota_stop.yaml`
- Formal verdict surface: `experiments/shenji_agent/league_<timestamp>/rating.json`
- Required sidecars: `promotion.json`, `exploitability.json`, `participants.json`, `summary.md`, `match_results.json`, `map_list.json`

If `program.md` and a lighter development config disagree, `program.md` wins for any "can we stop training / can we claim the mainline is ready" question.

## Hard-stop criterion

Run:

```bash
python -m training.selfplay --config configs/selfplay_sota_stop.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

The project is allowed to stop the mainline training loop only when all of the following are true in two consecutive evaluation windows:

1. Protocol validity
   - `participants.json` shows `candidate` and `incumbent` are different artifact directories.
   - `match_results.json`, `rating.json`, `promotion.json`, `exploitability.json`, `summary.md`, and `map_list.json` all exist.
   - `map_list.json` records exactly `100` paired map items sampled from `land_wargame_sdk/Data/Data/scenarios/*.json`, with each item carrying `scenario`, `map_id`, `bucket`, and `order_index`.
   - The sampling protocol is unique-first random sampling without replacement; only if the scenario pool is smaller than `100` may the remaining slots be backfilled with repeats.
   - The paired matches are executed in the saved `map_list.json` order; the result is invalid if the run silently skips or reorders items.

2. Primary combat advantage
   - `rating.json.overall.point_estimate >= 0.60`
   - `rating.json.overall.ci95[0] > 0.55`
   - `rating.json.buckets.complex.point_estimate >= 0.60`
   - `rating.json.buckets.complex.ci95[0] > 0.55`

3. Derived strength floor
   - `promotion.json.elo_delta_lower_bound >= 35`
   - Interpretation: the 95% lower-bound win rate must still correspond to a positive Elo-equivalent gap, not just the point estimate.

4. Robustness against exploiters
   - `promotion.json.max_exploiter_win_rate <= 0.45`
   - No edited or mixed-strategy exploiter in `exploitability.json` may win more than 45% of its games against the candidate.

5. No bucket collapse
   - `promotion.json.max_bucket_regression <= 0.03`
   - Interpretation: no bucket is allowed to fall more than 3 percentage points below the neutral `0.50` line.

6. Learned-policy engagement
   - `promotion.json.learned_path_coverage >= 0.30`
   - `promotion.json.move_path_coverage >= 0.20`
   - `promotion.json.shoot_path_coverage >= 0.20`
   - Rationale: in this repo the learned policy currently affects `Move` and `Shoot`; if coverage is too low, a headline win-rate gain is not attributable to the learned path.

7. Stability
   - The full rule above must be satisfied in two consecutive league windows.
   - Do not stop on a single lucky run.

## Why these numbers

- `0.55` lower-bound floor: inherited from the classic self-play gating idea used by `AlphaGo Zero`, then strengthened here by applying it to the lower confidence bound instead of a single raw win-rate threshold.
- `0.60` point estimate on both overall and complex buckets: this is stricter than simple promotion and prevents easy or medium maps from hiding failure on the hardest terrain.
- `100` paired map items: the formal comparison unit is now `100` ordered paired map items, not `64/128` single-sided or ratio-driven bucket samples. This keeps the arena budget manageable while forcing every accepted comparison to finish a full saved map list and produce a stable final win-rate summary.
- Unique-first map sampling: the map list is redrawn every formal comparison window, but it must first maximize map diversity and only reuse scenarios when the pool itself is smaller than `100`.
- `0.45` exploiter cap: adapted from league-style robustness logic. If a fixed exploiter can still win nearly half the time, the mainline is not yet hard to exploit.
- Coverage thresholds: these are repo-specific inference thresholds derived from `ai/policy.py`, not copied from a paper.

## What is not a stop metric

These numbers remain diagnostics, not stop conditions:

- `metrics.json.intent.val_immediate_accuracy`
- `metrics.json.situation.val_accuracy`
- `metrics.json.tactical.val_accuracy`
- training loss
- a single `run_evaluation.py` score

They are useful for debugging data contracts and ablations, but SOTA-style stop decisions must come from closed-loop league evidence.

## Current gap to the stop line

The latest archived league run in this repo is still far from the new stop rule:

- `experiments/shenji_agent/league_20260327_141724/rating.json`
  - `overall.point_estimate = 0.5000`
  - `overall.ci95[0] = 0.5000`
  - `complex.point_estimate = 0.5000`
- `experiments/shenji_agent/league_20260327_141724/promotion.json`
  - `accepted = false`
- The archived league artifact predates the newer coverage-enriched mainline, so it is not valid evidence for the current stop gate.

Therefore the project is not stop-ready as of 2026-04-23.

## Execution note

- `training/evaluate.py` now enforces `elo_delta_lower_min`, `max_bucket_regression`, `max_exploiter_win_rate`, `min_move_path_coverage`, and `min_shoot_path_coverage`.
- `training/selfplay.py` now uses a saved `map_list.json` as the canonical execution order for formal candidate-vs-incumbent comparison.
- `configs/selfplay_baseline.yaml` stays as a lighter development gate, but the paired-map protocol itself is the same 100-item ordered comparison contract.
- Any final "stop / submit / freeze mainline" decision must use `configs/selfplay_sota_stop.yaml` plus the two-window rule above.
