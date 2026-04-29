# Objective Label Source Audit

## Status
Iteration 008 audit complete for currently available artifacts and local datasets.

## Sources Checked
- `datasets/`: WG-StateGraph, tactical feature dataset, intent dataset.
- `experiments/shenji_agent/**/match_results.json`: historical self-play summaries and policy stats.
- `experiments/shenji_agent/**/diagnostics.json`, `move_diagnostics.json`: none present in the latest run tree before Iteration 008.
- `experiments/shenji_agent/**/summary.md`: score/gate summaries only.
- `land_wargame_sdk/Data/Data/scenarios/*.json`: scenario initial operators, cities, city value/owner, positions, unit value/type/blood.
- SDK observation schema visible through `ai/agent.py`: `operators`, `cities`, `valid_actions`, `reward`, `time`.
- `training/data.py`: intent, tactical, and graph-level situation dataset loaders.
- `training/selfplay.py`: before Iteration 008, match-level score and policy stats only; Iteration 008 adds action/objective sidecars.
- `ai/agent.py`: action generation and valid/generated action type diagnostics.
- `ai/policy.py`: move/shoot score inputs, route samples, trace samples.
- Tactical dataset: action-like tabular labels, but no rollout score deltas.
- Situation graph dataset: graph-level binary labels plus separate final score file.
- Intent dataset: trajectory frames for movement intent, not score or city capture labels.

## Required Questions
1. Per-turn score: not available in existing historical artifacts; available at SDK observation time as `reward` and now logged by Iteration 008 diagnostics.
2. Final score: available in `match_results.json` as final leg scores; WG-StateGraph has `score.csv` with final red/blue occupy/remain/attack/total/win fields.
3. City ownership changes: not available in historical self-play artifacts; initial city owner/value exists in scenarios; Iteration 008 diagnostics now logs city control timeline.
4. Occupy actions: historical artifacts had aggregate valid/generated counts only; Iteration 008 action outcomes log selected Occupy actions and ownership changes.
5. Damage/kill events: WG-StateGraph has `new-Human-Human-judge.csv` with hit/damage fields; historical self-play did not log per-turn damage; Iteration 008 diagnostics infer damage/kills from blood deltas.
6. Unit positions over time: intent dataset has trajectories; historical self-play artifacts did not store per-step positions; Iteration 008 diagnostics inspect per-step state.
7. Operator value/type: available in scenario/operator fields and observations.
8. Map city value: available in scenario `cities[].value`.
9. Distance-to-objective delta: constructible from scenario cities plus policy move route samples or per-step state.
10. City capture delta: not constructible from old artifacts, constructible from new per-step city owner diagnostics.
11. Damage dealt/taken delta: not constructible from old artifacts, constructible from new per-step operator blood diagnostics.
12. Action-level reward proxy: constructible with priority actual score delta, city capture, distance progress, damage/kill delta, final outcome credit, then heuristic.
13. Self-play replay labels: historical replay is insufficient because no per-step state was persisted; new diagnostics can generate replay-derived labels.
14. Scenario/map heuristic labels: constructible for distance-to-valuable-city and city value/owner.
15. Tactical dataset city/node/objective labels: partial only. It has operator type, start/target point, main/secondary objective point columns and binary sample labels, but no direct score delta/capture outcome.
16. Immediately constructible labels: distance-to-city progress from move route samples, final score credit, scenario city value/owner metadata, and new smoke-derived per-step score/city/damage labels.
17. Not constructible from old artifacts: true per-action score delta, occupy success, city ownership timeline, damage/kill deltas, and unit position timeline. Missing fields are per-step state snapshots and action outcome logs.

## Label Builder Result From Existing Artifacts
Command:
`PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages:. python3.10 scripts/build_objective_labels.py --input experiments/shenji_agent --scenario-root land_wargame_sdk/Data/Data/scenarios --output-dir experiments/shenji_agent/objective_labels/latest`

Output:
- labels: `1480`
- confidence: `1453 medium`, `27 low`, `0 high`
- label kinds: `1453 distance_to_objective_delta`, `27 final_outcome_credit`
- action types: `1453 Move`, `27 Shoot`
- score/capture/damage labels from old artifacts: `0`

## Audit Conclusion
Existing artifacts can produce only heuristic/medium distance-progress labels and low final-outcome credit labels. They cannot prove score-producing action quality because every historical medium leg was `0.0-0.0` and no old per-turn state exists. Iteration 008 therefore adds action outcome/objective progress diagnostics and uses objective-aware policy scoring as the immediate executable path.
