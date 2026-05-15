# V3E Route Reliability Audit

## Route Reference Search

Search completed into `reports/v3e_claim_freeze/route_reference_grep.txt` (~150k lines, 17.6MB). Matches span V3B/V3D per-game metrics, league rating sidecar data, V2 top-tier gap audit, experiment update log, ai_reviewer drafts, and action diagnostics.

Key evidence sources identified:
- `reports/v3b_formal_tactical_ablation/per_game_metrics/*.json` — pre-auth V3B data (all zero moves)
- `reports/v3d_authenticated_tactical_ablation/per_game_metrics/*.json` — authenticated V3D data (non-zero moves)
- `experiments/shenji_agent/league_20260501_170204/rating.json` — full league evaluation (66.3% empty rate)
- `reports/v2_top_tier_upgrade/top_tier_gap_audit.md` — earlier route empty rate documentation
- `reports/experiment_update_log.md` — Chinese-language route failure documentation
- `reports/v3a_tactical_ablation/action_diagnostic/action_diagnostic.json` — synthetic diagnostic (not live)

## Route Diagnostic Recheck

Re-ran `scripts/round005_route_diagnostic.py` targeting `experiments/shenji_agent/league_20260501_170204`:

| Metric | Value |
|---|---|
| Total move attempts | 23,580 |
| Empty route count | 15,641 |
| Empty route rate | **66.3%** |
| By type: Fly | 7,182 / 8,716 (82.4% empty) |
| By type: Maneuver | 7,595 / 10,093 (75.2% empty) |
| By type: Walk | 864 / 4,771 (18.1% empty) |
| Failure reason | all `no_valid_move_candidate` |

This is the **full league evaluation** (100 map-scenario pairs, ~500 games). It represents the v1-v2 era route reliability baseline.

## V3D Per-Game Route Metrics (CORRECTED)

**Critical correction to prior audit:** The existing audit report erroneously claimed V3D `total_move_attempts=0` for all 30 games. This is **false**. The V3D per-game metrics show:

| Scenario | Seeds | Tactical ON moves | Tactical OFF moves | Total moves | Empty routes | Empty rate |
|---|---|---|---|---|---|---|
| 1231 | 5 | 35 | 25 | 60 | 0 | 0.0% |
| 1531 | 5 | 95 | 119 | 214 | 0 | 0.0% |
| 1910631192 | 5 | 158 | 158 | 316 | 0 | 0.0% |
| **Total** | **15** | **288** | **302** | **590** | **0** | **0.0%** |

Every V3D game has detailed per-attempt route evidence in `move_route_samples` arrays and per-game `move_route_aggregate` sections. All 590 move attempts used learned policy (learned_path_coverage=1.0). All route_failure_reasons are empty.

The prior audit's zero-move-claim appears to have confused V3B data (which DID have zero moves due to SDK auth blocking) with V3D data (authenticated replay with real battle-phase engagement).

## V3B Route Metrics (for comparison)

V3B was auth-blocked. All 30 games had `total_move_attempts=0`. The `route_empty_rate=0.0` in V3B is indeed a zero-denominator artifact. This is correctly documented in `reports/v3b_formal_tactical_ablation/05_multiseed_eval.md` ("SDK authentication blocks full game replay... The game environment stays in deployment stage").

## Q&A

**Q1: Was route empty rate computed over actual move attempts?**
Yes for V3D. Each game's `route_metrics.total_move_attempts` > 0 (range 4-35), and `route_empty_rate = route_empty_count / total_move_attempts`. The denominator is per-game non-zero. The aggregate in statistics.json computes the mean of per-pair rates. For V3B, the answer is no — denominator was zero.

**Q2: How many move attempts occurred?**
V3D: 590 total (288 tactical-on, 302 tactical-off). Full league: 23,580. V3B: 0.

**Q3: How many route-planned moves occurred?**
V3D: 590 successful routes (100% learned_path_coverage, 0 empty). Full league: 7,939 successful routes out of 23,580 attempts (15,641 empty).

**Q4: How many route-empty events occurred?**
V3D: 0 (non-trivial — denominator 590 > 0). Full league: 15,641. V3B: 0 (trivial — denominator 0).

**Q5: Did battle phase progress enough for route diagnostics to be meaningful?**
V3D: Partially. All 30 games hit the 1801-step maximum (not natural termination). However, the presence of 590 move attempts with tactical invocations (351 nonzero across ON runs) confirms battle phase was entered and meaningful move-route decisions occurred. Compare to full league where 129,890 turns with move occurred across ~500 games — V3D's sample is smaller but non-trivial.
Limitation: All games timed out at the step cap, so we don't know how route planning would behave in longer games or games approaching natural termination.

**Q6: Is V3D route_empty_rate=0.0 comparable to earlier high empty-route evidence?**
Not directly comparable due to different scenario sets. The 66.3% empty rate comes from the full league evaluation using ~100 map-scenario pairs. V3D used only 3 scenarios (1231, 1531, 1910631192) — a tiny subset. The V3D scenarios may not be representative of the conditions that trigger `no_valid_move_candidate` failures. However, within those 3 scenarios across 5 seeds, the 0% empty rate is robust and non-artifactual.

**Q7: Did earlier high empty-rate come from different runner, different phase, different sidecar, or different denominator?**
All of the above:
- **Runner**: The league evaluation (`league_20260501_170204`) used a different evaluation pipeline (full league runner) vs V3D's paired ablation runner
- **Phase**: League evaluation runs ~500 games across 100 map-scenario pairs; V3D runs 30 games across 3 scenarios
- **Sidecar**: League data comes from `rating.json/policy.move_route_aggregate`; V3D data comes from per-game `route_metrics` and `move_route_aggregate` in per-game JSON files
- **Denominator**: League denominator = 23,580 attempts; V3D denominator = 590 attempts
- **Software version**: League was run with the earlier v1-v2 era codebase; V3D uses the V3 codebase (same policy weights but potentially different infrastructure)

**Q8: Can route reliability be claimed as repaired, or only "no degradation observed in V3D tactical ablation"?**
Only: "In the authenticated V3D tactical ablation runs, no route-empty degradation was observed between tactical-on and tactical-off modes. Additionally, 0% route-empty rate was observed across 590 authenticated move-route decisions in 3 evaluation scenarios."

Route reliability cannot be claimed as broadly repaired because:
1. Only 3 scenarios tested (vs ~100 in the league evaluation)
2. Only 590 move attempts (vs 23,580 in the league)
3. All games timed out at step limit
4. The 66.3% empty rate from the league evaluation has not been re-evaluated with the V3 codebase

## Data Quality Notes

- The V3D per-game files contain `move_route_samples` arrays with per-attempt diagnostics (operator_id, target_source, route_len, route_empty, route_failure_reason, selected_by_learned). These are usable for detailed analysis.
- The `move_route_aggregate` sections in V3D files contain per-game total_attempts, empty_route_count, and route_failure_reasons. All show 0 failures.
- League sidecar `rating.json` route data is aggregated without per-attempt samples but includes breakdown by move type (Fly/Maneuver/Walk) and failure reasons.
- The `on_learned_move` and `off_learned_move` fields in the V3D multiseed_ablation.json's action_divergence section are all 0 — this is a different metric from the move_route_aggregate and appears to track "learned action selection" which may use a different definition.

## Classification

```
ROUTE_NO_DEGRADATION_OBSERVED_ONLY
```

**Justification:**
- V3D provides genuine route metrics: 590 move attempts with 0 empty routes across authenticated replay
- However, this is limited to 3 scenarios — insufficient to claim broad route reliability
- The route_empty_rate=0.0 is a non-trivial result (non-zero denominator), unlike V3B
- No route-empty degradation from tactical mode (both ON and OFF have 0% empty)
- Earlier 66.3% empty rate from full league evaluation is not re-evaluated with V3 codebase
- `ROUTE_RELIABILITY_SUPPORTED` would require broader scenario coverage or direct comparison with the league scenario set

## Allowed Wording

> In the authenticated V3D tactical ablation runs, no route-empty degradation was observed between tactical modes, with 0% empty rate across 590 move-route decisions in 3 evaluation scenarios. However, broader route reliability remains unproven: the full league evaluation (100 scenarios, 23,580 attempts) showed 66.3% empty rate, and this has not been re-evaluated with the V3 codebase.

## Forbidden Claims

- Route reliability is fixed
- Route planner is robust
- Empty-route issue is solved
- Route reliability improved (cannot be claimed from 3-scenario subset alone)
