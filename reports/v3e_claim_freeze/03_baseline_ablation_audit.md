# V3E Baseline and Ablation Coverage Audit

Date: 2026-05-15
Branch: main (0f5bd0d)

## Objective

Inventory every available type of baseline and ablation evidence to determine whether coverage is sufficient for a manuscript. This audit searches across `reports/`, `paper/v3_evidence_package/`, `paper/prewrite_package/`, `experiments/shenji_agent/`, `configs/`, `scripts/`, and `ai/weights*`.

## Inventoried Variants

### a) Rule incumbent baseline

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `experiments/shenji_agent/league_20260501_170204/participants.json` (candidate=challenger, incumbent=baseline), `rating.json` (pe=0.87), `promotion.json` (accepted=true). Weights: `ai/weights_incumbent_rule/model_manifest.json` (snapshot_role=bootstrap_rule_baseline, 0 managed files = pure rule) |
| **Executable** | Yes -- `training/selfplay.py` with `configs/selfplay_baseline.yaml` (paired_matches=100) |
| **Formal or smoke** | FORMAL multi-game league evaluation (100 paired maps) |
| **N games/pairs** | 100 paired matches, 2 exploiter types (4 matches each) |
| **Outcome metric** | Win rate 0.87 (95% CI [0.80, 0.93]), Elo delta lower bound 240.8, complex wr 0.82 (CI [0.64, 0.95]), learned_path_coverage=0.37, move_path_coverage=0.34, shoot_path_coverage=1.0 |
| **Limitations** | Single SDK, single domain. Incumbent is rule-only with no model weights (bootstrap_rule_baseline profile), not a strong external baseline. Candidate falls back to rule on ~66% of move actions (empty route rate). |
| **Can it support a manuscript claim?** | YES. The candidate convincingly beats the rule incumbent. This is the core evaluation result. |

### b) Tactical on vs tactical off (V3D ablation)

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `reports/v3d_authenticated_tactical_ablation/multiseed_ablation.json`, `statistics.json`, `04_statistics.md`, `05_claim_gate.md`, `FINAL_DECISION.md`. Raw data: `experiments/shenji_agent/v3b_formal_tactical_ablation/` |
| **Executable** | Yes -- `scripts/run_v3b_formal_tactical_ablation.py` |
| **Formal or smoke** | FORMAL authenticated paired multi-seed evaluation |
| **N games/pairs** | 15 pairs = 30 games (5 seeds x 3 scenarios: 1231, 1531, 1910631192) |
| **Outcome metric** | Score delta mean -1.27 (bootstrap 95% CI [-50.07, 34.82]), Cohen's d -0.015, win rate identical 0.533, route empty rate 0.0 for all 30 games. Tactical nonzero counts: ON=351, OFF=0. Action divergence confirmed (nonzero tactical scores in all ON runs). |
| **Limitations** | Outcome-level contribution weak/inconclusive. Only 3 scenarios. All timeout at 1801 steps (max 2000). High variance (std 85.84). No move/shoot learned actions executed in either mode. |
| **Can it support a manuscript claim?** | YES -- as action-level/diagnostic evidence only ("tactical branch changes action selection under authenticated paired evaluation; outcome-level contribution weak or inconclusive"). Not as a main result. |

### c) Intent-only ablation

| Field | Value |
|---|---|
| **Exists** | No |
| **Evidence path** | N/A |
| **Executable** | No config or script exists for intent-only evaluation |
| **Formal or smoke** | N/A |
| **N games/pairs** | N/A |
| **Outcome metric** | N/A |
| **Limitations** | All training configs enable both intent AND situation together. `supervised_baseline.yaml` only toggles tactical (enabled=false). No config sets situation.enabled=false or intent.enabled=false individually. |
| **Can it support a manuscript claim?** | NO -- does not exist |

### d) Situation-only ablation

| Field | Value |
|---|---|
| **Exists** | No |
| **Evidence path** | N/A |
| **Executable** | No config or script exists for situation-only evaluation |
| **Formal or smoke** | N/A |
| **N games/pairs** | N/A |
| **Outcome metric** | N/A |
| **Limitations** | Like intent-only, no config toggles intent/situation independently. |
| **Can it support a manuscript claim?** | NO -- does not exist |

### e) No-learned fallback

| Field | Value |
|---|---|
| **Exists** | Partial (tracked as diagnostic, not as experimental variant) |
| **Evidence path** | `ai/policy.py` (fallback logic in LearnedPolicy), rating.json tracks `fallback_reasons: no_valid_move_candidate`, `learned_path_coverage: ~0.37` |
| **Executable** | Not as a standalone ablation experiment. The configs always have intent+situation enabled. |
| **Formal or smoke** | Tracked as a diagnostic metric during promotion evaluation |
| **N games/pairs** | N/A -- not an experimental variant |
| **Outcome metric** | Fallback rate tracked: 15641/23580 move attempts had empty route (66.3% empty rate). Candidate uses rule-based fallback when learned policy has no valid candidate. |
| **Limitations** | The rule incumbent serves as a proxy "no-learned" baseline at the agent level, but there is no experimental condition where the candidate's learned components are disabled while keeping the candidate architecture otherwise identical. |
| **Can it support a manuscript claim?** | PARTIALLY -- candidate-vs-incumbent comparison serves as proxy for learned-vs-non-learned |

### f) Objective-only baseline

| Field | Value |
|---|---|
| **Exists** | No |
| **Evidence path** | `scripts/build_objective_labels.py` exists. Rating.json shows objective_score as a component of combined_score (mean 130.4 for move). |
| **Executable** | No formal evaluation exists for objective-only play |
| **Formal or smoke** | N/A |
| **N games/pairs** | N/A |
| **Outcome metric** | N/A |
| **Limitations** | Objective scores are tracked as part of the combined scoring pipeline but no objective-only baseline evaluation was conducted. |
| **Can it support a manuscript claim?** | NO -- does not exist |

### g) Random/legal baseline

| Field | Value |
|---|---|
| **Exists** | No |
| **Evidence path** | N/A |
| **Executable** | No random-legal baseline script, config, or experiment |
| **Formal or smoke** | N/A |
| **N games/pairs** | N/A |
| **Outcome metric** | N/A |
| **Limitations** | Not implemented at any level. |
| **Can it support a manuscript claim?** | NO -- does not exist |

### h) Route-aware / route-disabled variants

| Field | Value |
|---|---|
| **Exists** | No |
| **Evidence path** | Route metrics tracked in `rating.json` (move_route_aggregate, empty_route_count). V3D shows route_empty_rate=0.0 in all 30 games. Route diagnostics: `reports/v3e_claim_freeze/02_route_reliability_audit.md`. |
| **Executable** | No route-specific ablation config or experiment exists |
| **Formal or smoke** | Route metrics tracked within promotion evaluation, not as separate ablation |
| **N games/pairs** | Route metrics across 100-game leagues and 30-game V3D evaluation |
| **Outcome metric** | V3D: 0% empty rate. Latest league: 15641/23580 empty moves (66.3% empty route rate). |
| **Limitations** | No route-on vs route-off ablation. Route reliability is measured but not causally isolated. High empty route rate in full league evaluation. |
| **Can it support a manuscript claim?** | PARTIALLY -- route metrics are tracked but no comparison variant exists for causal claims |

### i) Exploiter / robustness evaluation

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `experiments/shenji_agent/league_20260501_170204/exploitability.json`. Also present in many other league directories. |
| **Executable** | As part of the promotion-gate evaluation pipeline (config promotion_gate.max_exploiter_win_rate) |
| **Formal or smoke** | SMOKE -- 4 matches per exploiter, part of promotion gate evaluation |
| **N games/pairs** | 4 matches per exploiter type (aggressive, occupy). Latest: aggressive_exploiter wr=0.0, occupy_exploiter wr=0.25. |
| **Outcome metric** | max_exploiter_win_rate=0.25 (gate threshold 0.45 passed). Both exploiters have learned_path_coverage=1.0. |
| **Limitations** | Only 2 exploiter types, only 4 matches each, embedded within promotion evaluation not a standalone robustness study. Not statistically powered for independent claims. |
| **Can it support a manuscript claim?** | WEAKLY -- sufficient to show exploiter resistance passes promotion gate, but underpowered for standalone robustness claim |

### j) Static training metrics

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `experiments/shenji_agent/*/summary.md` (9 training run summaries with accuracy metrics). Key file: `20260428_235555/summary.md` (only run with tactical enabled). |
| **Executable** | Already produced by `training/supervised.py` |
| **Formal or smoke** | Diagnostic metrics from individual supervised training sessions |
| **N data points** | 9 training summaries |
| **Outcome metric** | intent_immediate_accuracy: 0.618-0.622 (all runs), tactical_accuracy: 0.962 (one run, other 8 skipped/disabled), situation_accuracy: 0.545-0.591 (all runs) |
| **Limitations** | Only accuracy metrics (no loss curves, no per-epoch tracking). All runs use limited training data (4096 intent samples). Tactical accuracy only measured once. Intent accuracy only measured on medium-complexity maps (complex and simple show 0.0). |
| **Can it support a manuscript claim?** | WEAKLY -- provides basic model quality evidence but not comprehensive or linked to game-level outcome |

### k) Authenticated replay smoke (V3C)

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `reports/v3c_sdk_auth_recovery/replay_smoke.json`, `03_replay_smoke.md` |
| **Executable** | Yes -- `scripts/run_v3b_formal_tactical_ablation.py --smoke --out experiments/shenji_agent/v3c_smoke` |
| **Formal or smoke** | SMOKE (1 pair) |
| **N games/pairs** | 1 pair (1 game each for tactical-on and tactical-off) |
| **Outcome metric** | Auth fixed: deployment+battle phases entered. Tactical nonzero: ON=3, OFF=0. Game: timeout at 200 steps. |
| **Limitations** | Single-pair smoke only, limited steps (200), not formal evidence. |
| **Can it support a manuscript claim?** | Infrastructure verification only -- not manuscript-worthy on its own |

### l) Authenticated multi-seed replay (V3D formal)

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `reports/v3d_authenticated_tactical_ablation/multiseed_ablation.json`, `statistics.json`, `04_statistics.md` |
| **Executable** | Yes -- `scripts/run_v3b_formal_tactical_ablation.py` |
| **Formal or smoke** | FORMAL multi-seed authenticated paired evaluation |
| **N games/pairs** | 15 pairs = 30 games (5 seeds x 3 scenarios) |
| **Outcome metric** | Score delta mean -1.27, bootstrap 95% CI [-50.07, 34.82], Cohen's d -0.015, win rate delta 0.0 |
| **Limitations** | 3 scenarios only, outcome inconclusive, no move/shoot learned actions observed |
| **Can it support a manuscript claim?** | YES -- as the core tactical ablation result (action-level/diagnostic) |

### m) Two-window promotion evidence

| Field | Value |
|---|---|
| **Exists** | Partial (single window only) |
| **Evidence path** | `experiments/shenji_agent/league_20260501_170204/` and ~30 other league dirs. `league_state.json` shows consecutive_windows=0. |
| **Executable** | Yes -- via full promotion evaluation pipeline |
| **Formal or smoke** | FORMAL league evaluations (100 games each) |
| **N games/pairs** | 100 per league for the latest; many leagues exist with various outcomes |
| **Outcome metric** | Latest: accepted=true, consecutive_windows=0, champion never updated from `ai/weights`. Historical: all 8 entries in historical_mains are `ai/weights`. |
| **Limitations** | Two-window hard-stop success NOT proven. Single-window only. One earlier league (league_20260501_085914) showed consecutive_windows=1 but latest resets to 0. |
| **Can it support a manuscript claim?** | PARTIALLY -- single window supports "one verified promotion window exists" but not two-window |

### n) Pre-write package baseline evidence

| Field | Value |
|---|---|
| **Exists** | Yes |
| **Evidence path** | `paper/prewrite_package/04_claim_to_evidence_matrix.md`, `05_results_for_tables.md`, `results_tables.json`, `formal_artifacts_index.json`, `claim_to_evidence_matrix.json` |
| **Executable** | Read-only evidence package (snapshot at V2 level) |
| **Formal or smoke** | Pre-write inventory of available evidence |
| **N data points** | 6 formal leagues inventoried with sidecar presence/absence analysis |
| **Outcome metric** | Systematic claim-to-evidence matrix with 14 claims tracked. Results tables CSV/JSON. |
| **Limitations** | Pre-write quality, snapshotted before V3D tactical ablation. Does not add new experimental evidence. Claim C10 (ablation) rated "partially supported -- preliminary baseline/ablation scaffolding." |
| **Can it support a manuscript claim?** | YES -- as organizational package for paper drafting, with conservative claim boundaries already defined |

## Summary Table

| Variant | Exists | Formal/Smoke | N | Supports claim? |
|---|---|---|---|---|
| Rule incumbent baseline | YES | FORMAL | 100 maps | YES -- core evaluation |
| Tactical on/off (V3D) | YES | FORMAL | 30 games | YES -- diagnostic only |
| Intent-only ablation | NO | -- | -- | NO |
| Situation-only ablation | NO | -- | -- | NO |
| No-learned fallback | PARTIAL | Diagnostic | -- | PARTIAL -- proxy only |
| Objective-only baseline | NO | -- | -- | NO |
| Random/legal baseline | NO | -- | -- | NO |
| Route variants | NO | -- | -- | NO |
| Exploiter robustness | YES | SMOKE | 8 matches | WEAKLY |
| Training metrics | YES | DIAGNOSTIC | 9 runs | WEAKLY |
| Auth smoke (V3C) | YES | SMOKE | 1 pair | Infrastructure only |
| Auth multiseed (V3D) | YES | FORMAL | 30 games | YES -- core ablation |
| Two-window promotion | PARTIAL | FORMAL | 100/league | PARTIAL -- single only |
| Prewrite baselines | YES | INVENTORY | 6 leagues | Organizational |

## Evidence sources checked

- `reports/v3d_authenticated_tactical_ablation/` -- full V3D report set (11 files)
- `reports/v3c_sdk_auth_recovery/` -- V3C auth recovery (6 files)
- `reports/v3b_formal_tactical_ablation/` -- V3B pre-auth protocol (8 files)
- `reports/v3a_tactical_ablation/` -- V3A tactical diagnostic (9 files)
- `paper/v3_evidence_package/` -- V3 evidence matrix, status, results (7 files)
- `paper/prewrite_package/` -- Pre-write claim matrix, artifact inventory (20+ files)
- `experiments/shenji_agent/league_20260501_170204/` -- Latest league sidecars (11 files)
- `configs/*.yaml` -- 11 config files inspected for ablation/enablement flags
- `ai/weights/` and `ai/weights_incumbent_rule/` -- Model manifests
- `experiments/shenji_agent/*/summary.md` -- 9 training run summaries
- `experiments/shenji_agent/*/metrics.json` -- Training metrics (map catalogs)
- `experiments/shenji_agent/model_quality_*.json` -- Model quality diagnostics

## Classification

```
BASELINE_COVERAGE_INSUFFICIENT_FOR_STRONG_PAPER
```

## Justification

The project has exactly two formal comparison types:
1. **Rule incumbent vs learned candidate** (100-game promotion league with 87% win rate) -- strong for what it is.
2. **Tactical-on vs tactical-off** (30-game authenticated paired ablation) -- outcome weak but action-level executability and route reliability established.

All other standard baselines and ablations are missing or underpowered:
- No intent-only ablation (cannot isolate component contributions)
- No situation-only ablation (cannot isolate component contributions)
- No objective-only baseline (cannot measure objective score contribution)
- No random/legal action baseline (most basic sanity check)
- No route-on vs route-off ablation (cannot causally measure route impact)
- No learned policy completely disabled ablation
- No comprehensive robustness study (only 8 exploiter games, 2 types)
- No two-window promotion success

This is insufficient for a paper making strong comparative or ablation claims. The manuscript must position baselines as "preliminary" and "limited to rule-incumbent comparison with diagnostic-level tactical ablation."

## Allowed claim

> The project includes a rule-incumbent comparison and authenticated tactical-on/off ablation, but broader baseline and component-ablation coverage remains limited.

## Forbidden claims

- The method outperforms strong baselines
- The method is SOTA
- The method is broadly validated across baseline families
- Comprehensive ablation study completed
- Component contributions isolated and validated
- Route planner reliability is proven
- Two-window promotion success
