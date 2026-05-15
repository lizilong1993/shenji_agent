# V3B Outcome Ablation Status

**Date:** 2026-05-15

## Terminal State

**TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE**

## Overview

The V3B formal tactical outcome ablation was executed via SDK replay. The paired evaluation infrastructure was built, verified, and run across 15 pairs (3 scenarios × 5 seeds).

## What was built

| Component | Status | Location |
|-----------|--------|----------|
| Formal replay runner | Operational | `scripts/run_v3b_formal_tactical_ablation.py` |
| Paired pairing module | Operational | `scripts/v3b_pairing.py` |
| Metrics + bootstrap | Operational | `scripts/v3b_metrics.py` |
| Evaluation protocol | Defined | `reports/v3b_formal_tactical_ablation/01_protocol.md` |

## Results

| Metric | Value |
|--------|-------|
| Total games attempted | 30 (15 pairs × 2 modes) |
| Total games completed | 30 |
| Total errors | 0 |
| Score delta (mean) | 0.0 |
| Bootstrap 95% CI | [0.0, 0.0] |
| Route empty rate delta | 0.0 |

## Key finding

Tactical on/off produced identical outcomes across all 15 pairs. This is because the SDK environment authentication blocks full game replay — the game never progresses past deployment stage. The `.engine_config` is bound to MAC `02:42:ac:11:00:02`, but the current environment has MAC `02:42:ac:11:00:07`.

## Evidence outputs

| Output | Location |
|--------|----------|
| Start state | `reports/v3b_formal_tactical_ablation/00_start_state.md` |
| Protocol | `reports/v3b_formal_tactical_ablation/01_protocol.md` |
| Action divergence | `reports/v3b_formal_tactical_ablation/03_action_divergence_validation.md` |
| Smoke eval | `reports/v3b_formal_tactical_ablation/04_smoke_eval.md` |
| Multi-seed eval | `reports/v3b_formal_tactical_ablation/05_multiseed_eval.md` |
| Route audit | `reports/v3b_formal_tactical_ablation/06_route_reliability_audit.md` |
| Claim boundary | `reports/v3b_formal_tactical_ablation/07_claim_boundary.md` |
| Final decision | `reports/v3b_formal_tactical_ablation/FINAL_DECISION.md` |
| Paired results (full) | `experiments/shenji_agent/v3b_formal_tactical_ablation/paired_results.json` |
| Paired results (summary) | `experiments/shenji_agent/v3b_formal_tactical_ablation/paired_results_summary.json` |
| Bootstrap statistics | `experiments/shenji_agent/v3b_formal_tactical_ablation/bootstrap_statistics.json` |
