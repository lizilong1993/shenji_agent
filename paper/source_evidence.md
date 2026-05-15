# Source Evidence

| Topic | Evidence paths | Safe claim boundary |
|---|---|---|
| Agent implementation | `ai/agent.py`, `ai/policy.py` | implemented local agent with fallback logic |
| Model definitions | `ai/models.py` | standard LSTM/attention and GraphSAGE-style components |
| Training entrypoints | `training/supervised.py`, `training/selfplay.py`, `train_models.py` | entrypoints exist; not all produce formal paper results |
| Evaluation entrypoints | `run_evaluation.py`, `training/evaluate.py`, `training/selfplay.py` | local SDK evaluation implemented |
| Promotion sidecars | `experiments/shenji_agent/league_*/promotion.json` | sidecar-supported promotion evidence only |
| Candidate checkpoints | `ai/weights/intention_lstm.pth`, `ai/weights/situation_gcn.pth` | intent/situation checkpoints restored |
| SDK maps/scenarios | `land_wargame_sdk/Data/Data/maps/`, `land_wargame_sdk/Data/Data/scenarios/` | restored local SDK assets |
| Route diagnostics | `rating.json` policy route aggregates; `scripts/round005_route_diagnostic.py` | route reliability unproven |
| Tactical diagnostics | `scripts/run_v3a_tactical_action_diagnostic.py`; `reports/v3a_tactical_ablation/` | action-level contribution supported; outcome-level ablation executable via LegacyIntentModel |
| Baseline/ablation status | `ai/weights_incumbent_rule/model_manifest.json`, `configs/supervised_tactical_ablation.yaml` | partial baseline/ablation support |
| Known limitations | `paper/prewrite_package/04_claim_to_evidence_matrix.md` | unsupported claims are forbidden |
| Intent checkpoint fix | `reports/v3a_intent_checkpoint_fix/`; `scripts/inspect_intent_checkpoint_architecture.py`; `ai/models.py` (LegacyIntentModel, LegacySituationModel); `ai/policy.py` (auto-detection); `tests/test_v3a_intent_checkpoint_compatibility.py`; `paper/v3_evidence_package/INTENT_CHECKPOINT_COMPATIBILITY_STATUS.md` | LegacyIntentModel adapter restores intent checkpoint compatibility; full-agent replay unblocked; outcome ablation executable pending SDK replay loop |
| V3B formal outcome ablation | `reports/v3b_formal_tactical_ablation/` (full report set); `scripts/run_v3b_formal_tactical_ablation.py`; `scripts/v3b_pairing.py`; `scripts/v3b_metrics.py`; `experiments/shenji_agent/v3b_formal_tactical_ablation/` (raw data); `paper/v3_evidence_package/V3B_OUTCOME_ABLATION_STATUS.md` | Formal SDK-backed paired evaluation completed (15 pairs). Outcome contribution INCONCLUSIBLE due to SDK auth (MAC mismatch). Replay infrastructure operational and verified. |
| V3C SDK auth replay recovery | `reports/v3c_sdk_auth_recovery/` (full report set); `scripts/check_sdk_auth_environment.py`; `scripts/rebind_engine_config.py`; `ai/models.py` (LegacySituationModel backbone compatibility fix); `experiments/shenji_agent/v3c_smoke/` (smoke data) | SDK replay authentication restored at smoke level. |
| V3D authenticated tactical outcome ablation | `reports/v3d_authenticated_tactical_ablation/` (full report set); `experiments/shenji_agent/v3d_authenticated_tactical_ablation/` (raw data); `paper/v3_evidence_package/TACTICAL_CONTRIBUTION_STATUS.md`; `paper/v3_evidence_package/V3_READINESS.md`; `paper/v3_evidence_package/claim_to_evidence_matrix_v3.md`; `paper/v3_evidence_package/results_tables_v3.csv`; `paper/v3_evidence_package/results_tables_v3.json` | Authenticated paired tactical-on/off evaluation executed successfully (15 pairs, 30 games). Tactical branch invoked (nonzero scores in all ON runs). Route empty rate 0.0 across all runs. Outcome delta mean=-1.27, CI [-50.07, 34.82]. Outcome contribution weak or inconclusive under tested protocol. |
