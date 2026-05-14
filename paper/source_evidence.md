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
