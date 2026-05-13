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
| Tactical diagnostics | `scripts/round005_tactical_diagnostic.py`; manifests | tactical contribution unproven |
| Baseline/ablation status | `ai/weights_incumbent_rule/model_manifest.json`, `configs/supervised_tactical_ablation.yaml` | partial baseline/ablation support |
| Known limitations | `paper/prewrite_package/04_claim_to_evidence_matrix.md` | unsupported claims are forbidden |
| Asset provenance | `paper/prewrite_package/01_asset_provenance.md` | restored assets documented |
