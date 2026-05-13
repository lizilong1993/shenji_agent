# Methods Package

1. Environment and SDK setup: local Python environment with `land_wargame_sdk/land_wargame_train_env-4.1.0-*.whl`; SDK auth may be required.
2. Scenario/map assets: `land_wargame_sdk/Data/Data/scenarios/` and `land_wargame_sdk/Data/Data/maps/`.
3. Agent architecture: `ai/agent.py`, `ai/policy.py`.
4. Learned components: `ai/models.py`, `ai/weights/intention_lstm.pth`, `ai/weights/situation_gcn.pth`.
5. Fallback logic: rule/objective fallback in `ai/policy.py` and `ai/agent.py`.
6. Route planner: `Map.gen_move_route` in `ai/agent.py`.
7. Evaluation protocol: `training/selfplay.py` and `configs/selfplay_*.yaml`.
8. Promotion gate: `promotion.json` sidecars under `experiments/shenji_agent/league_*`.
9. Sidecar validation: `tests/test_promotion_artifact_contract.py` and recovered Round-005 diagnostics.
10. Reviewer/audit process: claim matrix, formal artifact index, and validation log in this package.
11. Reproducibility commands: `python -m unittest discover -v`; `python -m compileall paper/scripts scripts ai tests training`; diagnostics listed in `11_validation_log.md`.
12. Known unavailable components: tactical contribution proof, two-window proof, protocol-matched public baseline/SOTA evidence.
