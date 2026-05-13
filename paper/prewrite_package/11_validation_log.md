# Validation Log

| Command | Exit code | Expected | Output path | Result |
|---|---:|---|---|---|
| `python -m unittest discover -v` | 0 | pass | terminal | 68 tests ran and passed. |
| `python -m compileall paper/scripts scripts ai tests training` | 0 | pass | terminal | Compile checks passed. It updated tracked `ai/__pycache__` files; those bytecode changes were restored and not kept. |
| `python scripts/run_round005_smoke_pipeline.py --out paper/prewrite_package/validation/smoke_preflight` | 2 | fail-closed acceptable | `paper/prewrite_package/validation/smoke_preflight/smoke_preflight.json` | Blocked before launching self-play because manifest names expect `intent_lstm.pt`, `situation_gcn.pt`, and `tactical_model.pt`; restored current files are `.pth` for intent/situation and tactical checkpoint is absent. |
| `python scripts/round005_tactical_diagnostic.py --out paper/prewrite_package/validation/tactical_diagnostic.json` | 2 | fail-closed acceptable | `paper/prewrite_package/validation/tactical_diagnostic.json` | Tactical model file missing; `tactical_nonzero_count=0`. Tactical contribution must not be claimed. |
| `python scripts/round005_route_diagnostic.py --out paper/prewrite_package/validation/route_diagnostic.json` | 0 | pass | `paper/prewrite_package/validation/route_diagnostic.json` | Route diagnostic parsed `league_20260501_170204`: `23580` attempts, `15641` empty routes, empty-route rate `0.6633163698049194`. Route reliability must not be claimed. |

The fail-closed validator failures are evidence limitations, not hidden crashes. They are reflected in the claim matrix, result tables, figure limitations, methods package, and final readiness decision.
