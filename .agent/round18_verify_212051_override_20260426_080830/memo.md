# Round 18 Verify 212051 Override

- generated_at: `2026-04-26T08:15:23.513441Z`
- scenario: `2120511121.json`
- env_backend: `train_env.TrainEnv`
- resolver map_id: `21`
- expected resolver map_id: `21`
- baseline unresolved map: `map_212`

## Before vs After
- before `map_212` move/learned coverage: `0.0388` / `0.0436`
- after `map_21` move/learned coverage: `1.0000` / `1.0000`
- before `empty_route_with_no_reachable_city`: `1710`
- after `empty_route_with_no_reachable_city`: `0`
- decisive_recovery: `True`

## Fit
- `map_212`: cities `2/5`, operators `37/58` in bounds
- `map_21`: cities `5/5`, operators `58/58` in bounds

