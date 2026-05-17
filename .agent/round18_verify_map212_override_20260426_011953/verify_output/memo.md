# Round 18 Runtime Verification: map_212 Explicit Overrides

- generated_at: `2026-04-26T01:25:50.057478Z`
- scenarios: `2120131194.json, 2120131294.json`
- env_backend: `train_env.TrainEnv`
- requested_device_env: `cuda`

## 2120131194.json
- resolved map: `map_94` bucket `simple`
- fit: `cities 5/5`, `operators 54/54`
- candidate coverage: `move 1.0000`, `learned 1.0000`, `shoot 1.0000`
- candidate empty_route_with_no_reachable_city: `0`
- pre-patch map_212 comparison: `move 0.0291 -> 1.0000`, `learned 0.0297 -> 1.0000`, `no_reachable 1634 -> 0`

## 2120131294.json
- resolved map: `map_94` bucket `simple`
- fit: `cities 5/5`, `operators 61/61`
- candidate coverage: `move 1.0000`, `learned 1.0000`, `shoot 1.0000`
- candidate empty_route_with_no_reachable_city: `0`
- pre-patch map_212 comparison: `move 0.0185 -> 1.0000`, `learned 0.0193 -> 1.0000`, `no_reachable 2394 -> 0`
