# 2120511121 observation capture memo

- Date: 2026-04-25
- Target scenario: `land_wargame_sdk/Data/Data/scenarios/2120511121.json`
- Goal: recover one real observation/replay frame with `scenario_id = 2120511121` and read authoritative `observation["terrain_id"]`

## Search locations used

- Local scenario/map/runtime data:
  - `land_wargame_sdk/Data/Data/scenarios/2120511121.json`
  - `land_wargame_sdk/Data/Data/maps/map_{21,53,212,221}/`
  - `replays/`
  - `.agent/round18_*`
- Installed SDK / runtime entrypoints:
  - `/remote-home/lizilong/venvs/shenji/lib/python3.10/site-packages/train_env/`
  - `train_env.TrainEnv`
  - `train_env.env.wgloader.loader.Loader`
  - `train_env.env.wgloader.file_loader.FileLoader`
  - `train_env.env.authenticate.authenticator.Authenticator`
- Upstream/public interfaces:
  - `https://wargame.ia.ac.cn/docs/reference/observations/`
  - `https://wargame.ia.ac.cn/docs/reference/apis/`
  - `https://wargame.ia.ac.cn/docs/tutorials/replay_system/`
  - `https://wargame.ia.ac.cn/newReplay`

## Strongest evidence

1. `terrain_id` is an authoritative runtime observation field, but `TrainEnv.setup(...)` requires caller-supplied `scenario_data`, `basic_data`, `cost_data`, `see_data`, and `player_info`.
   - This means local offline environment setup cannot infer the true terrain from the scenario alone.
2. The installed SDK exposes `Loader`, but in this environment it does not bind `2120511121` to any concrete source:
   - `Loader.__doc__` says it can load from file / database / existing data.
   - For all tested constructor forms around `2120511121` (int, string, absolute scenario path, dicts with attempted `source` / `SOURCE` values), the resulting loader instance remained only `{'scenario': ...}`.
   - `load_terrain()` returned `None`.
   - `load_clock()`, `load_operator()`, and `load_weapon()` returned `None`.
   - `load_cities()` failed with `AttributeError: 'Loader' object has no attribute 'str_dir_cities'`.
   - So the local SDK shell is present, but no usable scenario-backed source binding is exposed here for this target scenario.
3. `FileLoader` is not authoritative for terrain recovery:
   - When given the same `2120511121.json` scenario and different local map bundles, `get_terrain_id()` echoed the supplied map:
     - `map_21 -> 21`
     - `map_212 -> 212`
     - `map_53 -> 53`
     - `map_221 -> 221`
   - Therefore `FileLoader` cannot recover the true terrain unless the caller already knows it.
4. The official replay system is an authoritative upstream entrypoint, but it is gated:
   - Official docs: `https://wargame.ia.ac.cn/docs/tutorials/replay_system/`
   - Docs say to open `http://wargame.ia.ac.cn/newReplay` and upload a replay JSON after login.
   - Live check from this environment: `https://wargame.ia.ac.cn/newReplay` redirects to `https://wargame.ia.ac.cn/login`.
   - No local environment variables, cookies, or saved credentials for this platform were found.

## Decisive conclusion

- No authoritative observation/replay frame for `2120511121` was recovered.
- No authoritative `terrain_id` was recovered.
- Exact blocker:
  - There is no local replay/observation export for `2120511121`.
  - The local SDK cannot self-resolve `2120511121 -> terrain_id`; it needs an already-bound data source.
  - The official replay/upstream path is login-gated, and no usable credentials/session are available in this environment.

## Minimal safe next action

- Obtain one real replay JSON or one authenticated platform session for a run of `2120511121`, then read `observation["terrain_id"]` directly from the first frame or replay payload.
- Until that exists, overriding `2120511121` to any map remains unsafe.
