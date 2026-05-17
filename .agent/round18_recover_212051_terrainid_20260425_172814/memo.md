# Recover 212051 Terrain ID

- generated_at: `2026-04-25T17:28:14Z`
- target: `land_wargame_sdk/Data/Data/scenarios/2120511121.json`

## Search Locations Used

- Repo-local scenario/data/docs:
  - `land_wargame_sdk/Data/Data/scenarios/2120511121.json`
  - `land_wargame_sdk/run_offline_games.py`
  - `docs/docs.md`
- Repo-local historical artifacts:
  - `experiments/shenji_agent/league_*/map_list.json`
  - `experiments/shenji_agent/league_*/match_results.json`
  - `experiments/shenji_agent/league_*/partial_match_results.json`
  - `.agent/round18_probe_*`
- Repo-local runtime artifacts:
  - `replays/replay_1770632023.2594852.zip`
  - `logs/evaluation/*.log`
  - `logs/evaluation/results.json`
- Remote-environment sibling backup:
  - `/remote-home/lizilong/shenji_agent_backup226422/**`
  - including its `replays/`, `logs/`, and `training/cache/map_catalog.json`
- Internet/public search:
  - web search for `2120511121`, `2120511121.json`, `terrain_id`, `replay`, and official `wargame.ia.ac.cn` references

## Result

- Authoritative `terrain_id` for `2120511121.json`: **not found**

## Strongest Evidence

- The authoritative field is runtime observation/replay `terrain_id`, not `scenario_id`:
  - `/remote-home/lizilong/shenji_agent/docs/docs.md`
  - `https://wargame.ia.ac.cn/docs/reference/observations/`
- Official local runner also requires scenario and map data separately, so the scenario file is not self-describing for terrain binding:
  - `/remote-home/lizilong/shenji_agent/land_wargame_sdk/run_offline_games.py`
- Local replay proof of authority channel exists, but only for another scenario:
  - `/remote-home/lizilong/shenji_agent/replays/replay_1770632023.2594852.zip`
  - first frame contains `scenario_id=1231`, `terrain_id=123`
- For the target scenario, no local replay/observation/export/log frame with both:
  - `scenario_id = 2120511121`
  - explicit `terrain_id`

## Why Override Remains Unsafe

- All repo `map_id=212` records are downstream of repo inference, not upstream truth.
- The target scenario JSON contains no `terrain_id`.
- No public replay/catalog/source was found that explicitly binds `2120511121` to a terrain id.
- `map_21` may be a strong hypothesis from family structure, but it is still weaker than a real runtime `terrain_id` frame.

## Safest Next Action

- Acquire one real runtime observation or replay frame for `2120511121.json` from the SDK/offline runner and read `observation["terrain_id"]` directly.
- Until that exists, do **not** hard-code a silent override.
