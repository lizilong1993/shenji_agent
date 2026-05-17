# Progress Log

- 2026-04-27 02:11:27 UTC: Created read-only diagnosis workspace and recorded scope/constraints.
- 2026-04-27 02:18 UTC: Collected detached-launch evidence for `.agent/round18_compare_post_212051fix_20260426_082101*`; confirmed missing `status.json` and empty launcher/stdout/stderr logs.
- 2026-04-27 02:21 UTC: Collected `league_20260426_082209` run-dir evidence; confirmed non-terminal cutoff at `91 / 100` with only `map_list.json`, `progress.json`, and `partial_match_results.json`.
- 2026-04-27 02:24 UTC: Collected `round18_compare_relaunch_20260426_160631*` and `league_20260426_160721` lineage evidence; confirmed a fresh relaunch started from `0 / 100` and later exited `0`.
- 2026-04-27 02:27 UTC: Compared `082209` and `160721` artifacts.
  - `map_list.json` is byte-identical across both run dirs.
  - Core match outcomes/order align across the first `91` matches.
  - Policy stats differ immediately, so `160721` is a fresh replay, not a continuation.
- 2026-04-27 02:29 UTC: Computed recoverable partial summary for `082209` from `partial_match_results.json`.
- 2026-04-27 02:30 UTC: Wrote final diagnosis memo.
