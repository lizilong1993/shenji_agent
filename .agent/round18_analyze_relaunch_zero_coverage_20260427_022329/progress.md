# Progress Log

- 2026-04-27 02:23:29 UTC: Created fresh analysis workspace and recorded scope/constraints.
- 2026-04-27 02:29 UTC: Collected `league_20260426_160721` terminal artifacts and relaunch sidecars.
- 2026-04-27 02:31 UTC: Collected `league_20260426_082209` partial evidence and confirmed same candidate/incumbent SHAs.
- 2026-04-27 02:36 UTC: Compared per-match policy stats across both runs.
  - `160721`: candidate/opponent zero counts in all `100` matches.
  - `082209`: candidate/opponent non-zero counts in all completed `91` matches.
- 2026-04-27 02:41 UTC: Inspected `training/selfplay.py`, `ai/agent.py`, and `ai/policy.py` to confirm where counts are produced and aggregated.
- 2026-04-27 02:46 UTC: Confirmed `map_list.json` is byte-identical across the two run dirs and the first `91` matches align on core outcome/order fields.
- 2026-04-27 02:48 UTC: Confirmed candidate/incumbent artifact mtimes predate both compares, reducing the chance of mid-lineage artifact replacement.
- 2026-04-27 02:50 UTC: Wrote final memo and evidence summary.
