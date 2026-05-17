# Round18 Analyze Relaunch Zero Coverage

Timestamp: 2026-04-27 02:23:29 UTC

Scope:
- Read-only diagnosis of whether `experiments/shenji_agent/league_20260426_160721` is a trustworthy terminal result for the post-212051fix branch.
- Compare it against the interrupted `experiments/shenji_agent/league_20260426_082209`.
- Inspect `.agent/round18_compare_relaunch_20260426_160631*` and relevant compare-side code paths only as needed.
- Write artifacts only inside this directory.

Plan:
1. Done: Collected terminal metrics and per-match policy-stats structure from `league_20260426_160721`.
2. Done: Compared against `league_20260426_082209` partial evidence, especially the first 91 pairs and policy-stat fields.
3. Done: Inspected compare-side code paths that produce participant loading and policy-stat aggregation.
4. Done: Decided that the zero-coverage result is internally consistent with `160721` sidecars but not trustworthy as a branch result; it most likely reflects a runtime action-path issue.
5. Done: Wrote concise memo and evidence files here.

Constraint note:
- I am not modifying mainline code, data, or repo-wide `.agent/*.md` files because the user restricted writes to this fresh analysis directory only.
