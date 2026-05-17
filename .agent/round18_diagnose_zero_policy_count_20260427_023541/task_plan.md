# Round18 Diagnose Zero Policy Count

Timestamp: 2026-04-27 02:35:41 UTC

Scope:
- Read-only diagnosis of the zero-policy-count anomaly centered on `experiments/shenji_agent/league_20260426_160721`.
- Compare against `experiments/shenji_agent/league_20260426_082209` and relaunch artifacts.
- Inspect only the relevant code-path details needed to push from symptom to likely cause.
- Write artifacts only in this directory.

Plan:
1. Done: Compared 082209 vs 160721 launch commands, sidecars, and run-dir evidence for environment/relaunch differences.
2. Done: Compared the exact commit-loaded agent/policy code for candidate and incumbent SHAs against the working-tree code.
3. Done: Identified runtime action-path degeneration as the most likely cause class.
4. Done: Wrote concise memo and evidence here.

Constraint note:
- I am not modifying mainline code, data, or repo-wide `.agent/*.md` files because the user restricted writes to this fresh diagnosis directory only.
