# Stalemate Action Diagnosis

## Latest Runs
- Pre-Iteration-008 medium reference: `experiments/shenji_agent/league_20260429_001512`.
- Objective smoke: `experiments/shenji_agent/league_20260429_032914`.
- Objective medium attempt: `experiments/shenji_agent/league_20260429_033459` interrupted after `3/5` paired matches because per-step diagnostics made heartbeat serialization too expensive.

## Pre-Iteration-008 Medium Evidence
- Match scores: all five paired matches ended `0.0-0.0`.
- Promotion: `accepted=false`.
- Candidate learned/move/shoot coverage: `1.0 / 1.0 / 1.0`.
- Valid/generated `Shoot`: `28 / 28`.
- Generated action aggregate included `Move: 137`, `Shoot: 28`.
- This proves learned move/shoot plumbing was active, but did not produce score.

## Objective Smoke Evidence
- Run: `league_20260429_032914`.
- Scores: two paired matches, all legs `0.0-0.0`.
- Promotion file says `accepted=true`, but this is smoke-only and not A-complete evidence.
- Candidate learned/move/shoot coverage: `0.0 / 0.0 / 0.0`.
- Candidate valid/generated action counts: empty.
- Action outcome diagnostics: `11520` action records; `303: 9520`, `333: 1920`, `GetOn: 80`.
- Occupy opportunities: `0`.
- Occupy successes: `0`.
- Score delta actions: `0`.
- Damage actions: `0`.
- City timeline: no city value change; unique controlled city value states stayed at `0.0/0.0`.
- SDK printed repeated `did not pass authentication`, and the run remained in deployment/setup-like action flow rather than exposing normal Move/Shoot/Occupy actions.

## Objective Medium Attempt Evidence
- Run: `league_20260429_033459`.
- Status: `interrupted`.
- Completed paired matches: `3/5`.
- Completed scores: all completed legs `0.0-0.0`.
- Promotion: `accepted=false`, failure `compare_interrupted`.
- Candidate valid/generated action counts in completed pairs: empty.
- The run produced a 46MB `match_results.json`/`partial_match_results.json` before the storage fix because detailed per-step diagnostics were embedded in match stats. This has been fixed in commit `e23a98f`.
- The same repeated SDK authentication messages appeared.

## Required Action-Level Questions
1. Action type distribution: pre-Iteration medium had learned Move/Shoot; objective smoke/partial medium had deployment actions only (`303`, `333`, `GetOn`) and no valid action map counts.
2. Move/Occupy/Shoot ratio: pre-Iteration medium had Move `137`, Shoot `28`, Occupy `0`; objective smoke had Move/Shoot/Occupy `0`.
3. Occupy opportunity count: objective smoke `0`.
4. Occupy selected count: objective smoke `0`.
5. Occupy success count: objective smoke `0`.
6. City owner changes: objective smoke `0`.
7. Units reaching city but not occupying: not observed in objective smoke because normal movement actions were not exposed.
8. Damage dealt: objective smoke `0`.
9. Kills: objective smoke `0`.
10. Score before/after: always `0.0 -> 0.0`.
11. Candidate vs incumbent objective progress: no separation in objective smoke or completed objective medium pairs.
12. Repeated action/oscillation: objective smoke stayed in setup/deployment action records rather than showing movement oscillation.
13. No-score shoot: pre-Iteration medium showed learned shoot but no score; objective smoke did not expose shoot.
14. No-progress move: pre-Iteration medium showed learned move but no score; objective smoke did not expose move.
15. Mutual attrition: objective smoke showed no damage; pre-Iteration medium did not have damage sidecars.
16. Scoring function advantage recording: Iteration 008 now records score/city/damage deltas, but collected smoke deltas are all zero.
17. Scenario victory mismatch: still plausible. The policy optimizes route/city heuristics, but available runs either produce no score from learned moves/shots or fail to enter normal action phase under current SDK authentication behavior.

## Conclusion
The current deeper blocker is twofold: old artifacts are insufficient for score-producing labels, and current objective self-play did not expose normal Move/Shoot/Occupy actions because the SDK run stayed in setup/deployment flow with repeated authentication messages. No reliable score-producing labels were collected.
