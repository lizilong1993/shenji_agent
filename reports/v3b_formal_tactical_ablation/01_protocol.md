# V3B Phase 1 — Evaluation Protocol

**Date:** 2026-05-15

## Tactical modes

| Mode | `tactical_mode` parameter | Behavior |
|------|--------------------------|----------|
| On | `"on"` | Policy loads tactical model, computes tactical scores, adds tactical contribution to action scores |
| Off | `"off"` | Policy skips tactical branch entirely (line 581: `if self.tactical_mode == "off": return 0.0`) |

Only `tactical_mode` changes between paired runs. All other parameters are identical.

## Pairing protocol

Each paired replay pair MUST share:

| Property | Constraint |
|----------|-----------|
| seed | Same integer seed for random initialization |
| scenario | Same scenario JSON file |
| map | Same map directory |
| player_info | Same player setup |
| env_step_info | Same env configuration |
| agent configuration | Same policy profile, artifact dir |
| **tactical_mode** | **CHANGES** — on vs off |

## Minimum evaluation size

**Target:** >= 5 seeds, >= 3 scenarios, >= 2 replay windows
**Minimum acceptable:** >= 3 seeds, >= 2 scenarios (if SDK runtime cost is high)

Each seed runs a complete game with tactical on AND tactical off, yielding a paired comparison.

## Scenario selection

Selected scenarios for evaluation:

| Scenario | Map | Difficulty | Rationale |
|----------|-----|-----------|-----------|
| `1231.json` | `map_123` | Standard | Previously used in SDK tests |
| `1531.json` | `map_53` | Standard | Different map topology |
| `1910631192.json` | `map_19` | Standard | Additional coverage |

Fallback: at minimum 2 scenarios if 3 exceeds runtime budget.

## Seed selection

Seeds: `[0, 1, 2, 3, 4]` — target 5 seeds.
Minimum acceptable: `[0, 1, 2]` — 3 seeds.

## Required metrics

### Primary metrics

| Metric | Definition |
|--------|-----------|
| win_rate | Fraction of games won by the agent |
| score_delta | Tactical-on score minus tactical-off score (paired) |
| rating_delta | Rating change difference if available |

### Secondary metrics

| Metric | Definition |
|--------|-----------|
| action_divergence_count | Number of turns where tactical on/off select different actions |
| tactical_invocation_count | Number of times tactical_score() is called and returns nonzero |
| route_empty_rate | Fraction of steps where route planner returns no valid route |
| move_success_rate | Fraction of move actions that successfully execute |
| policy_fallback_rate | Fraction of steps where policy falls back to default action |
| replay_completion_rate | Fraction of games that complete without crash/abort |

### Reliability metrics

| Metric | Definition |
|--------|-----------|
| crash_count | Number of games that crash |
| replay_abort_rate | Fraction of games aborted mid-way |
| invalid_action_count | Number of invalid action attempts |

## Statistical methods

| Method | Application |
|--------|------------|
| Paired difference | Per-seed tactical-on minus tactical-off |
| Bootstrap 95% CI | Resample paired deltas 1000 times |
| Mean ± std | Aggregate statistics across seeds |
| Effect size | Cohen's d if applicable (mean / std of paired deltas) |

## Transparency requirements

Report:
- Total games attempted
- Completed games
- Failed games
- Excluded games with exact exclusion reasons

No hidden filtering. Failed runs are recorded, not deleted.

## Output files

| File | Content |
|------|---------|
| `protocol.json` | Machine-readable protocol definition |
| `per_game_metrics/` | Per-game raw metrics (JSON) |
| `paired_results.json` | Paired tactical on/off results |
| `bootstrap_statistics.json` | Bootstrap confidence intervals |
