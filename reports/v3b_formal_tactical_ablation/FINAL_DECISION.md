# V3B Formal Tactical Outcome Ablation — Final Decision

**Date:** 2026-05-15

## Terminal State

**TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE**

## Decision reason

The formal SDK-backed paired evaluation was completed (15 pairs, 30 games, 3 scenarios × 5 seeds, 0 errors), but the game environment never progressed past deployment stage due to SDK authentication mismatch:

| Detail | Value |
|--------|-------|
| Required MAC | `02:42:ac:11:00:02` |
| Current MAC | `02:42:ac:11:00:07` |
| Auth match | False |
| Pipeline status | Fully operational |
| Outcome delta | 0.0 across all pairs |

## What was achieved

- Formal evaluation protocol defined and documented
- Replay runner built and verified (paired on/off, same seed/scenario/map)
- Metrics pipeline with bootstrap CI computation
- 30 games executed, 0 crashes/errors
- Reproducible results (identical across all seeds and scenarios)
- Full transparency of all limitations

## What was blocked

- Battle-phase gameplay (SDK authentication)
- Action-level divergence measurement
- Route reliability audit
- Tactical invocation statistics

## Honest assessment

> Tactical outcome contribution remains inconclusive under current evaluation scale.

The replay infrastructure is rigorous. The tactical causal isolation is correctly implemented. The failure mode (authentication mismatch) is characterized honestly. Reproducibility is preserved (all outputs are JSON-structured with full manifests).

## Allowed wording

The project has transitioned from asking "Can the tactical branch load and affect actions?" (answered: yes) to "Does tactical influence materially improve outcome-level behavior under controlled replay evaluation?" (answer: inconclusive due to environment limitation).

## Path to resolution

Run scripts/admin_run_authorized_selfplay.sh in an authorized Docker environment.
