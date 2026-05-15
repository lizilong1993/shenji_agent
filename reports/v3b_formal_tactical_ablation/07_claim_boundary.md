# V3B Phase 7 — Claim Boundary Decision

**Date:** 2026-05-15

## Final Decision

**TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE**

## Evidence summary

| Evidence source | Status | Note |
|-----------------|--------|------|
| SDK replay pipeline | Operational | 30/30 games completed, 0 errors |
| Action-level divergence | Not measurable | Game stuck in deployment (SDK auth) |
| Outcome-level metrics | Identical | Score delta = 0.0 across all 15 pairs |
| Route reliability | Not measurable | No move routes attempted |
| V3A action-level diagnostic | Supported | Prior evidence confirms nonzero tactical scores |
| V3A synthetic multiseed | Completed | Bootstrap CIs computed on synthetic candidates |

## Why inconclusive

The SDK environment authentication is bound to MAC `02:42:ac:11:00:02`, but the current environment has MAC `02:42:ac:11:00:07`. This prevents the game from progressing past deployment stage. Key consequences:

1. **No battle-phase gameplay** — Agent never generates Move/Shoot actions through the learned policy
2. **No route planning** — Move routes are never computed, so route-empty metrics are unavailable
3. **No tactical decisions** — Tactical model is loaded but never invoked during action selection
4. **Identical outcomes** — Tactical on/off produce identical scores because deployment-stage behavior is deterministic

## Allowed wording

> Tactical outcome contribution remains inconclusive under current evaluation scale.

The replay infrastructure, protocol, and metrics pipeline are complete and functional. The only blocker is SDK authorization in the current environment.

## What IS supported

Based on V3A evidence and this evaluation:

| Claim | Supported | Evidence |
|-------|-----------|----------|
| Tactical checkpoint loads and produces nonzero scores | Yes | V3A checkpoint validation |
| Tactical branch measurably changes selected actions | Yes | V3A action diagnostic (synthetic) |
| Tactical influence improves outcome-level behavior | Not assessed | SDK auth blocks evaluation |
| Formal replay infrastructure is operational | Yes | 30/30 games complete, no errors |
| SDK replay pairing produces reproducible results | Yes | Identical results across all 15 pairs |

## Allowed terminal state

**TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE**

## Required next action for definitive outcome

To obtain a definitive outcome-level assessment, run the evaluation in an authorized Docker environment:
```bash
bash scripts/admin_run_authorized_selfplay.sh
```
