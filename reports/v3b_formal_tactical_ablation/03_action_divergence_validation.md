# V3B Phase 3 — Action Divergence Validation

**Date:** 2026-05-15

## Result

**TACTICAL_BRANCH_OUTCOME_TEST_NOT_MEANINGFUL** — Action divergence cannot be measured because SDK replay stays in deployment stage.

## Constraint

The SDK authentication is bound to MAC `02:42:ac:11:00:02`, but the current environment has MAC `02:42:ac:11:00:07`. Without authentication, the game environment does not progress past deployment stage (stage=1), so no tactical action-level decisions are executed during the game loop.

| Check | Value |
|-------|-------|
| SDK auth MAC | `02:42:ac:11:00:02` |
| Current MAC | `02:42:ac:11:00:07` |
| Environment ready | `False` |
| Game stage all steps | 1 (deployment only) |

## Attempted measurement

- Paired tactical-on vs tactical-off replay runs (15 pairs across 3 scenarios × 5 seeds)
- All 30 games completed with `max_step=1800`
- Policy diagnostics collected at end of each game
- Game never transitions to stage 2 (battle phase)

## Diagnostics collected

Since the policy's `use_ai_optimization` is enabled and models are loaded, the policy diagnostics are available but the action-level decision pipeline (Move/Shoot selection through `_generate_action`) was never exercised because the game never left deployment stage.

## Conclusion

**TACTICAL_BRANCH_OUTCOME_TEST_NOT_MEANINGFUL** — tactical on/off produces effectively identical outcomes under the current evaluation setup because the SDK authentication blocks full game replay. Action-level divergence cannot be validated in this environment.

To proceed, run the evaluation inside an authorized Docker environment matching the SDK's bound MAC address.
