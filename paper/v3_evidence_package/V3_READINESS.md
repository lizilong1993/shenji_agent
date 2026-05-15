# V3 Readiness

**State:** `IN_PROGRESS`

## Claim classification updates

| Claim | V2 status | V3A status | V3A-FIX status | V3B status | V3C status | V3D status | Change |
|-------|-----------|------------|----------------|------------|------------|------------|--------|
| tactical contribution | forbidden | partially supported (action-level only) | partially supported (action-level only, outcome executable) | inconclusive (auth-blocked) | auth recovered | weak/inconclusive (outcome) | upgraded from V3B (auth no longer blocks) |
| route reliability | forbidden | forbidden (unchanged) | forbidden (unchanged) | inconclusive (no move routes) | — | measured (0% empty) | unproven but measured |
| two-window success | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |
| baseline superiority | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |
| method novelty | unsupported | unsupported (unchanged) | unsupported (unchanged) | unsupported (unchanged) | — | — | — |
| reproducibility | partially repaired | same (unchanged) | same (unchanged) | same (unchanged) | — | — | — |
| public SOTA | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |
| top-tier readiness | forbidden | forbidden (unchanged) | forbidden (unchanged) | forbidden (unchanged) | — | — | — |

## Completed this round

### V3D (authenticated tactical outcome ablation)
- ✅ Authenticated paired tactical-on/off evaluation completed (15 pairs, 30 games)
- ✅ 15/15 pairs completed, 0 errors
- ✅ Tactical branch invoked in all tactical-on runs (nonzero scores), zero in all tactical-off runs
- ✅ Route metrics recorded (0% empty rate across all runs)
- ✅ Score deltas computed: mean=-1.27, CI [-50.07, 34.82]
- ✅ Claim gate decision: TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE
- ✅ V3 evidence package updated conservatively
- ✅ Tests and compileall pass

### V3C (SDK auth replay recovery)
- ✅ SDK MAC binding diagnosed and classified: SDK_MAC_BINDING_CONFIRMED
- ✅ `.engine_config` rebind applied via `authenticator.set_mac_time()`
- ✅ `mac_match=true`, `auth_environment_ready=true`
- ✅ Minimal authenticated replay smoke completed (1 pair)
- ✅ Authentication passed, deployment and battle phases entered
- ✅ Final state: `SDK_REPLAY_AUTH_RECOVERED`

### V3B (formal tactical outcome ablation)
- ✅ Formal replay runner built (`scripts/run_v3b_formal_tactical_ablation.py`)
- ✅ Paired evaluation protocol defined (3 scenarios × 5 seeds)
- ✅ Metrics pipeline with bootstrap CIs built

### V3A-FIX (intent checkpoint compatibility)
- ✅ LegacyIntentModel + LegacySituationModel implemented
- ✅ Policy auto-detects old vs new checkpoint format
- ✅ Full-agent compatibility smoke: PASS

### V3A (tactical ablation)
- ✅ Tactical on/off switch implemented
- ✅ Action-level diagnostic: nonzero variegated scores
- ✅ Smoke and multi-seed ablation completed

## Remaining blockers

1. ~~**SDK authentication**~~ — **RESOLVED**. MAC binding recovered.
2. **Route reliability unproven** — empty rate is 0% under V3D (good sign) but formal route reliability evaluation not yet conducted.
3. **Long-horizon and generalization evaluation** — not yet attempted.
4. **Outcome-level tactical contribution** — evaluated; result is WEAK_OR_INCONCLUSIVE.
