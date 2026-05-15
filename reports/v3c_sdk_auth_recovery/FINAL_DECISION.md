# V3C SDK Authentication / Replay Recovery — Final Decision

**Date:** 2026-05-15

## Final State

**SDK_REPLAY_AUTH_RECOVERED**

## Decision reason

SDK authentication (MAC binding) was successfully recovered in the current environment. The minimal replay smoke test confirmed:

1. **Authentication**: Both tactical-on and tactical-off runs printed "did pass authentication"
2. **Deployment phase**: Entered and passed
3. **Battle phase**: Entered — 3 move attempts recorded in both runs
4. **Tactical mode operation**: Confirmed — `tactical_nonzero_count=3` (on) vs `tactical_nonzero_count=0` (off)
5. **Route metrics**: Collected with `route_empty_rate=0.0`
6. **Game completion**: Both games timed out at 200 steps (standard smoke limit), with winner Blue (132 vs 104)

## What was done

| Phase | Action | Status |
|-------|--------|--------|
| 1 | Snapshot + backup branch | Done |
| 2 | SDK auth audit — classified MAC binding | Done |
| 3 | MAC recovery — used `authenticator.set_mac_time()` to rebind `.engine_config` | Done |
| 4 | Minimal authenticated replay smoke (1 pair) | Done |
| 5 | Infrastructure validation (87 tests, compileall) | Done |
| 6 | Final decision | Done |

## Recovery method

The SDK authenticator's public `set_mac_time()` API method was used to rebind the `.engine_config` to the current container MAC (`02:42:ac:11:00:07`). Equivalent to running `scripts/rebind_engine_config.py --apply`.

- No SDK binaries were modified
- No authenticator code was patched
- No network interfaces were changed
- No tactical claims were modified

## Remaining limitation

The minimal smoke timed out at 200 steps rather than completing naturally. This is consistent with V3B's observed behavior (30 games all timed out) and is now understood to be a game mechanic (deployment+battle phases exceed 200 steps for scenario 1231), not an authentication issue.

## Path forward

To complete formal outcome ablation, run with higher `--timeout-steps` (e.g., 2000 as in V3B) or the default timeout.
