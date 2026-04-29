# Iteration 010

## Goal

Reproduce the SDK authentication blocker in the formal repository after merging validated diagnostics, then document legal MAC-resolution paths.

## Current HEADs

- Formal repository HEAD: `0a9c7ff3736b5dd095ad8b1abc9fab164a477c3d`
- Fresh source HEAD: `0a9c7ff3736b5dd095ad8b1abc9fab164a477c3d`
- Closeout commit: `5d29772 docs: finalize sdk auth blocked evidence`
- Merge strategy: fast-forward from `fresh/main`
- Merge result: successful; formal repository now contains the validated Iteration 009 code, tests, configs, diagnostics, and reports.

## Test Result

```bash
python3.10 -m pytest tests -q
```

Result: `121 passed in 7.96s`.

## Smoke Command

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Smoke run path: `experiments/shenji_agent/league_20260429_102023`

## Auth And MAC Evidence

- SDK backend: `train_env.TrainEnv`
- SDK auth config path: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`
- SDK auth config exists: true
- SDK auth config bound MAC: `02:42:ac:11:00:45`
- Current container non-loopback MAC: `02:42:ac:11:00:02`
- Auth status: `failed_mac_mismatch`
- Auth failure records: `640`

## Battle Phase And Action Evidence

- `battle_phase_entered`: `false`
- Phase distribution: `deployment=640`
- Aggregate valid action counts: `Move=15360`, `WeaponLock=12800`, `ChangeState=16640`, `GetOn=5120`, `GetOff=1280`, `JMPlan=3840`
- Aggregate selected action counts: `303=5056`, `333=640`, `GetOn=64`
- Selected `Move` / `Shoot` / `Occupy`: `0`
- No battle action reason: `auth_failed_or_unverified_stage_stuck`

## Promotion Gate

`promotion.json` rejected promotion:

- `battle_phase_not_entered`
- `no_move_shoot_occupy_selected_actions`

This matches the fresh clone blocker and confirms the gate is rejecting setup/deployment-only evidence.

## Legal Resolution Work

Checked:

- Current interfaces and MACs with `ip link` and `/sys/class/net/*/address`.
- SDK auth config location through diagnostics.
- `Dockerfile`, `docker-compose.yml`, `.devcontainer/devcontainer.json`.
- Environment variables matching `ENGINE|AUTH|LICENSE|MAC|TRAIN|SDK|OFFLINE`.
- Project docs and runner scripts for offline/auth/license references.

Added:

- `docker-compose.auth-mac.yml`
- `scripts/run_selfplay_authorized_mac.sh`
- `docs/sdk_auth_runbook.md`
- `experiments/shenji_agent/sdk_auth_resolution_plan.md`

Current environment cannot safely mutate the already-running container's network MAC in place. The minimal legal external action is to recreate/run the Compose `dev` service with the authorized MAC override, or obtain a regenerated `.engine_config` for the actual runtime MAC.

The current container also lacks Docker CLI access (`docker` and `docker-compose` commands are not installed), so the Compose override must be executed by the host/admin environment.

## Next Target

If a container with MAC `02:42:ac:11:00:45` is available, rerun the battle-phase smoke and continue objective labels / policy scoring / formal promotion. Otherwise close as blocked with hard evidence.
