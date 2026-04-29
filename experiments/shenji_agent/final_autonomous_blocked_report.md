# Shenji Agent Final Autonomous Blocked Report

## Final Status

B blocked-with-hard-evidence after Iteration 010.

Validated fresh-clone work has been merged back into the formal repository, and all subsequent verification was run from `/remote-home/lizilong/shenji_agent`. The remaining blocker is external SDK MAC authentication. Repository code now proves the failure, prevents setup/deployment-only promotion evidence, and provides a legal Compose-based path for an authorized runtime MAC.

## Repository State

- Formal repository path: `/remote-home/lizilong/shenji_agent`
- Fresh source HEAD merged: `0a9c7ff3736b5dd095ad8b1abc9fab164a477c3d`
- Formal HEAD after merge-back: `0a9c7ff3736b5dd095ad8b1abc9fab164a477c3d`
- Closeout evidence commit: `5d29772 docs: finalize sdk auth blocked evidence`
- Merge strategy: fast-forward from `fresh/main`
- Merge-back report: `experiments/shenji_agent/iteration_010_merge_back.md`
- Formal pre-merge tracked/index changes were preserved in Git stash entries:
  - `stash@{1}: pre-fresh-merge tracked formal worktree backup`
  - `stash@{0}: pre-fresh-merge untracked collision backup`

## New Iteration 010 Files

- `docker-compose.auth-mac.yml`
- `scripts/run_selfplay_authorized_mac.sh`
- `docs/sdk_auth_runbook.md`
- `experiments/shenji_agent/sdk_auth_resolution_plan.md`
- `experiments/shenji_agent/iteration_010_merge_back.md`
- `experiments/shenji_agent/iteration_010.md`
- `experiments/shenji_agent/league_20260429_102023/*`

## New Commits In Formal Repository

- `a33cdee fix: restore learned move and shoot engagement in selfplay`
- `9e36d05 fix: add move route diagnostics, fallback strategy, and context-aware occupy`
- `1860b86 fix: passthrough move route diagnostics and update manifest commit`
- `33c6b71 feat: add feature normalizer, tactical model, and training-compatible situation features`
- `ff4804f Expose self-play action type diagnostics`
- `ecd962e feat: add objective labels and diagnostics`
- `3d3e69a feat: add battle phase valid action diagnostics`
- `0a9c7ff docs: refresh iteration 009 final head`
- `5d29772 docs: finalize sdk auth blocked evidence`

## Tests

Command:

```bash
python3.10 -m pytest tests -q
```

Result: `121 passed in 7.96s`.

The first formal run failed at collection because `scikit-learn` was missing from the environment despite being declared in `requirements.txt`; installing it with `python3.10 -m pip install --user scikit-learn` restored the validated test baseline.

## SDK Auth Source

- Authentication failure string is not implemented in repository Python source.
- SDK authentication extension: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so`
- SDK engine config: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`
- `.engine_config` exists: true
- `.engine_config` bound MAC: `02:42:ac:11:00:45`
- Current non-loopback container MAC: `02:42:ac:11:00:02`
- Current auth status: `failed_mac_mismatch`

## Formal Smoke Evidence

Command:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Artifact path: `experiments/shenji_agent/league_20260429_102023`

- SDK output: repeated `did not pass authentication`
- `battle_phase_entered`: `false`
- phase distribution: `deployment=640`
- auth failure records: `640`
- aggregate valid action counts: `Move=15360`, `WeaponLock=12800`, `ChangeState=16640`, `GetOn=5120`, `GetOff=1280`, `JMPlan=3840`
- selected Move/Shoot/Occupy actions: `0`
- selected action counts: `303=5056`, `333=640`, `GetOn=64`
- promotion accepted: `false`
- promotion failures: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`

## Legal Resolution Attempt

Checked:

- Current network interfaces with `ip link` and `/sys/class/net/*/address`.
- SDK auth config path and decoded bound MAC through repository diagnostics.
- `Dockerfile`, `docker-compose.yml`, and `.devcontainer/devcontainer.json`.
- Environment variables matching auth/license/MAC/offline patterns.
- Project docs and runner scripts for `TrainEnv`, offline usage, auth, license, and engine config references.

Added:

- `docker-compose.auth-mac.yml`, setting service `dev` `mac_address: "02:42:ac:11:00:45"`.
- `scripts/run_selfplay_authorized_mac.sh`, which refuses to run promotion smoke unless the current MAC matches or instructs the authorized Compose command.
- `docs/sdk_auth_runbook.md`, documenting verification and fallback.

The current already-running container cannot be safely changed to a different MAC in place from repository code. The legal path is to recreate/run the configured service with the authorized MAC, or obtain a regenerated `.engine_config` for the actual runtime MAC.

Additional execution blocker: this container does not include Docker CLI access (`docker` and `docker-compose` are not installed), so the Compose override cannot be applied from inside the current runtime.

## Minimal External Action Required

Use an authorized container with MAC `02:42:ac:11:00:45`:

```bash
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml up -d --build dev
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec dev bash -lc \
  'cd /app && python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule'
```

If the platform cannot provide that MAC, request a new `.engine_config` from the SDK provider for the actual non-loopback runtime MAC.

## Once Auth Is Fixed

Run:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
python3.10 -m training.selfplay --config configs/selfplay_smoke_objective_quality.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
python3.10 scripts/build_objective_labels.py --input <latest_objective_smoke_dir> --output experiments/shenji_agent/objective_labels/latest
```

Continue objective scorer / policy integration only after `auth_status` is not `failed_mac_mismatch`, `battle_phase_entered=true`, and selected Move/Shoot/Occupy counts are non-zero.

## Git Status Notes

Expected uncommitted entries after closeout are limited to generated/tracked pycache modifications from test execution and pre-existing untracked local artifacts under `.agent`, `.codex`, local weights, helper scripts, and other local project assets. These were not included in commits because they are generated, local-only, or preserved pre-existing work.
