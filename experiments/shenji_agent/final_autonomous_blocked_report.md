# Shenji Agent Final Autonomous Blocked Report

## Final Status

B blocked-with-hard-evidence after Iteration 011.

All legal engineering work that can be executed inside `/remote-home/lizilong/shenji_agent` without host/container-creation privileges has been completed. The remaining blocker is external SDK MAC authentication, not repository code.

## Repository State

- Formal repository path: `/remote-home/lizilong/shenji_agent`
- Starting HEAD for Iteration 011: `483a3cf205dc2aca5693a80c5dec913c70286c1e`
- Iteration 011 tooling/evidence commit: `b25ab657ee43c09e479556ecc0bf77600efbb11c`
- Latest current-environment smoke artifact: `experiments/shenji_agent/league_20260429_105446`
- Latest preflight JSON: `experiments/shenji_agent/sdk_auth_environment_check.json`
- Stash entries preserved:
  - `stash@{0}: On main: post-test pycache generated backup`
  - `stash@{1}: On main: pre-fresh-merge untracked collision backup`
  - `stash@{2}: On main: pre-fresh-merge tracked formal worktree backup`

## Why This Is Not A Code Pipeline Blocker

- Tests pass: `128 passed`.
- Learned engagement was fixed in prior iterations.
- Move route diagnostics and fallback routing were fixed.
- Objective diagnostics and label construction are implemented.
- Battle-phase valid-action diagnostics are implemented.
- Invalid setup/deployment-only evidence is rejected by promotion gates.
- Current blocker is SDK auth: the real backend remains in deployment/setup because SDK authentication fails.

## Current SDK Auth Evidence

- SDK backend: `train_env.TrainEnv`
- SDK version printed at runtime: `4.1.0`
- Authenticator extension: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so`
- Engine config: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`
- Decoded bound MAC: `02:42:ac:11:00:45`
- Current container MAC: `02:42:ac:11:00:02`
- Preflight status: `blocker_reason=mac_mismatch`
- Smoke auth status: `failed_mac_mismatch`
- Runtime output: repeated `did not pass authentication`

## Latest Smoke Evidence

Command:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Artifact: `experiments/shenji_agent/league_20260429_105446`

- `auth_status=failed_mac_mismatch`
- `bound_mac=02:42:ac:11:00:45`
- `current_macs=["02:42:ac:11:00:02"]`
- `battle_phase_entered=false`
- `turns_with_move=640`
- `turns_with_shoot=0`
- `turns_with_occupy=0`
- `selected_move_actions=0`
- `selected_shoot_actions=0`
- `selected_occupy_actions=0`
- `promotion_accepted=false`
- promotion failures: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`

## Why Current Environment Cannot Reach A Completed

- The user/runtime does not have host/container-creation privileges from inside this container.
- The current non-loopback MAC does not match the SDK `.engine_config` bound MAC.
- SDK auth fails before normal battle execution.
- TrainEnv does not enter battle phase under failed auth.
- No selected `Move` / `Shoot` / `Occupy` actions occur.
- Self-play evidence from this environment is setup/deployment-only and is correctly invalid for promotion.

## Automatic Research Completed

- Local docs/code search across `README.md`, `docs`, `scripts`, `configs`, `training`, `ai`, `land_wargame_sdk`, `Dockerfile`, `docker-compose.yml`, and `.devcontainer`.
- Generated local evidence:
  - `experiments/shenji_agent/local_auth_research_grep.txt`
  - `experiments/shenji_agent/local_auth_research_files.txt`
  - `experiments/shenji_agent/sdk_auth_auto_research.md`
- Public research:
  - official Wargame docs;
  - official TrainEnv API docs;
  - Docker Compose `mac_address` service documentation.
- SDK resource inspection:
  - wheel metadata;
  - installed dist-info;
  - visible package resources;
  - non-invasive `strings`/`ldd` inspection.
- No official offline auth bypass, env var, config selector, license regeneration CLI, or non-Docker current-permission workaround was found.
- Added preflight, admin handoff, and continuation scripts.

## New Handoff Tools

- `scripts/check_sdk_auth_environment.py`
  - Read-only JSON preflight.
  - Checks current MAC list, `.engine_config`, latest auth/phase/promotion diagnostics, blocker reason, next action, and commands after fix.

- `scripts/admin_run_authorized_selfplay.sh`
  - Host/admin entry point.
  - Checks Docker/Compose availability, starts the service with `docker-compose.auth-mac.yml`, verifies MAC `02:42:ac:11:00:45`, runs preflight and battle smoke, then validates auth/battle/action evidence.

- `scripts/continue_after_auth_fixed.sh`
  - Continuation path after auth is fixed.
  - Runs preflight, battle smoke, objective smoke, objective label build, tests, and formal self-play continuation.
  - Refuses to proceed if auth fails, battle phase is absent, or selected `Move` / `Shoot` / `Occupy` actions are zero.

## Exact Actions For A Host/Admin Operator

Enter the repository:

```bash
cd /remote-home/lizilong/shenji_agent
```

Check environment:

```bash
python3.10 scripts/check_sdk_auth_environment.py --output experiments/shenji_agent/sdk_auth_environment_check.json
```

Start authorized MAC self-play:

```bash
bash scripts/admin_run_authorized_selfplay.sh
```

Alternative direct wrapper:

```bash
bash scripts/run_selfplay_authorized_mac.sh
```

Acceptance checks:

```text
auth_status != failed_mac_mismatch
battle_phase_entered=true
selected Move/Shoot/Occupy > 0
```

Continue the task after auth is fixed:

```bash
bash scripts/continue_after_auth_fixed.sh
```

## If Authorized MAC Still Fails

- Ensure the runtime really has MAC `02:42:ac:11:00:45`.
- If it still fails, request a regenerated `.engine_config` for the actual runtime MAC.
- Or contact the SDK/platform authorization owner for official support.
- Do not patch, replace, or bypass `authenticator*.so`.
- Do not fake `.engine_config`, `auth_status`, battle phase, valid actions, or promotion evidence.

## Verification

Commands run:

```bash
python3.10 -m pytest tests/test_sdk_auth_environment_check.py -q
python3.10 -m pytest tests/test_auth_handoff_scripts.py -q
python3.10 -m pytest tests/test_phase_action_diagnostics.py -q
python3.10 -m pytest tests/test_selfplay_battle_phase_contract.py -q
python3.10 -m pytest tests/test_mainline_contract.py -q
python3.10 -m pytest tests -q
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Results:

- Targeted tests passed.
- Full tests passed: `128 passed`.
- Current-environment smoke reproduced SDK MAC auth blocker.

## Worktree Notes

Do not delete preserved user/local artifacts. The worktree contains pre-existing untracked files under `.agent`, `.codex`, local package files, local weights, and helper scripts. Test execution also leaves tracked `ai/__pycache__/*.pyc` modifications in this environment; these are generated artifacts and are not part of this handoff commit.
