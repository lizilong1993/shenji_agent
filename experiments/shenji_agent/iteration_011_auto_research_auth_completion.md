# Iteration 011 Auto Research Auth Completion

Date: 2026-04-29

## Status

B blocked-with-hard-evidence.

The current container cannot complete SDK-authenticated battle promotion because the installed `.engine_config` is bound to MAC `02:42:ac:11:00:45`, while the current container MAC is `02:42:ac:11:00:02`. All legal work that does not require host/container-creation privileges has been executed and committed as handoff-ready tooling and evidence.

## TODO

- [x] Confirm formal repository state.
- [x] Confirm current tests status.
- [x] Confirm latest auth blocker evidence.
- [x] Inspect stash state without deleting it.
- [x] Perform full local SDK/auth/offline/license/MAC documentation search.
- [x] Inspect `.engine_config` paths and references.
- [x] Inspect SDK package resources and wheel metadata.
- [x] Check whether an official offline mode exists.
- [x] Check whether a non-Docker run path exists.
- [x] Check whether environment variables or config can select engine_config/license.
- [x] Check whether authorized preflight can run under current permissions.
- [x] If network/search is available, research public official/legal material.
- [x] Add or improve SDK auth environment preflight.
- [x] Add or improve no-host-permission runbook.
- [x] Add admin one-command validation script.
- [x] Add continue-after-auth-fixed promotion continuation script.
- [x] Run tests.
- [x] Run current-environment smoke to reproduce blocker.
- [x] If a legal path is found, execute it where current permissions allow.
- [ ] If auth passes, continue objective labels, self-play, formal promotion.
- [x] If all legal paths fail, write stronger blocked report.
- [ ] Commit.
- [x] Continue next loop until completed or hard-blocked.

Auth did not pass in the current environment, so the auth-passed continuation item remains intentionally unchecked.

## Phase 1 State Confirmation

- `pwd`: `/remote-home/lizilong/shenji_agent`
- Starting HEAD: `483a3cf205dc2aca5693a80c5dec913c70286c1e`
- Tracked diff at start: empty.
- Stash:
  - `stash@{0}: On main: post-test pycache generated backup`
  - `stash@{1}: On main: pre-fresh-merge untracked collision backup`
  - `stash@{2}: On main: pre-fresh-merge tracked formal worktree backup`
- Disk: `/remote-home` had approximately `2.9T` available.
- Worktree contained many pre-existing untracked local artifacts under `.agent`, `.codex`, local weights, package files, and helper scripts. These were not modified or deleted.
- Current user/runtime has no host/container-creation privileges from inside this container.

## Existing Evidence Read

Read and summarized:

- `experiments/shenji_agent/final_autonomous_blocked_report.md`
- `experiments/shenji_agent/sdk_auth_resolution_plan.md`
- `experiments/shenji_agent/sdk_auth_blocked_analysis.md`
- `experiments/shenji_agent/authentication_failure_analysis.md`
- `experiments/shenji_agent/valid_action_pipeline_analysis.md`
- `docs/sdk_auth_runbook.md`
- `docker-compose.auth-mac.yml`
- `scripts/run_selfplay_authorized_mac.sh`

Existing evidence already showed:

- `auth_status=failed_mac_mismatch`
- `.engine_config` bound MAC: `02:42:ac:11:00:45`
- current container MAC: `02:42:ac:11:00:02`
- `battle_phase_entered=false`
- selected `Move/Shoot/Occupy=0`
- promotion rejected by `battle_phase_not_entered` and `no_move_shoot_occupy_selected_actions`

## Phase 2 Local Research

Generated:

- `experiments/shenji_agent/local_auth_research_grep.txt`
- `experiments/shenji_agent/local_auth_research_files.txt`
- `experiments/shenji_agent/sdk_auth_auto_research.md`

Findings:

- Local docs describe offline/local `TrainEnv` usage, not an auth-free mode.
- No repository config or script was found for license regeneration.
- No SDK auth environment variable was found.
- SDK visible strings reference `.engine_config`, `getmac`, `pass_MAC_check`, `did pass authentication`, and `did not pass authentication`.
- Installed package metadata is `land_wargame_train_env==4.1.0` with dependencies `getmac`, `numpy`, and `pandas`.
- Wheel contains `train_env/env/authenticate/.engine_config` and `authenticator*.so`.
- Non-invasive inspection only; no SDK binary was patched or reverse engineered.

## Phase 3 Public Research

Generated:

- `experiments/shenji_agent/public_auth_research.md`

Public official/legal sources checked:

- `https://wargame.ia.ac.cn/docs/`
- `https://wargame.ia.ac.cn/docs/reference/apis/`
- `https://docs.docker.com/reference/compose-file/services/#mac_address`

Findings:

- Official docs confirm local/offline SDK use through `TrainEnv`.
- No official public auth bypass, `.engine_config` regeneration CLI, or auth environment variable was found.
- Docker Compose documents `mac_address`, supporting the existing legal authorized-container route.

## Phase 4 Engineering Delivery

Added:

- `scripts/check_sdk_auth_environment.py`
- `scripts/admin_run_authorized_selfplay.sh`
- `scripts/continue_after_auth_fixed.sh`
- `tests/test_sdk_auth_environment_check.py`
- `tests/test_auth_handoff_scripts.py`

Updated:

- `docs/sdk_auth_runbook.md`
- `experiments/shenji_agent/final_autonomous_blocked_report.md`

Preflight output:

- `experiments/shenji_agent/sdk_auth_environment_check.json`
- `auth_environment_ready=false`
- `blocker_reason=mac_mismatch`
- `next_action=Run in an authorized environment whose non-loopback MAC is 02:42:ac:11:00:45, or request a regenerated .engine_config from the SDK provider for the actual runtime MAC.`

## Phase 5 Tests

Targeted tests:

```bash
python3.10 -m pytest tests/test_sdk_auth_environment_check.py -q
python3.10 -m pytest tests/test_auth_handoff_scripts.py -q
python3.10 -m pytest tests/test_phase_action_diagnostics.py -q
python3.10 -m pytest tests/test_selfplay_battle_phase_contract.py -q
python3.10 -m pytest tests/test_mainline_contract.py -q
```

Results:

- `3 passed`
- `4 passed`
- `5 passed`
- `5 passed`
- `35 passed`

Full test command:

```bash
python3.10 -m pytest tests -q
```

Result: `128 passed in 21.57s`.

## Phase 6 Current-Environment Smoke

Command:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Artifact: `experiments/shenji_agent/league_20260429_105446`

Results:

- SDK printed repeated `did not pass authentication`.
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
- failures: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`

## Conclusion

The only remaining legal path is to run the project in an authorized environment whose runtime MAC is `02:42:ac:11:00:45`, or to obtain a regenerated `.engine_config` from the SDK/platform owner for the actual runtime MAC. Once that is done, run:

```bash
cd /remote-home/lizilong/shenji_agent
bash scripts/admin_run_authorized_selfplay.sh
bash scripts/continue_after_auth_fixed.sh
```
