# Shenji Agent Final Autonomous Blocked Report

> **Superseded by `final_autonomous_completion_report.md` after auth path fix and formal promotion success.** This document is preserved for historical reference.

## Final Status

B blocked-with-hard-evidence after Iteration 012 (HISTORICAL — superseded by A completed).

The repository has been advanced as far as legally executable in the current container. All current-permission auth paths were researched and attempted. The remaining blocker is SDK MAC authentication outside the repository's present permissions, not missing policy/test/report work.

## Repository State

- Formal repository path: `/remote-home/lizilong/shenji_agent`
- Iteration 012 starting HEAD: `45ad227fd5bee068a491d3d458941d4818334c58`
- Full promotion runner commit: `17eb19b8573e8fd14ea54063e4b32886d9a0e072`
- Latest current-environment smoke artifact: `experiments/shenji_agent/league_20260429_112500`
- Latest preflight JSON: `experiments/shenji_agent/sdk_auth_environment_check.json`
- Current fail-closed full-runner report: `experiments/shenji_agent/full_promotion_after_auth_report.md`
- Stash entries preserved:
  - `stash@{0}: On main: post-test pycache generated backup`
  - `stash@{1}: On main: pre-fresh-merge untracked collision backup`
  - `stash@{2}: On main: pre-fresh-merge tracked formal worktree backup`

## Current SDK Auth Evidence

- SDK backend: `train_env.TrainEnv`
- SDK runtime version: `4.1.0`
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

Artifact: `experiments/shenji_agent/league_20260429_112500`

- `auth_status=failed_mac_mismatch`
- `bound_mac=02:42:ac:11:00:45`
- `current_macs=["02:42:ac:11:00:02"]`
- `battle_phase_entered=false`
- `turns_with_setup_deployment_only=640`
- `turns_with_move=640`
- `turns_with_shoot=0`
- `turns_with_occupy=0`
- `selected_move_actions=0`
- `selected_shoot_actions=0`
- `selected_occupy_actions=0`
- `promotion_accepted=false`
- promotion failures: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`

## Current Permission Matrix

- User identity: `uid=0(root) gid=0(root) groups=0(root)` inside the container.
- Effective capabilities do not include `CAP_NET_ADMIN` or `CAP_SYS_ADMIN`.
- User namespace creation fails: `unshare --user --map-root-user true` -> `Operation not permitted`.
- Docker CLI/Compose: unavailable.
- Docker socket: absent.
- Podman/rootless Podman: unavailable.
- Buildah/Nerdctl/Singularity/Apptainer: unavailable.
- Slurm/qsub/srun/kubectl: unavailable.
- Visible network namespaces: one current namespace only.
- Current interfaces: loopback and `eth0@if192` with MAC `02:42:ac:11:00:02`.

Current permissions allow tests, diagnostics, self-play smoke, and report/script generation. They do not allow creating a container with a selected MAC, changing the current interface MAC, or entering an already-authorized namespace.

## Exhausted Legal Paths

- Current direct Python/venv path: fails `mac_mismatch`.
- Current `scripts/run_selfplay_authorized_mac.sh`: refuses because MAC is `02:42:ac:11:00:02`, not `02:42:ac:11:00:45`.
- Docker/Compose authorized MAC path: not executable here because Docker CLI/socket are unavailable.
- Podman/rootless container path: not executable because Podman is absent and user namespaces are disabled.
- Existing authorized interface/container/namespace path: searched current interfaces, `/var/run/netns`, `/proc/*/ns/net`, Docker/Podman listings where possible; none found.
- Direct host/non-Docker SDK path: same current container MAC, same auth failure.
- Remote runner path: project scripts contain `lizilong@146.56.220.99:21427`, but SSH with `BatchMode=yes` failed with `Permission denied (publickey,password)`.
- Slurm/qsub/srun/kubectl path: tools absent.
- Official offline/local mode: local and public docs describe local `TrainEnv` usage but no auth bypass, alternate license env var, or regeneration CLI.

Evidence files:

- `experiments/shenji_agent/authorized_environment_search.md`
- `experiments/shenji_agent/sdk_auth_auto_research.md`
- `experiments/shenji_agent/public_auth_research.md`
- `experiments/shenji_agent/local_auth_research_grep.txt`
- `experiments/shenji_agent/local_auth_research_files.txt`
- `experiments/shenji_agent/current_interfaces.txt`
- `experiments/shenji_agent/current_mac_addresses.txt`

## One-Command Continuation Path

These scripts are present and tested:

- `scripts/check_sdk_auth_environment.py`
- `scripts/admin_run_authorized_selfplay.sh`
- `scripts/continue_after_auth_fixed.sh`
- `scripts/run_full_promotion_after_auth.sh`

For an authorized runtime, run:

```bash
cd /remote-home/lizilong/shenji_agent
bash scripts/run_full_promotion_after_auth.sh
```

The runner fails closed and does not fake success. In an auth-ready environment it performs:

- SDK auth preflight.
- Battle smoke.
- Objective smoke.
- Objective label build.
- High-confidence score/capture/damage/kill/city-control label validation.
- Objective-aware policy scoring validation.
- Full test suite.
- Formal self-play with `configs/selfplay_sota_stop.yaml`.
- Promotion gate validation:
  - `overall_point_estimate >= 0.60`
  - `overall_ci95_lower >= 0.55`
  - `complex_point_estimate >= 0.60`
  - `complex_ci95_lower >= 0.55`
  - `learned_path_coverage >= 0.30`
  - `move_path_coverage >= 0.20`
  - `shoot_path_coverage >= 0.20`
  - `elo_delta_lower >= 35`
  - `max_bucket_regression <= 0.03`
  - `max_exploiter_win_rate <= 0.45`
  - required consecutive windows
- Final completion report generation.

Current run of that script failed at preflight, as expected, with `blocker_reason=mac_mismatch`.

## Authorized Environment Requirements

The remaining external condition is exactly one of:

- Run in a legal runtime whose non-loopback MAC is `02:42:ac:11:00:45`.
- Or obtain an official regenerated `.engine_config` for the actual runtime MAC.

After that condition is satisfied, required first-pass evidence is:

- `auth_status != failed_mac_mismatch`
- `battle_phase_entered=true`
- selected `Move` / `Shoot` / `Occupy` actions greater than zero

If a runtime has MAC `02:42:ac:11:00:45` and SDK auth still fails, the only legal next step is SDK/platform owner support or official license regeneration. Do not patch, replace, bypass, or spoof SDK auth artifacts.

## Verification

Targeted tests run:

```bash
python3.10 -m pytest tests/test_sdk_auth_environment_check.py -q
python3.10 -m pytest tests/test_auth_handoff_scripts.py -q
python3.10 -m pytest tests/test_full_promotion_after_auth_script.py -q
python3.10 -m pytest tests/test_phase_action_diagnostics.py -q
python3.10 -m pytest tests/test_selfplay_battle_phase_contract.py -q
python3.10 -m pytest tests/test_mainline_contract.py -q
```

Results:

- `3 passed`
- `4 passed`
- `5 passed`
- `5 passed`
- `5 passed`
- `35 passed`

Full suite:

```bash
python3.10 -m pytest tests -q
```

Result: `133 passed in 5.45s`.

Smoke:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Result: current environment reproduces SDK MAC auth blocker in `experiments/shenji_agent/league_20260429_112500`.

## Worktree Notes

Do not delete preserved user/local artifacts. The worktree contains pre-existing untracked files under `.agent`, `.codex`, local package files, local weights, and helper scripts. Test execution leaves tracked `ai/__pycache__/*.pyc` modifications in this environment; these are generated artifacts and are not part of the committed handoff.

The report/evidence files under `experiments/shenji_agent` are intentionally force-added where needed because the repository ignores `experiments/` by default.

