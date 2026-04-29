# Iteration 012 Auto Completion Push

Date: 2026-04-29
Repository: `/remote-home/lizilong/shenji_agent`
Starting HEAD: `45ad227fd5bee068a491d3d458941d4818334c58`
Runner commit: `17eb19b8573e8fd14ea54063e4b32886d9a0e072`
Latest smoke artifact: `experiments/shenji_agent/league_20260429_112500`
Status: B blocked-with-hard-evidence

## TODO

- [x] Confirm repo status.
- [x] Confirm tests status.
- [x] Confirm latest auth blocker evidence.
- [x] Check current environment permission matrix.
- [x] Check CAP_NET_ADMIN / user namespace / rootless container / Docker socket / podman.
- [x] Check whether an authorized network interface already exists.
- [x] Check whether an authorized container/namespace already exists.
- [x] Check project remote runner / slurm / detached scripts.
- [x] Check SDK offline/local mode.
- [x] Check engine_config search path.
- [x] Complete local documentation/source research.
- [x] Complete public documentation research where available.
- [x] Try all legal executable auth paths in the current environment.
- [ ] If auth passes, continue battle/objective/formal promotion pipeline.
- [x] If auth remains blocked, improve one-command authorized-environment continuation.
- [x] Run tests.
- [x] Run current-environment smoke.
- [x] Commit executable continuation runner.
- [x] Update final report.
- [x] If still blocked, document exhausted paths and the single remaining external condition.

The unchecked auth-pass pipeline item did not become executable: no legal current-permission path produced `auth_status != failed_mac_mismatch`.

## Phase 1: Baseline State And Permission Matrix

Commands executed/reviewed: `pwd`, `git status --short`, `git log --oneline -n 50`, `git rev-parse HEAD`, `git diff --stat`, `git stash list`, `df -h`, `id`, `groups`, `uname -a`, `ip link`, capability status, Docker/Compose/Podman checks, `unshare`, Docker socket check, and relevant environment variables.

Key results:

- `pwd`: `/remote-home/lizilong/shenji_agent`
- Starting HEAD: `45ad227fd5bee068a491d3d458941d4818334c58`
- Recent commits include `45ad227 docs: finalize iteration 011 auth handoff status` and `b25ab65 chore: add sdk auth preflight and handoff tools`.
- Worktree already contained pre-existing generated pycache modifications and many untracked `.agent`/local artifacts; these were not reverted.
- Stashes preserved:
  - `stash@{0}: On main: post-test pycache generated backup`
  - `stash@{1}: On main: pre-fresh-merge untracked collision backup`
  - `stash@{2}: On main: pre-fresh-merge tracked formal worktree backup`
- Disk: `/remote-home` has about `2.9T` available.
- Identity: `uid=0(root) gid=0(root) groups=0(root)` inside the container.
- Kernel: Linux `5.4.0-216-generic`.
- Interfaces: `eth0@if192` MAC `02:42:ac:11:00:02`; loopback only otherwise.
- Capabilities decoded from `CapEff=00000000a80425fb`: no `CAP_NET_ADMIN`; no `CAP_SYS_ADMIN`.
- `docker`, `docker compose`, `docker-compose`, and `podman`: unavailable.
- `/var/run/docker.sock`: absent.
- `unshare --user --map-root-user true`: `Operation not permitted`.

Conclusion: current process can run repo diagnostics/tests/smoke but cannot create an authorized container, attach a Docker socket, create a user/net namespace, or change the current interface MAC.

## Phase 2: Authorized Environment Search

Evidence files:

- `experiments/shenji_agent/current_interfaces.txt`
- `experiments/shenji_agent/current_mac_addresses.txt`
- `experiments/shenji_agent/authorized_environment_search.md`

Results:

- No current interface has MAC `02:42:ac:11:00:45`.
- Docker/Podman container and network listing paths were unavailable because CLIs and sockets are absent.
- `/var/run/netns` was empty/absent.
- `/proc/*/ns/net` showed 7 visible processes in 1 network namespace only.
- No accessible existing container/namespace with the bound MAC was found.

## Phase 3: Local SDK/Auth Research

Evidence files:

- `experiments/shenji_agent/local_auth_research_grep.txt`
- `experiments/shenji_agent/local_auth_research_files.txt`
- `experiments/shenji_agent/sdk_auth_auto_research.md`

Results:

- Local docs and `land_wargame_sdk/run_offline_games.py` use `TrainEnv`; no offline auth bypass was found.
- `training/selfplay.py` uses real `train_env.TrainEnv` on Linux and rejects mock backend for formal self-play.
- Installed SDK package origin: `/usr/local/lib/python3.10/dist-packages/train_env/__init__.cpython-310-x86_64-linux-gnu.so`.
- Installed auth config: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`.
- Decoded config: `02:42:ac:11:00:45,1777298501090988694`.
- Non-invasive SDK strings show `.engine_config`, `pass_MAC_check`, `getmac`, `did pass authentication`, and `did not pass authentication`.
- `ldd` on the authenticator extension showed only libc/libpthread dependencies; the `file` command is not installed in this container.
- No SDK binary was modified, patched, replaced, or reverse engineered.

## Phase 4: Public Auth Research

Evidence file: `experiments/shenji_agent/public_auth_research.md`

Sources checked:

- Official platform docs: `https://wargame.ia.ac.cn/docs/`
- Official TrainEnv/API docs: `https://wargame.ia.ac.cn/docs/reference/apis/`
- Docker Compose service `mac_address`: `https://docs.docker.com/reference/compose-file/services/#mac_address`
- Docker container run `--mac-address`: `https://docs.docker.com/reference/cli/docker/container/run/`

No official public source documented an auth bypass, alternate license env var, local offline auth mode, or `.engine_config` regeneration CLI. Docker docs confirm the legal container-MAC route requires container creation privileges.

## Phase 5: Legal Auth Path Attempts

Attempted paths:

- Current direct preflight: `blocker_reason=mac_mismatch`, `mac_match=false`.
- Current direct wrapper: `bash scripts/run_selfplay_authorized_mac.sh` refused because current MAC is `02:42:ac:11:00:02`.
- Docker/Compose path: not executable because CLI/socket are absent.
- Podman/rootless path: not executable because Podman is absent and user namespaces are disabled.
- Existing namespace/container path: no accessible matching namespace/container found.
- Direct non-Docker Python path: already current path; current MAC does not match bound MAC.
- Remote runner path: project scripts found a default SSH target, but `ssh -p 21427 -o BatchMode=yes ... lizilong@146.56.220.99` failed with `Permission denied (publickey,password)`.
- Slurm/qsub/srun/kubectl path: tools absent.
- Official offline/local mode: no bypass or alternate auth route found.

No legal current-permission path produced SDK auth success.

## Phase 6: Promotion Pipeline Status

Not executed beyond preflight in `scripts/run_full_promotion_after_auth.sh` because current auth preflight is hard-failing with `mac_mismatch`.

The runner is ready for an authorized environment and will execute:

```bash
bash scripts/run_full_promotion_after_auth.sh
```

It checks SDK auth, battle smoke, objective smoke, high-confidence objective labels, objective-aware scoring, tests, formal self-play against `configs/selfplay_sota_stop.yaml`, promotion gates, and consecutive windows.

## Phase 7: Continuation Automation

Implemented and committed in `17eb19b8573e8fd14ea54063e4b32886d9a0e072`:

- `scripts/run_full_promotion_after_auth.sh`
- `tests/test_full_promotion_after_auth_script.py`
- `scripts/check_sdk_auth_environment.py` now includes the full runner in post-fix commands.
- `scripts/admin_run_authorized_selfplay.sh` now points operators to the full runner.
- `scripts/bootstrap_remote.py` no longer uses destructive `git reset --hard`; it uses `git pull --ff-only`.
- `docs/sdk_auth_runbook.md` documents the full promotion runner.

Current-environment fail-closed report:

- `experiments/shenji_agent/full_promotion_after_auth_report.md`
- Status: `failed`
- Reason: `preflight rejected current environment`
- Next action: run from a runtime whose non-loopback MAC matches the official engine config binding.

## Phase 8: Tests

Targeted tests:

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

Final full suite:

```bash
python3.10 -m pytest tests -q
```

Result after script/test changes: `133 passed in 5.45s`.

## Phase 9: Current Environment Smoke

Command:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Latest artifact: `experiments/shenji_agent/league_20260429_112500`

Files present:

- `summary.md`
- `promotion.json`
- `rating.json`
- `participants.json`
- `phase_action_diagnostics.json`
- `valid_action_timeline.json`
- `auth_status_diagnostics.json`

Observed:

- `auth_status=failed_mac_mismatch`
- `bound_mac=02:42:ac:11:00:45`
- `current_macs=["02:42:ac:11:00:02"]`
- `battle_phase_entered=false`
- `selected Move/Shoot/Occupy=0`
- `promotion_accepted=false`
- promotion rejection reasons: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`

## Phase 10: Final Status

B blocked-with-hard-evidence.

All legal current-permission auth paths were researched and attempted. The only remaining external condition is an authorized runtime whose non-loopback MAC is `02:42:ac:11:00:45`, or an official regenerated `.engine_config` for the current runtime MAC.

Once that condition is satisfied, continue with:

```bash
cd /remote-home/lizilong/shenji_agent
bash scripts/run_full_promotion_after_auth.sh
```
