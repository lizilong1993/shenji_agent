# SDK Auth Auto Research

Date: 2026-04-29

Iteration 012 update: current-environment paths were rechecked after adding the full promotion continuation runner. The latest smoke artifact is now `experiments/shenji_agent/league_20260429_112500`, and auth remains blocked by the same MAC mismatch.

## Local Search Scope

Generated evidence files:

- `experiments/shenji_agent/local_auth_research_grep.txt`
- `experiments/shenji_agent/local_auth_research_files.txt`
- `experiments/shenji_agent/sdk_auth_environment_check.json`

Searched repository paths: `README.md`, `docs`, `scripts`, `configs`, `training`, `ai`, `land_wargame_sdk`, `Dockerfile`, `docker-compose.yml`, and `.devcontainer`.

## Findings

1. Relevant local docs/scripts/config:
   - `docs/docs.md` and `README.md` document offline/local SDK usage through `run_offline_games.py` and `TrainEnv`.
   - `training/selfplay.py` imports real `train_env.TrainEnv` on Linux and rejects mock `MockTrainEnv` for formal self-play.
   - `docker-compose.auth-mac.yml` sets the service MAC to the `.engine_config` bound value.
   - `scripts/run_selfplay_authorized_mac.sh` refuses to run unless the current MAC already matches or instructs the authorized Compose path.
   - Handoff tools are `scripts/check_sdk_auth_environment.py`, `scripts/admin_run_authorized_selfplay.sh`, `scripts/continue_after_auth_fixed.sh`, and `scripts/run_full_promotion_after_auth.sh`.

2. Official offline mode:
   - The local SDK docs use "offline" to mean local single-machine inference/development with `TrainEnv`.
   - No local documentation, config, environment variable, or wheel metadata was found for an offline mode that bypasses SDK authentication.

3. Environment variables:
   - Repository search found no SDK env var such as `ENGINE_CONFIG`, `LICENSE`, `AUTH`, or `OFFLINE` that selects a different auth config or disables authentication.
   - The new preflight supports `SDK_ENGINE_CONFIG` only as a diagnostic override for tests/preflight; it does not change SDK runtime behavior.

4. Engine config search path:
   - Visible SDK strings include `/.engine_config`, `.engine_config`, `mac_time_location`, `pass_MAC_check`, `getmac`, `did pass authentication`, and `did not pass authentication`.
   - Installed config path is `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`.
   - Decoded value is `02:42:ac:11:00:45,1777298501090988694`.

5. License regeneration path:
   - No local regeneration tool, API, CLI, or documentation was found.
   - Legal recovery requires either running with MAC `02:42:ac:11:00:45` or obtaining a regenerated `.engine_config` from the SDK/platform owner.

6. Non-Docker run path:
   - Direct venv/container execution is already the current path and still uses installed `train_env.TrainEnv`.
   - Current direct path fails because the runtime MAC is `02:42:ac:11:00:02`.
   - No non-Docker local path was found that changes the SDK-visible MAC without host/container-network control.

7. Current-permission legal path:
   - Read-only preflight, diagnostics, test execution, current-environment smoke reproduction, and handoff script generation are executable.
   - Applying the authorized MAC to the already-running container is not executable from repository code without host/container creation control.

8. Iteration 012 environment checks:
   - Docker, Docker Compose, Docker socket, Podman, Buildah, Nerdctl, Singularity, Apptainer, Slurm, qsub, srun, and kubectl were unavailable.
   - The current container lacks `CAP_NET_ADMIN` and `CAP_SYS_ADMIN`.
   - User namespace creation fails with `Operation not permitted`.
   - `/proc/*/ns/net` shows one visible network namespace only.
   - No current interface or namespace exposes MAC `02:42:ac:11:00:45`.
   - The default remote runner `lizilong@146.56.220.99:21427` exists in project scripts but SSH access from this environment failed with `Permission denied (publickey,password)`.

## SDK Package Inspection

- Installed package origin: `/usr/local/lib/python3.10/dist-packages/train_env/__init__.cpython-310-x86_64-linux-gnu.so`
- Dist-info metadata:
  - Name: `land_wargame_train_env`
  - Version: `4.1.0`
  - Requires: `getmac`, `numpy`, `pandas`
- Wheel resources include:
  - `train_env/env/authenticate/.engine_config`
  - `train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so`
  - `train_env/env/train_env.cpython-310-x86_64-linux-gnu.so`
  - `train_env/env/train_engine.cpython-310-x86_64-linux-gnu.so`
- Non-invasive dependency inspection shows `authenticator*.so` links only standard libc/libpthread in this environment.

No SDK binary was patched, replaced, or reverse engineered.

## Latest Current-Environment Experiment

Command:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Artifact: `experiments/shenji_agent/league_20260429_112500`

Observed:

- `auth_status=failed_mac_mismatch`
- `bound_mac=02:42:ac:11:00:45`
- `current_macs=["02:42:ac:11:00:02"]`
- `battle_phase_entered=false`
- `turns_with_move=640`
- `turns_with_shoot=0`
- `turns_with_occupy=0`
- `selected Move/Shoot/Occupy=0`
- `promotion_accepted=false`
- failures: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`

## Next Experiment

The only discovered legal experiment that can change the result is to create/run the container with the authorized MAC and then execute:

```bash
bash scripts/admin_run_authorized_selfplay.sh
bash scripts/continue_after_auth_fixed.sh
bash scripts/run_full_promotion_after_auth.sh
```

If that still fails with the authorized MAC present, request a regenerated `.engine_config` or official SDK/platform authorization instructions.
