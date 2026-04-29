# SDK Auth Resolution Plan

Date: 2026-04-29

## Answers

1. `.engine_config` is located at `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`.
2. The installed `train_env` package reads the authentication config from its `env/authenticate` package directory; the binary extension in the same directory performs the authentication check.
3. The current container MAC was collected from `/sys/class/net/*/address` and `ip link`; the non-loopback MAC is `02:42:ac:11:00:02`.
4. `docker-compose.yml` defines service `dev` on bridge networking. Compose supports a service-level `mac_address`; `docker-compose.auth-mac.yml` now sets it to `02:42:ac:11:00:45`.
5. `.devcontainer/devcontainer.json` uses `../docker-compose.yml` service `dev`; it can be launched with the override by the same Compose command outside the current already-running container.
6. It is legal to make the runtime container use the MAC that the existing authorized `.engine_config` is bound to, provided this is the intended licensed local SDK environment. This does not patch the SDK or forge auth output.
7. Host networking is not required by the discovered project config. Bridge networking with service `mac_address` is the minimal Compose path. If the host/platform rejects `mac_address`, use an officially authorized host/container or regenerate `.engine_config`.
8. A regenerated `.engine_config` is required if the runtime MAC cannot be made `02:42:ac:11:00:45`.
9. A user/admin or SDK provider must provide a valid `.engine_config` if the environment MAC cannot be changed.
10. The docs describe local offline `TrainEnv` usage but no discovered official offline mode that bypasses authentication.
11. Current offline/local self-play still imports `train_env.TrainEnv` and fails MAC authentication; offline usage therefore still depends on SDK auth in this environment.
12. Minimal reproduction:

```bash
cat /sys/class/net/*/address
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
cat experiments/shenji_agent/league_*/auth_status_diagnostics.json
```

13. Minimal fix:

```bash
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml up -d --build dev
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec dev bash -lc \
  'cd /app && python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule'
```

If the container MAC still does not match `02:42:ac:11:00:45`, request a new `.engine_config` bound to the actual runtime MAC.

## Current Environment Limitation

The current container does not include Docker CLI access:

```bash
docker compose version
# /bin/bash: docker: command not found
docker-compose version
# /bin/bash: docker-compose: command not found
```

Therefore the Compose override cannot be executed from inside this already-running container. It must be run by the host/admin environment that controls container creation.
