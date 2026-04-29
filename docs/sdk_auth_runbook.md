# SDK Authentication Runbook

## Current Evidence

- SDK backend: `train_env.TrainEnv`
- SDK wheel authentication extension: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so`
- SDK engine config: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`
- Engine config exists: yes
- Engine config bound MAC: `02:42:ac:11:00:45`
- Current container MAC: `02:42:ac:11:00:02`
- Reproduced run: `experiments/shenji_agent/league_20260429_105446`
- Auth status: `failed_mac_mismatch`
- Battle phase entered: `false`
- Promotion gate result: rejected with `battle_phase_not_entered` and `no_move_shoot_occupy_selected_actions`

## No-Host-Permission Path

These commands are safe in the current unprivileged container. They do not modify the SDK, do not require Docker, and do not fake promotion evidence:

```bash
cd /remote-home/lizilong/shenji_agent
python3.10 scripts/check_sdk_auth_environment.py --output experiments/shenji_agent/sdk_auth_environment_check.json
python3.10 -m pytest tests -q
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

Expected current result:

- `auth_environment_ready=false`
- `blocker_reason=mac_mismatch`
- `auth_status=failed_mac_mismatch`
- `battle_phase_entered=false`
- selected `Move` / `Shoot` / `Occupy` actions equal `0`

This proves the remaining blocker is SDK authorization, not repository policy/diagnostic code.

## Legal Resolution Path

Do not patch or replace the SDK authentication extension, do not edit `.engine_config`, and do not fake battle-phase evidence.

The project has a Compose service named `dev` using bridge networking. A dedicated override is provided at `docker-compose.auth-mac.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml up -d --build dev
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec dev bash -lc \
  'cd /app && cat /sys/class/net/*/address && python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule'
```

Expected precondition: the container reports `02:42:ac:11:00:45` on its non-loopback interface.

If Compose is unavailable or the platform disallows service-level `mac_address`, run in a container or host that is officially authorized for the `.engine_config` MAC, or request a regenerated `.engine_config` for the actual runtime MAC.

For a one-command host/admin validation, run:

```bash
cd /remote-home/lizilong/shenji_agent
bash scripts/admin_run_authorized_selfplay.sh
```

The script checks Docker access, starts the Compose service with the authorized MAC override, verifies the container MAC, runs SDK auth preflight, runs battle smoke, and fails if auth still fails, battle phase is not entered, or selected `Move` / `Shoot` / `Occupy` actions remain zero.

## Verification

After the MAC matches, run:

```bash
bash scripts/continue_after_auth_fixed.sh
```

The continuation script performs:

1. SDK auth preflight.
2. Battle phase smoke.
3. Battle/action evidence validation.
4. Objective smoke.
5. Objective label build.
6. Test suite.
7. Formal self-play continuation.

Any run is useful promotion evidence only if:

- `auth_status` is no longer `failed_mac_mismatch`.
- `battle_phase_entered` is `true`.
- selected `Move` / `Shoot` / `Occupy` actions are greater than zero.
- `promotion.json` no longer fails `battle_phase_not_entered` or `no_move_shoot_occupy_selected_actions`.

## If Authorized MAC Still Fails

If the container reports `02:42:ac:11:00:45` and the SDK still prints `did not pass authentication`, do not patch the SDK. Use one of the official/legal paths:

- Request a regenerated `.engine_config` for the actual runtime MAC.
- Request SDK/platform support for the current host/container environment.
- Run on an officially authorized host/container image supplied by the SDK/platform owner.
