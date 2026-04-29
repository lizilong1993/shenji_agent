# SDK Authentication Runbook

## Current Evidence

- SDK backend: `train_env.TrainEnv`
- SDK wheel authentication extension: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so`
- SDK engine config: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`
- Engine config exists: yes
- Engine config bound MAC: `02:42:ac:11:00:45`
- Current container MAC: `02:42:ac:11:00:02`
- Reproduced run: `experiments/shenji_agent/league_20260429_102023`
- Auth status: `failed_mac_mismatch`
- Battle phase entered: `false`
- Promotion gate result: rejected with `battle_phase_not_entered` and `no_move_shoot_occupy_selected_actions`

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

## Verification

After the MAC matches, run:

```bash
python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

The run is useful promotion evidence only if:

- `auth_status` is no longer `failed_mac_mismatch`.
- `battle_phase_entered` is `true`.
- selected `Move` / `Shoot` / `Occupy` actions are greater than zero.
- `promotion.json` no longer fails `battle_phase_not_entered` or `no_move_shoot_occupy_selected_actions`.
