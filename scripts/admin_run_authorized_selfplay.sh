#!/usr/bin/env bash
set -euo pipefail

AUTHORIZED_MAC="${AUTHORIZED_MAC:-02:42:ac:11:00:45}"
SERVICE="${SDK_AUTH_COMPOSE_SERVICE:-dev}"
CONFIG="${SELFPLAY_CONFIG:-configs/selfplay_smoke_battle_phase.yaml}"
CANDIDATE="${SELFPLAY_CANDIDATE:-ai/weights}"
INCUMBENT="${SELFPLAY_INCUMBENT:-ai/weights_incumbent_rule}"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    echo "Docker Compose is not available in this environment." >&2
    return 127
  fi
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is not available. Run this script from the host/admin environment that can create containers." >&2
  exit 127
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not reachable. Run from a host/admin shell with Docker access." >&2
  exit 125
fi

echo "Starting ${SERVICE} with authorized MAC ${AUTHORIZED_MAC}..."
compose -f docker-compose.yml -f docker-compose.auth-mac.yml up -d --build "${SERVICE}"

echo "Verifying container MAC..."
compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec -T "${SERVICE}" bash -lc \
  "cat /sys/class/net/*/address | grep -qi '^${AUTHORIZED_MAC}$'"

echo "Running SDK auth preflight inside authorized container..."
compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec -T "${SERVICE}" bash -lc \
  "cd /app && python3.10 scripts/check_sdk_auth_environment.py --output experiments/shenji_agent/sdk_auth_environment_check.json"

echo "Running battle smoke inside authorized container..."
compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec -T "${SERVICE}" bash -lc \
  "cd /app && python3.10 -m training.selfplay --config '${CONFIG}' --candidate '${CANDIDATE}' --incumbent '${INCUMBENT}'"

echo "Validating auth, battle phase, and selected Move/Shoot/Occupy evidence..."
compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec -T "${SERVICE}" bash -lc \
  "cd /app && python3.10 - <<'PY'
import json
from pathlib import Path

root = Path('experiments/shenji_agent')
runs = [p for p in root.glob('league_*') if p.is_dir() and (p / 'phase_action_diagnostics.json').exists()]
if not runs:
    raise SystemExit('No self-play run artifact found.')
run = max(runs, key=lambda p: p.stat().st_mtime)
auth = json.loads((run / 'auth_status_diagnostics.json').read_text(encoding='utf-8'))
phase = json.loads((run / 'phase_action_diagnostics.json').read_text(encoding='utf-8'))
summary = phase.get('summary', {})
statuses = [str(item.get('status')) for item in auth.get('auth_statuses', []) if isinstance(item, dict)]
selected = (
    int(summary.get('selected_move_actions', 0))
    + int(summary.get('selected_shoot_actions', 0))
    + int(summary.get('selected_occupy_actions', 0))
)
failures = []
if any(status.startswith('failed') for status in statuses):
    failures.append('auth_status=' + ','.join(sorted(set(statuses))))
if summary.get('battle_phase_entered') is not True:
    failures.append('battle_phase_entered=false')
if selected <= 0:
    failures.append('selected_move_shoot_occupy_actions=0')
if failures:
    raise SystemExit('Authorized self-play validation failed: ' + '; '.join(failures) + f'; run={run}')
print(f'authorized_selfplay_ready=true run={run} selected_move_shoot_occupy_actions={selected}')
PY"

echo "Authorized smoke completed. To continue promotion work, run inside the container:"
echo "  cd /app && bash scripts/continue_after_auth_fixed.sh"
echo "For full formal promotion, run:"
echo "  cd /app && bash scripts/run_full_promotion_after_auth.sh"
