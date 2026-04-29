#!/usr/bin/env bash
set -euo pipefail

AUTHORIZED_MAC="${AUTHORIZED_MAC:-02:42:ac:11:00:45}"
CONFIG="${SELFPLAY_CONFIG:-configs/selfplay_smoke_battle_phase.yaml}"
CANDIDATE="${SELFPLAY_CANDIDATE:-ai/weights}"
INCUMBENT="${SELFPLAY_INCUMBENT:-ai/weights_incumbent_rule}"

current_macs="$(cat /sys/class/net/*/address 2>/dev/null | grep -iv '^00:00:00:00:00:00$' || true)"

echo "authorized_mac=${AUTHORIZED_MAC}"
echo "current_macs=${current_macs//$'\n'/,}"

if printf '%s\n' "${current_macs}" | grep -qi "^${AUTHORIZED_MAC}$"; then
  python3.10 -m training.selfplay \
    --config "${CONFIG}" \
    --candidate "${CANDIDATE}" \
    --incumbent "${INCUMBENT}"
  exit 0
fi

cat <<EOF
Current container MAC does not match the SDK .engine_config binding.

Run from the project root in an authorized Docker environment:

  docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml up -d --build dev
  docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml exec dev bash -lc \\
    'cd /app && python3.10 -m training.selfplay --config ${CONFIG} --candidate ${CANDIDATE} --incumbent ${INCUMBENT}'

If authentication still fails after the container has MAC ${AUTHORIZED_MAC},
request a regenerated .engine_config from the SDK provider for the actual runtime MAC.
EOF

exit 2
