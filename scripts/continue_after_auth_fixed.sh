#!/usr/bin/env bash
set -euo pipefail

CONFIG_BATTLE="${SELFPLAY_BATTLE_CONFIG:-configs/selfplay_smoke_battle_phase.yaml}"
CONFIG_OBJECTIVE="${SELFPLAY_OBJECTIVE_CONFIG:-configs/selfplay_smoke_objective_quality.yaml}"
CANDIDATE="${SELFPLAY_CANDIDATE:-ai/weights}"
INCUMBENT="${SELFPLAY_INCUMBENT:-ai/weights_incumbent_rule}"
OUTPUT_ROOT="${SELFPLAY_OUTPUT_ROOT:-experiments/shenji_agent}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-${OUTPUT_ROOT}/sdk_auth_environment_check.json}"

latest_run_dir() {
  python3.10 - <<'PY'
from pathlib import Path
root = Path("experiments/shenji_agent")
candidates = [
    p for p in root.glob("league_*")
    if p.is_dir() and (p / "phase_action_diagnostics.json").exists()
]
print(max(candidates, key=lambda p: p.stat().st_mtime) if candidates else "")
PY
}

check_battle_ready() {
  local run_dir="$1"
  python3.10 - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
phase = json.loads((run_dir / "phase_action_diagnostics.json").read_text(encoding="utf-8"))
summary = phase.get("summary", {})
auth_path = run_dir / "auth_status_diagnostics.json"
auth = json.loads(auth_path.read_text(encoding="utf-8")) if auth_path.exists() else {}
statuses = [str(item.get("status")) for item in auth.get("auth_statuses", []) if isinstance(item, dict)]
selected = (
    int(summary.get("selected_move_actions", 0))
    + int(summary.get("selected_shoot_actions", 0))
    + int(summary.get("selected_occupy_actions", 0))
)
failures = []
if "failed_mac_mismatch" in statuses:
    failures.append("auth_status=failed_mac_mismatch")
if summary.get("battle_phase_entered") is not True:
    failures.append("battle_phase_entered=false")
if selected <= 0:
    failures.append("selected_move_shoot_occupy_actions=0")
if failures:
    print("Battle smoke is not promotion-quality: " + ", ".join(failures))
    print(f"run_dir={run_dir}")
    raise SystemExit(2)
print(f"battle_ready=true run_dir={run_dir} selected_move_shoot_occupy_actions={selected}")
PY
}

echo "Preflight: ${PREFLIGHT_JSON}"
python3.10 scripts/check_sdk_auth_environment.py --output "${PREFLIGHT_JSON}"

if PREFLIGHT_JSON="${PREFLIGHT_JSON}" python3.10 - <<'PY'
import json
import os
from pathlib import Path
payload = json.loads(Path(os.environ["PREFLIGHT_JSON"]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("blocker_reason") in (None, "battle_phase_not_entered") else 1)
PY
then
  echo "Preflight permits smoke continuation."
else
  echo "SDK auth environment is not ready; refusing to run promotion continuation."
  exit 2
fi

echo "Running battle phase smoke..."
python3.10 -m training.selfplay \
  --config "${CONFIG_BATTLE}" \
  --candidate "${CANDIDATE}" \
  --incumbent "${INCUMBENT}"

BATTLE_RUN="$(latest_run_dir)"
if [[ -z "${BATTLE_RUN}" ]]; then
  echo "No battle smoke artifact found under ${OUTPUT_ROOT}."
  exit 2
fi
check_battle_ready "${BATTLE_RUN}"

echo "Running objective smoke..."
python3.10 -m training.selfplay \
  --config "${CONFIG_OBJECTIVE}" \
  --candidate "${CANDIDATE}" \
  --incumbent "${INCUMBENT}"

OBJECTIVE_RUN="$(latest_run_dir)"
if [[ -z "${OBJECTIVE_RUN}" ]]; then
  echo "No objective smoke artifact found under ${OUTPUT_ROOT}."
  exit 2
fi
check_battle_ready "${OBJECTIVE_RUN}"

LABEL_DIR="${OBJECTIVE_LABEL_DIR:-experiments/shenji_agent/objective_labels/latest_auth_fixed}"
python3.10 scripts/build_objective_labels.py \
  --input "${OBJECTIVE_RUN}" \
  --output-dir "${LABEL_DIR}"

python3.10 - <<'PY' "${LABEL_DIR}/summary.json"
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(summary.get("label_count", 0)) <= 0:
    print("Objective labels were not produced; refusing formal continuation.")
    raise SystemExit(2)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "Running tests..."
python3.10 -m pytest tests -q

FORMAL_CONFIG="${SELFPLAY_FORMAL_CONFIG:-configs/selfplay_smoke_objective_quality.yaml}"
echo "Running formal self-play continuation with ${FORMAL_CONFIG}..."
python3.10 -m training.selfplay \
  --config "${FORMAL_CONFIG}" \
  --candidate "${CANDIDATE}" \
  --incumbent "${INCUMBENT}"

FORMAL_RUN="$(latest_run_dir)"
check_battle_ready "${FORMAL_RUN}"

echo "Artifacts:"
echo "battle_run=${BATTLE_RUN}"
echo "objective_run=${OBJECTIVE_RUN}"
echo "label_dir=${LABEL_DIR}"
echo "formal_run=${FORMAL_RUN}"
