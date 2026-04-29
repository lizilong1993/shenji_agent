#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${SELFPLAY_OUTPUT_ROOT:-experiments/shenji_agent}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-${OUTPUT_ROOT}/sdk_auth_environment_check.json}"
CONFIG_BATTLE="${SELFPLAY_BATTLE_CONFIG:-configs/selfplay_smoke_battle_phase.yaml}"
CONFIG_OBJECTIVE="${SELFPLAY_OBJECTIVE_CONFIG:-configs/selfplay_smoke_objective_quality.yaml}"
FORMAL_CONFIG="${SELFPLAY_FORMAL_CONFIG:-configs/selfplay_sota_stop.yaml}"
CANDIDATE="${SELFPLAY_CANDIDATE:-ai/weights}"
INCUMBENT="${SELFPLAY_INCUMBENT:-ai/weights_incumbent_rule}"
LABEL_DIR="${OBJECTIVE_LABEL_DIR:-${OUTPUT_ROOT}/objective_labels/latest_after_auth}"
RUN_REPORT="${FULL_PROMOTION_RUN_REPORT:-${OUTPUT_ROOT}/full_promotion_after_auth_report.md}"
FINAL_REPORT="${FULL_PROMOTION_FINAL_REPORT:-${OUTPUT_ROOT}/final_autonomous_completion_report.md}"

BATTLE_RUN=""
OBJECTIVE_RUN=""
FORMAL_RUN=""
FAILURE_MESSAGE=""
NEXT_ACTION=""
STATUS="running"

write_report() {
  STATUS="${STATUS}" \
  FAILURE_MESSAGE="${FAILURE_MESSAGE}" \
  NEXT_ACTION="${NEXT_ACTION}" \
  PREFLIGHT_JSON="${PREFLIGHT_JSON}" \
  CONFIG_BATTLE="${CONFIG_BATTLE}" \
  CONFIG_OBJECTIVE="${CONFIG_OBJECTIVE}" \
  FORMAL_CONFIG="${FORMAL_CONFIG}" \
  CANDIDATE="${CANDIDATE}" \
  INCUMBENT="${INCUMBENT}" \
  BATTLE_RUN="${BATTLE_RUN}" \
  OBJECTIVE_RUN="${OBJECTIVE_RUN}" \
  FORMAL_RUN="${FORMAL_RUN}" \
  LABEL_DIR="${LABEL_DIR}" \
  RUN_REPORT="${RUN_REPORT}" \
  FINAL_REPORT="${FINAL_REPORT}" \
  python3.10 - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def load_json(path: str) -> dict:
    p = Path(path)
    if not path or not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

def git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], check=True, text=True, capture_output=True).stdout.strip()
    except Exception:
        return ""

status = os.environ["STATUS"]
run_report = Path(os.environ["RUN_REPORT"])
final_report = Path(os.environ["FINAL_REPORT"])
preflight = os.environ["PREFLIGHT_JSON"]
battle_run = os.environ["BATTLE_RUN"]
objective_run = os.environ["OBJECTIVE_RUN"]
formal_run = os.environ["FORMAL_RUN"]
label_dir = os.environ["LABEL_DIR"]

promotion = load_json(str(Path(formal_run) / "promotion.json")) if formal_run else {}
rating = load_json(str(Path(formal_run) / "rating.json")) if formal_run else {}
label_summary = load_json(str(Path(label_dir) / "summary.json"))
preflight_payload = load_json(preflight)

lines = [
    "# Full Promotion After Auth Report",
    "",
    f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
    f"- status: `{status}`",
    f"- head: `{git(['rev-parse', 'HEAD'])}`",
    f"- preflight_json: `{preflight}`",
    f"- battle_config: `{os.environ['CONFIG_BATTLE']}`",
    f"- objective_config: `{os.environ['CONFIG_OBJECTIVE']}`",
    f"- formal_config: `{os.environ['FORMAL_CONFIG']}`",
    f"- candidate: `{os.environ['CANDIDATE']}`",
    f"- incumbent: `{os.environ['INCUMBENT']}`",
    f"- battle_run: `{battle_run}`",
    f"- objective_run: `{objective_run}`",
    f"- label_dir: `{label_dir}`",
    f"- formal_run: `{formal_run}`",
    f"- auth_environment_ready: `{preflight_payload.get('auth_environment_ready')}`",
    f"- blocker_reason: `{preflight_payload.get('blocker_reason')}`",
    f"- label_count: `{label_summary.get('label_count')}`",
    f"- label_kind_counts: `{label_summary.get('label_kind_counts')}`",
    f"- promotion_accepted: `{promotion.get('accepted')}`",
    f"- promotion_failures: `{promotion.get('failures')}`",
    f"- overall_point_estimate: `{rating.get('overall', {}).get('point_estimate')}`",
    f"- overall_ci95_lower: `{(rating.get('overall', {}).get('ci95') or [None])[0]}`",
    f"- learned_path_coverage: `{rating.get('policy', {}).get('learned_path_coverage')}`",
]
if os.environ.get("FAILURE_MESSAGE"):
    lines.extend(
        [
            f"- failure: `{os.environ['FAILURE_MESSAGE']}`",
            f"- next_action: `{os.environ['NEXT_ACTION']}`",
        ]
    )
run_report.parent.mkdir(parents=True, exist_ok=True)
run_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
if status == "completed":
    final_report.parent.mkdir(parents=True, exist_ok=True)
    final_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

fail() {
  FAILURE_MESSAGE="$1"
  NEXT_ACTION="$2"
  STATUS="failed"
  echo "FAILED: ${FAILURE_MESSAGE}" >&2
  echo "artifact_report=${RUN_REPORT}" >&2
  echo "next_action=${NEXT_ACTION}" >&2
  write_report
  exit 2
}

latest_run_dir() {
  local required_file="$1"
  python3.10 - "${OUTPUT_ROOT}" "${required_file}" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
required = sys.argv[2]
candidates = [p for p in root.glob("league_*") if p.is_dir() and (p / required).exists()]
print(max(candidates, key=lambda p: p.stat().st_mtime) if candidates else "")
PY
}

validate_preflight() {
  python3.10 - "${PREFLIGHT_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blocker = payload.get("blocker_reason")
if payload.get("mac_match") is not True:
    raise SystemExit(
        "preflight rejected: current MACs do not match engine_config bound MAC; "
        f"blocker={blocker}; next={payload.get('next_action')}"
    )
if blocker not in (None, "battle_phase_not_entered"):
    raise SystemExit(f"preflight rejected: blocker={blocker}; next={payload.get('next_action')}")
print("preflight_ready=true")
PY
}

validate_battle_ready() {
  local run_dir="$1"
  python3.10 - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
phase_path = run_dir / "phase_action_diagnostics.json"
auth_path = run_dir / "auth_status_diagnostics.json"
if not phase_path.exists():
    raise SystemExit(f"missing phase_action_diagnostics.json in {run_dir}")
phase = json.loads(phase_path.read_text(encoding="utf-8"))
auth = json.loads(auth_path.read_text(encoding="utf-8")) if auth_path.exists() else {}
summary = phase.get("summary", {})
statuses = [str(item.get("status")) for item in auth.get("auth_statuses", []) if isinstance(item, dict)]
selected = (
    int(summary.get("selected_move_actions", 0))
    + int(summary.get("selected_shoot_actions", 0))
    + int(summary.get("selected_occupy_actions", 0))
)
failures = []
failed_statuses = [status for status in statuses if status.startswith("failed")]
if failed_statuses:
    failures.append("auth_status=" + ",".join(sorted(set(failed_statuses))))
if summary.get("battle_phase_entered") is not True:
    failures.append("battle_phase_entered=false")
if selected <= 0:
    failures.append("selected_move_shoot_occupy_actions=0")
if failures:
    raise SystemExit(f"battle validation failed for {run_dir}: " + "; ".join(failures))
print(f"battle_ready=true run_dir={run_dir} selected_move_shoot_occupy_actions={selected}")
PY
}

validate_objective_progress() {
  local run_dir="$1"
  python3.10 - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
progress = json.loads((run_dir / "objective_progress_diagnostics.json").read_text(encoding="utf-8"))
outcomes = json.loads((run_dir / "action_outcome_diagnostics.json").read_text(encoding="utf-8"))
city = json.loads((run_dir / "city_control_timeline.json").read_text(encoding="utf-8"))
progress_summary = progress.get("summary", {})
outcome_summary = outcomes.get("summary", {})
city_summary = city.get("summary", {})
progress_count = (
    int(progress_summary.get("score_delta_records", 0))
    + int(progress_summary.get("damage_records", 0))
    + int(outcome_summary.get("occupy_successes", 0))
    + int(outcome_summary.get("score_delta_actions", 0))
    + int(outcome_summary.get("damage_actions", 0))
    + int(city_summary.get("city_value_change_records", 0))
)
if progress_count <= 0:
    raise SystemExit(f"objective progress validation failed for {run_dir}: no real score/capture/damage/city-control progress")
print(f"objective_progress_ready=true run_dir={run_dir} progress_count={progress_count}")
PY
}

validate_label_quality() {
  python3.10 - "${LABEL_DIR}/summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
kind_counts = summary.get("label_kind_counts", {})
confidence_counts = summary.get("confidence_counts", {})
accepted_kinds = {"score_delta", "city_capture_delta", "occupy_success", "damage_kill_delta"}
high_kind_count = sum(int(kind_counts.get(kind, 0)) for kind in accepted_kinds)
if int(summary.get("label_count", 0)) <= 0:
    raise SystemExit("objective label quality failed: no labels produced")
if int(confidence_counts.get("high", 0)) <= 0 or high_kind_count <= 0:
    raise SystemExit(
        "objective label quality failed: requires high-confidence score/capture/damage/kill/city-control labels; "
        f"summary={summary}"
    )
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

validate_objective_scoring() {
  local run_dir="$1"
  python3.10 - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path
rating = json.loads((Path(sys.argv[1]) / "rating.json").read_text(encoding="utf-8"))
score_means = rating.get("policy", {}).get("score_means", {})
sample_count = sum(int(score_means.get(action, {}).get("samples", 0)) for action in ("move", "shoot"))
objective_magnitude = sum(abs(float(score_means.get(action, {}).get("objective_score", 0.0))) for action in ("move", "shoot"))
if sample_count <= 0:
    raise SystemExit("objective scoring validation failed: no move/shoot scoring samples")
if objective_magnitude <= 0.0:
    raise SystemExit("objective scoring validation failed: objective_score was not applied to move/shoot scoring")
print(f"objective_scoring_ready=true sample_count={sample_count} objective_magnitude={objective_magnitude:.6f}")
PY
}

validate_formal_gates() {
  local run_dir="$1"
  python3.10 - "${run_dir}" "${FORMAL_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path
from training.common import load_config, repo_root

run_dir = Path(sys.argv[1])
config = load_config(repo_root() / sys.argv[2])
gate = config["promotion_gate"]
rating = json.loads((run_dir / "rating.json").read_text(encoding="utf-8"))
promotion = json.loads((run_dir / "promotion.json").read_text(encoding="utf-8"))
overall = rating.get("overall", {})
complex_bucket = rating.get("buckets", {}).get("complex", {})
policy = rating.get("policy", {})
checks = {
    "overall_point_estimate": float(overall.get("point_estimate", 0.0)) >= float(gate["overall_point_estimate_min"]),
    "overall_ci95_lower": float((overall.get("ci95") or [0.0])[0]) >= float(gate["overall_ci_lower_min"]),
    "complex_point_estimate": float(complex_bucket.get("point_estimate", 0.0)) >= float(gate["complex_point_estimate_min"]),
    "complex_ci95_lower": float((complex_bucket.get("ci95") or [0.0])[0]) >= float(gate["complex_ci_lower_min"]),
    "learned_path_coverage": float(policy.get("learned_path_coverage", 0.0)) >= float(gate["min_learned_path_coverage"]),
    "move_path_coverage": float(policy.get("move_path_coverage", 0.0)) >= float(gate["min_move_path_coverage"]),
    "shoot_path_coverage": float(policy.get("shoot_path_coverage", 0.0)) >= float(gate["min_shoot_path_coverage"]),
    "elo_delta_lower": float(promotion.get("elo_delta_lower_bound", 0.0)) >= float(gate["elo_delta_lower_min"]),
    "max_bucket_regression": float(promotion.get("max_bucket_regression", 1.0)) <= float(gate["max_bucket_regression"]),
    "max_exploiter_win_rate": float(promotion.get("max_exploiter_win_rate", 1.0)) <= float(gate["max_exploiter_win_rate"]),
    "battle_phase_entered": promotion.get("battle_phase_entered") is True,
    "selected_move_shoot_occupy_actions": int(promotion.get("selected_move_shoot_occupy_actions", 0)) > 0,
    "promotion_accepted": promotion.get("accepted") is True,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"formal promotion gates failed for {run_dir}: {failed}; promotion_failures={promotion.get('failures')}")
print("formal_gates_ready=true")
PY
}

formal_required_consecutive() {
  python3.10 - "${FORMAL_CONFIG}" <<'PY'
import sys
from training.common import load_config, repo_root
config = load_config(repo_root() / sys.argv[1])
print(int(config.get("promotion_gate", {}).get("consecutive_windows", 1)))
PY
}

echo "Preflight auth check..."
python3.10 scripts/check_sdk_auth_environment.py --output "${PREFLIGHT_JSON}" \
  || fail "preflight command failed" "Run in an authorized environment and inspect ${PREFLIGHT_JSON}."
validate_preflight \
  || fail "preflight rejected current environment" "Run from a runtime whose non-loopback MAC matches the official engine_config binding."
write_report

echo "Running battle smoke..."
python3.10 -m training.selfplay \
  --config "${CONFIG_BATTLE}" \
  --candidate "${CANDIDATE}" \
  --incumbent "${INCUMBENT}" \
  || fail "battle smoke command failed" "Inspect the latest ${OUTPUT_ROOT}/league_* artifact and SDK output."
BATTLE_RUN="$(latest_run_dir phase_action_diagnostics.json)"
[[ -n "${BATTLE_RUN}" ]] || fail "battle smoke produced no artifact" "Check ${OUTPUT_ROOT} write permissions and self-play logs."
validate_battle_ready "${BATTLE_RUN}" \
  || fail "battle smoke did not produce authorized battle actions" "Fix SDK auth or battle-phase policy evidence before continuing."
write_report

echo "Running objective smoke..."
python3.10 -m training.selfplay \
  --config "${CONFIG_OBJECTIVE}" \
  --candidate "${CANDIDATE}" \
  --incumbent "${INCUMBENT}" \
  || fail "objective smoke command failed" "Inspect the latest ${OUTPUT_ROOT}/league_* artifact and SDK output."
OBJECTIVE_RUN="$(latest_run_dir objective_progress_diagnostics.json)"
[[ -n "${OBJECTIVE_RUN}" ]] || fail "objective smoke produced no artifact" "Check ${OUTPUT_ROOT} write permissions and self-play logs."
validate_battle_ready "${OBJECTIVE_RUN}" \
  || fail "objective smoke did not remain in battle phase" "Fix auth/battle evidence before building objective labels."
validate_objective_progress "${OBJECTIVE_RUN}" \
  || fail "objective smoke lacks real objective progress" "Run longer or fix policy/scenario selection so battle produces score/capture/damage/city-control progress."
write_report

echo "Building objective labels..."
python3.10 scripts/build_objective_labels.py \
  --input "${OBJECTIVE_RUN}" \
  --output-dir "${LABEL_DIR}" \
  || fail "objective label build failed" "Inspect ${OBJECTIVE_RUN} diagnostics and ${LABEL_DIR}."
validate_label_quality \
  || fail "objective labels failed quality gate" "Collect real battle objective progress before training or formal promotion."

echo "Validating objective-aware policy scoring loop..."
validate_objective_scoring "${OBJECTIVE_RUN}" \
  || fail "objective-aware policy scoring was not active" "Check ai/policy.py objective scoring diagnostics and model manifests."
write_report

echo "Running tests..."
python3.10 -m pytest tests -q \
  || fail "test suite failed" "Fix tests before formal self-play promotion."
write_report

required_consecutive="$(formal_required_consecutive)"
max_attempts="${FULL_PROMOTION_MAX_FORMAL_ATTEMPTS:-${required_consecutive}}"
accepted_streak=0
attempt=1
while [[ "${attempt}" -le "${max_attempts}" ]]; do
  echo "Running formal self-play attempt ${attempt}/${max_attempts} with ${FORMAL_CONFIG}..."
  python3.10 -m training.selfplay \
    --config "${FORMAL_CONFIG}" \
    --candidate "${CANDIDATE}" \
    --incumbent "${INCUMBENT}" \
    || fail "formal self-play command failed on attempt ${attempt}" "Inspect the latest ${OUTPUT_ROOT}/league_* artifact and SDK output."
  FORMAL_RUN="$(latest_run_dir promotion.json)"
  [[ -n "${FORMAL_RUN}" ]] || fail "formal self-play produced no promotion artifact" "Check ${OUTPUT_ROOT} write permissions and self-play logs."
  validate_battle_ready "${FORMAL_RUN}" \
    || fail "formal self-play did not keep authorized battle actions" "Fix auth/battle evidence before promotion."
  validate_objective_scoring "${FORMAL_RUN}" \
    || fail "formal self-play did not use objective-aware scoring" "Check policy scoring diagnostics in ${FORMAL_RUN}."
  if validate_formal_gates "${FORMAL_RUN}"; then
    accepted_streak=$((accepted_streak + 1))
    echo "formal_accepted_streak=${accepted_streak}/${required_consecutive}"
  else
    fail "formal promotion gates failed on attempt ${attempt}" "Inspect ${FORMAL_RUN}/promotion.json and improve policy/scorer before rerunning."
  fi
  write_report
  if [[ "${accepted_streak}" -ge "${required_consecutive}" ]]; then
    STATUS="completed"
    write_report
    echo "full_promotion_completed=true"
    echo "battle_run=${BATTLE_RUN}"
    echo "objective_run=${OBJECTIVE_RUN}"
    echo "label_dir=${LABEL_DIR}"
    echo "formal_run=${FORMAL_RUN}"
    echo "final_report=${FINAL_REPORT}"
    exit 0
  fi
  attempt=$((attempt + 1))
done

fail "formal accepted streak ${accepted_streak}/${required_consecutive} did not satisfy consecutive windows" \
  "Increase FULL_PROMOTION_MAX_FORMAL_ATTEMPTS or inspect ${FORMAL_RUN}/promotion.json."
