from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTINUE_SCRIPT = ROOT / "scripts" / "continue_after_auth_fixed.sh"
ADMIN_SCRIPT = ROOT / "scripts" / "admin_run_authorized_selfplay.sh"
RUN_AUTHORIZED_SCRIPT = ROOT / "scripts" / "run_selfplay_authorized_mac.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_continue_after_auth_fixed_contains_required_continuation_commands() -> None:
    text = _read(CONTINUE_SCRIPT)

    assert "scripts/check_sdk_auth_environment.py" in text
    assert "configs/selfplay_smoke_battle_phase.yaml" in text
    assert "configs/selfplay_smoke_objective_quality.yaml" in text
    assert "scripts/build_objective_labels.py" in text
    assert "python3.10 -m pytest tests -q" in text
    assert "python3.10 -m training.selfplay" in text
    assert "battle_phase_entered=false" in text


def test_admin_script_validates_docker_mac_auth_battle_and_valid_actions() -> None:
    text = _read(ADMIN_SCRIPT)

    assert "docker compose" in text or "docker-compose" in text
    assert "docker-compose.auth-mac.yml" in text
    assert "02:42:ac:11:00:45" in text
    assert "/sys/class/net/*/address" in text
    assert "auth_status_diagnostics.json" in text
    assert "phase_action_diagnostics.json" in text
    assert "battle_phase_entered" in text
    assert "selected_move_shoot_occupy_actions" in text


def test_handoff_scripts_do_not_require_sudo_or_patch_sdk() -> None:
    combined = "\n".join(_read(path) for path in [CONTINUE_SCRIPT, ADMIN_SCRIPT, RUN_AUTHORIZED_SCRIPT])
    lowered = combined.lower()

    assert "sudo" not in lowered
    assert "reset --hard" not in lowered
    assert "git checkout" not in lowered
    assert "git restore" not in lowered
    assert "authenticator*.so" not in lowered
    assert "authenticator.cpython" not in lowered
    assert "promotion_gate:" not in lowered
    assert "overall_point_estimate_min" not in lowered
    assert "overall_ci_lower_min" not in lowered


def test_admin_script_refuses_to_fake_success() -> None:
    text = _read(ADMIN_SCRIPT)

    assert "raise SystemExit" in text
    assert "Authorized self-play validation failed" in text
    assert "selected <= 0" in text
    assert "status.startswith('failed')" in text
