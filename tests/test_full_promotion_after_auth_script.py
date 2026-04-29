from __future__ import annotations

import os
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_promotion_after_auth.sh"
BOOTSTRAP_REMOTE = ROOT / "scripts" / "bootstrap_remote.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_full_promotion_runner_exists_and_is_executable() -> None:
    assert SCRIPT.exists()
    mode = os.stat(SCRIPT).st_mode
    assert mode & stat.S_IXUSR


def test_full_promotion_runner_contains_required_pipeline_steps() -> None:
    text = _read(SCRIPT)

    assert "scripts/check_sdk_auth_environment.py" in text
    assert "configs/selfplay_smoke_battle_phase.yaml" in text
    assert "configs/selfplay_smoke_objective_quality.yaml" in text
    assert "scripts/build_objective_labels.py" in text
    assert "validate_label_quality" in text
    assert "validate_objective_scoring" in text
    assert "python3.10 -m pytest tests -q" in text
    assert "configs/selfplay_sota_stop.yaml" in text
    assert "validate_formal_gates" in text
    assert "final_autonomous_completion_report.md" in text


def test_full_promotion_runner_fails_closed_without_claiming_success() -> None:
    text = _read(SCRIPT)

    assert "fail()" in text
    assert "exit 2" in text
    assert "full_promotion_completed=true" in text
    assert "STATUS=\"completed\"" in text
    assert text.index("STATUS=\"completed\"") < text.index("full_promotion_completed=true")
    assert "promotion.get(\"accepted\") is True" in text


def test_full_promotion_runner_does_not_require_host_or_sdk_mutation() -> None:
    lowered = _read(SCRIPT).lower()

    assert "sudo" not in lowered
    assert "reset --hard" not in lowered
    assert "git checkout" not in lowered
    assert "git restore" not in lowered
    assert "authenticator" not in lowered
    assert ".engine_config" not in lowered
    assert "mac_address:" not in lowered
    assert "overall_point_estimate_min:" not in lowered
    assert "overall_ci_lower_min:" not in lowered


def test_remote_bootstrap_uses_non_destructive_git_update() -> None:
    text = _read(BOOTSTRAP_REMOTE)

    assert "pull --ff-only" in text
    assert "reset --hard" not in text
