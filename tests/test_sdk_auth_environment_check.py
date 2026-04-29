from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_sdk_auth_environment.py"


def _run_check(tmp_path: Path, engine_config: Path) -> dict[str, object]:
    output = tmp_path / "check.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--engine-config",
            str(engine_config),
            "--experiments-root",
            str(tmp_path / "experiments"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.stderr == ""
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == payload
    return payload


def test_missing_engine_config_does_not_crash(tmp_path: Path) -> None:
    payload = _run_check(tmp_path, tmp_path / "missing.engine_config")

    assert payload["schema_version"] == 1
    assert payload["engine_config"]["exists"] is False
    assert payload["auth_environment_ready"] is False
    assert payload["blocker_reason"] == "engine_config_missing"
    assert "official SDK .engine_config" in payload["next_action"]


def test_mac_mismatch_is_identified(tmp_path: Path) -> None:
    engine_config = tmp_path / ".engine_config"
    encoded = base64.b64encode(b"de:ad:be:ef:00:01,1777298501090988694").decode("ascii")
    engine_config.write_text(encoded, encoding="utf-8")

    payload = _run_check(tmp_path, engine_config)

    assert payload["engine_config"]["bound_mac"] == "de:ad:be:ef:00:01"
    assert payload["mac_match"] is False
    assert payload["blocker_reason"] == "mac_mismatch"
    assert "regenerated .engine_config" in payload["next_action"]


def test_json_schema_contains_stable_handoff_fields(tmp_path: Path) -> None:
    payload = _run_check(tmp_path, tmp_path / "missing.engine_config")

    for key in [
        "current_macs",
        "engine_config",
        "latest_auth_status_diagnostics",
        "latest_phase_action_diagnostics",
        "auth_environment_ready",
        "blocker_reason",
        "next_action",
        "run_after_fix_commands",
        "safety",
    ]:
        assert key in payload
    assert any("continue_after_auth_fixed.sh" in item for item in payload["run_after_fix_commands"])
    assert payload["safety"] == {
        "mutates_system": False,
        "patches_sdk": False,
        "fakes_auth_or_battle_evidence": False,
    }
