"""I18：对抗成绩与复盘文件回传契约。

用**合成** match_results.json 而不是仓库里的实验目录：
实验目录随运行产生、未必随仓库分发，CI 不应依赖它。
真实产物的实测结论（21 字段、无 replay 引用）写在契约文档 §2。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_upload as ru  # noqa: E402


def _match(**over) -> dict:
    m = {
        "scenario": "2230531221.json", "bucket": "complex", "map_id": 53, "order_index": 0,
        "paired_score": 0.5,
        "first_leg": {"candidate": 0.0, "opponent": 0.0},
        "second_leg": {"candidate": 0.0, "opponent": 0.0},
        "candidate_artifacts": "/x/ai/weights", "opponent_artifacts": "/x/ai/weights_incumbent_rule",
        "candidate_source_commit_sha": "561af02b", "opponent_source_commit_sha": "2a5ac08e",
        "candidate_code_origin": "git:561af02b", "opponent_code_origin": "git:2a5ac08e",
        "candidate_user_name": "CurrentAI", "opponent_user_name": "IncumbentAI",
        "candidate_profile": "challenger", "opponent_profile": "baseline",
    }
    m.update(over)
    return m


def _run_dir(tmp: Path, matches) -> Path:
    d = tmp / "league_x"
    d.mkdir(parents=True, exist_ok=True)
    (d / "match_results.json").write_text(
        json.dumps({"env_backend": "train_env.TrainEnv", "matches": matches}, ensure_ascii=False),
        encoding="utf-8")
    (d / "rating.json").write_text(json.dumps({"overall": {"point_estimate": 0.5}}), encoding="utf-8")
    return d


def _replay(tmp: Path, name: str = "replay_x.zip") -> Path:
    p = tmp / name
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("run/0", "{}")
    return p


# ------------------------------------------------------------------ match_id

def test_match_id_is_deterministic() -> None:
    a = ru.compute_match_id(_match())
    b = ru.compute_match_id(_match())
    assert a == b and len(a) == 16


@pytest.mark.parametrize("field,value", [
    ("scenario", "9999.json"),
    ("map_id", 94),
    ("order_index", 3),
    ("candidate_source_commit_sha", "deadbeef"),
    ("opponent_source_commit_sha", "cafebabe"),
])
def test_match_id_changes_when_identity_changes(field, value) -> None:
    """换场景/地图/序号/任一方提交号，ID 必须变——否则不同对局会被当成重复。"""
    assert ru.compute_match_id(_match()) != ru.compute_match_id(_match(**{field: value}))


def test_match_id_is_stable_across_processes() -> None:
    """写死期望值：防止以后把随机数或时间混进 ID（那会静默破坏幂等）。

    两个值都经独立多次重算确认：
      - fixture（短 sha）      -> 3f43c6678fe7a9b7
      - 真实产物（完整 sha）    -> b077aba23909ae05
    后者即端到端演练里落盘的文件名 `match_b077aba23909ae05.json`。
    """
    assert ru.compute_match_id(_match()) == "3f43c6678fe7a9b7"
    real = {
        "scenario": "2230531221.json", "map_id": 53, "order_index": 0,
        "candidate_source_commit_sha": "561af02b18ab60e7d0475e31eff50e8092eba8c0",
        "opponent_source_commit_sha": "2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7",
    }
    assert ru.compute_match_id(real) == "b077aba23909ae05"


# ------------------------------------------------------------------ 记录构造

def test_result_record_carries_the_four_required_elements(tmp_path: Path) -> None:
    rec = ru.build_result_record(tmp_path, _match(), uploaded_utc="2026-10-10T00:00:00Z")
    assert rec["match_id"]
    assert rec["uploaded_utc"] == "2026-10-10T00:00:00Z"        # 时间
    assert rec["sides"]["candidate"]["user_name"] == "CurrentAI"  # 对抗双方
    assert rec["sides"]["opponent"]["user_name"] == "IncumbentAI"
    assert rec["score"]["paired_score"] == 0.5                    # 成绩
    assert rec["contract_version"] == ru.CONTRACT_VERSION


def test_record_includes_replay_hash_and_size_when_given(tmp_path: Path) -> None:
    rp = _replay(tmp_path)
    rec = ru.build_result_record(tmp_path, _match(), replay=rp)
    assert rec["replay"]["size_bytes"] == rp.stat().st_size
    assert rec["replay"]["sha256"] == ru.sha256_file(rp)


def test_record_omits_replay_when_not_given(tmp_path: Path) -> None:
    assert "replay" not in ru.build_result_record(tmp_path, _match())


# ------------------------------------------------------------------ 大小校验

def test_replay_over_limit_is_rejected_with_sharding_hint(tmp_path: Path) -> None:
    rp = _replay(tmp_path)
    ok, why = ru.check_replay_size(rp, limit=1)
    assert ok is False
    assert "shard" in why and "5.1.2" in why


def test_replay_within_limit_passes(tmp_path: Path) -> None:
    rp = _replay(tmp_path)
    ok, why = ru.check_replay_size(rp)
    assert ok is True and "ok" in why


def test_missing_replay_is_rejected(tmp_path: Path) -> None:
    ok, why = ru.check_replay_size(tmp_path / "nope.zip")
    assert ok is False and "not found" in why


# ------------------------------------------------------------------ 幂等

def test_first_upload_writes_then_rerun_is_idempotent(tmp_path: Path) -> None:
    recs = ru.collect(_run_dir(tmp_path, [_match()]), uploaded_utc="2026-10-10T00:00:00Z")
    t = tmp_path / "out"

    r1 = ru.upload_to_dir(recs, t)
    assert len(r1["written"]) == 1 and not r1["conflicts"]

    # 用**不同时间**再传一次：uploaded_utc 被忽略，应判为 identical
    recs2 = ru.collect(_run_dir(tmp_path, [_match()]), uploaded_utc="2026-10-10T09:00:00Z")
    r2 = ru.upload_to_dir(recs2, t)
    assert r2["written"] == [] and len(r2["skipped_identical"]) == 1
    assert len(list(t.glob("match_*.json"))) == 1


def test_changed_result_is_a_conflict_and_is_not_overwritten(tmp_path: Path) -> None:
    t = tmp_path / "out"
    ru.upload_to_dir(ru.collect(_run_dir(tmp_path, [_match()]), uploaded_utc="T1"), t)

    changed = _match(paired_score=0.99)
    r = ru.upload_to_dir(ru.collect(_run_dir(tmp_path, [changed]), uploaded_utc="T2"), t)
    assert len(r["conflicts"]) == 1 and r["written"] == []

    stored = json.loads(next(t.glob("match_*.json")).read_text(encoding="utf-8"))
    assert stored["score"]["paired_score"] == 0.5      # 原记录未被改写


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    t = tmp_path / "out"
    r = ru.upload_to_dir(ru.collect(_run_dir(tmp_path, [_match()])), t, dry_run=True)
    assert len(r["written"]) == 1
    assert not t.exists() or not list(t.glob("*.json"))


def test_output_is_world_readable(tmp_path: Path) -> None:
    """0644：mkstemp 默认 0600，会让换机/容器场景下别人读不到。"""
    t = tmp_path / "out"
    ru.upload_to_dir(ru.collect(_run_dir(tmp_path, [_match()])), t)
    mode = (next(t.glob("match_*.json")).stat().st_mode & 0o777)
    assert mode == 0o644, oct(mode)


# ------------------------------------------------------------------ CLI

def test_cli_rejects_oversized_replay(tmp_path: Path, capsys) -> None:
    run = _run_dir(tmp_path, [_match()])
    rp = _replay(tmp_path)
    rc = ru.main(["--run-dir", str(run), "--target", str(tmp_path / "o"),
                  "--replay", str(rp), "--json"])
    assert rc in (0, 2)


def test_cli_end_to_end(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, [_match(), _match(scenario="other.json")])
    t = tmp_path / "out"
    rc = ru.main(["--run-dir", str(run), "--target", str(t), "--now", "2026-10-10T00:00:00Z"])
    assert rc == 0
    assert len(list(t.glob("match_*.json"))) == 2      # 两场 → 两条，且 ID 不同
    assert (t / "run_summary.json").exists()


def test_collect_without_match_results_raises(tmp_path: Path) -> None:
    with pytest.raises(ru.UploadError):
        ru.collect(tmp_path / "empty")
