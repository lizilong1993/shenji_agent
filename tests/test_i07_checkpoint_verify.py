"""I07：检查点验证脚本 + manifest 六类字段。

判据 2 是「`python3 scripts/verify_checkpoint.py --weights ai/weights` 退出码 0」，
故这里**真的跑那个脚本**（子进程），而不是重新实现一遍它的逻辑 ——
重实现会掩盖"脚本本身跑不起来"这类问题。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFY = REPO / "scripts" / "verify_checkpoint.py"


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_checkpoint", VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────── 判据 2：脚本在真实权重上退出码 0

def test_verify_script_passes_on_repo_weights() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    wd = REPO / "ai" / "weights"
    if not wd.is_dir():
        pytest.skip("ai/weights not present")
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", str(wd), "--json"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, "退出码应为 0\n%s\n%s" % (r.stdout[-2000:], r.stderr[-2000:])
    rep = json.loads(r.stdout)
    assert rep["ok"] is True and rep["failed"] == 0


def test_verify_script_reports_hash_and_dimensions() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    wd = REPO / "ai" / "weights"
    if not wd.is_dir():
        pytest.skip("ai/weights not present")
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", str(wd), "--json"],
                       capture_output=True, text=True, timeout=300)
    rep = json.loads(r.stdout)
    # 每个托管文件都要有 sha256，第三方才能逐字节核对
    assert rep.get("files"), "未记录任何文件哈希"
    for name, info in rep["files"].items():
        assert len(info["sha256"]) == 64, "%s 的 sha256 长度不对" % name
        assert info["bytes"] > 0
    # 归一化器维度必须与模型输入匹配（清单点名要求 feature_dim=108）
    norm = [c for c in rep["checks"] if "归一化器 feature_dim == 模型输入维度" in c["name"]]
    assert norm and norm[0]["ok"], "归一化器 feature_dim 与模型输入不匹配"


def test_verify_script_exit_codes() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    # 不存在的目录 -> 用法错误 2
    r = subprocess.run([sys.executable, str(VERIFY), "--weights", "/nonexistent-xyz"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2


# ───────────────────────── 历史产物降级通过

def test_missing_environment_is_legacy_not_failure(tmp_path: Path) -> None:
    """【风险与回滚】：新字段可选，缺失时按**历史产物**降级通过并标注。

    旧权重不该因为"没有新字段"被判成坏权重 —— 那样验证脚本会永远红着，没人再看它。
    """
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    v = _load_verify()
    # ⚠ 必须把托管文件造齐：否则这个用例测的是「文件缺失」，而不是「缺 environment」。
    # 我第一版没造，于是它失败了一次 —— 那是**测试写错**，不是被测逻辑错。
    (tmp_path / "intent_lstm.pt").write_bytes(b"placeholder")
    (tmp_path / "situation_gcn.pt").write_bytes(b"placeholder")
    (tmp_path / "situation_feature_normalizer.json").write_text(
        json.dumps({"feature_dim": 108}), encoding="utf-8")
    (tmp_path / "model_manifest.json").write_text(json.dumps({
        "generated_at": "20260428_235813",
        "source_commit_sha": "c3bd538603afe1f020bea7709523c588fb8b69a0",
        "history_len": 8, "horizons": 5,
        "intent_lstm": "intent_lstm.pt",
        "situation_gcn": "situation_gcn.pt",
        "situation_feature_normalizer": "situation_feature_normalizer.json",
        "managed_files": [],
    }, ensure_ascii=False), encoding="utf-8")
    rep = v.verify(tmp_path)
    legacy_checks = [c for c in rep["checks"] if "历史产物降级通过" in c["name"]]
    assert legacy_checks and legacy_checks[0]["ok"], \
        "缺 environment 时应按历史产物**判通过**并标注"
    assert any("legacy_artifact" in w for w in rep["warnings"]), "未标注 legacy_artifact"


def test_missing_manifest_fails(tmp_path: Path) -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    v = _load_verify()
    rep = v.verify(tmp_path)
    assert rep["ok"] is False and rep["failed"] >= 1


# ───────────────────────── manifest 构建：六类字段

def test_manifest_has_all_six_field_classes() -> None:
    if not VERIFY.exists():
        pytest.skip("verify_checkpoint.py not present")
    from training.supervised import _build_model_manifest

    cfg = {"intent": {"history_len": 8, "horizons": 5}}
    metrics = {"situation": {"normalizer": {
        "kind": "per_dimension_zscore", "fitted_on": "train",
        "feature_dim": 108, "zero_std_count": 0, "training_applied_normalizer": True}}}
    m = _build_model_manifest(cfg, metrics, tactical_skipped=False)
    env = m["environment"]
    assert m["source_commit_sha"]                     # 1 代码状态
    assert "requirements_lock_sha256" in env          # 2 数据/依赖哈希（值可 None）
    assert m["history_len"] == 8 and m["horizons"] == 5   # 3 超参
    assert env["python_version"] and "torch_version" in env  # 4 依赖版本
    assert env["platform"]                            # 5 环境
    assert env["experiment_id"]                       # 6 实验身份


def test_experiment_id_is_deterministic_and_derived() -> None:
    """同输入必须同 ID（不含随机数），否则第三方无法判断"是不是同一次实验"。"""
    from training.supervised import _experiment_id
    a = _experiment_id("20260101_000000")
    b = _experiment_id("20260101_000000")
    assert a == b, "实验 ID 不确定，幂等/复跑核验会失效"
    assert a.startswith("20260101_000000-")


# ───────────────────────── 保护 copy_publish_artifacts 不被误伤

def _path_like_top_level_strings(manifest, run_dir):
    """返回「会被 `copy_publish_artifacts` 当作路径复制」的顶层字符串键。

    判据必须**对齐真实危害**：`copy_publish_artifacts` 复制的是 `run_dir / <值>`，
    且 `src.exists()` 才复制。故只有"值是相对路径**且** run_dir 下真实存在"的键
    才会被误复制。

    ⚠ 我第一版用硬编码白名单快照，与危害不对齐：它会把 I06 要写入的合法字符串
    字段（`objective_mode` / `objective_label_source`）判成 offender，
    于是**本测试会在 I06 落地时立刻失败** —— 把冲突从文本合并期搬到测试红灯期。
    第二版改成拿 repo_root 当 run_dir，又会因为 `objective_label_source` 在仓里
    确实存在而误报。**只有按真实 run_dir 判定才对。**
    """
    from training.common import MANIFEST_NON_PATH_KEYS

    out = []
    for key, value in manifest.items():
        if key in MANIFEST_NON_PATH_KEYS or not isinstance(value, str):
            continue
        if (Path(run_dir) / value).exists():
            out.append((key, value))
    return out


def test_environment_is_a_dict_and_not_a_path_like_string(tmp_path: Path) -> None:
    """新信息必须收进 `environment` dict，**且不得新增"会被当路径"的顶层字符串键**。

    ⚠ 这里**引用生产代码的跳过集合**（`training.common.MANIFEST_NON_PATH_KEYS`），
    而不是抄一份白名单快照。

    为什么必须这样（这是被评审实测证伪过的教训）：我第一版用硬编码白名单，
    于是 I06 要写回的 `objective_mode` / `objective_label_source`（合法的字符串字段）
    会被判成"offender" —— **本测试会在 I06 落地时立刻失败**，
    把冲突从"文本合并期"搬到"测试红灯期"，正好挡住它声称要兼容的那个任务。
    对 `copy_publish_artifacts` 而言，键名本身无害，**危害在于"值是字符串会被当路径"**；
    故判据应对齐危害，而非对齐一份快照。
    """
    from training.common import MANIFEST_NON_PATH_KEYS
    from training.supervised import _build_model_manifest

    cfg = {"intent": {"history_len": 8, "horizons": 5}}
    metrics = {"situation": {"normalizer": {}}}
    m = _build_model_manifest(cfg, metrics, tactical_skipped=True)

    # ① 新信息确实在 environment 里（本次改动的核心主张）
    assert isinstance(m["environment"], dict)
    assert "environment" in MANIFEST_NON_PATH_KEYS, (
        "生产代码的跳过集合未包含 environment —— 那它会被 copy_publish_artifacts 当路径")

    # ② 顶层字符串值：只有"**确实会被解析为 run_dir 下真实路径**"的才算危害。
    #    这里用 repo_root 当 run_dir 的代理：manifest 里的相对路径若真存在，才判 offender。
    # 用 tmp_path 当 run_dir：与 copy_publish_artifacts 的真实语义一致。
    # 这样 I06 写入 objective_* 这类**不在 run_dir 下的**字符串字段不会误报。
    offenders = _path_like_top_level_strings(m, tmp_path)
    assert offenders == [], (
        "顶层字符串值在 run_dir 下真实存在，会被 copy_publish_artifacts 复制: %s" % offenders)

    # 反向自检：**真的会被误复制的形态必须被抓到**（否则本守卫是恒真的）
    bait = dict(m)
    (tmp_path / "bait_value").write_text("x", encoding="utf-8")
    bait["a_new_scalar"] = "bait_value"
    assert _path_like_top_level_strings(bait, tmp_path), (
        "守卫抓不到真实误复制形态 —— 那它等于没有保护")

    # ③ 既有键的类型未被改动（纯加法的机器保证）
    for k in ("history_len", "horizons"):
        assert isinstance(m[k], int)
    assert isinstance(m.get("managed_files"), list)


# ───────────────────────── 评审整改：负向用例（此前完全缺失）

def _fixture_dir(tmp_path: Path, **manifest_overrides):
    """造一个带真占位产物的目录，并按需改写 manifest。"""
    import hashlib
    import json as _json

    from training.supervised import _build_model_manifest

    for name in ("intent_lstm.pt", "situation_gcn.pt", "tactical_model.pt"):
        (tmp_path / name).write_bytes(b"placeholder-" + name.encode())
    norm = {"feature_dim": 108}
    (tmp_path / "situation_feature_normalizer.json").write_text(
        _json.dumps(norm), encoding="utf-8")
    m = _build_model_manifest({"intent": {"history_len": 8, "horizons": 5}},
                              {"situation": {"normalizer": norm}}, tactical_skipped=False)
    # 填上真实哈希，模拟"已发布"的 manifest
    m["environment"]["artifact_sha256"] = {
        n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
        for n in ("intent_lstm.pt", "situation_gcn.pt", "tactical_model.pt",
                  "situation_feature_normalizer.json")
    }
    m.update(manifest_overrides)
    (tmp_path / "model_manifest.json").write_text(
        _json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return m


def _run_verify(tmp_path: Path):
    return subprocess.run([sys.executable, str(VERIFY), "--weights", str(tmp_path), "--json"],
                          capture_output=True, text=True, timeout=300)


def test_hash_mismatch_is_detected(tmp_path: Path) -> None:
    """**判据 3 的负向用例**：声明的哈希与实际字节不符必须失败。

    这是评审实测出来的缺陷 —— 原实现只算哈希、从不比对，
    伪造哈希后仍 EXIT=0，等于「这份权重是不是 manifest 说的那一份」无从回答。
    """
    _fixture_dir(tmp_path)
    import json as _json

    m = _json.loads((tmp_path / "model_manifest.json").read_text(encoding="utf-8"))
    m["environment"]["artifact_sha256"]["intent_lstm.pt"] = "0" * 64
    (tmp_path / "model_manifest.json").write_text(_json.dumps(m), encoding="utf-8")

    r = _run_verify(tmp_path)
    assert r.returncode == 1, "伪造哈希必须判失败，实际 exit=%d" % r.returncode
    rep = json.loads(r.stdout)
    assert rep["ok"] is False
    assert any("哈希" in c["name"] and not c["ok"] for c in rep["checks"]), rep["checks"]


def test_declared_but_missing_weight_is_rejected(tmp_path: Path) -> None:
    """声明了文件名却不存在 → 失败。**不得静默回退别名**（评审实测的缺陷）。"""
    _fixture_dir(tmp_path, intent_lstm="this_file_does_not_exist.pt")
    r = _run_verify(tmp_path)
    assert r.returncode == 1
    rep = json.loads(r.stdout)
    assert any("intent_lstm" in c["name"] and not c["ok"] for c in rep["checks"]), rep["checks"]


def test_declared_but_missing_tactical_is_rejected(tmp_path: Path) -> None:
    """`tactical_model` 只是**未声明**时才可选；**声明了却缺失必须失败**。"""
    _fixture_dir(tmp_path, tactical_model="tactical_model.pt")
    (tmp_path / "tactical_model.pt").unlink()
    r = _run_verify(tmp_path)
    assert r.returncode == 1, "声明了 tactical 却缺失，应失败"
    rep = json.loads(r.stdout)
    assert any("tactical_model" in c["name"] and not c["ok"] for c in rep["checks"])


def test_malformed_inputs_fail_cleanly_without_traceback(tmp_path: Path) -> None:
    """畸形输入必须**干净失败**：退出码 1 且 stdout 仍是可解析的 JSON。

    `--json` 与退出码是被机器消费的；崩溃会让调用方拿不到 checks/failed/warnings，
    恰与「输出差异明细」相悖。评审用四种畸形态证伪过原实现。
    """
    cases = {
        "bad_json": "{oops",
        "manifest_is_list": "[1,2,3]",
        "manifest_is_string": '"hello"',
    }
    for label, payload in cases.items():
        d = tmp_path / label
        d.mkdir()
        (d / "model_manifest.json").write_text(payload, encoding="utf-8")
        r = _run_verify(d)
        assert "Traceback" not in (r.stderr or ""), "%s 崩溃了: %s" % (label, r.stderr[-300:])
        assert r.returncode == 1, "%s exit=%d" % (label, r.returncode)
        json.loads(r.stdout)          # 必须是可解析 JSON


def test_non_integer_feature_dim_fails_cleanly(tmp_path: Path) -> None:
    import json as _json

    _fixture_dir(tmp_path)
    (tmp_path / "situation_feature_normalizer.json").write_text(
        _json.dumps({"feature_dim": "abc"}), encoding="utf-8")
    r = _run_verify(tmp_path)
    assert "Traceback" not in (r.stderr or ""), r.stderr[-300:]
    assert r.returncode == 1
    rep = json.loads(r.stdout)
    assert rep["ok"] is False


def test_publish_records_hashes_of_published_bytes(tmp_path: Path) -> None:
    """发布流程必须在**复制完成后**按发布目录的实际字节填哈希。

    两个必须同时成立的性质：
      ① 本次训练写出的规范名被真正复制（不回看发布位旧文件 → 否则新权重被静默漏发布）；
      ② 声明的哈希 == 发布目录里那份文件的字节。
    """
    import hashlib
    import json as _json

    from training.common import copy_publish_artifacts
    from training.supervised import _build_model_manifest

    run_dir = tmp_path / "run"
    pub = tmp_path / "pub"
    run_dir.mkdir()
    for name in ("intent_lstm.pt", "situation_gcn.pt", "tactical_model.pt",
                 "situation_feature_normalizer.json"):
        (run_dir / name).write_text("CONTENT-" + name, encoding="utf-8")
    m = _build_model_manifest({"intent": {"history_len": 8, "horizons": 5}},
                              {"situation": {"normalizer": {}}}, tactical_skipped=False)
    # 构造期不填哈希（那会描述发布位旧文件）
    assert m["environment"]["artifact_sha256"] == {}

    copy_publish_artifacts(run_dir, pub, m)

    # ① 本次产物被复制
    for name in ("intent_lstm.pt", "situation_gcn.pt", "tactical_model.pt"):
        assert (pub / name).is_file(), "新权重未被发布（静默漏发布）: %s" % name
        assert (pub / name).read_text(encoding="utf-8") == "CONTENT-" + name

    # ② manifest 里的哈希 == 发布目录的实际字节
    written = _json.loads((pub / "model_manifest.json").read_text(encoding="utf-8"))
    declared = written.get("artifact_sha256") or written["environment"]["artifact_sha256"]
    assert declared, "发布后未记录任何哈希"
    for name, dig in declared.items():
        assert dig == hashlib.sha256((pub / name).read_bytes()).hexdigest(), \
            "%s 的声明哈希与发布字节不符" % name
