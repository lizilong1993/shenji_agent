from __future__ import annotations

import errno
import importlib
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _optional_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:  # pragma: no cover
        return None


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """固定随机种子；默认**同时开启确定性算法**（I07）。

    ## 为什么加 `deterministic`

    `31007 5.3.1 d)` 要求「为增强实验可重复性，训练初始化应允许设置随机数种子。
    在环境模拟、模型初始化、数据混洗等环节使用固定种子，**可重复相同训练过程验证结果**」。

    `docs/checkpoint_policy.md` 记录的缺口是：
    「同 seed 两次训练权重**逐字节一致**既未实现也无守护」，
    并注明「需**先工程化确定性**再谈守护」。

    ## 实测事实（第 55 轮）

    | 项 | 实测 |
    | --- | --- |
    | CPU 同 seed 两次训练权重 | **已逐字节一致**（SHA-256 相同） |
    | `torch.backends.cudnn.deterministic` | **False**（未设） |
    | `torch.are_deterministic_algorithms_enabled()` | **False** |
    | `CUBLAS_WORKSPACE_CONFIG` | **未设** |
    | CUDA 可用 | **True** |

    **即：CPU 路径本来就一致，但 CUDA 路径缺少确定性保证。**
    故此处补上 CUDA 侧的三项设置 —— 否则「逐字节一致」在 GPU 上不成立，
    而本机有 GPU，训练**可能**落到 GPU 上。

    ## `deterministic=False` 的用途

    某些算子**没有**确定性实现（如部分 scatter/atomic 操作），
    强开 `use_deterministic_algorithms(True)` 会让它们直接抛错。
    故保留关掉的开关，供"能跑但要快"的场景使用 ——
    **但那种场景下不得声称"逐字节可复现"**。

    ## 副作用说明（如实记录）

    `use_deterministic_algorithms(True)` 会让缺少确定性实现的算子**抛异常**
    （而非静默给不确定结果）。这是**有意的**：宁可报错，也不要"看起来可复现、实际不可复现"。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch = _optional_torch()
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not deterministic:
        # ⚠ 必须**显式关掉**，否则"关"只是"不开启"——
        # 我第一版直接 `return`，于是前一次 set_seed(默认 True) 打开的
        # `use_deterministic_algorithms` 会**残留**，该开关成了假的。
        # 测试 `test_deterministic_false_is_available_as_an_escape_hatch` 抓出了这一点。
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        except Exception:  # noqa: BLE001
            pass
        return
    # ── CUDA 确定性（本机 CUDA 可用，故这几项不是摆设）───────────────
    # cuBLAS 在 CUDA >= 10.2 时需要该环境变量才保证确定性；
    # 官方要求取值为 `:4096:8` 或 `:16:8`（前者显存占用更高）。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001 - 非 CUDA 构建可能没有这些属性
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:  # noqa: BLE001 - 老版本/特殊构建可能不支持
        pass


def load_config(path: Path | str) -> Dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def dump_yaml(data: Dict[str, Any], path: Path | str) -> None:
    out_path = Path(path)
    if yaml is not None:
        out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    out_path = Path(path)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    retry_errnos = {errno.ESTALE, errno.EINTR}
    last_error: OSError | None = None
    for attempt in range(6):
        tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.{attempt}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
            os.replace(tmp_path, out_path)
            return
        except OSError as exc:
            last_error = exc
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if exc.errno not in retry_errnos:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.5))
    if last_error is not None:
        raise last_error


def discover_dir(parent: Path | str, *, contains: Optional[str] = None, required_file: Optional[str] = None) -> Path:
    base = Path(parent)
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if contains and contains not in child.name:
            continue
        if required_file and not (child / required_file).exists():
            continue
        return child
    raise FileNotFoundError(f"Could not discover directory in {base} contains={contains} required_file={required_file}")


def safe_device():
    torch = _optional_torch()
    if torch is None:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# manifest 里**不是文件路径**的键。`copy_publish_artifacts` 遍历 manifest 时
# 会把任何字符串值当作 `run_dir` 下的路径去复制，故这些键必须跳过。
#
# 抽成模块级常量而不是留在函数里的字面量：这样**生产代码与测试引用同一个集合**，
# 测试才能真正否决「新增顶层字符串键」这类回归 —— 否则测试只能维护自己的一份
# 快照白名单，加字段时改测试即可通过，守卫形同虚设。
MANIFEST_NON_PATH_KEYS = frozenset({
    "generated_at", "managed_files", "environment",
    # I06 步骤 3：标签出处是**元数据**（相对仓库路径），不是 run_dir 下的产物
    "objective_label_source",
})


def copy_publish_artifacts(run_dir: Path, publish_dir: Path, manifest: Dict[str, Any]) -> None:
    ensure_dir(publish_dir)
    managed_files = manifest.get("managed_files", [])
    if isinstance(managed_files, list):
        for name in managed_files:
            if not isinstance(name, str):
                continue
            target = publish_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    for name, rel_path in manifest.items():
        if name in MANIFEST_NON_PATH_KEYS or not isinstance(rel_path, str):
            continue
        src = run_dir / rel_path
        if src.exists():
            target = publish_dir / src.name
            target.write_bytes(src.read_bytes())
    # ── I07：按**发布目录的实际字节**填入哈希，然后写 manifest ──────────────
    # 顺序很关键：必须在复制**之后**算，否则声明的是"复制前"的旧文件；
    # 而 manifest 又必须在填完哈希**之后**写，否则盘上的 manifest 里没有哈希。
    #
    # 为什么在这里而不是在 `_build_model_manifest()` 里：只有本函数知道
    # 「这次到底发布了哪些文件、字节是什么」。在 manifest 构造期去读 `ai/weights/`
    # 会把**上一次训练**的哈希写进本次 manifest，并可能让
    # `copy_publish_artifacts` 去找 `run_dir/<旧名>` 而漏发布新权重。
    try:
        import hashlib

        declared = {}
        for name in manifest.get("managed_files", []) or []:
            if not isinstance(name, str):
                continue
            f = publish_dir / name
            if f.is_file():
                declared[name] = hashlib.sha256(f.read_bytes()).hexdigest()
        if isinstance(manifest.get("environment"), dict):
            manifest["environment"]["artifact_sha256"] = declared
        manifest["artifact_sha256"] = declared
    except Exception:  # noqa: BLE001 - 发布不应因算哈希而失败
        pass

    save_json(manifest, publish_dir / "model_manifest.json")


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def git_head_sha(cwd: Optional[Path | str] = None) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd or repo_root()),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_show_text(spec: str, cwd: Optional[Path | str] = None) -> str:
    result = subprocess.run(
        ["git", "show", spec],
        cwd=str(cwd or repo_root()),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
