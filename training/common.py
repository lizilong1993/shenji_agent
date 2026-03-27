from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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


def safe_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def copy_publish_artifacts(run_dir: Path, publish_dir: Path, manifest: Dict[str, Any]) -> None:
    ensure_dir(publish_dir)
    for name, rel_path in manifest.items():
        if name == "generated_at" or not isinstance(rel_path, str):
            continue
        src = run_dir / rel_path
        if src.exists():
            target = publish_dir / src.name
            target.write_bytes(src.read_bytes())
    save_json(manifest, publish_dir / "model_manifest.json")


def env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default
