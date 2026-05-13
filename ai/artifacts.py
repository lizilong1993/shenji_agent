from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


MODEL_KEYS = ("intent_lstm", "intent_model", "situation_gcn", "situation_model", "tactical_model")

LEGACY_ALIASES = {
    "intent_lstm.pt": ("intention_lstm.pth",),
    "situation_gcn.pt": ("situation_gcn.pth",),
}


@dataclass(frozen=True)
class ArtifactResolution:
    key: str
    expected_name: str
    path: Path
    exists: bool
    required: bool
    resolved_by_alias: bool = False
    alias_for: Optional[str] = None

    @property
    def mismatch(self) -> bool:
        return self.resolved_by_alias


def model_entries(manifest: Mapping[str, Any]) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for key in MODEL_KEYS:
        value = manifest.get(key)
        if value:
            entries[key] = str(value)
    return entries


def _alias_candidates(expected_name: str, aliases: Mapping[str, Iterable[str]]) -> Iterable[str]:
    yield from aliases.get(expected_name, ())


def resolve_model_artifact(
    artifact_dir: Path,
    key: str,
    expected_name: str,
    *,
    required: bool = True,
    aliases: Mapping[str, Iterable[str]] = LEGACY_ALIASES,
) -> ArtifactResolution:
    expected_path = artifact_dir / expected_name
    if expected_path.exists():
        return ArtifactResolution(key=key, expected_name=expected_name, path=expected_path, exists=True, required=required)
    for alias_name in _alias_candidates(expected_name, aliases):
        alias_path = artifact_dir / alias_name
        if alias_path.exists():
            return ArtifactResolution(
                key=key,
                expected_name=expected_name,
                path=alias_path,
                exists=True,
                required=required,
                resolved_by_alias=True,
                alias_for=expected_name,
            )
    return ArtifactResolution(key=key, expected_name=expected_name, path=expected_path, exists=False, required=required)


def resolve_manifest_assets(artifact_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, ArtifactResolution]:
    return {
        key: resolve_model_artifact(artifact_dir, key, expected_name, required=True)
        for key, expected_name in model_entries(manifest).items()
    }


def missing_required_assets(resolutions: Mapping[str, ArtifactResolution]) -> Dict[str, ArtifactResolution]:
    return {key: value for key, value in resolutions.items() if value.required and not value.exists}


def naming_mismatches(resolutions: Mapping[str, ArtifactResolution]) -> Dict[str, ArtifactResolution]:
    return {key: value for key, value in resolutions.items() if value.mismatch}


def resolution_to_dict(resolution: ArtifactResolution) -> Dict[str, Any]:
    return {
        "key": resolution.key,
        "expected_name": resolution.expected_name,
        "resolved_path": str(resolution.path),
        "exists": resolution.exists,
        "required": resolution.required,
        "resolved_by_alias": resolution.resolved_by_alias,
        "alias_for": resolution.alias_for,
    }
