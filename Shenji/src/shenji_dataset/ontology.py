from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2] / "ontology" / "intent_ontology_v1.json"
)


def load_ontology(path: Path | str | None = None) -> dict[str, Any]:
    """Load the Shenji intent ontology from a UTF-8 JSON document."""

    ontology_path = Path(path) if path is not None else DEFAULT_ONTOLOGY_PATH
    with ontology_path.open("r", encoding="utf-8") as stream:
        ontology = json.load(stream)
    if not isinstance(ontology, dict):
        raise ValueError("ontology root must be a JSON object")
    return ontology


def intent_index(ontology: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index fine-grained intent entries by their stable code."""

    return {item["code"]: item for item in ontology.get("intents", [])}


def parent_index(ontology: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index parent intent categories by their stable code."""

    return {item["code"]: item for item in ontology.get("parent_categories", [])}
