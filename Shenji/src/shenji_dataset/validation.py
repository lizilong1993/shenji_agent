from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .ontology import intent_index


LABEL_KINDS = {"commanded", "realized", "human_inferred"}
SUBJECT_SCOPES = {"entity", "group", "company", "side"}
TARGET_TYPES = {
    "enemy_entity",
    "friendly_entity",
    "friendly_group",
    "objective_point",
    "terrain_region",
    "route",
    "own_survival",
    "none",
}
STATUSES = {"planned", "executing", "completed", "aborted", "switched"}
OUTCOMES = {"achieved", "failed", "partial", "not_triggered", "unknown"}
EVIDENCE_SOURCES = {
    "scenario_plan",
    "mission_command",
    "deterministic_rule",
    "human_review",
}
EVIDENCE_GRADES = {"A", "B", "C"}


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def validate_ontology(ontology: dict[str, Any]) -> list[str]:
    """Return all structural ontology errors without stopping at the first one."""

    errors: list[str] = []
    parents = ontology.get("parent_categories")
    intents = ontology.get("intents")
    if not isinstance(parents, list):
        return ["parent_categories must be a list"]
    if not isinstance(intents, list):
        return ["intents must be a list"]
    if len(parents) != 6:
        errors.append("ontology must contain exactly 6 parent categories")
    if len(intents) != 22:
        errors.append("ontology must contain exactly 22 fine intents")

    parent_codes = [item.get("code") for item in parents if isinstance(item, dict)]
    intent_codes = [item.get("code") for item in intents if isinstance(item, dict)]
    if len(parent_codes) != len(set(parent_codes)):
        errors.append("parent category codes must be unique")
    if len(intent_codes) != len(set(intent_codes)):
        errors.append("fine intent codes must be unique")

    parent_code_set = set(parent_codes)
    intent_code_set = set(intent_codes)
    declared_children: list[str] = []
    for parent in parents:
        if not isinstance(parent, dict):
            errors.append("each parent category must be an object")
            continue
        children = parent.get("children", [])
        if not isinstance(children, list):
            errors.append(f"parent {parent.get('code')} children must be a list")
            continue
        declared_children.extend(children)
        unknown = set(children) - intent_code_set
        if unknown:
            errors.append(
                f"parent {parent.get('code')} declares unknown children: {sorted(unknown)}"
            )
    child_duplicates = _duplicates(declared_children)
    if child_duplicates:
        errors.append(f"fine intents must belong to one parent: {sorted(child_duplicates)}")
    if set(declared_children) != intent_code_set:
        errors.append("parent children must cover every fine intent exactly once")

    for intent in intents:
        if not isinstance(intent, dict):
            errors.append("each fine intent must be an object")
            continue
        code = intent.get("code")
        parent_code = intent.get("parent_code")
        if parent_code not in parent_code_set:
            errors.append(f"intent {code} references unknown parent {parent_code}")
        if intent.get("auto_label_grade") not in EVIDENCE_GRADES:
            errors.append(f"intent {code} has invalid auto_label_grade")
        if not isinstance(intent.get("requires_plan_tag"), bool):
            errors.append(f"intent {code} requires_plan_tag must be boolean")
        parent = next((item for item in parents if item.get("code") == parent_code), None)
        if parent is not None and code not in parent.get("children", []):
            errors.append(f"intent {code} is not listed under parent {parent_code}")
    return errors


def validate_intent_segment(
    segment: dict[str, Any], ontology: dict[str, Any]
) -> list[str]:
    """Validate one temporal multi-label intent segment."""

    errors: list[str] = []
    required = {
        "segment_id",
        "label_kind",
        "subject_scope",
        "subject_id",
        "start_step",
        "end_step",
        "parent_intent",
        "primary_intent",
        "secondary_intents",
        "target_type",
        "target_id",
        "status",
        "outcome",
        "evidence_source",
        "evidence_grade",
        "evidence_refs",
        "confidence",
    }
    missing = sorted(required - set(segment))
    if missing:
        errors.append(f"intent segment is missing required fields: {missing}")
        return errors

    if segment["label_kind"] not in LABEL_KINDS:
        errors.append(f"invalid label_kind: {segment['label_kind']}")
    if segment["subject_scope"] not in SUBJECT_SCOPES:
        errors.append(f"invalid subject_scope: {segment['subject_scope']}")
    if segment["target_type"] not in TARGET_TYPES:
        errors.append(f"invalid target_type: {segment['target_type']}")
    if segment["status"] not in STATUSES:
        errors.append(f"invalid status: {segment['status']}")
    if segment["outcome"] not in OUTCOMES:
        errors.append(f"invalid outcome: {segment['outcome']}")
    if segment["evidence_source"] not in EVIDENCE_SOURCES:
        errors.append(f"invalid evidence_source: {segment['evidence_source']}")
    if segment["evidence_grade"] not in EVIDENCE_GRADES:
        errors.append(f"invalid evidence_grade: {segment['evidence_grade']}")

    start_step = segment["start_step"]
    end_step = segment["end_step"]
    if not isinstance(start_step, int) or isinstance(start_step, bool):
        errors.append("start_step must be an integer")
    if not isinstance(end_step, int) or isinstance(end_step, bool):
        errors.append("end_step must be an integer")
    if isinstance(start_step, int) and isinstance(end_step, int) and start_step > end_step:
        errors.append("start_step must be less than or equal to end_step")

    confidence = segment["confidence"]
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("confidence must be between 0 and 1 inclusive")

    intents = intent_index(ontology)
    primary = segment["primary_intent"]
    secondary = segment["secondary_intents"]
    if primary not in intents:
        errors.append(f"unknown primary_intent: {primary}")
    else:
        expected_parent = intents[primary]["parent_code"]
        if segment["parent_intent"] != expected_parent:
            errors.append(
                f"parent_intent must be {expected_parent} for primary intent {primary}"
            )
    if not isinstance(secondary, list):
        errors.append("secondary_intents must be a list")
        secondary = []
    if primary in secondary:
        errors.append("primary intent must not be repeated in secondary intents")
    if len(secondary) != len(set(secondary)):
        errors.append("secondary intents must be unique")
    unknown_secondary = sorted(set(secondary) - set(intents))
    if unknown_secondary:
        errors.append(f"unknown secondary intents: {unknown_secondary}")

    all_labels = {primary, *secondary}
    if "FEINT" in all_labels and segment["evidence_source"] not in {
        "scenario_plan",
        "human_review",
    }:
        errors.append("FEINT requires a scenario_plan or human_review evidence source")
    if not isinstance(segment["evidence_refs"], list):
        errors.append("evidence_refs must be a list")
    return errors


def validate_tables(
    tables: dict[str, list[dict[str, Any]]], ontology: dict[str, Any]
) -> list[str]:
    """Validate cross-table release invariants."""

    errors: list[str] = []
    required_tables = {
        "episodes",
        "entities",
        "states",
        "observations",
        "actions",
        "events",
        "relations",
        "intent_segments",
    }
    missing_tables = sorted(required_tables - set(tables))
    if missing_tables:
        return [f"missing canonical tables: {missing_tables}"]
    episode_ids = {row.get("episode_id") for row in tables["episodes"]}
    if None in episode_ids or not episode_ids:
        errors.append("episodes table must contain non-null episode_id values")
    for table_name, rows in tables.items():
        if not isinstance(rows, list):
            errors.append(f"table {table_name} must be a list")
            continue
        for row_index, row in enumerate(rows):
            if row.get("episode_id") not in episode_ids:
                errors.append(
                    f"{table_name}[{row_index}] references an unknown episode_id"
                )
    for row_index, segment in enumerate(tables["intent_segments"]):
        segment_errors = validate_intent_segment(segment, ontology)
        errors.extend(
            f"intent_segments[{row_index}]: {error}" for error in segment_errors
        )

    # ── Entity referential integrity ──────────────────────────────
    entity_ids: set[str] = set()
    for i, row in enumerate(tables.get("entities", [])):
        eid = row.get("entity_id")
        if eid is None:
            errors.append(f"entities[{i}] has null entity_id")
        else:
            entity_ids.add(str(eid))

    if entity_ids:
        for table_name, id_field in (
            ("states", "entity_id"),
            ("observations", "entity_id"),
            ("actions", "obj_id"),
            ("actions", "target_obj_id"),
        ):
            for i, row in enumerate(tables.get(table_name, [])):
                val = row.get(id_field)
                if val is not None and str(val) not in entity_ids:
                    errors.append(
                        f"{table_name}[{i}].{id_field}={val} does not reference a known entity_id"
                    )

        for i, segment in enumerate(tables.get("intent_segments", [])):
            if segment.get("subject_scope") == "entity":
                sid = segment.get("subject_id")
                if sid is not None and str(sid) not in entity_ids:
                    errors.append(
                        f"intent_segments[{i}].subject_id={sid} (entity scope) does not reference a known entity_id"
                    )

        for i, row in enumerate(tables.get("events", [])):
            for field in ("subject_id", "target_id"):
                val = row.get(field)
                if val is not None and str(val) not in entity_ids:
                    errors.append(
                        f"events[{i}].{field}={val} does not reference a known entity_id"
                    )

        for i, row in enumerate(tables.get("relations", [])):
            for field in ("subject_id", "object_id"):
                val = row.get(field)
                if val is not None and str(val) not in entity_ids:
                    errors.append(
                        f"relations[{i}].{field}={val} does not reference a known entity_id"
                    )

    return errors
