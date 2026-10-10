# Shenji Dataset v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, reproducible Shenji v1 dataset construction package inside the existing shenji_agent repository.

**Architecture:** A standalone `Shenji/` Python package records raw MiaoSuan episodes, normalizes them into eight canonical tables, validates hierarchical intent labels, and writes hashed release manifests. The implementation uses only the Python standard library by default, with optional pyarrow output for Parquet releases.

**Tech Stack:** Python 3.10+, pytest, JSON, JSONL, JSON Schema, optional pyarrow

## Global Constraints

- Preserve every pre-existing modified and untracked file.
- Add implementation files only under `Shenji/` and planning files under `docs/superpowers/`.
- Dataset name is `Shenji`; MiaoSuan appears only as `simulation_backend`.
- The ontology contains exactly six parents and 22 fine-grained intents.
- IDs and map-cell values remain categorical strings and are never perturbed.
- Use test-first implementation for Python behavior.
- Do not commit from this dirty worktree.

---

### Task 1: Intent ontology and validation contract

**Files:**
- Create: `Shenji/ontology/intent_ontology_v1.json`
- Create: `Shenji/tests/test_ontology.py`
- Create: `Shenji/src/shenji_dataset/ontology.py`
- Create: `Shenji/src/shenji_dataset/validation.py`

**Interfaces:**
- Produces: `load_ontology(path: Path | None = None) -> dict`
- Produces: `validate_ontology(ontology: dict) -> list[str]`
- Produces: `validate_intent_segment(segment: dict, ontology: dict) -> list[str]`

- [ ] **Step 1: Write the failing ontology tests**

```python
def test_ontology_has_six_parents_and_twenty_two_intents():
    ontology = load_ontology()
    assert len(ontology["parent_categories"]) == 6
    assert len(ontology["intents"]) == 22

def test_feint_requires_explicit_plan_or_human_review():
    feint = next(item for item in load_ontology()["intents"] if item["code"] == "FEINT")
    assert feint["requires_plan_tag"] is True
```

- [ ] **Step 2: Run the tests and confirm import failure**

Run: `py -3 -m pytest Shenji/tests/test_ontology.py -q`  
Expected: FAIL because `shenji_dataset` does not exist.

- [ ] **Step 3: Add the ontology and minimal validators**

```python
def validate_ontology(ontology: dict) -> list[str]:
    errors = []
    if len(ontology.get("parent_categories", [])) != 6:
        errors.append("ontology must contain exactly 6 parent categories")
    if len(ontology.get("intents", [])) != 22:
        errors.append("ontology must contain exactly 22 fine intents")
    return errors
```

- [ ] **Step 4: Run the ontology tests**

Run: `py -3 -m pytest Shenji/tests/test_ontology.py -q`  
Expected: all ontology tests pass.

### Task 2: Raw recorder and canonical normalizer

**Files:**
- Create: `Shenji/tests/test_recorder.py`
- Create: `Shenji/tests/test_normalizer.py`
- Create: `Shenji/src/shenji_dataset/recorder.py`
- Create: `Shenji/src/shenji_dataset/normalizer.py`

**Interfaces:**
- Produces: `EpisodeRecorder.record(state_by_side: dict, actions: list[dict] | None = None) -> None`
- Produces: `EpisodeRecorder.to_raw_episode() -> dict`
- Produces: `normalize_episode(raw_episode: dict, intent_plan: dict | None = None) -> dict[str, list[dict]]`

- [ ] **Step 1: Write recorder tests for truth, red, blue, and action capture**

```python
recorder.record({-1: truth, 0: red, 1: blue}, [{"type": 1, "obj_id": 7}])
raw = recorder.to_raw_episode()
assert raw["observations"]["truth"] == [truth]
assert raw["observations"]["red"] == [red]
assert raw["observations"]["blue"] == [blue]
```

- [ ] **Step 2: Run the recorder test and confirm the missing-module failure**

Run: `py -3 -m pytest Shenji/tests/test_recorder.py -q`  
Expected: FAIL because `recorder.py` does not exist.

- [ ] **Step 3: Implement the dependency-free recorder**

```python
def record(self, state_by_side, actions=None):
    self.observations["truth"].append(state_by_side.get(-1, state_by_side.get("truth")))
    self.observations["red"].append(state_by_side.get(0, state_by_side.get("red")))
    self.observations["blue"].append(state_by_side.get(1, state_by_side.get("blue")))
```

- [ ] **Step 4: Write failing normalizer tests for categorical IDs and eight tables**

```python
tables = normalize_episode(raw_episode, intent_plan)
assert set(tables) == {
    "episodes", "entities", "states", "observations",
    "actions", "events", "relations", "intent_segments",
}
assert tables["states"][0]["entity_id"] == "1001"
assert tables["states"][0]["hex_id"] == "501"
```

- [ ] **Step 5: Implement canonical normalization and MiaoSuan action mapping**

```python
ACTION_NAMES = {1: "MOVE", 2: "SHOOT", 5: "SEIZE", 8: "INDIRECT_FIRE", 9: "GUIDED_FIRE"}
```

- [ ] **Step 6: Run recorder and normalizer tests**

Run: `py -3 -m pytest Shenji/tests/test_recorder.py Shenji/tests/test_normalizer.py -q`  
Expected: all tests pass.

### Task 3: Release writer and command-line interface

**Files:**
- Create: `Shenji/tests/test_release.py`
- Create: `Shenji/src/shenji_dataset/release.py`
- Create: `Shenji/src/shenji_dataset/cli.py`
- Create: `Shenji/pyproject.toml`
- Create: `Shenji/src/shenji_dataset/__init__.py`

**Interfaces:**
- Produces: `write_release(tables: dict[str, list[dict]], output_dir: Path, formats: tuple[str, ...]) -> dict`
- Produces: `verify_release(output_dir: Path) -> list[str]`
- Produces: CLI subcommands `build`, `validate`, and `verify`

- [ ] **Step 1: Write a failing release integrity test**

```python
manifest = write_release(tables, tmp_path, formats=("jsonl",))
assert manifest["dataset"] == "Shenji"
assert verify_release(tmp_path) == []
```

- [ ] **Step 2: Run the release test and confirm failure**

Run: `py -3 -m pytest Shenji/tests/test_release.py -q`  
Expected: FAIL because `release.py` does not exist.

- [ ] **Step 3: Implement deterministic JSONL writing and SHA-256 manifests**

```python
payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
```

- [ ] **Step 4: Add optional Parquet output with a clear missing-dependency error**

```python
try:
    import pyarrow
except ImportError as exc:
    raise RuntimeError("Parquet output requires the 'parquet' extra") from exc
```

- [ ] **Step 5: Run release tests**

Run: `py -3 -m pytest Shenji/tests/test_release.py -q`  
Expected: all release tests pass.

### Task 4: Schemas, example episode, and benchmark metadata

**Files:**
- Create: `Shenji/schema/shenji_schema_v1.json`
- Create: `Shenji/examples/raw_episode.json`
- Create: `Shenji/examples/intent_plan.json`
- Create: `Shenji/benchmarks/tasks_v1.json`
- Create: `Shenji/splits/split_policy_v1.json`
- Create: `Shenji/annotations/ANNOTATION_GUIDE.md`
- Create: `Shenji/raw/README.md`
- Create: `Shenji/canonical/shenji_schema_v1/README.md`

**Interfaces:**
- Consumes: `normalize_episode`, `write_release`, and the intent ontology.
- Produces: one reproducible example release under `Shenji/canonical/shenji_schema_v1/example/`.

- [ ] **Step 1: Add a minimal episode containing truth, red, blue, actions, and commanded plus realized intents**

```json
{
  "metadata": {
    "episode_id": "shenji_example_001",
    "simulation_backend": "MiaoSuan"
  },
  "observations": {"truth": [], "red": [], "blue": []},
  "actions": []
}
```

- [ ] **Step 2: Build the example release**

Run: `py -3 -m shenji_dataset.cli build --raw examples/raw_episode.json --intent-plan examples/intent_plan.json --output canonical/shenji_schema_v1/example`  
Expected: eight JSONL files and `manifest.json`.

- [ ] **Step 3: Verify the generated example release**

Run: `py -3 -m shenji_dataset.cli verify canonical/shenji_schema_v1/example`  
Expected: `release verification passed`.

### Task 5: Documentation and full verification

**Files:**
- Create: `Shenji/README.md`
- Create: `Shenji/DATA_CARD.md`
- Create: `Shenji/VERSION`
- Create: `Shenji/validation/README.md`

**Interfaces:**
- Documents installation, simulator integration, build commands, schema tables, label semantics, limitations, and the next large-scale collection stage.

- [ ] **Step 1: Document the recorder integration point**

```python
recorder.record(state, actions)
recorder.save(Path("Shenji/raw/episodes/shenji_000001/raw_episode.json"))
```

- [ ] **Step 2: Run the complete isolated test suite**

Run: `py -3 -m pytest Shenji/tests -q`  
Expected: all tests pass.

- [ ] **Step 3: Validate and verify the checked-in example**

Run: `py -3 -m shenji_dataset.cli validate canonical/shenji_schema_v1/example`  
Expected: `dataset validation passed`.

- [ ] **Step 4: Review only newly created paths**

Run: `git status --short -- Shenji docs/superpowers`  
Expected: only the Shenji implementation and its approved design and plan are listed.
