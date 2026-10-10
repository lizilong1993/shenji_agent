# Shenji Pilot Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Shenji to reproducible MiaoSuan episode collection, derive evidence-safe Grade A intent labels, and produce resumable pilot releases with coverage reports.

**Architecture:** A manifest expands scenarios, seeds, and policy matchups into deterministic jobs. A generic episode runner captures synchronized state views through `EpisodeRecorder`, writes immutable raw artifacts, derives only directly supported intent segments, normalizes each episode, and records resumable status. MiaoSuan imports and NumPy asset loading remain behind runtime adapters so core behavior is testable without the simulator.

**Tech Stack:** Python 3.10+, standard library, existing NumPy and MiaoSuan runtime, unittest

## Global Constraints

- Preserve all pre-existing files outside `Shenji/` and `docs/superpowers/`.
- Keep the dataset name `Shenji`; use `MiaoSuan` only as backend metadata.
- Never infer Grade B or Grade C labels from one action.
- Never create `FEINT` without an explicit plan or human review.
- Use deterministic episode IDs from scenario, seed, and policy identities.
- Raw episodes are immutable after a successful run.
- A failed or interrupted job must remain resumable and must not be marked complete.
- Do not commit from the existing dirty worktree.

---

### Task 1: Pilot job manifest and deterministic expansion

**Files:**
- Create: `Shenji/tests/test_jobs.py`
- Create: `Shenji/src/shenji_dataset/jobs.py`
- Create: `Shenji/configs/pilot_manifest_v1.json`

**Interfaces:**
- Produces: `load_collection_manifest(path: Path) -> dict`
- Produces: `expand_jobs(manifest: dict) -> list[CollectionJob]`
- Produces: `CollectionJob.episode_id: str`

- [ ] Write tests proving matrix expansion, stable ordering, unique IDs, and invalid-policy rejection.
- [ ] Run `PYTHONPATH=Shenji/src python -m unittest Shenji/tests/test_jobs.py -v` and confirm missing-module failure.
- [ ] Implement immutable job objects and strict manifest validation.
- [ ] Re-run the job tests and confirm all pass.

### Task 2: Evidence-safe Grade A intent labeler

**Files:**
- Create: `Shenji/tests/test_labeling.py`
- Create: `Shenji/src/shenji_dataset/labeling.py`

**Interfaces:**
- Produces: `derive_grade_a_intent_plan(raw_episode: dict) -> dict`
- Consumes: mission commands from truth communication and direct MiaoSuan action types.

- [ ] Write tests for ASSAULT, HOLD, RECON, SEIZE, TARGET_GUIDANCE, FIRE_SUPPORT, ASSEMBLE, DISPERSE, and FORTIFY.
- [ ] Write a negative test proving unsupported actions never create Grade B labels or FEINT.
- [ ] Run the tests and confirm missing-module failure.
- [ ] Implement deterministic segment IDs, command de-duplication, and direct-action evidence references.
- [ ] Re-run the labeler tests and confirm all pass.

### Task 3: Resumable episode runner

**Files:**
- Create: `Shenji/tests/test_collection.py`
- Create: `Shenji/src/shenji_dataset/collection.py`
- Create: `Shenji/src/shenji_dataset/runtime.py`

**Interfaces:**
- Produces: `run_collection_job(job, dependencies, output_root, resume=True) -> dict`
- Produces: `load_miaosuan_assets(job, data_root) -> dict`
- Produces: `resolve_miaosuan_runtime(project_root) -> tuple[type, type]`

- [ ] Write a fake-environment test that captures truth, red, blue, and submitted actions.
- [ ] Write tests for successful status, failed status, canonical release creation, and resume behavior.
- [ ] Run the tests and confirm missing-module failure.
- [ ] Implement the generic loop with cleanup in `finally`, atomic status writes, and no mock fallback in the real runtime resolver.
- [ ] Re-run the collection tests and confirm all pass.

### Task 4: Collection and coverage CLI

**Files:**
- Create: `Shenji/tests/test_collect_cli.py`
- Create: `Shenji/src/shenji_dataset/collect_cli.py`
- Create: `Shenji/src/shenji_dataset/coverage.py`
- Modify: `Shenji/pyproject.toml`

**Interfaces:**
- Produces: `shenji-collect probe`, `shenji-collect run`, and `shenji-collect coverage`.

- [ ] Write CLI tests for probe diagnostics, dry-run expansion, one fake run, and coverage JSON.
- [ ] Run the tests and confirm missing behavior.
- [ ] Implement commands, machine-readable diagnostics, and nonzero exit codes on runtime failure.
- [ ] Re-run the CLI tests and confirm all pass.

### Task 5: Pilot documentation and verification

**Files:**
- Create: `Shenji/PILOT_RUNBOOK.md`
- Create: `Shenji/validation/runtime_probe_example.json`
- Modify: `Shenji/README.md`
- Modify: `Shenji/src/shenji_dataset/__init__.py`

**Interfaces:**
- Documents native Linux, WSL, Docker, and dry-run workflows without claiming mock data as research data.

- [ ] Run the complete Shenji unittest suite.
- [ ] Run compileall over source and tests.
- [ ] Re-verify and validate the checked-in canonical example.
- [ ] Run the collection manifest in dry-run mode.
- [ ] Attempt a MiaoSuan runtime probe and save the exact diagnostic.
- [ ] Confirm project-side hashes match the tested mirror and review changes only under approved paths.
