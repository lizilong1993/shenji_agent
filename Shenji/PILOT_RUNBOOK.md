# Shenji pilot collection runbook

## 1. Runtime requirement

The supplied simulator wheel is tagged for CPython 3.10 on manylinux. Real MiaoSuan collection therefore requires Linux, WSL with a Linux distribution, or a Linux container. Native Windows Python cannot load that wheel.

The collector never falls back to `MockTrainEnv` during a real run. A missing simulator produces a nonzero probe or run result and no research episode is marked complete.

## 2. Install

From the project root on Linux or WSL:

```bash
cd Shenji
python3.10 -m pip install -e .
python3.10 -m pip install numpy
python3.10 -m pip install ../land_wargame_sdk/land_wargame_train_env-4.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

The existing agent dependencies must also be installed from the project-level requirements file.

## 3. Inspect the job matrix

Dry-run expansion does not import MiaoSuan and does not write episode data:

```bash
shenji-collect run \
  --manifest configs/pilot_manifest_v1.json \
  --dry-run
```

The checked-in manifest expands to four jobs using scenario `1231`, two seeds, and two policy matchups. Episode IDs are deterministic and independent of manifest list ordering.

## 4. Probe the real runtime

```bash
shenji-collect probe \
  --manifest configs/pilot_manifest_v1.json \
  --project-root .. \
  --output validation/runtime_probe.json
```

The probe checks the real `TrainEnv` import, agent imports, scenario JSON, map JSON, movement-cost pickle, and visibility NPZ. A ready result has `"ready": true` and exits with code 0.

## 5. Run one episode

```bash
shenji-collect run \
  --manifest configs/pilot_manifest_v1.json \
  --project-root .. \
  --limit 1
```

Each completed episode creates:

```text
raw/pilot_v1/<episode_id>/
├── job.json
├── status.json
├── raw_episode.json
└── intent_plan.json

canonical/shenji_schema_v1/pilot_v1/<episode_id>/
├── episodes.jsonl
├── entities.jsonl
├── states.jsonl
├── observations.jsonl
├── actions.jsonl
├── events.jsonl
├── relations.jsonl
├── intent_segments.jsonl
└── manifest.json
```

The raw episode synchronizes full truth, red partial observation, blue partial observation, and submitted actions. Grade A realized intent is derived only from accepted actions recorded in the truth stream.

## 6. Resume and failure handling

Run the same command again to resume. Completed jobs are skipped before constructing the simulator. A failed job contains `status.json` with the exception type and message. If capture had already begun, `partial_raw_episode.json` is retained for diagnosis but is not treated as research data.

A complete `raw_episode.json` is immutable. If canonical conversion fails after capture, the next run reuses that raw episode and rebuilds labels and canonical tables without rerunning the simulation.

## 7. Coverage report

```bash
shenji-collect coverage \
  --raw-root raw/pilot_v1 \
  --targets configs/collection_plan_v1.json \
  --output validation/pilot_coverage.json
```

The report includes completed and failed jobs, counts by intent, label kind, evidence grade, outcome, scenario, and policy, plus the shortfall against pilot targets.

## 8. Promotion gate

Do not begin the full corpus until the four-job smoke batch satisfies all of the following:

1. Every job has synchronized truth, red, and blue streams.
2. Every canonical release passes integrity and semantic validation.
3. Submitted actions and accepted truth actions are distinguishable.
4. Repeating the collection command skips completed jobs.
5. Coverage statistics match manual inspection.
6. No mock episode appears under `raw/pilot_v1`.

After this gate, extend the manifest with additional maps, scenario variants, policy implementations, and intent-conditioned controllers.
