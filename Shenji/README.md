# Shenji

Shenji is a hierarchical multi-intent dataset and benchmark for situation awareness in wargame simulation. MiaoSuan is the initial simulation backend, but it is not the dataset name or a schema layer.

The repository currently contains the complete v1 data contract, collection recorder, normalizer, 22-intent ontology, release validator, benchmark definitions, collection targets, and one executable example release. Large-scale episode generation is the next runtime stage.

## Core properties

1. Full truth, red partial observation, and blue partial observation are synchronized by episode and step.
2. Eight canonical tables describe episodes, entities, states, observations, actions, events, relations, and temporal intent segments.
3. Six parent categories contain 22 fine-grained intents.
4. Commanded intent, realized intent, and human-inferred intent remain separate.
5. Intent is hierarchical, temporal, and multi-label.
6. Every release artifact has a recorded byte count, row count, and SHA-256 hash.
7. JSONL works without external runtime dependencies. Parquet output is optional.

## Install

From the `Shenji` directory:

```powershell
py -3 -m pip install -e .
```

For Parquet releases:

```powershell
py -3 -m pip install -e ".[parquet]"
```

## Build the example

```powershell
shenji-dataset build `
  --raw examples/raw_episode.json `
  --intent-plan examples/intent_plan.json `
  --output canonical/shenji_schema_v1/example

shenji-dataset verify canonical/shenji_schema_v1/example
shenji-dataset validate canonical/shenji_schema_v1/example
```

To produce both JSONL and Parquet, pass `--format jsonl --format parquet` to the build command.

## Capture a MiaoSuan episode

```python
from pathlib import Path
from shenji_dataset import EpisodeRecorder

metadata = {
    "episode_id": "shenji_000001",
    "scenario_id": "1231",
    "map_id": "map_123",
    "seed": 7,
    "policy_red": "red_policy_v1",
    "policy_blue": "blue_policy_v1",
}
recorder = EpisodeRecorder(metadata)
recorder.record(state, actions)
recorder.save(Path("raw/episodes/shenji_000001/raw_episode.json"))
```

The recorder expects the standard environment mapping with truth at `-1`, red at `0`, and blue at `1`.

## Main files

| Path | Purpose |
|---|---|
| `ontology/intent_ontology_v1.json` | Six parent categories and 22 fine intents |
| `schema/shenji_schema_v1.json` | Unified field and table contract |
| `configs/collection_plan_v1.json` | Pilot and full-release balance targets |
| `configs/pilot_manifest_v1.json` | Reproducible scenario, seed, and policy job matrix |
| `COLLECTION_PROTOCOL.md` | Batch simulation and audit procedure |
| `PILOT_RUNBOOK.md` | Runtime probe, dry-run, collection, resume, and coverage commands |
| `benchmarks/tasks_v1.json` | Five benchmark task definitions |
| `splits/split_policy_v1.json` | IID, cross-map, cross-policy, and partial-observation splits |
| `annotations/ANNOTATION_GUIDE.md` | Segment-level annotation rules |
| `DATA_CARD.md` | Scope, provenance, limitations, and release considerations |

## Test

The test suite uses the Python standard library:

```powershell
$env:PYTHONPATH = "src"
py -3 -m unittest discover -s tests -v
```

## Pilot collection

The pilot collector expands the checked-in manifest into four deterministic jobs. Inspect the job matrix without importing the simulator:

```powershell
shenji-collect run --manifest configs/pilot_manifest_v1.json --dry-run
```

Check the real MiaoSuan runtime and all required assets before collection:

```powershell
shenji-collect probe `
  --manifest configs/pilot_manifest_v1.json `
  --output validation/runtime_probe.json
```

Run one real episode first, then inspect coverage:

```powershell
shenji-collect run --manifest configs/pilot_manifest_v1.json --limit 1

shenji-collect coverage `
  --raw-root raw/pilot_v1 `
  --targets configs/collection_plan_v1.json `
  --output validation/pilot_coverage.json
```

Completed jobs are skipped on subsequent runs. Failed jobs retain `status.json` and `partial_raw_episode.json`, then resume safely without overwriting a completed raw episode.

Automatic pilot labeling is intentionally restricted to directly supported Grade A evidence. It covers `ASSAULT`, `HOLD`, `RECON`, `SEIZE`, `TARGET_GUIDANCE`, `FIRE_SUPPORT`, `ASSEMBLE`, `DISPERSE`, and `FORTIFY`. Grade B labels and `FEINT` require the separate rules and audit workflow described in the annotation guide.

## Current boundary

The checked-in example is a schema reference and smoke-test dataset, not the full research corpus. The full corpus requires running the collection protocol against the local simulator and meeting every class-balance and out-of-distribution gate in the collection plan.
