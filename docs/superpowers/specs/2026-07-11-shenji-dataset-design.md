# Shenji Dataset v1 Design

**Date:** 2026-07-11  
**Status:** Approved for implementation  
**Dataset name:** Shenji  
**Simulation backend:** MiaoSuan

## 1. Goal

Shenji v1 is a public, event-centric wargame situation-awareness dataset generated from one simulation source. It preserves full truth, red-side partial observations, blue-side partial observations, actions, derived events, and hierarchical multi-label intent segments in one unified schema.

The first implementation must provide a reproducible data contract and construction pipeline before large-scale simulation collection begins.

## 2. Design principles

1. Dataset identity and simulator identity are separated. The dataset is always named Shenji.
2. IDs and map cells are categorical values. They are never perturbed or numerically normalized.
3. Missing values remain null and are accompanied by observation provenance, rather than being filled with fabricated values.
4. Intent is segment-level and multi-label, not an independent single class at each frame.
5. Commanded intent and realized intent are stored separately. Human-inferred intent is supported for audited subsets.
6. Raw records are immutable. Canonical tables are reproducibly derived from raw records.
7. Every release includes a machine-readable manifest with row counts and SHA-256 hashes.

## 3. Intent ontology

The ontology contains six parent categories and 22 fine-grained intents.

| Parent category | Fine-grained intents |
|---|---|
| OFFENSIVE_ACTION | ASSAULT, BREAKTHROUGH, FLANK, ENCIRCLEMENT, PURSUIT, AMBUSH |
| AREA_CONTROL | SEIZE, HOLD, INTERDICT |
| RECON_INFORMATION | RECON, SURVEILLANCE, TARGET_GUIDANCE |
| SUPPORT_COORDINATION | REINFORCE, ESCORT, FIRE_SUPPORT |
| MANEUVER_POSTURE | REDEPLOY, ASSEMBLE, DISPERSE, FORTIFY |
| DISENGAGEMENT_DECEPTION | RETREAT, EVADE, FEINT |

Each intent entry defines operational criteria, positive evidence, exclusion rules, an automatic-label grade, and whether an explicit scenario plan tag is required.

## 4. Label model

Each intent segment contains:

- label kind: commanded, realized, or human_inferred
- subject scope: entity, group, company, or side
- temporal extent: start_step and end_step
- one primary intent and zero or more secondary intents
- target type and target ID
- execution status and outcome
- evidence source, evidence grade, evidence references, and confidence

Fine intents are not mutually exclusive. For example, SEIZE may be primary while FLANK and FIRE_SUPPORT are secondary.

## 5. Data layers

### Raw episode

A raw episode contains metadata, truth observations, red observations, blue observations, and submitted actions. An optional intent plan records hidden task intent assigned before execution.

### Canonical tables

The v1 schema produces:

- episodes
- entities
- states
- observations
- actions
- events
- relations
- intent_segments

JSONL is the dependency-free exchange format. Parquet is the publication format when the optional pyarrow dependency is available.

### Release structure

```text
Shenji/
├── raw/
├── canonical/shenji_schema_v1/
├── ontology/
├── schema/
├── annotations/
├── benchmarks/
├── splits/
├── validation/
├── examples/
├── src/shenji_dataset/
└── tests/
```

## 6. MiaoSuan mapping

MiaoSuan action types map into canonical action names without changing their IDs. Examples include movement, shooting, occupation, state changes, indirect fire, guided fire, aggregation, disaggregation, altitude change, correction radar, fortification entry, and task commands.

Action type 200 supplies coarse commanded mission anchors. Fine intent labels come from explicit hidden intent plans, deterministic event rules, or human review. FEINT must have an explicit plan tag or human verification.

The recorder consumes the environment state mapping directly:

- key -1 or truth stores full truth
- key 0 or red stores red observation
- key 1 or blue stores blue observation

## 7. Validation gates

A valid release must satisfy all of the following:

1. Exactly six parent categories and 22 unique fine intent codes.
2. Every intent segment references an ontology code and its correct parent.
3. Segment steps are ordered and confidence lies in the closed interval from 0 to 1.
4. Primary intent is not repeated in secondary intents.
5. All canonical rows reference a known episode.
6. Every release artifact listed in the manifest matches its SHA-256 hash.
7. Train, validation, and test splits are episode-disjoint.

## 8. First implementation boundary

This implementation creates the complete data contract, recorder, normalizer, validator, writer, command-line interface, example episode, example canonical release, benchmark definitions, and test suite. Large-scale episode generation remains a separate execution stage because it requires the local simulator runtime and scenario batch policy.
