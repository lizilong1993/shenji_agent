# Shenji v1 collection protocol

## Purpose

This protocol converts MiaoSuan simulation runs into auditable Shenji episodes. The collection process must preserve the hidden task plan, actual execution, and each side's partial observations without leaking hidden fields into benchmark inputs.

## Before a run

1. Assign a globally unique `episode_id`.
2. Record base scenario, scenario variant, map family, random seed, red policy, and blue policy.
3. Generate `intent_plan.json` before execution. Each planned segment identifies subject, target, time range, success condition, trigger condition, and fine-grained intent.
4. Select policy implementation independently from intent. Each intent must have at least three controller variants.
5. Reserve map and policy families for the intended split before starting large-scale generation.

## During a run

Create one `EpisodeRecorder` per episode. After environment setup and after every environment step, pass the complete state mapping and submitted actions to the recorder.

```python
from pathlib import Path
from shenji_dataset import EpisodeRecorder

recorder = EpisodeRecorder(metadata)
recorder.record(state, actions=[])

while not done:
    actions = red.step(state[0]) + blue.step(state[1])
    state, done = env.step(actions)
    recorder.record(state, actions)

recorder.save(Path("raw/episodes") / metadata["episode_id"] / "raw_episode.json")
```

The environment state must include full truth at key `-1`, red observation at key `0`, and blue observation at key `1`. Do not reconstruct missing views after the run.

## After a run

1. Normalize the raw episode into all eight canonical tables.
2. Preserve commanded segments from the hidden plan.
3. Derive realized segments from actions, events, relations, trajectories, and objective changes.
4. Mark planned tasks as achieved, failed, partial, not triggered, aborted, or switched.
5. Run release validation before adding the episode to a split.
6. Update class, controller, map, policy, outcome, and duration coverage statistics.

## Controller fingerprint control

Do not bind one fixed script to one intent. Randomize route family, timing, formation, unit composition, objective position, trigger thresholds, and opponent response. The cross-policy test split must contain policy implementations not observed during training.

## Difficult labels

`FEINT` must come from an explicit hidden plan or human review. `BREAKTHROUGH`, `ENCIRCLEMENT`, `INTERDICT`, `ESCORT`, `RETREAT`, and `EVADE` require multi-event temporal rules and sampled human audit. A single movement vector is insufficient.

## Pilot acceptance

The pilot ends only after all 22 intents reach the minimum targets in `configs/collection_plan_v1.json`. At least 20 percent of segments must represent failure, abortion, or intent switching. No single controller may contribute more than half of the examples for one intent.
