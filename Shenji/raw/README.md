# Raw Shenji episodes

Each episode is stored under `raw/episodes/<episode_id>/`. The required document is `raw_episode.json`, containing metadata, truth observations, red observations, blue observations, and submitted actions. An optional `intent_plan.json` stores hidden commanded intent and must never be exposed as a benchmark input.

Raw records are immutable after collection. Corrections are represented by a new release-level transformation or annotation revision, not by silently modifying the captured episode.

Large release files are intentionally not included in source control. The checked-in example under `examples/` is the executable format reference.
