# Canonical Shenji schema v1 releases

Each release directory contains eight canonical tables and `manifest.json`. JSONL is always supported. Parquet with zstd compression is recommended for public releases.

The eight tables are `episodes`, `entities`, `states`, `observations`, `actions`, `events`, `relations`, and `intent_segments`.

Run the following commands from the `Shenji` directory after installing the package:

```powershell
shenji-dataset verify canonical/shenji_schema_v1/example
shenji-dataset validate canonical/shenji_schema_v1/example
```

The `example/` release is generated from `examples/raw_episode.json` and `examples/intent_plan.json`.
