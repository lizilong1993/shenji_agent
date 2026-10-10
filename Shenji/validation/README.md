# Shenji validation

A publishable release must pass three independent gates.

## Integrity gate

`shenji-dataset verify <release>` checks artifact presence, byte counts, JSONL or Parquet row counts, and SHA-256 hashes against `manifest.json`.

## Semantic gate

`shenji-dataset validate <release>` checks the ontology, eight-table contract, episode references, segment timing, parent-child intent consistency, multi-label uniqueness, confidence bounds, and FEINT evidence restrictions.

## Split gate

Before publication, verify episode-disjoint splits, scenario-family grouping, map holdouts, policy holdouts, and label balance. Store the resulting statistics and the exact split-generation configuration beside the public release.

## Required release report

The validation report should contain schema version, ontology version, episode and row counts, per-intent segment counts, duration distributions, status and outcome distributions, controller contribution by intent, map and policy coverage, label-audit sample sizes, agreement statistics, and all validation warnings.
