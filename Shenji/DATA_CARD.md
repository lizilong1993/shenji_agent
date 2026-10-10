# Shenji dataset card

## Summary

Shenji is designed for state estimation, trajectory prediction, tactical activity recognition, hierarchical multi-label intent recognition, and intent-transition forecasting under partial observation. Version 1 uses MiaoSuan as its data-generation backend.

## Data composition

Each episode may contain full simulator truth, red partial observations, blue partial observations, submitted actions, adjudication events, entity relations, and temporal intent segments. The canonical representation contains eight tables linked by `episode_id`, step, and categorical entity identifiers.

## Intent labels

The ontology defines 22 fine intents under six parent categories. Labels are temporal segments. One primary intent and multiple secondary intents may coexist. Commanded, realized, and human-inferred labels are separate label kinds.

## Collection

Episodes are generated through scenario variation, stochastic seeds, multiple controller implementations, self-play policies, and later human replays. Hidden intent plans are saved as ground truth but are excluded from model inputs. Full collection targets are defined in `configs/collection_plan_v1.json`.

## Recommended uses

1. Partial-observation state estimation.
2. Entity and group trajectory prediction.
3. Tactical activity and event recognition.
4. Hierarchical multi-label intent recognition.
5. Intent transition and onset forecasting.
6. Cross-map and cross-policy generalization studies.

## Uses requiring caution

Results from the checked-in example must not be reported as benchmark performance. Models must not receive truth observations, hidden plans, future actions, or future intent labels when those fields belong to the prediction target. Random frame splits are prohibited because they leak episode and controller information.

## Limitations

The first source domain is synthetic and simulator-specific. Some fine labels depend on explicit scenario plans or deterministic operational definitions. Grade B labels require sampled human audit. FEINT cannot be inferred reliably from behavior alone. Human replay data may introduce policy and skill imbalance.

## Privacy and sensitivity

Synthetic episodes should not contain personal data. Human replay metadata must be anonymized before release. Scenario files, maps, and task descriptions require a separate sensitivity review before public upload, even when the recorded trajectories are synthetic.

## Licensing

The public data license and code license are intentionally not selected by this implementation. The project owner must choose them before public publication after confirming simulator asset redistribution rights and institutional policy.

## Maintenance

Ontology and schema versions are immutable after release. Corrections require a new version and a changelog. Every release must publish a manifest with row counts and SHA-256 hashes, plus a validation report and split-generation record.

## Current status

The v1 construction framework and executable reference episode are present. The large-scale corpus has not yet been generated and must satisfy the pilot, balance, out-of-distribution, and human-audit gates before being presented as Shenji v1 research data.
