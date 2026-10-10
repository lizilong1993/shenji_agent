# Shenji dataset card — v2 supplement

This supplement documents changes and additions for the Shenji v2 ontology, benchmark tasks, annotation protocol, collection plan, and split policy. It builds upon the v1 DATA_CARD.md.

## v2 ontology changes

- **File**: `ontology/intent_ontology_v2.json`
- **Status layers added**: BACKGROUND, UNKNOWN, AMBIGUOUS, OUT_OF_ONTOLOGY (independent from tactical hierarchy)
- **Per-intent annotation criteria**: necessary_conditions, sufficient_evidence, counterexamples, hard_negative, allowed_cooccurrence, conflicts_with, primary_vs_secondary, minimum_duration, boundary_rules, applicable_subject_scope
- **Confusion matrix**: Explicit discrimination rules for 6 confusion pairs (ASSAULT/SEIZE, HOLD/FORTIFY, RECON/SURVEILLANCE, RETREAT/EVADE, FLANK/BREAKTHROUGH, FIRE_SUPPORT/primary task)
- **FEINT**: remains Grade C with requires_plan_tag=true; failed attack must NOT be labeled FEINT

## v2 benchmark tasks

- **File**: `benchmarks/tasks_v2.json`
- Intent recognition split into 4 subtasks:
  - `commanded_intent_inference` — infer hidden command from partial observations only
  - `realized_activity_recognition` — recognize executed tactical activities
  - `intent_onset_forecasting` — predict intent before clear behavioral evidence
  - `command_realization_mismatch` — predict success/failure/switch/deviation
- New tracks:
  - `human_inferred_intent` — evaluate against human annotator labels
  - `open_set_intent` — allow abstention, handle UNKNOWN/AMBIGUOUS/BACKGROUND/OUT_OF_ONTOLOGY
- Every track specifies: observation_cutoff, allowed_fields, denied_fields, target_label_kind, prediction_subject, truth_usage_boundary

## v2 input contract

- **File**: `benchmarks/input_contract_v2.json`
- Field-level matrix: each field tagged with `label_evidence`, `model_input` (per-track), `target`, `available_time`
- Critical restrictions documented:
  - `mission_type` in action payload: NEVER available as model input
  - Type-200 actions: excluded from commanded_intent_inference
  - Action_type_id (5/8/9/14/15/18/314/315): excluded from commanded_intent_inference
  - Truth observations: never available to any model
  - Human-inferred track: only replay-visible data

## v2 annotation protocol

- **Files**: `annotation/annotation_protocol_v1.md`, `annotation/adjudication_schema_v1.json`
- Grade A: automatic deterministic rules (unchanged)
- Grade B: automated detection + human review (min 2 annotators)
- Grade C (FEINT): plan tag + human review
- Quality metrics: Krippendorff's alpha, boundary IoU, Jaccard similarity, ECE
- Hard negative construction for 6 confusion pairs
- Human-inferred benchmark: replay-visible data only, no truth

## v2 collection plan

- **File**: `configs/collection_plan_v2.json`
- Controllers per class: minimum 5, target 5-8 (was 3)
- Single controller cap: 20-25% (was 50%)
- Collection targets in effective samples (episodes, scenario families, controller families, map families) vs segment counts alone
- BACKGROUND/UNKNOWN quotas added
- FEINT quota separated (80 pilot, 800 v1; requires 5+ episodes minimum)
- Four phases: pilot → directed_expansion → ood_holdout → human_audit

## v2 split policy

- **File**: `splits/split_policy_v2.json`
- Five tracks: IID, cross_map, cross_policy, cross_scenario_family, visibility_stratified
- Cross-policy isolates by implementation family (not policy code alone)
- Automatic zero-leakage verification: scenario_family, controller_family, participant_pair, seed_family must not cross splits
- OOD test sets sealed and hashed before baseline training

## Baseline evaluation

- **Location**: `benchmarks/baseline/`
- Minimum gradient:
  - Level 0: Frequency / Random
  - Level 1: Label-rule oracle
  - Level 2: Metadata-only (map/policy ID)
  - Level 3: Kinematic + XGBoost
  - Level 4+: LSTM/TCN/Transformer (use existing training pipeline where available)
- Metrics: segment mAP@0.3/0.5/0.7 IoU, hierarchical F1, early-recognition curve, ECE, AUROC, bootstrap 95% CI
- Probes: metadata-only, controller-identification, label-rule oracle

## Current status (v2)

- [x] v2 ontology (intent_ontology_v2.json)
- [x] v2 benchmark tasks (tasks_v2.json)
- [x] v2 input contract (input_contract_v2.json)
- [x] v2 annotation protocol (annotation_protocol_v1.md + adjudication_schema_v1.json)
- [x] v2 collection plan (collection_plan_v2.json)
- [x] v2 split policy (split_policy_v2.json)
- [ ] Pilot corpus for all 22 classes: NOT YET GENERATED
- [ ] Baseline results: NOT YET RUN
- [ ] Grade B/C labelers: NOT YET IMPLEMENTED
- [ ] Human annotation sessions: NOT YET CONDUCTED
- [ ] Domain expert ontology review: NOT YET CONDUCTED
- [ ] License selection: NOT YET COMPLETED
- [ ] DOI / persistent download: NOT YET REGISTERED

## v2 release checklist

- [ ] All 22 intent classes have at least 3 pilot episodes with 2+ map families
- [ ] Grade B/C labelers produce deterministic first-pass labels meeting quality thresholds
- [ ] FEINT episodes have explicit plan tags and human review
- [ ] Human-inferred benchmark has 1000+ annotated segments from 2+ annotators
- [ ] Inter-annotator agreement Krippendorff's alpha >= 0.67 for Grade B
- [ ] Split zero-leakage verification passes for all 5 tracks
- [ ] Metadata-only baseline does not exceed chance + 5% on any task
- [ ] Controller-identification probe F1 < 0.3
- [ ] Baseline results for at least 3 model levels on IID + one OOD track
- [ ] Data license selected, code license selected, redistribution rights documented
- [ ] Persistent DOI registered
- [ ] Release manifest with SHA-256, row counts, version numbers
- [ ] Validator and split generator are version-locked and runnable from clean environment
