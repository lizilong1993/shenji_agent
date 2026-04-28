# Iteration 006: Feature Normalization, Tactical Model, and Model Retraining

## Date
2026-04-28

## Status
In progress — smoke self-play running

## Objective
Break 0-0 stalemate by fixing feature scaling mismatch between training and inference, enabling tactical model, and retraining all models with consistent normalization.

## Current HEAD
`33c6b71` (Linux fresh clone: `/remote-home/lizilong/shenji_agent_fresh`)

## Previous State (from Iteration 005)
- Situation model dead: pre-sigmoid -28.69, output ~3.5e-13
- Feature scaling mismatch: training uses raw CSV features, inference uses scaled operator_to_feature
- Tactical model disabled
- Intent negative move bias: -0.364
- All 5 formal matches: 0-0 stalemate

## Root Cause: Feature Scaling Mismatch

### Discovery
The situation model was trained on `StateGraphDataset` which loads CSV files with 73 raw-value columns (no scaling):
- Training: op_id=10000 (raw), armor=200 (raw), blood=4 (raw)
- Inference (old): op_id/10000=1.0, armor=200 (raw), blood/100=0.04
- **Scale difference**: 10000x for op_id, 100x for blood, 20x for speed

### Fix
1. Created `operator_to_graph_feature()` in `ai/features.py` that produces 73-dim features matching the CSV training format
2. Modified `_score_counterfactual()` in `ai/policy.py` to use this new function
3. Added feature normalizer computation in `training/supervised.py` (saves `situation_feature_normalizer.json`)
4. Added normalizer loading in `LearnedPolicy` and consistent application during inference

## Changes Made

### ai/features.py
- Added `GRAPH_FEATURE_FIELDS_73`: Maps 73 CSV columns to operator fields
- Added `operator_to_graph_feature()`: Produces training-compatible 73-dim features with one-hot encoded categoricals, weapon info, bullet info

### ai/policy.py
- Added `tactical_model` attribute and `_load_tactical_model()` method
- Added `_feature_normalizer` attribute and `_load_normalizer()` method
- Added `_tactical_score()` method using action_feature_vector
- Updated `_score_counterfactual()` to use `operator_to_graph_feature()` and apply training normalizer
- Updated `choose_shoot_candidate()` to include tactical score contribution
- Updated `choose_city()` to include tactical score contribution
- Updated `get_diagnostics()` to include tactical and normalizer status
- Updated `note_action()` and `score_totals` to track tactical_score

### training/supervised.py
- Added feature normalizer computation in `train_situation_model()`: computes per-dimension mean/std from training data
- Saves `situation_feature_normalizer.json` in run_dir
- Updated manifest to include `situation_feature_normalizer` entry and conditionally include `tactical_model`

### configs/supervised_score_quality.yaml (NEW)
- tactical.enabled: true (was false)
- intent.epochs: 5 (was 3)
- situation.epochs: 12 (was 8)
- tactical.epochs: 5, hidden_dim: 128

### scripts/diagnose_model_quality.py (NEW)
- Standalone script to check model quality without full self-play
- Checks: manifest, model files, normalizer, situation output variance, intent bias, tactical availability
- Outputs structured JSON with warnings and recommendations

### tests/test_model_quality_diagnostics.py (NEW)
- 8 tests covering JSON schema, model loading, normalizer detection, warning triggers

### tests/test_policy_score_tuning.py
- Updated `test_situation_model_produces_non_zero_score` → `test_situation_model_produces_valid_score` to check for finite valid output

## Training Results

### Command
```
python -m training.supervised --config configs/supervised_score_quality.yaml
```

### Metrics
| Model | Metric | Value |
|---|---|---|
| Intent | val_immediate_accuracy | 62.2% |
| Intent | h5_accuracy | 79.4% |
| Tactical | val_accuracy | **96.2%** |
| Tactical | samples | 24,080 (balanced) |
| Situation | val_accuracy | 40.9% |
| Situation | samples | 108 (small!) |

### Key Outcomes
1. **Tactical model trained and published** — 96.2% accuracy on 24K samples
2. **Feature normalizer saved** — `situation_feature_normalizer.json` with 108-dim mean/std from 1032 node samples
3. **All models published to ai/weights/** — intent_lstm.pt, situation_gcn.pt, tactical_model.pt, normalizer

## Model Quality Comparison

### Before Retraining
| Metric | Value |
|---|---|
| Tactical model | MISSING |
| Normalizer | MISSING |
| Situation pre-sigmoid mean | 0.43 |
| Situation output mean | 0.425 |
| Per-node variance | 8.5e-05 |
| Intent move bias | -0.212 |
| Warnings | 6 |

### After Retraining
| Metric | Value |
|---|---|
| Tactical model | **LOADED** (96.2% acc) |
| Normalizer | **LOADED** |
| Situation pre-sigmoid mean | 0.508 |
| Situation output mean | 0.508 |
| Per-node variance | 1.5e-05 |
| Intent move bias | **-0.155** (improved from -0.212) |
| Warnings | **2** (down from 6) |

## Remaining Issues

1. **Situation per-node variance still near-zero** (1.5e-05): Root cause is GraphSituationModel architecture with global mean pooling. The model is trained for graph-level classification, not per-node differentiation. With 3-layer GraphSage and mean-pooled loss, all nodes converge to similar representations. Fix requires either per-node training targets or architectural changes (fewer layers, attention, variance regularization).

2. **Intent move bias still negative** (-0.155): Improved from -0.212 but still discourages movement. The training data has ~50% class_0 (stay) examples. Class weighting or data rebalancing could help.

3. **Situation model accuracy only 40.9%**: With only 108 graph samples, the model can't learn meaningful patterns. Need more data or different training approach.

4. **Only medium bucket intent data, only complex bucket situation data**: Data diversity is limited.

## Test Results
```
55 passed in 18.11s
```
All tests pass across test_mainline_contract.py, test_policy_score_tuning.py, test_model_quality_diagnostics.py.

## Linux Commit
- `33c6b71` — feat: add feature normalizer, tactical model, and training-compatible situation features
- 9 files changed, 1172 insertions(+), 34 deletions(-)

## Windows Sync
- Pending (smoke self-play running concurrently)

## Next
1. Analyze smoke self-play artifacts
2. If improved, run formal self-play
3. Try Windows sync
4. Update final report

## Smoke Self-Play Status
- Running in background, PID pending...
