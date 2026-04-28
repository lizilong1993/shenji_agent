# Iteration 007 Normalized Retrain Report

## Command
`PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages TMPDIR=$PWD/.tmp_claude TEMP=$PWD/.tmp_claude TMP=$PWD/.tmp_claude python3.10 train_models.py --config configs/supervised_score_quality.yaml`

## Output
- Run dir: `experiments/shenji_agent/20260428_235555`
- Published artifacts dir: `ai/weights`
- Manifest source commit: `77849cd38e80fd3281e19e6f0ebfd074ad44fae5`
- Situation normalizer: `situation_feature_normalizer.json`
- Situation normalizer fitted on: `situation_train_split`
- Situation normalizer feature dim: `108`
- Situation zero-std count: `50`
- Situation training applied normalizer: `true`

## Training Metrics
- Intent immediate validation accuracy: `0.6221`
- Tactical validation accuracy: `0.9622`
- Situation validation accuracy: `0.5455`
- Situation samples: `108`
- Situation bucket counts: `{'complex': 108}`
- Tactical samples: `24080`

## Model Quality Before vs After
Before file: `experiments/shenji_agent/model_quality_before_iter007_normalized_retrain.json`

After file: `experiments/shenji_agent/model_quality_after_iter007_normalized_retrain.json`

| Metric | Before | After |
|---|---:|---:|
| normalizer_applied_in_training | `false` | `true` |
| situation_pre_sigmoid_mean | `0.505834` | `0.411455` |
| situation_pre_sigmoid_std | `0.005960` | `0.057212` |
| situation_output_mean | `0.508567` | `0.414247` |
| situation_per_node_variance_mean | `1.5056e-05` | `0.004074` |
| intent_move_bias_estimate | `-0.2365` | `-0.2010` |
| tactical_available | `true` | `true` |
| warning_count | `3` | `1` |

## Interpretation
The normalized retrain fixed the explicit training/inference normalizer mismatch. Situation per-node variance improved by roughly 270x, so the next step is smoke self-play before implementing a GNN architecture variant. Intent move bias remains negative and is still the only model-quality warning.

## Remaining Risk
Supervised diagnostics improved, but this is not promotion evidence. Smoke self-play must show non-0-0 score behavior or stronger candidate/baseline differentiation before formal self-play is justified.
