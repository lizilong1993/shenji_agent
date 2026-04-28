# Shenji Supervised Training Summary

- run_dir: `/remote-home/lizilong/shenji_agent_fresh/experiments/shenji_agent/20260428_235555`
- intent_immediate_accuracy: `0.6221`
- tactical_accuracy: `0.9622`
- situation_accuracy: `0.5455`
- complex_bucket_intent_accuracy: `0.0000`
- medium_bucket_intent_accuracy: `0.6221`
- simple_bucket_intent_accuracy: `0.0000`
- intent_bucket_counts_train: `{'medium': 4096}`
- intent_bucket_counts_val: `{'medium': 1024}`
- situation_bucket_counts: `{'complex': 108}`
- situation_normalizer: `{'path': 'situation_feature_normalizer.json', 'kind': 'per_dimension_zscore', 'fitted_on': 'situation_train_split', 'feature_dim': 108, 'num_samples': 1032, 'zero_std_count': 50, 'training_applied_normalizer': True}`