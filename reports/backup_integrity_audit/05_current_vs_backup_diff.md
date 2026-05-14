# Current vs Backup Diff

- Candidate: `/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`
- Current manifest entries: 1177
- Backup manifest entries: 3179
- Files in backup not current: 3004
- Files in current not backup: 1002
- Important files in backup not current: 53

## Important backup-only files (first 120)
```
ai/__pycache__/__init__.cpython-312.pyc	149
ai/__pycache__/agent.cpython-312.pyc	47182
ai/__pycache__/features.cpython-310.pyc	5058
ai/__pycache__/features.cpython-312.pyc	10296
ai/__pycache__/models.cpython-312.pyc	7143
ai/__pycache__/policy.cpython-310.pyc	8073
ai/__pycache__/policy.cpython-312.pyc	22587
ai/__pycache__/trt_utils.cpython-312.pyc	2995
ai/agent.py	36180
ai/features.py	6468
ai/models.py	4225
ai/policy.py	17910
ai/weights/intent_lstm.pt	3917845
ai/weights/intent_model.pt	3917845
ai/weights/model_manifest.json	151
ai/weights/situation_gcn.pt	125261
ai/weights/situation_model.pt	125261
ai/weights/tactical_model.pt	53013
experiments/shenji_agent/20260327_141522/intent_model.pt	3917845
experiments/shenji_agent/20260327_141522/situation_model.pt	125261
experiments/shenji_agent/20260327_141522/tactical_model.pt	53013
scripts/__pycache__/bootstrap_remote.cpython-312.pyc	3079
scripts/__pycache__/report_progress.cpython-312.pyc	3934
scripts/__pycache__/run_stop_autopilot.cpython-313.pyc	15162
scripts/__pycache__/sync_remote_artifacts.cpython-312.pyc	2963
scripts/bootstrap_remote.py	2024
scripts/run_stop_autopilot.py	11347
tests/__pycache__/__init__.cpython-312.pyc	196
tests/__pycache__/test_mainline_contract.cpython-312.pyc	12114
tests/test_mainline_contract.py	9872
training/__pycache__/__init__.cpython-310.pyc	285
training/__pycache__/__init__.cpython-312.pyc	165
training/__pycache__/__init__.cpython-313.pyc	247
training/__pycache__/common.cpython-310.pyc	3590
training/__pycache__/common.cpython-312.pyc	5778
training/__pycache__/common.cpython-313.pyc	6032
training/__pycache__/data.cpython-310.pyc	9778
training/__pycache__/data.cpython-312.pyc	18683
training/__pycache__/evaluate.cpython-310.pyc	4053
training/__pycache__/evaluate.cpython-312.pyc	12821
training/__pycache__/maps.cpython-310.pyc	6162
training/__pycache__/maps.cpython-312.pyc	8706
training/__pycache__/selfplay.cpython-310.pyc	6639
training/__pycache__/selfplay.cpython-312.pyc	19332
training/__pycache__/supervised.cpython-310.pyc	9826
training/__pycache__/supervised.cpython-312.pyc	19717
training/cache/map_catalog.json	6776
training/common.py	3298
training/data.py	12179
training/evaluate.py	9789
training/maps.py	4343
training/selfplay.py	16625
training/supervised.py	14834
```

## Current-only files (first 120)
```
.agent/context.md	2848
.agent/findings.md	3395
.agent/progress.md	4389
.agent/round18_smoke_post212051fix_20260427_024127/league/league_state.json	143
.agent/task_plan.md	2083
.claude/settings.local.json	535
.dockerignore	375
README.md	11128
ai/__pycache__/artifacts.cpython-310.pyc	3532
ai/__pycache__/features.cpython-310.pyc	7569
ai/__pycache__/policy.cpython-310.pyc	22747
ai/agent.py	48283
ai/artifacts.py	3114
ai/features.py	10274
ai/models.py	4683
ai/policy.py	32876
ai/weights/model_manifest.json	1306
ai/weights/situation_feature_normalizer.json	5916
ai/weights_incumbent_rule/model_manifest.json	197
configs/selfplay_formal_learned.yaml	513
configs/selfplay_medium_learned.yaml	512
configs/selfplay_smoke_battle_phase.yaml	505
configs/selfplay_smoke_learned_engagement.yaml	505
configs/selfplay_smoke_objective_quality.yaml	506
configs/selfplay_sota_stop.yaml	593
configs/supervised_baseline.yaml	787
configs/supervised_score_quality.yaml	931
configs/supervised_tactical_ablation.yaml	787
docker-compose.auth-mac.yml	209
docs/docs.md	114290
docs/sdk_auth_runbook.md	4582
experiments/shenji_agent/20260422_123838/config.yaml	723
experiments/shenji_agent/20260422_123838/metrics.json	7827
experiments/shenji_agent/20260422_123838/slice_report.json	305
experiments/shenji_agent/20260422_123838/summary.md	515
experiments/shenji_agent/20260423_024736/config.yaml	783
experiments/shenji_agent/20260423_024736/metrics.json	7969
experiments/shenji_agent/20260423_024736/slice_report.json	307
experiments/shenji_agent/20260423_024736/summary.md	516
experiments/shenji_agent/20260423_044528/config.yaml	783
experiments/shenji_agent/20260423_044528/metrics.json	7969
experiments/shenji_agent/20260423_044528/slice_report.json	307
experiments/shenji_agent/20260423_044528/summary.md	516
experiments/shenji_agent/20260423_051056/config.yaml	783
experiments/shenji_agent/20260423_051056/metrics.json	7969
experiments/shenji_agent/20260423_051056/slice_report.json	307
experiments/shenji_agent/20260423_051056/summary.md	516
experiments/shenji_agent/20260423_052221/config.yaml	783
experiments/shenji_agent/20260423_052221/metrics.json	7969
experiments/shenji_agent/20260423_052221/slice_report.json	307
experiments/shenji_agent/20260423_052221/summary.md	516
experiments/shenji_agent/20260423_093042/config.yaml	783
experiments/shenji_agent/20260423_093042/metrics.json	7969
experiments/shenji_agent/20260423_093042/slice_report.json	307
experiments/shenji_agent/20260423_093042/summary.md	516
experiments/shenji_agent/20260425_034746/config.yaml	783
experiments/shenji_agent/20260425_034746/metrics.json	7971
experiments/shenji_agent/20260425_034746/slice_report.json	307
experiments/shenji_agent/20260425_034746/summary.md	516
experiments/shenji_agent/20260428_235555/config.yaml	783
experiments/shenji_agent/20260428_235555/metrics.json	8984
experiments/shenji_agent/20260428_235555/situation_feature_normalizer.json	5916
experiments/shenji_agent/20260428_235555/slice_report.json	307
experiments/shenji_agent/20260428_235555/summary.md	742
experiments/shenji_agent/authentication_failure_analysis.md	3077
experiments/shenji_agent/authorized_environment_search.md	3540
experiments/shenji_agent/autonomous_project_run_initial_snapshot.md	6053
experiments/shenji_agent/completion_criteria.md	3274
experiments/shenji_agent/current_interfaces.txt	344
experiments/shenji_agent/current_mac_addresses.txt	36
experiments/shenji_agent/final_autonomous_blocked_report.md	8311
experiments/shenji_agent/final_autonomous_completion_report.md	5217
experiments/shenji_agent/full_promotion_after_auth_report.md	1092
experiments/shenji_agent/iteration_001.md	2841
experiments/shenji_agent/iteration_002.md	7821
experiments/shenji_agent/iteration_003.md	9125
experiments/shenji_agent/iteration_004.md	2971
experiments/shenji_agent/iteration_005.md	6399
experiments/shenji_agent/iteration_006.md	6325
experiments/shenji_agent/iteration_007.md	14747
experiments/shenji_agent/iteration_008.md	12242
experiments/shenji_agent/iteration_009.md	12628
experiments/shenji_agent/iteration_010.md	3176
experiments/shenji_agent/iteration_010_merge_back.md	2340
experiments/shenji_agent/iteration_011_auto_research_auth_completion.md	7213
experiments/shenji_agent/iteration_012_auto_completion_push.md	9877
experiments/shenji_agent/iteration_015_auth_recovered_full_promotion.md	6433
experiments/shenji_agent/iteration_015_final_completion_closeout.md	2808
experiments/shenji_agent/league_20260422_124556/config.yaml	503
experiments/shenji_agent/league_20260422_124556/exploitability.json	254
experiments/shenji_agent/league_20260422_124556/league_state.json	179
experiments/shenji_agent/league_20260422_124556/match_results.json	126144
experiments/shenji_agent/league_20260422_124556/participants.json	247
experiments/shenji_agent/league_20260422_124556/promotion.json	192
experiments/shenji_agent/league_20260422_124556/rating.json	1126
experiments/shenji_agent/league_20260422_124556/summary.md	255
experiments/shenji_agent/league_20260422_125405/config.yaml	503
experiments/shenji_agent/league_20260422_125405/exploitability.json	254
experiments/shenji_agent/league_20260422_125405/league_state.json	199
experiments/shenji_agent/league_20260422_125405/match_results.json	126224
experiments/shenji_agent/league_20260422_125405/participants.json	247
experiments/shenji_agent/league_20260422_125405/promotion.json	118
experiments/shenji_agent/league_20260422_125405/rating.json	1140
experiments/shenji_agent/league_20260422_125405/summary.md	254
experiments/shenji_agent/league_20260423_024928/config.yaml	450
experiments/shenji_agent/league_20260423_024928/exploitability.json	254
experiments/shenji_agent/league_20260423_024928/league_state.json	199
experiments/shenji_agent/league_20260423_024928/map_list.json	12309
experiments/shenji_agent/league_20260423_024928/match_results.json	791559
experiments/shenji_agent/league_20260423_024928/participants.json	250
experiments/shenji_agent/league_20260423_024928/promotion.json	330
experiments/shenji_agent/league_20260423_024928/rating.json	1147
experiments/shenji_agent/league_20260423_024928/summary.md	445
experiments/shenji_agent/league_20260423_032802/config.yaml	509
experiments/shenji_agent/league_20260423_032802/exploitability.json	254
experiments/shenji_agent/league_20260423_032802/league_state.json	199
experiments/shenji_agent/league_20260423_032802/map_list.json	12309
experiments/shenji_agent/league_20260423_032802/match_results.json	791559
experiments/shenji_agent/league_20260423_032802/participants.json	250
experiments/shenji_agent/league_20260423_032802/promotion.json	359
```

## Assessment
- Backup appears to be a partial/older or different layout: it lacks many current report and paper files, has no `.git`, and contains backup-only datasets/SDK assets.
- Backup contains critical missing asset class: tactical dataset.
- Backup does not prove source files should overwrite current files. Any restore should be selected and non-destructive only.
