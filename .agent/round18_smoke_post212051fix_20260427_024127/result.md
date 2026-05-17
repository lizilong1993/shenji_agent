# Round 18 Smoke Result

- run_dir: `.agent/round18_smoke_post212051fix_20260427_024127/runs/league_20260427_024628`
- command:
  - `CUDA_VISIBLE_DEVICES=1 SHENJI_POLICY_DEVICE=cuda /remote-home/lizilong/venvs/shenji/bin/python -m training.selfplay --config .agent/round18_smoke_post212051fix_20260427_024127/smoke_config.yaml --candidate .agent/round18_smoke_post212051fix_20260427_024127/candidate_weights --incumbent artifacts/best_agent/weights`
- candidate provenance:
  - `candidate_weights/model_manifest.json.source_commit_sha = 3ed3217d3635fc04d3c70495703331583e673a1f`
  - this worktree snapshot object contains the current `ai/agent.py` move-path fix
- seeded map evidence:
  - `map_list.json.items[0] = {"scenario": "2230531221.json", "map_id": 53, "bucket": "complex", "order_index": 0}`
- outcome:
  - `match_results.json.matches[0].candidate_policy_stats.counts.move = {"total": 107, "learned": 107}`
  - `match_results.json.matches[0].candidate_policy_stats.counts.shoot = {"total": 11, "learned": 11}`
  - `rating.json.policy.learned_path_coverage = 1.0`
  - `rating.json.policy.move_path_coverage = 1.0`
  - `rating.json.policy.shoot_path_coverage = 1.0`
- interpretation:
  - the zero-policy-count anomaly from `experiments/shenji_agent/league_20260426_160721` does not reproduce when the compare loads the current post-212051fix worktree snapshot for the candidate.
  - however, a plain `--candidate ai/weights` relaunch would still read `ai/weights/model_manifest.json.source_commit_sha = 2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`, so full relaunch safety depends on refreshing provenance to the fixed code snapshot first.
