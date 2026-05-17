# Round 18 GPU selfplay worker memo

Changed files:
- ai/policy.py
- tests/test_mainline_contract.py

Verification commands:
- `/remote-home/lizilong/venvs/shenji/bin/python -m unittest tests.test_mainline_contract.MainlineContractTest.test_learned_policy_configure_moves_models_to_cuda_when_available tests.test_mainline_contract.MainlineContractTest.test_learned_policy_configure_keeps_cpu_fallback_without_cuda`
- `CUDA_VISIBLE_DEVICES='' /remote-home/lizilong/venvs/shenji/bin/python - <<'PY'`
- `CUDA_VISIBLE_DEVICES=2 /remote-home/lizilong/venvs/shenji/bin/python - <<'PY'`

Evidence files:
- `.agent/round18_gpu_unittest.log`
- `.agent/round18_gpu_cpu_probe.json`
- `.agent/round18_gpu_cuda_probe.json`

Notes:
- The learned policy now picks `cuda` when `torch.cuda.is_available()` is true, otherwise `cpu`.
- Both real probes loaded `ai/weights` and executed `_intent_distribution()` plus `_score_counterfactual()` successfully.
- `training.selfplay` loads agent/policy code from `source_commit_sha`, so a formal compare will only pick up this fix after the compare points at a commit containing the updated `ai/policy.py`.
