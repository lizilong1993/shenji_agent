# Round 18 GPU Publish Memo

## Summary

- Committed files:
  - `ai/policy.py`
  - `tests/test_mainline_contract.py` (only the staged GPU-policy hunks; the worktree still has additional unstaged edits outside this commit)
- New commit:
  - `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`
- Artifact provenance refresh:
  - Reused existing supervised run artifacts from `experiments/shenji_agent/20260425_034746`
  - Republished into `ai/weights` with `training.common.copy_publish_artifacts`
  - Updated local ignored manifest `ai/weights/model_manifest.json` so `source_commit_sha` now points at `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`

## Exact Commands

```bash
git add -- ai/policy.py
git apply --cached .agent/round18_gpu_publish_tests.patch
git commit -m "Enable GPU policy inference for learned policy"
git rev-parse HEAD
/remote-home/lizilong/venvs/shenji/bin/python -m unittest tests.test_mainline_contract.MainlineContractTest.test_learned_policy_configure_moves_models_to_cuda_when_available tests.test_mainline_contract.MainlineContractTest.test_learned_policy_configure_keeps_cpu_fallback_without_cuda tests.test_mainline_contract.MainlineContractTest.test_learned_policy_configure_allows_explicit_cpu_override_with_cuda_visible > .agent/round18_gpu_publish_unittest.log 2>&1
/remote-home/lizilong/venvs/shenji/bin/python - <<'PY' > .agent/round18_gpu_publish_republish.json
import json
from pathlib import Path
from training.common import copy_publish_artifacts, git_head_sha, now_tag, repo_root

root = repo_root()
run_dir = root / 'experiments' / 'shenji_agent' / '20260425_034746'
publish_dir = root / 'ai' / 'weights'
manifest_path = publish_dir / 'model_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['generated_at'] = now_tag()
manifest['source_commit_sha'] = git_head_sha(root)
copy_publish_artifacts(run_dir, publish_dir, manifest)
result = {
    'run_dir': str(run_dir),
    'publish_dir': str(publish_dir),
    'source_commit_sha': manifest['source_commit_sha'],
    'generated_at': manifest['generated_at'],
    'managed_files': manifest.get('managed_files', []),
    'published_files': sorted(p.name for p in publish_dir.iterdir()),
}
print(json.dumps(result, indent=2, ensure_ascii=False))
PY
git cat-file -p 2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7:ai/policy.py | rg -n "_resolve_device|SHENJI_POLICY_DEVICE|requested_device|self._move_to_device"
```

## Evidence

- Unit test log: `.agent/round18_gpu_publish_unittest.log`
  - `Ran 3 tests in 0.313s`
  - `OK`
- Republish record: `.agent/round18_gpu_publish_republish.json`
  - `run_dir`: `/remote-home/lizilong/shenji_agent/experiments/shenji_agent/20260425_034746`
  - `publish_dir`: `/remote-home/lizilong/shenji_agent/ai/weights`
  - `source_commit_sha`: `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`
  - `generated_at`: `20260425_121557`
- Current local manifest: `ai/weights/model_manifest.json`
  - `source_commit_sha`: `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`
  - `intent_lstm`: `intent_lstm.pt`
  - `situation_gcn`: `situation_gcn.pt`
- Git object check:
  - `git cat-file` on `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7:ai/policy.py` shows `_resolve_device`, `SHENJI_POLICY_DEVICE`, `requested_device`, and `_move_to_device`

## Why This Is Sufficient

- `training.selfplay` resolves candidate code from `model_manifest.json.source_commit_sha`, not from the dirty working tree.
- The new commit contains the GPU-enabled `ai/policy.py`.
- `ai/weights/model_manifest.json` now points to that commit.
- Reusing the existing run artifacts is sufficient because the weights themselves did not change; only the code provenance needed to advance to the new commit.

## Caveats

- `ai/weights/model_manifest.json` and `ai/weights/*.pt` are ignored by git, so the provenance refresh is local workspace state, not part of git history.
- `tests/test_mainline_contract.py` still has additional unstaged worktree edits beyond the committed GPU hunks; they were intentionally excluded from the commit.
- GPU inference still depends on runtime availability: `policy_profile != "baseline"` and `torch.cuda.is_available()` must be true unless overridden with `SHENJI_POLICY_DEVICE=cpu`.
