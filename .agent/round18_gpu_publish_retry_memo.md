# Round 18 GPU Publish Retry Memo

## Result

- No new git commit was created in this retry.
- No second `ai/weights` republish was performed in this retry.
- Existing state already satisfies the requested handoff for the next formal compare:
  - `HEAD` is `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`
  - `HEAD` commit message is `Enable GPU policy inference for learned policy`
  - `HEAD` commit touches only:
    - `ai/policy.py`
    - `tests/test_mainline_contract.py`
  - local `ai/weights/model_manifest.json` currently points `source_commit_sha` at that same commit
  - `training.selfplay` resolves participant code from `model_manifest.json.source_commit_sha` and then loads `ai/*.py` from git objects

## Exact Commands Run In This Retry

```bash
git status --short
git status --short --branch
rg -n "source_commit_sha|model_manifest|publish" -S .
rg --files | rg "(^|/)ai/policy.py$|(^|/)tests/test_mainline_contract.py$|(^|/)model_manifest.json$|(^|/)weights|publish|artifact"
git diff -- ai/policy.py tests/test_mainline_contract.py
git log --oneline --decorate -n 8 -- ai/policy.py tests/test_mainline_contract.py
sed -n '1,240p' ai/policy.py
sed -n '1,240p' training/common.py
sed -n '330,410p' training/supervised.py
sed -n '1,220p' ai/weights/model_manifest.json
git show --stat --oneline --name-only 2a5ac08
git diff --stat 2a5ac08^ 2a5ac08 -- ai/policy.py tests/test_mainline_contract.py ai/weights/model_manifest.json
git ls-files ai/weights/model_manifest.json ai/policy.py tests/test_mainline_contract.py
sed -n '1,160p' .agent/round18_gpu_publish_memo.md
sed -n '1,200p' .agent/round18_gpu_publish_republish.json
sed -n '260,340p' training/selfplay.py
git check-ignore -v ai/weights/model_manifest.json ai/weights/intent_lstm.pt ai/weights/situation_gcn.pt
git cat-file -p 2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7:ai/policy.py | sed -n '80,150p'
sed -n '1,220p' .agent/round18_compare_20260425_040546.command.txt
sed -n '1,220p' .agent/round18_compare_20260425_040546.memo.md
git rev-parse HEAD
git diff-tree --no-commit-id --name-only -r HEAD
git status --short -- ai/policy.py tests/test_mainline_contract.py ai/weights/model_manifest.json
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/remote-home/lizilong/shenji_agent')
manifest = json.loads((root / 'ai/weights/model_manifest.json').read_text(encoding='utf-8'))
print(json.dumps({
  'source_commit_sha': manifest.get('source_commit_sha'),
  'generated_at': manifest.get('generated_at'),
  'intent_lstm': manifest.get('intent_lstm'),
  'situation_gcn': manifest.get('situation_gcn')
}, indent=2))
PY
python3 - <<'PY'
from pathlib import Path
text = Path('/remote-home/lizilong/shenji_agent/training/selfplay.py').read_text(encoding='utf-8')
needles = {
  'loads_manifest_from_artifacts': 'manifest_path = Path(artifacts) / "model_manifest.json"',
  'reads_source_commit_sha': 'commit_sha = str(manifest.get("source_commit_sha", "")).strip()',
  'loads_agent_class_from_commit': '"agent_class": _load_agent_class_from_commit(source_commit_sha)',
}
for key, needle in needles.items():
    print(f'{key}={needle in text}')
PY
git show --no-patch --format='%H %s' HEAD
```

## Evidence

- Existing publish memo confirms the earlier minimal publish path:
  - reused run dir: `/remote-home/lizilong/shenji_agent/experiments/shenji_agent/20260425_034746`
  - refresh method: `training.common.copy_publish_artifacts`
  - recorded in: `.agent/round18_gpu_publish_memo.md`
  - structured record: `.agent/round18_gpu_publish_republish.json`
- Current manifest state:
  - `source_commit_sha`: `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7`
  - `generated_at`: `20260425_121557`
  - `intent_lstm`: `intent_lstm.pt`
  - `situation_gcn`: `situation_gcn.pt`
- `training.selfplay` verification in this retry:
  - `loads_manifest_from_artifacts=True`
  - `reads_source_commit_sha=True`
  - `loads_agent_class_from_commit=True`
- Current `HEAD` verification in this retry:
  - `2a5ac08ef5dc7a6cc9694aa7cbbd5e1ceb5f0bb7 Enable GPU policy inference for learned policy`
  - changed files in commit:
    - `ai/policy.py`
    - `tests/test_mainline_contract.py`

## Why No Further Action Was Taken

- The requested git commit already exists and is the current `HEAD`.
- The local candidate artifact manifest already points at that commit.
- `ai/weights/model_manifest.json` is ignored by git, so the provenance refresh is workspace-local and already effective for the next compare launched from this checkout.
- Re-running the same republish would only churn ignored local files and timestamps without improving correctness.

## Caveats

- `tests/test_mainline_contract.py` still has additional unstaged worktree edits; they are outside the already-landed minimal commit and were not touched in this retry.
- `ai/weights/model_manifest.json` and `ai/weights/*.pt` are ignored files, so provenance is not visible in git history.
- A compare launched before the `20260425_121557` manifest refresh would not benefit from this provenance; a compare launched after it should.
