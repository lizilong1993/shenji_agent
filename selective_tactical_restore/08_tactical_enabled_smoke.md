# Tactical-Enabled Smoke

Date: 2026-05-14

## Result

**SKIPPED — torch unavailable in analysis environment**

## Required metrics (not collected)

| Metric | Value |
|---|---|
| episodes/games attempted | N/A |
| episodes/games completed | N/A |
| tactical_loaded | Not testable |
| tactical_reached_count | Not testable |
| tactical_nonzero_score_count | Not testable |
| tactical_changed_action_count | Not testable |
| route_empty_count | N/A |
| route_empty_rate | N/A |
| errors | Torch import hangs (CUDA lib) |

## Why skipped

Both validators pass (checkpoint COMPATIBLE, dataset COMPATIBLE). The smoke requires
torch to load the model and run inference. In this analysis environment, `import torch`
hangs indefinitely due to a CUDA library initialization issue. This is a runtime
environment limitation, not an asset problem.

## Resolution

The smoke can be run in a production training environment where torch works normally.
Run:
```bash
python scripts/run_tactical_enabled_smoke.py \
  --checkpoint ai/weights/tactical_model.pt \
  --dataset "datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv"
```
