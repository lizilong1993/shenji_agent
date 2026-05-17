# Memo: Zero Policy Count Likely Cause

## Most Likely Cause Class

- The zero-policy-count anomaly is most likely a **runtime action-path degeneration** in the relaunch compare.
- More specifically: the relaunch appears to have run through an environment/backend episode shape that never produced meaningful `Move`/`Shoot` accounting on either side, despite models being loaded.

## Strongest Concrete Evidence

1. The relaunch command is effectively the same compare:
   - same cwd
   - same `training.selfplay` command
   - same candidate/incumbent artifact pair
   - same `SHENJI_POLICY_DEVICE=cuda`
   - only GPU ordinal changed: `CUDA_VISIBLE_DEVICES=0` vs `1`
2. The relaunch run `160721` says models loaded successfully:
   - candidate profile `challenger`
   - opponent profile `incumbent`
   - `intent_model_loaded_count = 2`
   - `situation_model_loaded_count = 2`
   - candidate `inference_devices = ['cuda']`
3. Yet every one of its `100` matches reports:
   - candidate `move.total = 0`, `shoot.total = 0`
   - opponent `move.total = 0`, `shoot.total = 0`
   - `trace_samples = []`
4. The interrupted control run `082209` has:
   - same source SHAs
   - same artifact dirs
   - same `map_list.json`
   - same first `91` core match order/outcome fields
   - but non-zero policy counts and traces in all completed matches
5. The relaunch also completed about `6.59x` faster than the interrupted run, which strongly suggests a different runtime episode path rather than just different post-processing.

## Source Bucket

- This anomaly is most likely from **runtime action path differences**.
- It does **not** look primarily like:
  - a launcher/relaunch command issue
  - a post-hoc stats-collection / summary-merging issue
- A backend/env degeneration is still plausible, but the observable failure surface is the runtime action path, not the launcher wrapper.

## Single Best Next Action

- Run a **single paired-map detached replay of order_index `0` (`2230531221.json`) with raw per-step diagnostics** captured before aggregation:
  - `observation.time.stage`
  - count of `valid_actions`
  - count of `controllable_ops`
  - `use_ai_optimization`
  - raw `get_policy_stats()` from both agents before `training/selfplay.py` merges them
- This one action would immediately separate:
  - env/backend degeneration such as stage-1-only or no-action episodes
  - from a pure learned-path selection failure
  - from any hidden merge/reporting issue
