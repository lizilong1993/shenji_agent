# Asset Loss Prevention

Do not run destructive sync or cleanup commands against this repository or its asset directories:

- `rsync --delete`
- `rsync --delete-excluded`
- `rsync --delete-missing-args`
- `robocopy /MIR`
- `git reset --hard`
- `git clean -fdx`

Before any cross-machine sync, create a branch or archive copy and verify `ai/weights/`, `ai/weights_incumbent_rule/`, `land_wargame_sdk/Data/Data/`, and `experiments/shenji_agent/` after transfer.
