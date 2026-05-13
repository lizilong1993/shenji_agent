# Dangerous Sync Audit

The mandated recursive grep over `.` was started, but the repository contains large restored SDK/data trees and the process was stopped after it became too slow for an interactive prewrite task. A scoped `rg` audit excluding `.git/` and `land_wargame_sdk/Data/` completed.

Command:

```bash
rg --glob '!.git/**' --glob '!land_wargame_sdk/Data/**' -n -- '--delete|--delete-excluded|--delete-missing-args|robocopy .* /MIR|rm -rf|git reset --hard|git clean -fdx|Remove-Item' . | head -200
```

Findings:

- `Dockerfile` contains `rm -rf /var/lib/apt/lists/*`, a normal container image cleanup outside project assets.
- Goal files contain explicit warnings or historical restore instructions for `git reset --hard`, `git clean -fdx`, `rsync --delete`, and `robocopy /MIR`.
- `paper/prewrite_package/asset_loss_prevention.md` and the package generator repeat the warnings.
- No active repository sync script with destructive defaults was identified in the scoped audit.
