# Current Repo State

- Captured at UTC: 2026-05-14T03:47:29Z
- Current repo path: /remote-home/lizilong/shenji_agent
- Branch: main
- HEAD: 33fc11f3eec0993b61494a60bd37e7e09e9f38f1

## git status --short
```
?? goals/codex_goal_backup_integrity_audit_before_v2.md
```

## git log --oneline --decorate -n 30
```
33fc11f (HEAD -> main, backup/before_backup_integrity_audit_20260513_082959) upgrade evidence toward top-tier submission candidate
586ffbc (backup/before_v2_top_tier_upgrade_20260513_072051) version 1 completed
89d2c0e prepare manuscript prewrite evidence package
5e414f7 (backup/before_paper_prewrite_package_20260513_045909) Add paper-facing project reports
b542b98 Preserve formal promotion handoff evidence
65db518 docs: add research project report for promotion results
095bb38 fix: count exploiter gate by true win rate
fb45105 fix: keep heartbeat json writes nonblocking
9087c0b fix: retry atomic json writes on stale handles
b752402 fix: evaluate exploiters with paired objective scoring
da065ce fix: use objective evidence for reward-tied promotion scoring
27cd94e fix: use objective evidence for reward-tied promotion scoring
9e880c5 fix: preserve objective scores in promotion ratings
0d448d3 fix: prefer user-local sdk package for selfplay
dab4a84 fix: align auth diagnostics with sdk engine config path
b7271b5 docs: finalize iteration 012 auth blocker evidence
17eb19b chore: add full promotion continuation runner
45ad227 docs: finalize iteration 011 auth handoff status
b25ab65 chore: add sdk auth preflight and handoff tools
483a3cf docs: refresh iteration 010 final status
5d29772 docs: finalize sdk auth blocked evidence
0a9c7ff (fresh/main) docs: refresh iteration 009 final head
0c62b37 docs: record iteration 009 sdk auth blocker
e7cc4f3 chore: point weights at battle phase diagnostics code
3d3e69a feat: add battle phase valid action diagnostics
1459339 docs: refresh iteration 008 final head
789ce7f docs: finalize iteration 008 blocked evidence
e23a98f fix: keep objective diagnostics out of match stats
1d7cdf3 fix: correct objective policy manifest sha
d3680ae chore: publish objective policy manifest
```

## Latest relevant commits
- V1 commit: searched in log above; no explicit V1 tag/branch identified in this pass.
- V2 SAFE_STOP commit: searched in log above; no explicit V2 SAFE_STOP tag/branch identified in this pass.
- Current HEAD: 33fc11f3eec0993b61494a60bd37e7e09e9f38f1

## Current Known Blockers To Audit
ai/weights/tactical_model.pt
ai/weights/tactical_model.pth
Wargame_Feature_Dataset.csv
ai/weights_incumbent_rule/intent_lstm.pt
ai/weights_incumbent_rule/intent_lstm.pth
ai/weights_incumbent_rule/intention_lstm.pt
ai/weights_incumbent_rule/intention_lstm.pth
ai/weights_incumbent_rule/situation_gcn.pt
ai/weights_incumbent_rule/situation_gcn.pth
land_wargame_sdk/Data/Data/scenarios/*.json
land_wargame_sdk/Data/Data/maps/map_*/basic.json
land_wargame_sdk/Data/Data/maps/map_*/cost.pickle
land_wargame_sdk/Data/Data/maps/map_*/*see.npz
experiments/shenji_agent/league_*
training/*.py
scripts/*.py
ai/*.py
tests/*.py
paper/prewrite_package/
paper/v2_evidence_package/
reports/v2_top_tier_upgrade/
```
```text
reports/backup_integrity_audit/00_current_repo_state.md
```
```text
reports/backup_integrity_audit/01_backup_candidates.md
```
```text
reports/backup_integrity_audit/02_quarantine_location.md
```
```text
reports/backup_integrity_audit/03_backup_identity.md
```
```text
LIKELY_CORRECT_SHENJI_AGENT
POSSIBLY_CORRECT_BUT_INCOMPLETE
WRONG_PROJECT_OR_WRONG_VERSION
UNREADABLE_BACKUP
```
```text
reports/backup_integrity_audit/04_missing_file_search.md
reports/backup_integrity_audit/missing_file_search.json
```
```text
reports/backup_integrity_audit/05_current_vs_backup_diff.md
```
```text
reports/backup_integrity_audit/06_backup_checksums.md
```
```text
reports/backup_integrity_audit/RESTORE_DECISION.md
reports/backup_integrity_audit/RESTORE_DECISION.json
```
```text
BACKUP_VALID_RESTORE_RECOMMENDED
```
```text
reports/backup_integrity_audit/selected_restore_paths.txt
```
```text
reports/backup_integrity_audit/08_selected_restore_log.md
```
```text
reports/backup_integrity_audit/FINAL_COMPLETENESS_REPORT.md
```
```text
PROJECT_COMPLETENESS_CONFIRMED_RESUME_V3
PROJECT_PARTIALLY_IMPROVED_RESUME_WITH_BLOCKERS
BACKUP_INVALID_KEEP_CURRENT_PROJECT
SAFE_STOP_NEED_CORRECT_BACKUP
```
```text
reports/backup_integrity_audit/SAFE_STOP.md
