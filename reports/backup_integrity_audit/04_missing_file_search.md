# Missing File Search

- Candidate: `/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`

## Raw critical-file search
```
./ai/weights/intent_lstm.pt	3917845 bytes
./ai/weights/intention_lstm.pth	13532645 bytes
./ai/weights/situation_gcn.pt	125261 bytes
./ai/weights/situation_gcn.pth	125601 bytes
./ai/weights/tactical_model.pt	53013 bytes
./configs/supervised_tactical_ablation.yaml	727 bytes
./datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv	409055700 bytes
./experiments/shenji_agent/20260327_141522/tactical_model.pt	53013 bytes
```

| File / asset class | Found in current repo? | Found in backup? | Backup path | Size | Restore candidate? |
| --- | ---: | ---: | --- | ---: | --- |
| tactical checkpoint | no | yes | /remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215/experiments/shenji_agent/20260327_141522/tactical_model.pt | 53013 | yes |
| tactical dataset | no | yes | /remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215/datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv | 409055700 | yes |
| incumbent intent checkpoint | yes | yes | /remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215/ai/weights/intention_lstm.pth | 13532645 | no |
| incumbent situation checkpoint | yes | yes | /remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215/ai/weights/situation_gcn.pt | 125261 | no |
| scenarios | yes (50) | yes (50) | land_wargame_sdk/Data/Data/scenarios/*.json | n/a | no |
| maps basic.json | yes (16) | yes (16) | land_wargame_sdk/Data/Data/maps/map_*/basic.json | n/a | no |
| maps cost.pickle | yes (16) | yes (16) | land_wargame_sdk/Data/Data/maps/map_*/cost.pickle | n/a | no |
| see arrays | yes (16) | yes (16) | land_wargame_sdk/Data/Data/maps/map_*/*see.npz | n/a | no |
| league artifacts | yes (61) | yes (1) | experiments/shenji_agent/league_* | n/a | no |
