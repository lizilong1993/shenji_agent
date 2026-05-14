# Candidate Asset Inventory

Quarantine source:

`/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`

| Asset | Candidate found? | Current repo had it before restore? | Size | SHA256 | Restore needed? |
|---|---:|---:|---:|---|---:|
| `ai/weights/tactical_model.pt` | yes | no | 53,013 bytes | `9fc261e560112e33c4a79c7b40a47a0ebf1314595ad37970cb3b6eef6209cb04` | yes |
| experiment tactical checkpoint | yes | no | 53,013 bytes | `9fc261e560112e33c4a79c7b40a47a0ebf1314595ad37970cb3b6eef6209cb04` | yes, as comparison copy |
| `Wargame_Feature_Dataset.csv` | yes | no | 409,055,700 bytes | `824dbb0f81ef4510d762a0d616c4cb3ed45be8b54ec4117c47b86d516ee4e718` | yes |

The two checkpoint candidates are byte-identical. Listings are recorded in `candidate_asset_listing.txt`, `current_weights_listing.txt`, and `current_dataset_listing.txt`.
