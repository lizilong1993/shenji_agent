# Quarantine Location

- Source backup path: `/remote-home/lizilong/shenji_agent_backup`
- Quarantine path: `/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`
- Copy command used: `cp -al /remote-home/lizilong/shenji_agent_backup /remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_hardlink_041215`
- Rationale: full `rsync -a` attempt stalled in NFS wait after a partial copy; hard-link copy creates an isolated directory tree without duplicating file contents.
- Previous incomplete copy: `/remote-home/lizilong/recovery/shenji_backup_audit/candidate_shenji_agent_backup_20260514_035020` (left in recovery area, not used as audit candidate).
- Source file count: 3179
- Quarantine file count: 3179
- Copy succeeded: yes
- Destructive options used: no
- Current project overwritten: no
