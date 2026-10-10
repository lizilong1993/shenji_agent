#!/usr/bin/env bash
# Transfer Grade B precision audit package from remote to local.
# Usage: bash scripts/transfer_audit.sh [remote_user@host] [remote_audit_dir]
# Default: remote_user@host=lizilong@100.73.109.52

set -euo pipefail

REMOTE="${1:-lizilong@100.73.109.52}"
SRC="/tmp"

echo "=== Running remote precision audit script ==="
ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 "$REMOTE" \
  "cd /remote-home/lizilong/shenji_agent && /remote-home/lizilong/venvs/shenji/bin/python3.10 run_grade_b_audit.py"

echo "=== Finding newest audit directory on remote ==="
LATEST=$(ssh -o ConnectTimeout=15 "$REMOTE" \
  "ls -1dt /tmp/grade_b_precision_audit_* 2>/dev/null | head -1")

if [ -z "$LATEST" ]; then
  echo "ERROR: No audit directories found on remote at $SRC/grade_b_precision_audit_*"
  exit 1
fi

TS=$(basename "$LATEST" | sed 's/grade_b_precision_audit_//')
echo "Latest audit: $TS"

echo "=== Packaging on remote ==="
ssh -o ConnectTimeout=30 -o ServerAliveInterval=15 "$REMOTE" \
  "cd /tmp && tar czf grade_b_precision_audit_${TS}.tar.gz grade_b_precision_audit_${TS}/ && sha256sum grade_b_precision_audit_${TS}.tar.gz"

LOCAL_DEST="D:/WHUSAI/Projects/SituationAwareness/code/shenji_agent/artifacts/grade_b_precision_audit_${TS}"
mkdir -p "$LOCAL_DEST"

echo "=== Transferring to local: $LOCAL_DEST ==="
scp -o ConnectTimeout=30 -o ServerAliveInterval=15 -r "${REMOTE}:${LATEST}/" "$LOCAL_DEST/"

# Also pull the compressed archive for integrity
scp -o ConnectTimeout=30 -o ServerAliveInterval=15 "${REMOTE}:/tmp/grade_b_precision_audit_${TS}.tar.gz" "${LOCAL_DEST}/"

echo "=== Verifying manifest ==="
MANIFEST="${LOCAL_DEST}/_manifest.json"
if [ -f "$MANIFEST" ]; then
  python3.10 -c "
import json, hashlib, sys
m = json.load(open('${MANIFEST}'))
errors = []
for fname, fmeta in m.get('files', {}).items():
    fpath = '${LOCAL_DEST}/' + fname
    try:
        actual = hashlib.sha256(open(fpath, 'rb').read()).hexdigest()
        if actual != fmeta['sha256']:
            errors.append(f'SHA256 MISMATCH: {fname}')
    except FileNotFoundError:
        errors.append(f'MISSING: {fname}')
if errors:
    print('VERIFY FAILED:')
    for e in errors:
        print('  ', e)
    sys.exit(1)
else:
    print('All files verified OK')
"
else
  echo "WARNING: No _manifest.json found in audit package"
fi

echo ""
echo "=== Done ==="
echo "Local artifacts: $LOCAL_DEST"
echo ""
echo "Next steps:"
echo "  1. Open $LOCAL_DEST/_README.md"
echo "  2. Review per-class positive and hard_negative samples"
echo "  3. Mark TP/FP/TN/FN in _all_samples.csv"
echo "  4. Report precision back to the team"
