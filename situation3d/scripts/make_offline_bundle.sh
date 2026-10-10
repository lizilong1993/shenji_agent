#!/usr/bin/env bash
# 生成可传输的离线资产包（tar）+ 清单 + 校验和。
#
# 为什么还要 tar
# --------------
# 资产被 `.gitignore` 忽略，**不在版本控制内**；单有清单无法重建资产
# （清单只证明「现有文件对不对」，不携带文件本体）。
# 故 C1 的「离线可复现包」= 清单 + 校验和 + **资产归档** + 恢复脚本。
#
# 用法：
#   bash scripts/make_offline_bundle.sh [输出目录] [situation3d 根]
#
# 产物（默认输出到 dist/）：
#   offline_assets_manifest.json   清单（相对路径 / 字节数 / SHA-256 / 类别）
#   offline_assets_manifest.csv    同上，CSV（便于 diff 与人工核对）
#   offline_assets.tar.gz          资产归档
#   offline_assets.tar.gz.sha256   归档的 SHA-256
#   BUNDLE_README.md               包内说明（含一行恢复命令）
set -euo pipefail

OUT="${1:-dist}"
ROOT="${2:-.}"
mkdir -p "$OUT"
cd "$ROOT"

echo "[1/4] 生成清单与包版本号 ..."
python3 scripts/package_offline_assets.py --root . --out "$OUT/offline_assets_manifest.json" \
  --csv "$OUT/offline_assets_manifest.csv"

VER=$(python3 -c "import json,sys;print(json.load(open('$OUT/offline_assets_manifest.json'))['package_version'])")
TAR="$OUT/offline_assets.tar.gz"

echo "[2/4] 打包资产（版本 $VER）..."
# 只打包清单里列出的路径；-C . 保证归档内是相对路径，
# 这样恢复端不需要知道打包机的绝对路径。
python3 - "$OUT/offline_assets_manifest.json" <<'PY' > /tmp/bundle_paths.txt
import json, sys
m = json.load(open(sys.argv[1], encoding='utf-8'))
cats = sorted({e['path'].split('/')[0] + '/' + e['path'].split('/')[1] for e in m['entries']})
# 逐类别顶级目录打包（比逐文件快几个数量级，且清单已能校验多打的文件）
for c in sorted({p for p in cats}):
    print(c)
PY
# 逐类别目录列表：去重后只保留顶层类别目录
python3 - "$OUT/offline_assets_manifest.json" <<'PY' > /tmp/bundle_dirs.txt
import json, sys, os
m = json.load(open(sys.argv[1], encoding='utf-8'))
tops = set()
for e in m['entries']:
    parts = e['path'].split('/')
    # 顶层目录可能是 web/xxx 或 engine/xxx 两层
    tops.add('/'.join(parts[:2]) if parts[0] in ('web', 'engine') else parts[0])
for t in sorted(tops):
    print(t)
PY
mapfile -t DIRS < /tmp/bundle_dirs.txt
echo "     打包目录: ${DIRS[*]}"
tar -czf "$TAR" "${DIRS[@]}"

echo "[3/4] 计算归档校验和 ..."
sha256sum "$TAR" | awk '{print $1"  offline_assets.tar.gz"}' > "$TAR.sha256"
cat "$TAR.sha256" | sed 's/^/     /'

echo "[4/4] 写包内说明 ..."
ARCH_SHA=$(cut -d' ' -f1 < "$TAR.sha256")
cat > "$OUT/BUNDLE_README.md" <<EOF
# 离线资产包（package_version \`$VER\`）

| 文件 | 说明 |
| --- | --- |
| \`offline_assets_manifest.json\` | 资产清单：相对路径 / 字节数 / SHA-256 / 类别 / 是否可选 |
| \`offline_assets_manifest.csv\` | 同上，CSV（便于 diff） |
| \`offline_assets.tar.gz\` | 资产归档，sha256 = \`$ARCH_SHA\` |
| \`offline_assets.tar.gz.sha256\` | 归档校验和 |

## 恢复步骤

\`\`\`bash
# 1) 校验归档本身
sha256sum -c offline_assets.tar.gz.sha256

# 2) 解到目标 situation3d 根目录
tar -xzf offline_assets.tar.gz -C <situation3d 根目录>

# 3) 逐项校验（相对清单，含 SHA-256）
python3 scripts/verify_restore_offline_assets.py \\
    --manifest offline_assets_manifest.json --target <situation3d 根目录> --mode verify

# 4) 若有缺失且手边有另一份拷贝，可只补差的
python3 scripts/verify_restore_offline_assets.py \\
    --manifest offline_assets_manifest.json --target <situation3d 根目录> \\
    --source <另一份拷贝> --mode restore
\`\`\`

## 注意

- \`engine/data/intent_model.json\` 是**可选**产物；缺失时规则回落，**不参与包版本号**。
- 资产的许可与来源见 \`docs/reproduce_offline.md\`。
- 归档用 \`tar -C <root>\` 生成，内部为相对路径，**不绑定打包机的绝对路径**。
EOF

echo
echo "完成。产物："
ls -lh "$OUT" | sed 's/^/  /'
echo "package_version: $VER"
