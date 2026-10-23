#!/usr/bin/env bash
# 把演示环境更新到 main 上的某个已验证 commit（只允许 fast-forward）。
# 更新演示前必须先征得用户同意；脚本默认只做预检，带 --yes 才真正执行。
# 用法: demo_update.sh <commit> [--yes] [--with-data]
#   --with-data  同时把 main 工作树的案例数据同步到演示（里程碑包含新案例/数据时用）
set -u

MAIN_WT=/remote-home/lizilong/shenji_agent
DEMO_WT=/remote-home/lizilong/shenji_agent_demo
SVC=$MAIN_WT/situation3d/ops/situation_service.sh

target=""; yes=0; with_data=0
for a in "$@"; do
  case "$a" in
    --yes) yes=1 ;;
    --with-data) with_data=1 ;;
    -*) echo "未知参数: $a"; exit 2 ;;
    *) target=$a ;;
  esac
done
[ -n "$target" ] || { echo "用法: $0 <commit> [--yes] [--with-data]"; exit 2; }

die() { echo "拒绝更新: $*"; exit 1; }

# 1) 演示工作树必须干净（只检查已跟踪文件；运行期数据/缓存允许存在）
git -C "$DEMO_WT" diff --quiet || die "演示工作树有未提交修改"
git -C "$DEMO_WT" diff --cached --quiet || die "演示工作树有已暂存修改"

# 2) 目标必须是 main 上的 commit，且是当前 demo 的后代（快进）
target=$(git -C "$MAIN_WT" rev-parse --verify "$target^{commit}") || die "commit 不存在"
git -C "$MAIN_WT" merge-base --is-ancestor "$target" main || die "$target 不在 main 上"
git -C "$MAIN_WT" merge-base --is-ancestor demo "$target" || die "$target 不是 demo 的快进更新（先确认 main 已合入基于 demo 的提交）"

echo "== 演示更新计划 =="
echo "当前 demo: $(git -C "$DEMO_WT" rev-parse --short HEAD)  $(git -C "$DEMO_WT" log -1 --format=%s)"
echo "目标    : $(git -C "$DEMO_WT" rev-parse --short "$target")  $(git -C "$MAIN_WT" log -1 --format=%s "$target")"
echo "新增提交:"
git -C "$DEMO_WT" log --oneline demo.."$target" | sed 's/^/  /'
[ "$with_data" = 1 ] && echo "案例数据: 将从 $MAIN_WT 同步到演示"

if [ "$yes" != 1 ]; then
  echo
  echo "这是预检。确认更新前必须先征得用户同意，然后带 --yes 重新执行。"
  exit 3
fi

# 3) 备份当前稳定版本，更新，重启，冒烟；失败自动回退
backup="demo-backup-$(date +%Y%m%d_%H%M%S)"
git -C "$DEMO_WT" tag "$backup" demo
echo "已打备份 tag: $backup"

git -C "$DEMO_WT" merge --ff-only "$target" || die "fast-forward 合并失败（未改动 demo）"

if [ "$with_data" = 1 ]; then
  src=$MAIN_WT/situation3d/web/data
  dst=$DEMO_WT/situation3d/web/data
  release=$dst/demo_release.json
  rel_case=""; rel_hash=""
  if [ -f "$release" ]; then
    rel_case=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("case_id") or "")' "$release") \
      || die "读取 demo_release.json 失败"
    rel_hash=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("trajectory_sha256") or "")' "$release") \
      || die "读取 demo_release.json 失败"
    [ -n "$rel_case" ] && [ -n "$rel_hash" ] || die "demo_release.json 缺少 case_id/trajectory_sha256"
    if [ -n "$rel_case" ]; then
      actual=$(sha256sum "$dst/situations/$rel_case.json" 2>/dev/null | cut -d" " -f1)
      [ "$actual" = "$rel_hash" ] || die "演示端 $rel_case 与 demo_release.json 哈希不一致"
      echo "发布产物校验通过: $rel_case $rel_hash"
    fi
  fi
  for f in "$src"/situations/*.json; do
    case "$f" in *.bak*) continue ;; esac
    base=$(basename "$f" .json)
    if [ -n "$rel_case" ] && [ "$base" = "$rel_case" ]; then
      echo "跳过已发布案例（保持选定推演结果）: $base"
      continue
    fi
    cp -p "$f" "$dst/situations/"
  done
  cp -p "$src/base.json" "$dst/base.json" 2>/dev/null || true
  cp -p "$src/ion.json" "$dst/ion.json" 2>/dev/null || true
  if [ -n "$rel_case" ]; then
    actual=$(sha256sum "$dst/situations/$rel_case.json" 2>/dev/null | cut -d" " -f1)
    [ "$actual" = "$rel_hash" ] || die "同步后发布产物校验失败: $rel_case"
  fi
  echo "案例数据已同步（已发布的选定结果未被覆盖）"
fi

"$SVC" demo restart || true
if ! "$SVC" demo smoke; then
  echo "!! 冒烟失败，回退到 $backup"
  git -C "$DEMO_WT" checkout -B demo "$backup"
  "$SVC" demo restart || true
  "$SVC" demo smoke || true
  exit 1
fi

echo "演示已更新到 $(git -C "$DEMO_WT" rev-parse --short HEAD)，回退命令："
echo "  git -C $DEMO_WT checkout -B demo $backup && $SVC demo restart && $SVC demo smoke"
