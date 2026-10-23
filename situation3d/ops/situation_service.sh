#!/usr/bin/env bash
# 演示/开发服务管理。
#   演示 demo: 端口 8848（Tailscale Funnel 对外），代码取自 demo 分支 worktree
#   开发 dev : 端口 8858（仅 tailnet/内网），代码取自 main 分支主工作树
# 用法: situation_service.sh <demo|dev> <start|stop|restart|status|smoke>
set -u

MAIN_WT=/remote-home/lizilong/shenji_agent
DEMO_WT=/remote-home/lizilong/shenji_agent_demo
TILE_DB_DIR=$MAIN_WT/situation3d/tiles_db
TILES3D_DIR=$MAIN_WT/situation3d/tiles3d_db

role=${1:-}
action=${2:-}
case "$role" in
  demo) WT=$DEMO_WT; PORT=8848 ;;
  dev)  WT=$MAIN_WT;  PORT=8858 ;;
  *) echo "用法: $0 <demo|dev> <start|stop|restart|status|smoke>"; exit 2 ;;
esac
LOG=/tmp/situation3d_${role}_${PORT}.log
PIDF=/tmp/situation3d_${role}_${PORT}.pid

# 端口当前占用者的 pid（权威来源；pid 文件只做记录）
port_pid() {
  ss -ltnp 2>/dev/null | awk -v p="$PORT" 'index($4, ":" p) && substr($4, length($4)-length(p)) == ":" p { if (match($0, /pid=[0-9]+/)) print substr($0, RSTART+4, RLENGTH-4) }' | head -1
}

start() {
  local pid
  pid=$(port_pid)
  if [ -n "$pid" ]; then echo "[$role] 已在运行 (pid $pid, :$PORT)"; return 0; fi
  local mode_args=""
  if [ "$role" = "dev" ]; then
    # 8858 是测试端：开启真实推演 API 与运行库；8848 保持只读 demo。
    mode_args="--mode test --runs-dir $WT/situation3d/var/sim_runs"
  fi
  ( cd "$WT/situation3d" || exit 1
    TILE_DB_DIR="$TILE_DB_DIR" TILES3D_DIR="$TILES3D_DIR" \
    setsid nohup nice -n 10 ionice -c3 python3 server/serve.py --port "$PORT" $mode_args \
      > "$LOG" 2>&1 < /dev/null &
    disown )
  for _ in $(seq 1 20); do
    pid=$(port_pid); [ -n "$pid" ] && break; sleep 0.5
  done
  if [ -n "$pid" ]; then
    echo "$pid" > "$PIDF"
    echo "[$role] 已启动 pid $pid :$PORT（$(git -C "$WT" rev-parse --short HEAD) $WT）"
  else
    echo "[$role] 启动失败，日志尾部："; tail -20 "$LOG"; return 1
  fi
}

stop() {
  local pid; pid=$(port_pid)
  if [ -z "$pid" ]; then rm -f "$PIDF"; echo "[$role] 未运行"; return 0; fi
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  rm -f "$PIDF"
  echo "[$role] 已停止 (pid $pid)"
}

smoke() {
  local fail=0
  ok() { echo "  ok   $1"; }
  bad() { echo "  FAIL $1"; fail=1; }
  curl -sf -m 10 -o /dev/null "http://127.0.0.1:$PORT/" && ok "首页 /" || bad "首页 /"
  curl -sf -m 10 "http://127.0.0.1:$PORT/src/app.js" | head -c 1 | grep -q . && ok "/src/app.js" || bad "/src/app.js"
  local cases; cases=$(curl -sf -m 10 "http://127.0.0.1:$PORT/api/situations")
  echo "$cases" | grep -q '"cases"' && ok "案例索引 ($(echo "$cases" | grep -o '"id"' | wc -l) 个)" || bad "案例索引"
  local id; id=$(echo "$cases" | grep -o '"id": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
  if [ -n "$id" ]; then
    curl -sf -m 60 -o /dev/null "http://127.0.0.1:$PORT/api/situation?id=$id&part=core" && ok "案例 core: $id" || bad "案例 core: $id"
  else
    bad "无可用案例"
  fi
  local pid; pid=$(port_pid)
  if [ -n "$pid" ]; then
    local cwd; cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
    [ "$cwd" = "$WT/situation3d" ] && ok "运行目录 $cwd" || bad "运行目录 $cwd（应为 $WT/situation3d）"
  else
    bad "进程不存在"
  fi
  [ "$fail" = 0 ] && echo "[$role] smoke PASS" || echo "[$role] smoke FAIL"
  return "$fail"
}

status() {
  local pid; pid=$(port_pid)
  if [ -z "$pid" ]; then echo "[$role] 未运行 (:$PORT)"; return 0; fi
  echo "[$role] pid $pid :$PORT  cwd=$(readlink "/proc/$pid/cwd")"
  echo "  git: $(git -C "$WT" branch --show-current) $(git -C "$WT" rev-parse --short HEAD)"
  echo "  etime: $(ps -o etime= -p "$pid" | tr -d ' ')"
}

case "$action" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  smoke) smoke ;;
  *) echo "用法: $0 <demo|dev> <start|stop|restart|status|smoke>"; exit 2 ;;
esac
