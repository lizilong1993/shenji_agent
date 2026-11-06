"""把一次成品运行导出成可离线打开的 HTML 包。

为什么是 HTML 而不是视频：编码视频要引入 ffmpeg 依赖与几百 MB 的中间件，
而这份包要解决的问题是"把一场推演带走给别人看"——单文件 HTML 双击就能开，
轨迹、事件、指标都在里面，离线可用，也不需要对方的浏览器装任何东西。

包的结构是一条最小的自包含页面：内联的 JSON 数据 + 一个 canvas 播放器。
不引用外部资源，不加载 Cesium：目的是"看轨迹发生了什么"，不是"重建三维场景"。
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def build_payload(run_dir: str,
                  run: Optional[dict] = None,
                  trajectory: Optional[dict] = None,
                  summary: Optional[dict] = None) -> Dict[str, Any]:
    """从运行目录读三份产物，拼成导出包需要的最小数据。"""
    run = run or _load_json(os.path.join(run_dir, "run.json")) or {}
    trajectory = trajectory or _load_json(
        os.path.join(run_dir, "trajectory.json")) or {}
    summary = summary or _load_json(
        os.path.join(run_dir, "summary.json")) or {}
    frames: List[dict] = trajectory.get("frames") or []
    targets: List[dict] = trajectory.get("targets") or []
    events: List[dict] = []
    nodes = _load_json(os.path.join(run_dir, "nodes.json")) or {}
    live_events = _load_json(os.path.join(run_dir, "live", "events.json")) or {}
    for row in (live_events.get("events") or []):
        if isinstance(row, dict):
            events.append(row)
    if not events:
        # 成品运行的 events 在 summary/trajectory 里各有一份，取能用的那个
        for row in (trajectory.get("events") or summary.get("events") or []):
            if isinstance(row, dict):
                events.append(row)
    return {
        "schema": "situation3d.replay.v1",
        "case_id": run.get("case_id") or trajectory.get("case_id") or "",
        "run_id": run.get("run_id") or "",
        "status": run.get("status") or "",
        "seed": run.get("seed"),
        "config": run.get("config") or {},
        "metrics": run.get("metrics") or {},
        "targets": targets,
        "frames": frames,
        "events": events,
        "nodes": nodes.get("nodes") or [],
    }


# 播放器：只画经纬度散点与轨迹线，颜色按阵营。刻意做得简单——
# 它的用途是"回看"，不是替代三维态势。
_PLAYER = """
<div id="hud"></div>
<canvas id="c"></canvas>
<div id="bar">
  <button id="play">播放</button>
  <input id="seek" type="range" min="0" value="0" step="1">
  <span id="tick"></span>
  <span id="legend"></span>
</div>
<div id="events"></div>
<style>
  html,body{margin:0;height:100%;font:13px/1.5 system-ui,sans-serif;color:#20344c;background:#f6f9fc}
  #hud{padding:10px 14px;border-bottom:1px solid #dbe6f2;background:#fff}
  #hud b{font-size:14px}
  #hud span{color:#66788f;margin-left:12px}
  #c{display:block;width:100%;height:calc(100% - 190px);background:#0f1a26}
  #bar{display:flex;gap:10px;align-items:center;padding:8px 14px;background:#fff;border-top:1px solid #dbe6f2}
  #seek{flex:1}
  #legend{margin-left:auto;color:#66788f}
  #events{height:110px;overflow:auto;padding:6px 14px;background:#fff;border-top:1px solid #dbe6f2}
  .ev{font-size:12px;padding:2px 0}
  .ev .t{color:#66788f;display:inline-block;min-width:52px}
</style>
<script>
const DATA = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const seek = document.getElementById('seek'), playBtn = document.getElementById('play');
const frames = DATA.frames || [];
// 帧里只有位置与状态，阵营在 targets 定义里：先建索引，否则所有目标都会被
// 画成同一种颜色（第一版就踩了这个坑，红方整批消失）。
const SIDE = {};
(DATA.targets || []).forEach(function(t){ if (t.id) SIDE[t.id] = t.affiliation || ''; });
function sideOf(row){ return row.affiliation || SIDE[row.id] || ''; }
let idx = 0, timer = null;

function bounds() {
  let w=1e9,s=1e9,e=-1e9,n=-1e9;
  (DATA.targets||[]).forEach(function(t){
    if (t.lon==null||t.lat==null) return;
    w=Math.min(w,t.lon); e=Math.max(e,t.lon);
    s=Math.min(s,t.lat); n=Math.max(n,t.lat);
  });
  if (w>e||s>n) return {w:119,s:22,e:121,n:24};
  const padX=(e-w)*0.08||0.05, padY=(n-s)*0.08||0.05;
  return {w:w-padX,s:s-padY,e:e+padX,n:n+padY};
}
const B = bounds();

function project(lon,lat){
  const x=(lon-B.w)/(B.e-B.w)*cv.width;
  const y=cv.height-(lat-B.s)/(B.n-B.s)*cv.height;
  return [x,y];
}

function draw(){
  const f = frames[idx] || {targets:[]};
  ctx.clearRect(0,0,cv.width,cv.height);
  const rows = f.targets || [];
  // 轨迹线：把该帧之前的位置连起来
  ctx.lineWidth = 1;
  const hist = frames.slice(Math.max(0, idx-120), idx+1);
  rows.forEach(function(t){
    if (sideOf(t) !== 'red') return;
    ctx.strokeStyle = 'rgba(200,53,44,.45)';
    ctx.beginPath();
    let started=false;
    hist.forEach(function(h){
      (h.targets||[]).forEach(function(p){
        if (p.id!==t.id||p.lon==null) return;
        const q=project(p.lon,p.lat);
        if(!started){ctx.moveTo(q[0],q[1]);started=true;} else ctx.lineTo(q[0],q[1]);
      });
    });
    ctx.stroke();
  });
  rows.forEach(function(t){
    if (sideOf(t) !== 'blue') return;
    ctx.strokeStyle = 'rgba(53,102,159,.45)';
    ctx.beginPath();
    let started=false;
    hist.forEach(function(h){
      (h.targets||[]).forEach(function(p){
        if (p.id!==t.id||p.lon==null) return;
        const q=project(p.lon,p.lat);
        if(!started){ctx.moveTo(q[0],q[1]);started=true;} else ctx.lineTo(q[0],q[1]);
      });
    });
    ctx.stroke();
  });
  // 当前位置
  rows.forEach(function(t){
    if (t.lon==null||t.lat==null) return;
    const q=project(t.lon,t.lat);
    ctx.beginPath();
    ctx.arc(q[0],q[1], t.destroyed? 2.5 : 4, 0, Math.PI*2);
    const side = sideOf(t);
    ctx.fillStyle = t.destroyed ? '#6b7785'
      : (side==='red' ? '#c8352c' : (side==='blue' ? '#35669f' : '#9aa4b2'));
    ctx.fill();
  });
  document.getElementById('tick').textContent =
    'tick ' + idx + ' / ' + Math.max(0, frames.length-1);
  document.getElementById('hud').innerHTML =
    '<b>' + (DATA.case_id||'') + '</b>'
    + '<span>run ' + (DATA.run_id||'') + '</span>'
    + '<span>seed ' + (DATA.seed==null?'-':DATA.seed) + '</span>'
    + '<span>目标 ' + (DATA.targets||[]).length + '</span>'
    + '<span>事件 ' + (DATA.events||[]).length + '</span>';
}

function renderEvents(){
  const box=document.getElementById('events');
  const now = idx;
  const rows=[]; let shown=0;
  (DATA.events||[]).forEach(function(e){
    const t = Number(e.tick);
    if (!(t<=now) || shown>=60) return;
    shown++;
    rows.push('<div class="ev"><span class="t">T' + t + '</span>'
      + (e.type||'') + (e.description? '：'+e.description : '') + '</div>');
  });
  box.innerHTML = rows.reverse().join('') || '<div class="ev">尚无事件</div>';
}

function setIdx(v){
  idx = Math.max(0, Math.min(frames.length-1, v|0));
  seek.value = String(idx);
  draw(); renderEvents();
}

seek.max = String(Math.max(0, frames.length-1));
seek.addEventListener('input', function(){ setIdx(Number(seek.value)); });
playBtn.addEventListener('click', function(){
  if (timer) { clearInterval(timer); timer=null; playBtn.textContent='播放'; return; }
  playBtn.textContent='暂停';
  timer = setInterval(function(){
    if (idx >= frames.length-1) { clearInterval(timer); timer=null; playBtn.textContent='播放'; return; }
    setIdx(idx+1);
  }, 100);
});
function resize(){
  cv.width = cv.clientWidth; cv.height = cv.clientHeight;
  draw();
}
window.addEventListener('resize', resize);
resize(); setIdx(0);
</script>
"""


def render(payload: Dict[str, Any], title: Optional[str] = None) -> str:
    """生成完整的单文件 HTML。数据用 JSON 内联，转义 ``</script>`` 防截断。"""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # 内联进 <script> 时唯一的危险序列是 "</script>"，转义斜杠即可。
    data = data.replace("</", "<\\/")
    heading = html.escape(title or (payload.get("case_id") or "推演回放"))
    body = _PLAYER.replace("__DATA__", data)
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>" + heading + "</title></head><body>" + body + "</body></html>"
    )
