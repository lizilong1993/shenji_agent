/**
 * 渲染调度：把"每帧无条件重算"改成"有事才重算"。
 *
 * 问题
 * ----
 * app.js 的帧循环里 `syncEntities()` 是**无条件每帧执行**的：
 *
 *   function loop(nowMs) {
 *     ...
 *     syncEntities();            // ← 不管有没有变化都跑
 *     requestAnimationFrame(loop);
 *   }
 *
 * syncEntities 本身不重（就是给每个目标写几个 .show 标志、必要时累积一个航迹点），
 * 但它是幂等的：没有变化时跑一遍的结果与上一帧完全一样。投影数据量大、图层变多
 * 之后，"什么都没变却每帧遍历全部目标"就是纯粹的浪费，而且会让帧时间随目标数
 * 线性上涨。GEV 用 renderGovernor 解决同一件事，这里借其思路，但**不照搬**
 * 它那套自绘渲染循环——Cesium 自己的渲染循环已经处理好了瓦片调度、时钟、
 * 尺寸变化，我们不接管它，只去掉自己这层空转。
 *
 * 做法：脏标记 + 一次性消费
 * --------------------------
 *   mark(reason)  声明"状态变了，需要重算"
 *   take()        取出"是否需要重算"并清除标记
 *
 * 必须在**所有**会改变 syncEntities 结果的入口调 mark：时间轴跳转、播放推进、
 * 目标数据装载、图层/过滤开关、选中变化。漏一处的表现是"改了不生效、刷新才生效"，
 * 比性能问题难查得多——所以 reason 会被记下来，debug() 能看出最近是谁引起的。
 *
 * 为什么不用 Cesium 的 requestRenderMode
 * -------------------------------------
 * Cesium 1.145 确实内置了 `requestRenderMode`，但那个模式要求**每一处**改动场景
 * 的地方都调 `scene.requestRender()`，漏一处就是永久不刷新。本项目动场景的地方
 * 有二十多处（图元、实体、影像层、地形、天气、overview），逐个改的风险远大于
 * 收益。这里只裁掉自己这层空转，把 Cesium 的渲染循环原样保留，改动面最小。
 */
(function () {
  'use strict';

  var dirty = true;              // 首帧必须先跑一次
  var reasons = [];              // 本轮累积的原因（去重后保留最近几条）
  var stats = { marks: 0, runs: 0, skips: 0 };

  function mark(reason) {
    dirty = true;
    stats.marks++;
    var r = String(reason || 'unspecified');
    if (reasons.indexOf(r) < 0) {
      reasons.push(r);
      if (reasons.length > 8) reasons.shift();
    }
  }

  /**
   * 取用脏标记：返回 true 表示"这一帧需要重算"，并清除标记。
   * 调用方据此决定跑不跑重活。
   */
  function take() {
    if (!dirty) { stats.skips++; return false; }
    dirty = false;
    reasons.length = 0;
    stats.runs++;
    return true;
  }

  /** 强制下一次 take() 返回 true（例如重建场景之后）。 */
  function invalidate(reason) { mark(reason || 'invalidate'); }

  function isDirty() { return dirty; }

  function debug() {
    return {
      dirty: dirty,
      pendingReasons: reasons.slice(),
      stats: { marks: stats.marks, runs: stats.runs, skips: stats.skips },
      skipRatio: (stats.runs + stats.skips) > 0
        ? +(stats.skips / (stats.runs + stats.skips)).toFixed(3) : null,
    };
  }

  function reset() { dirty = true; reasons.length = 0; reasons.push('reset'); stats = { marks: 0, runs: 0, skips: 0 }; }

  window.RenderGovernor = {
    mark: mark,
    take: take,
    invalidate: invalidate,
    isDirty: isDirty,
    debug: debug,
    reset: reset,
  };
})();