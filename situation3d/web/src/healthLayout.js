/**
 * healthLayout.js — 血条的屏幕避让排布（纯函数，可在 Node 里直接测）
 *
 * 问题
 * ----
 * `app.js` 给每个目标的血条写死 `pixelOffset: (0, 17)`，谁都不让谁。标牌早就
 * 有 `declutterLabels()` 做避让，血条没有。密集编队时（`taiwan_dense_transfer_v1`
 * 有 80 个目标）血条会叠成一条，读不出谁是谁的血。
 *
 * 做法
 * ----
 * 与标牌避让同一套思路，但单独一路：血条比标牌小、锚在图标正下方，需要
 * 的候选位是"绕锚点找最近的空位"，而不是"按行往下排"。gp-rts 的
 * `layoutHealthBars()` 用了 9 行 × 3 列 + 3 行下方候选、代价
 * `|dx| + |dy|*1.1`；这里沿用这个形状，但：
 *   - 保持"贴锚点"优先：代价里的纵向权重更大，宁可横挪也不想离目标太远；
 *   - 加 `maxOffset` 上限：被推到 40 px 以外就干脆不画，而不是画到别人身上；
 *   - 视口边界钳制后仍然超出的直接丢弃。
 *
 * 纯函数，不碰 Cesium：`app.js` 负责投影取位，这里只算"该放哪"。
 */
'use strict';

(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.HealthLayout = api;
}(typeof window !== 'undefined' ? window : null, function () {

  var DEFAULT_OPTIONS = {
    // 血条与锚点之间的基础间距（图标半高 + 2 px），与 app.js 的 17 px 一致。
    gapPx: 17,
    // 候选位步进：血条高 4 px，步进 5 px + 2 px 间距就能一行行错开；
    // 取 6 px 会在 44 px 上限内只挤出 5 行，同一位置的第 3 个血条就无处可放。
    stepX: 36,
    stepY: 5,
    rows: 12,
    // 横向寻找距离的倍数，-N..N。
    cols: 2,
    // 离锚点超过这个像素距离就不画了（宁可没有，也不画到别人身上）。
    maxOffsetPx: 68,
    // 视口内边距。
    marginPx: 4,
    padPx: 2,
  };

  function rectsOverlap(a, b, pad) {
    return !(a.x + a.w + pad < b.x || b.x + b.w + pad < a.x ||
             a.y + a.h + pad < b.y || b.y + b.h + pad < a.y);
  }

  /**
   * @param {Array} bars    [{id, anchorX, anchorY, w, h, priority}]
   *                        anchorX/Y 是血条**默认位置的左上角**（已含 gapPx），
   *                        不是图标中心——调用方先做好这一步，本函数只管避让，
   *                        这样 Node 测试可以直接给坐标，不必知道 gapPx。
   * @param {Object} viewport {w, h}
   * @param {Array}  obstacles 已占用的矩形 [{x,y,w,h}]（例如编队图标）
   * @param {Object} options 覆盖 DEFAULT_OPTIONS
   * @returns {Array} [{id, x, y, w, h}]，按 priority 降序；放不下的不出现。
   */
  function layout(bars, viewport, obstacles, options) {
    var o = {};
    for (var k in DEFAULT_OPTIONS) o[k] = DEFAULT_OPTIONS[k];
    if (options) for (var k2 in options) o[k2] = options[k2];

    var occupied = (obstacles || []).slice();
    var out = [];

    // 优先级降序；同优先级按屏幕纵向位置、再按 id 定序，保证同一输入同一输出。
    var sorted = (bars || []).slice().sort(function (a, b) {
      return (b.priority - a.priority)
        || (a.anchorY - b.anchorY)
        || (String(a.id) < String(b.id) ? -1 : 1);
    });

    sorted.forEach(function (bar) {
      var w = bar.w, h = bar.h;
      var best = null;
      for (var row = 0; row <= o.rows && !best; row++) {
        var dy = row * o.stepY;
        for (var col = 0; col <= o.cols * 2; col++) {
          // 0, -1, +1, -2, +2 … 由近及远
          var c = (col === 0) ? 0 : (col % 2 ? -Math.ceil(col / 2) : col / 2);
          var dx = c * o.stepX;
          var rect = {
            x: Math.round(bar.anchorX + dx),
            y: Math.round(bar.anchorY + dy),
            w: w, h: h,
          };
          if (rect.x < o.marginPx || rect.y < o.marginPx ||
              rect.x + rect.w > viewport.w - o.marginPx ||
              rect.y + rect.h > viewport.h - o.marginPx) continue;
          var offset = Math.abs(dx) + Math.abs(dy) * 1.1;
          if (offset > o.maxOffsetPx) continue;
          var hit = false;
          for (var i = 0; i < occupied.length; i++) {
            if (rectsOverlap(rect, occupied[i], o.padPx)) { hit = true; break; }
          }
          if (hit) continue;
          best = rect;
          break;
        }
      }
      if (!best) return;
      occupied.push(best);
      out.push({ id: bar.id, x: best.x, y: best.y, w: best.w, h: best.h });
    });

    return out;
  }

  /**
   * 血条相对锚点的 pixelOffset。
   *
   * 血条 billboard 是 `verticalOrigin: TOP` + 默认水平居中：它的**顶边中点**落
   * 在 anchor + offset 处，所以矩形左上角 x 对应 `offsetX - w/2`、y 对应
   * `offsetY`。反过来解出 offset 即为下式。
   */
  function offsetFor(layoutRect, bar) {
    return {
      x: Math.round(layoutRect.x - bar.anchorX),
      y: Math.round(layoutRect.y - bar.anchorY),
    };
  }

  return {
    DEFAULT_OPTIONS: DEFAULT_OPTIONS,
    layout: layout,
    offsetFor: offsetFor,
    rectsOverlap: rectsOverlap,
  };
}));
