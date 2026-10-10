# C5 地图质量排查：根因与证据（2026-10-10）

> 本文修正了 `reports/shenji_agent_promotion_v3_report.md` 对空路径问题的**根因归因**。
> 结论：**map_212 的邻接数据没有缺边**；真正的三个成因是
> ①**它是 16 张地图里唯一缺 `neighbors` 键的**（导致回退路径抛异常）；
> ②**场景→地图解析**曾把它错配给不属于它的场景（已由前序工作修复并实测）；
> ③评估里**没有空路径率指标**，症状长期不可监控。

---

## 1. 先用正确的坐标编码重算（此前结论被推翻的过程）

`ai/agent.py` 的 `Map.is_valid()` 是 `row, col = divmod(pos, 100)`（`:80`），
即 **`pos = row*100 + col`**。

**我第一次用 `r*列数+c` 解码，得出「7153 对邻接不对称、7301 处悬空」——那是错的**，
纯属编码假设错误产生的伪影。改用 `divmod(pos,100)` 后：

| 地图 | 格子 | 有向邻接边 | 不对称 | 悬空 | 自环 | 孤立节点 | 连通分量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **map_212** | 2500 | 14602 | **0** | **0** | **0** | **0** | **1（2500/2500）** |
| map_19 | 4620 | 27178 | **0** | **0** | **0** | **0** | 1 |
| 其余 14 张 | — | — | **0** | **0** | **0** | **0** | 1 |

→ **16 张地图的邻接表在结构上全部干净**。promotion 报告里
「根因判为地图 neighbor 缺边」**不成立**。

（验算示例：cell(0,0) 的 `nb=[1,-1,-1,-1,-1,100]` → `divmod(1,100)=(0,1)` 东邻、
`divmod(100,100)=(1,0)` 南邻，与 oddr 六邻方位完全一致。）

---

## 2. 成因一：`map_212` 是唯一缺 `neighbors` 键的地图（已复现）

`Map.get_neighbors()`（`ai/agent.py:86-90`）读的是 **`cell["neighbors"]`**。

**16 张地图里 15 张有 `neighbors`，只有 `map_212` 没有** —— 它用的是简写键 `nb`：

| 地图 | `neighbors` | `nb` |
| --- | --- | --- |
| map_123 / 221 | ✅ | ✅ |
| map_19、21、29、43、53、82、83、84、86、92、94、96、9601 | ✅ | — |
| **map_212** | **❌** | **✅** |

**实测复现（`source py310env.sh`，numpy 1.26.4 / py 3.10.9）**：

```text
=== map_212 ===
  get_neighbors(0)          RAISED KeyError: 'neighbors'
  get_neighbors(100)        RAISED KeyError: 'neighbors'
  gen_move_route_bfs(0,101) RAISED KeyError: 'neighbors'
  gen_move_route(0,101,0)   -> [100, 101]          ← cost 路径正常
=== map_19（规范键）===
  get_neighbors(0)          -> [1, -1, -1, -1, -1, 100]
  gen_move_route_bfs(0,101) -> [1, 101]
```

### 2.1 为什么这会造成**静默**空路径

两条路径取邻接的来源**不同**：

| 路径 | 邻接来源 | map_212 上 |
| --- | --- | --- |
| `gen_move_route()`（主路径） | `self.cost[mode][row][col]`（**cost.pickle**） | **正常** |
| `gen_move_route_bfs()`（回退路径） | `self.get_neighbors()` → `cell["neighbors"]` | **抛 KeyError** |

而 `map_212` 的 `cost.pickle[0][0][0] = {1: 3, 100: 3}` 与 `basic.json` 的 `nb` **邻接 id 完全对齐**，
所以**主路径能跑通、只在回退时炸**。回退路径的调用点是 `ai/agent.py:901`
（`bfs_route = self.map.gen_move_route_bfs(bop["cur_hex"], city_id)`）。
这正是「静默产生空路径」的典型形态：**主路径掩盖了数据缺陷，直到某个场景走回退才暴露**。

---

## 3. 成因二：场景→地图解析曾把 map_212 错配（前序工作已修复并实测）

`.agent/round18_verify_212051_override_20260426_080830/` 的实测记录：

| 指标 | 修前（错配到 map_212） | 修后（解析到 map_21） |
| --- | --- | --- |
| move coverage | 0.0388 | **1.0000** |
| learned coverage | 0.0436 | **1.0000** |
| `empty_route_with_no_reachable_city` | 1710 | **0** |
| cities 在界内 | 2/5 | **5/5** |
| operators 在界内 | 37/58 | **58/58** |

同类记录还有 `2120131194.json`（→ `map_94`：move 0.0291 → 1.0000，no_reachable 1634 → 0）
与 `2120131294.json`（→ `map_94`：0.0185 → 1.0000，2394 → 0）。

**读法**：这些场景的坐标**在 map_212 上大量越界**（cities 2/5、operators 37/58），
所以「空路径」不是因为 212 缺边，而是因为**它本来就不是这些场景的地图**。

---

## 4. 成因三：评估里没有空路径率指标

`DATA_QUALITY_AUDIT.md` 记「canonical_v4 从原始 raw episode 一次性生成（**56 episodes，15 map_212 跳过**）」
—— 已经**跳过了 15 个 map_212 episode**，但那是**数据侧**的规避，
**评估输出里仍没有「空路径率」这个可监控指标**，所以 66% 这个数字只能靠事后统计发现。

---

## 5. 全量地图质量扫描（新增工具，C5 步骤 2/6 交付）

新增 `scripts/check_map_quality.py`（可直接跑）：

```text
map          id     grid      cells  edges   asym     iso     verdict
map_123      123    100x100   10000  59202   0        0       OK
map_19       19     70x66     4620   27178   0        0       OK
map_21       21     92x77     7084   41830   0        0       OK
map_212      212    50x50     2500   14602   0        0       FAIL
map_221      221    70x60     4200   24682   0        0       OK
map_29/43/53/82/83/84/86/92/94/96/9601                        OK
FAIL map_212
     - 缺少 cell['neighbors'] —— ai/agent.py Map.get_neighbors() 会 KeyError（本图用的是别名键 ['nb']）
RESULT: 1/16 maps FAIL
```

`--strict` 时有 FAIL 即非 0 退出，可作质量门。

---

## 6. 三个成因对应的处置建议

| 成因 | 建议处置 | 我的本轮动作 |
| --- | --- | --- |
| ① map_212 缺 `neighbors` 键 | **两种可选**：(a) 给 map_212 补 `neighbors` 键（**不动** `nb`，值同源、机械等价）；(b) 黑名单排除 map_212 并记录 | **未改数据**（见 §7 理由）；已由扫描器**持续暴露** |
| ② 场景→地图错配 | 已有修复；应补**回归用例**钉住「212xxxx → map_21/map_94」 | 未做（属测试新增） |
| ③ 无空路径率指标 | 在评估输出里加 `empty_route_rate` 与 `empty_route_with_no_reachable_city` 计数 | **未做**（属 `training/evaluate.py` 改动） |

---

## 7. 为什么本轮**没有**动数据与运行期代码（如实说明）

1. **风险不对称**：C5 自己的【风险与回滚】写明「修改邻接数据会改变训练分布导致历史指标不可比」。
   而 `map_212` 在当前解析器下**已不再被正常场景选中**（成因二的修复），
   动它的收益不确定、风险却实在。
2. **已有两套一次性补丁脚本**：`scripts/_fix_empty_routes.py` 与 `scripts/_fix_empty_routes_v2.py`
   **用正则直接改写 `ai/agent.py`**。在它们之上再加一处运行期改动，
   会让我无法确定 `agent.py` 的最终形态是否仍可解释。**故本轮不碰 `ai/agent.py`。**
3. **一次只交付一件可验证的事**：本轮交付**扫描器 + 根因订正**（都可复验）；
   数据修复与评估指标应各自作为独立一步，并配套回归用例。

---

## 8. 边界

- 本文结论基于：`divmod(pos,100)` 编码（实读 `ai/agent.py`）、16 张地图的全量扫描、
  以及 `py310env.sh` 环境下的**实际复现**（KeyError 原文见 §2）。
- **未**对 `cost.pickle` 的**边权数值**做合理性判定（只验证邻接 id 对齐）——
  「边权是否合理」需要地图语义，属评审范畴。
- **未**运行 `test_move_route_diagnostics.py` 等测试（本轮未改运行期代码，故无回归面）。
