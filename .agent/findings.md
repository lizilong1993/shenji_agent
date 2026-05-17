# Findings

## Round 1 关键结果

1. Round 1 假设成立。
   - `training/data.py` 改为按文件名第二段解析 map id 后，fresh supervised run `experiments/shenji_agent/20260423_024736/slice_report.json` 显示：
     - `intent_bucket_counts.val = {'medium': 1024}`
     - `situation_bucket_counts = {'complex': 108}`
   - `state graph` 不再被误归到 `simple` 桶。
2. Round 1 正式比较成功。
   - `experiments/shenji_agent/league_20260423_024928`
   - `map_list.json` 含 100 个 ordered paired 项，`duplicates_used = true`
   - `match_results.json` 含 100 个 paired 结果
   - candidate vs bootstrap strongest baseline 的 `overall.point_estimate = 1.0`
   - 因此本轮 candidate 胜过当前 strongest baseline，已接受为新的 strongest snapshot
3. hard-stop 仍未满足。
   - `experiments/shenji_agent/league_20260423_032802`
   - 100 paired 协议有效完成
   - `promotion.json.failures = ['move_path_coverage']`
   - 其余阈值都满足，但 `move_path_coverage = 0.0 < 0.20`

## 当前最关键新瓶颈

下一轮最关键问题已经从数据合同转移为 strongest snapshot / self-play protocol：

- Round 1 成功后 strongest snapshot 已更新到 `artifacts/best_agent/weights`
- 但 hard-stop 窗口的 `participants.json` 仍显示 incumbent profile = `baseline`
- `ai/policy.py` 中 `LearnedPolicy.configure()` 对 `policy_profile == "baseline"` 直接 return，不加载模型
- 这意味着后续 arena 并没有真正拿 strongest learned snapshot 当 incumbent，而是继续拿规则回退当 incumbent

在修正这个协议前，“candidate 是否打赢当前 strongest learned agent”这个核心问题并没有被真实比较。

## 为什么下一轮选 strongest snapshot / incumbent profile 这条线

- 它直接影响用户要求的唯一默认比较规则：candidate 必须打 strongest learned incumbent，而不是规则回退。
- 如果这条线不先修，后续 win-rate 提升都不能严格解释为“打赢当前 strongest agent”。
- 这属于单一、高价值、可归因的 self-play gate / strongest snapshot 管理问题。

## 关键证据来源

- `program.md`
- `configs/selfplay_baseline.yaml`
- `configs/selfplay_sota_stop.yaml`
- `training/data.py`
- `training/supervised.py`
- `training/selfplay.py`
- `training/evaluate.py`
- `ai/policy.py`
- `tests/test_mainline_contract.py`
- `experiments/shenji_agent/20260423_024736/slice_report.json`
- `experiments/shenji_agent/league_20260423_024928/rating.json`
- `experiments/shenji_agent/league_20260423_024928/promotion.json`
- `experiments/shenji_agent/league_20260423_032802/rating.json`
- `experiments/shenji_agent/league_20260423_032802/promotion.json`
- `experiments/shenji_agent/league_20260423_032802/participants.json`

## 已验证 / 未验证

- 已验证：
  - Round 1 数据合同修正后，fresh supervised buckets 合理化。
  - Round 1 candidate 在正式 100 paired 对抗中打赢 bootstrap strongest baseline。
  - hard-stop 当前只差 `move_path_coverage`。
  - strongest snapshot 更新后，incumbent 仍按 `baseline` profile 运行，未真实加载 strongest learned weights。
- 未验证：
  - 修正 incumbent profile / strongest snapshot 合同后，candidate-vs-strongest learned arena 的真实胜率。
  - 在真实 strongest learned incumbent 下，`move_path_coverage = 0.0` 是否仍然存在。

## Paper Draft Evidence 2026-05-11

### 已验证

- 当前本地目录不是 Git 仓库：`git status --short --branch` 返回 `fatal: not a git repository`。
- `paper-orchestra` junction 存在，但目标 `SKILL.md` 缺失；可读的论文写作入口是 `.orchestra/skills/20-ml-paper-writing/SKILL.md`。
- 可读 NeurIPS 模板来自 `.orchestra/skills/20-ml-paper-writing/templates/neurips2025/`。
- `program.md` 明确项目没有 protocol-matched public leaderboard，因此论文不能声称公开 SOTA。
- 正式结果目录 `experiments/shenji_agent/league_20260501_170204` 包含：
  - `summary.md`
  - `rating.json`
  - `promotion.json`
  - `participants.json`
  - `exploitability.json`
- 正式结果显示：
  - `overall.point_estimate = 0.87`
  - `overall.ci95[0] = 0.800000011920929`
  - `complex.point_estimate = 0.8181818181818182`
  - `complex.ci95[0] = 0.6363636255264282`
  - `learned_path_coverage = 0.37277940409832777`
  - `move_path_coverage = 0.3366836301950806`
  - `shoot_path_coverage = 1.0`
  - `accepted = true`
  - `max_exploiter_win_rate = 0.25`
  - `battle_phase_entered = true`
  - `selected_move_shoot_occupy_actions = 94682`
- `participants.json` 显示 candidate 使用 `CurrentAI/challenger/ai/weights`，incumbent 使用 `IncumbentAI/baseline/ai/weights_incumbent_rule`，且后端是 `train_env.TrainEnv`。
- `ai/models.py` 中可见 LSTM 意图模型、GraphSAGE-style situation model、tactical scorer。
- `ai/policy.py` 中可见模型加载、feature normalizer、policy diagnostics、move/shoot coverage 统计和 objective scoring diagnostics。

### 未验证/阻塞

- 本地 `training/` 目录当前只有 `cache/`，`scripts/` 和 `tests/` 为空；`reports/experiment_results.md` 提到的训练/评估脚本与 `139 passed` 不能在当前本地树复跑。
- 指定 bootstrap/complete-task 脚本缺失，无法按治理规则记录本轮 lifecycle。
- 当前任务无法做 Git commit，因为当前目录不是 Git 仓库。

## AI Reviewer Evidence 2026-05-11

### 已验证

- Draft Detective 已作为外部项目放在 `C:\Users\lizilong\Desktop\WHUAI\tools\draft-detective`，没有复制进 Shenji 仓库。
- Draft Detective 本地 git head 为 `9292652`，且存在 `README.md`、`pyproject.toml`、`docker-compose.yml`。
- `scripts/run_ai_reviewer.py` 使用标准库完成：
  - 读取 `configs/ai_reviewer.yaml`
  - 发现最新 `experiments/shenji_agent/league_*`
  - 检查 `rating.json`、`promotion.json`、`exploitability.json`、`participants.json`、`map_list.json`、`summary.md`
  - 输出 claim JSONL、unsupported/unverifiable 清单、TSV 证据矩阵和 Draft Detective 输入包
- `experiments/shenji_agent/league_20260501_170204/map_list.json` 是 dict，正式 map 项在 `items` 字段，数量为 100；脚本已兼容顶层 list 与 `items` dict 两种形态。
- 本轮生成的 `reports/ai_reviewer/review_summary.md` 明确 `LLM review executed: False`，避免把本地规则审查冒充为 Draft Detective LLM 审稿。
- 远程 `/remote-home/lizilong/shenji_agent/reports` 归 `root:root`，当前 SSH 用户不能创建 `reports/ai_reviewer`；脚本新增 `--output-dir` 后可写入 `.agent/ai_reviewer_remote_probe` 完成远程 dry-run。
- 远程 `/remote-home/lizilong/shenji_agent/.agent/context.md`、`findings.md`、`progress.md` 归 `root:root`，当前 SSH 用户不能覆盖；`.agent/` 目录本身可新建子目录。

### 未验证/阻塞

- 未运行 Draft Detective 官方 `uv run pytest` 或服务端真实审稿；该项目依赖较重且需要运行时/API 配置，本轮验证范围限于外部仓库安装面和 Shenji 本地适配器。
- 规则指定 bootstrap/complete-task 脚本路径缺失，不能写入本轮 lifecycle 状态。
- 当前目录不是 Git 仓库，不能提交 commit。
