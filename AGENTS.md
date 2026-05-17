# Shenji Agent Rules

## 工作方式

- 先读本文件，再读 `.agent/context.md`。
- 先规划、执行、验证，再汇报；未知写未知，未验证写未验证。
- 项目稳定事实写入 `.agent/context.md`，任务过程证据写入 `.agent/progress.md` 或 `.agent/findings.md`。
- 复杂任务可使用 `.agent/task_plan.md`、`.agent/findings.md`、`.agent/progress.md` 和 `.agent/plans/`；简单任务不为凑流程新增文件。
- 默认由主 agent 直接完成可控工作；只有任务复杂度和并行收益明确时才启用 sub-agents。不要恢复“所有具体任务必须委派 subagents”的旧规则。
- 不再要求任务开始前运行 `codex-runtime-ops.ps1 -Action bootstrap-task`；该脚本在当前环境不可用，不能作为进入执行态的前置条件。
- 修改后能跑就跑、能测就测；没有 fresh verification evidence，不要声称已修复或已完成。
- 有可提交变更且仓库状态安全时，任务收尾执行一次 `git commit`；若不能提交，必须说明原因。

## 项目真值

- 研究主真值以 `.agent/context.md` 当前记录为准。
- 停机和 SOTA-style 结论以 `program.md`、`configs/selfplay_sota_stop.yaml`、正式 `experiments/shenji_agent/league_*` sidecars 为准。
- 重要 sidecars 包括 `rating.json`、`promotion.json`、`exploitability.json`、`participants.json`、`match_results.json`、`map_list.json`、`summary.md`。
- 不能用 README、旧计划、单次训练 loss、单个轻量评估或 mock 结果替代正式 league 证据。
- 当前 AI 审稿人入口为 `scripts/run_ai_reviewer.py --config configs/ai_reviewer.yaml`；它只做研究证据审查和 Draft Detective 输入打包，不替代正式训练、评测或 hard-stop 判定。

## 远程与同步

- 默认远程工作根：`/remote-home/lizilong/shenji_agent`。
- 默认远程 Python：`/remote-home/lizilong/venvs/shenji`。
- 远程入口通常为 `ssh lizilong@146.56.220.99 -p 21431`；本机 Windows 下优先使用 Git Bash `ssh.exe` / `scp.exe`。
- 需要远程验证时，命令按容器/远程用户可执行方式写，不假设宿主机权限。
- 常规小文件变更由 agent 负责同步；大体量数据、权重、实验产物不默认交互式上传。
- 最终说明必须包含远程执行步骤，或明确写“远程无需执行”。

## 文件治理

- 不再依赖或新增项目根 `PROJECT_CONTEXT.md`。
- 不在项目根新增 `task_plan.md`、`findings.md`、`progress.md` 作为新工作面；已有历史文件只按需要读取。
- `references.bib` 是本项目文献入口。实际影响分析、设计、训练、评测、SOTA 对标或 benchmark 判断的文献，应写入该文件并补 `project_note`。
- `reports/ai_reviewer/` 是本地 AI 审稿输出目录；远程若 `reports/` 无写权限，可用 `--output-dir .agent/<run_dir>` 临时输出。
- **Paper 代码隔离**：所有与论文撰写、审稿、评分相关的脚本、配置、输出文件必须放在 `paper/` 目录下。具体而言：
  - `paper/scripts/` — 论文专用脚本（审稿、评分、生成）
  - `paper/configs/` — 论文专用配置
  - `paper/reports/ai_reviewer/` — 论文审稿输出
  - 项目根目录的 `scripts/` 和 `configs/` 不应包含论文专用代码；根目录 `scripts/run_ai_reviewer.py` 只是一个向后兼容的转发器
  - 论文审稿应始终从 `paper/` 目录运行：`cd paper && python scripts/run_ai_reviewer.py --config configs/ai_reviewer.yaml [--llm]`
