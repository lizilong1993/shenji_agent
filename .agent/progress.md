# Progress

## Round 1 Complete

- 时间：2026-04-23 UTC
- 当前循环轮次：1 已完成
- Round 1 commit：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- Round 1 假设：按文件名第二段修正 supervised map-id 合同
- Round 1 是否成功：是

## strongest agent

- strongest agent：accepted learned mainline
- strongest commit SHA：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- strongest weights snapshot 路径：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- strongest weights 已更新来源：`/remote-home/lizilong/shenji_agent/ai/weights`

## 资源检查与同步

- 本轮开始前做了哪些同步：
  - 修复 `git` safe.directory
  - 检查 `origin/main`
  - bootstrap strongest snapshot 到 `artifacts/best_agent/weights`
- 同步了哪些关键资源：
  - strongest weight snapshot
- 哪些资源已确认齐全：
  - `program.md`
  - `configs/`
  - `training/`
  - `scripts/`
  - `tests/`
  - `datasets/`
  - `land_wargame_sdk/Data/Data/`
  - `artifacts/best_agent/weights`
- 哪些资源仍缺失：
  - `AGENTS.md`
  - `references.bib`
- 缺失说明：
  - 上述两项不在工作树、git 历史或 `origin/main`，无可同步上游
- 本轮开始前资源是否齐全：训练与评测所需资源齐全
- 本轮是否做了同步：是

## Fresh verification evidence

- `python -m compileall ai training scripts tests`：通过
- `python -m unittest tests.test_mainline_contract -v`：8 项测试通过
- 数据合同探针：
  - `state_bucket_counts {'complex': 108}`
  - `intent_bucket_counts_sample {'medium': 512}`
- fresh supervised run：
  - 路径：`experiments/shenji_agent/20260423_024736`
  - `situation_bucket_counts = {'complex': 108}`
  - `intent_bucket_counts.val = {'medium': 1024}`

## Round 1 正式比较

- 正式比较目录：`experiments/shenji_agent/league_20260423_024928`
- 本轮地图列表是否重新生成：是
- 地图总数：100
- 是否存在补重复抽样：是
- 实际完成了多少项：100
- 最终 `map_list.json` 路径：`/remote-home/lizilong/shenji_agent/experiments/shenji_agent/league_20260423_024928/map_list.json`
- participant：
  - candidate=`/remote-home/lizilong/shenji_agent/ai/weights`
  - incumbent=`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- 最终汇总胜率：`1.0000`
- 本轮领先 strongest baseline 的幅度：`+0.5000` 相对中性线；相对 bootstrap strongest baseline 为明确提升
- revert 状态：无需 revert

## hard-stop 检查

- hard-stop 检查目录：`experiments/shenji_agent/league_20260423_032802`
- 本轮地图列表是否重新生成：是
- 地图总数：100
- 是否存在补重复抽样：是
- 实际完成了多少项：100
- 最终 `map_list.json` 路径：`/remote-home/lizilong/shenji_agent/experiments/shenji_agent/league_20260423_032802/map_list.json`
- hard-stop 结果：未通过
- `promotion.json.failures`：`['move_path_coverage']`
- hard-stop 最终汇总胜率：`1.0000`

## 当前距离 stop rule 还差什么

- `move_path_coverage >= 0.20` 目前未满足，当前为 `0.0`
- 两个连续窗口满足 hard-stop 的稳定性条件目前也未满足
- stronger-learned incumbent 的 compare protocol 仍未修正：
  - strongest snapshot 已是 learned weights
  - 但 `participants.json` 里的 incumbent 仍使用 `baseline` profile

## Round 2 入口

- 下一轮准备改的方向：strongest snapshot / incumbent profile 合同
- 原因：不先修正这个协议，后续 candidate-vs-strongest learned 比较不可信

## Round 2 Start

- 时间：2026-04-23 UTC
- 当前循环轮次：2
- strongest agent：accepted learned mainline
- strongest commit SHA：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- strongest weights snapshot 路径：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- 本轮开始前资源是否齐全：是
- 是否执行了同步：否
- 本轮资源检查结论：代码、配置、数据、strongest snapshot 均齐全，无需同步
- 本轮新增 strongest snapshot 元数据：
  - `artifacts/best_agent/weights/model_manifest.json -> snapshot_role=accepted_learned_mainline`
  - `ai/weights_incumbent_rule/model_manifest.json -> snapshot_role=bootstrap_rule_baseline`
- 本轮轻量验证：
  - `python -m compileall ai training scripts tests`：通过
  - `python -m unittest tests.test_mainline_contract -v`：10 项通过

## Paper Draft Task 2026-05-11

- 已读取项目 `AGENTS.md` 与 `.agent/context.md`。
- 已尝试执行规则指定 bootstrap，结果：指定脚本不存在，未产生本轮成功 bootstrap 记录。
- 已确认当前目录不是 Git 仓库，本轮无法提交 commit。
- 已确认 PaperOrchestra junction 目标缺失；已改用 `.orchestra/skills/20-ml-paper-writing`。
- 已确认独立 nature skill 不存在；已读取 `scientific-writing` 作为 Nature/Science 风格参考。
- 已复制 NeurIPS 本地模板文件到 `paper/`：
  - `paper/neurips.sty`
  - `paper/extra_pkgs.tex`
  - `paper/Makefile`
- 下一步：写入 `paper/main.tex`、`paper/references.bib`、`paper/source_evidence.md`、`paper/README.md` 并编译验证。

## Paper Draft Task Verification 2026-05-11

- 已写入论文产物：
  - `paper/main.tex`
  - `paper/references.bib`
  - `paper/source_evidence.md`
  - `paper/README.md`
  - `paper/figures/system_architecture.tex`
  - `paper/figures/promotion_gate.tex`
  - `paper/main.pdf`
- 引用验证：
  - DOI/arXiv BibTeX 已写入 `paper/references.bib`
  - `paper/main.blg` 显示 `You've used 8 entries` 且 `warning$ -- 0`
- 编译验证：
  - `latexmk -pdf -interaction=nonstopmode main.tex`：失败，原因是本机 TeX Live `runscript.tlu` wrapper 内部错误：`attempt to concatenate a nil value`
  - 手动链路 `pdflatex -> bibtex -> pdflatex -> pdflatex`：通过，生成 `paper/main.pdf`
  - `paper/main.log` 未发现 undefined citations、undefined references 或 LaTeX fatal error
- 治理收尾：
  - `complete-task` 指定脚本仍缺失，无法记录 workflow candidate
  - 当前目录不是 Git 仓库，无法执行 commit

## AI Reviewer Integration 2026-05-11

- 已补项目根 `AGENTS.md` 最小治理骨架；`.agent/context.md` 已存在。
- 已读取 `AGENTS.md` 与 `.agent/context.md`。
- 已尝试执行规则指定 bootstrap：
  - `C:\Users\lizilong\.codex\automations\weekly-skill-trigger-optimization\codex-runtime-ops.ps1`
  - 结果：脚本路径不存在，未产生本轮成功 bootstrap 记录。
- 已安装外部 Draft Detective：
  - 路径：`C:\Users\lizilong\Desktop\WHUAI\tools\draft-detective`
  - git head：`9292652`
- 已新增 Shenji 适配层：
  - `configs/ai_reviewer.yaml`
  - `scripts/run_ai_reviewer.py`
  - `references.bib`
  - `reports/ai_reviewer/*`
- 本地验证：
  - `python -m compileall scripts`：通过
  - `python scripts\run_ai_reviewer.py --dry-run --config configs\ai_reviewer.yaml`：通过，生成 40 条 claim，Draft Detective 存在
  - `python scripts\run_ai_reviewer.py --config configs\ai_reviewer.yaml`：通过，最新 league 为 `experiments/shenji_agent/league_20260501_170204`
  - 输出统计：supported=2，partially_supported=23，unsupported=0，unverifiable=15
- 远程同步/验证：
  - 已成功同步：`AGENTS.md`、`references.bib`、`configs/ai_reviewer.yaml`、`scripts/run_ai_reviewer.py`
  - 未能同步：远程 `.agent/context.md`、`.agent/progress.md`、`.agent/findings.md` 归 `root:root`，当前 SSH 用户无覆盖权限
  - 未能写入默认远程输出：远程 `reports/` 归 `root:root`，当前 SSH 用户无法创建 `reports/ai_reviewer`
  - 已用覆盖输出目录验证远程脚本：`/remote-home/lizilong/venvs/shenji/bin/python scripts/run_ai_reviewer.py --dry-run --config configs/ai_reviewer.yaml --output-dir .agent/ai_reviewer_remote_probe`：通过，生成 40 条 claim，unsupported=0，unverifiable=15
  - 远程 Draft Detective 状态为未安装；按计划远程不默认安装 Draft Detective
- 外部工具最小验证：
  - `uv --version`：`uv 0.9.22`
  - Draft Detective `README.md`、`pyproject.toml`、`docker-compose.yml` 存在
  - 未执行真实 LLM 审稿；原因：未配置 Draft Detective 运行时服务/API key，本轮只验证外部仓库安装与本地证据审稿适配器。
- 当前目录不是 Git 仓库，无法执行 `git commit`。
