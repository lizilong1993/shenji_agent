#!/bin/bash
# 准备 JS 引擎（I08/I09 的**运行时**测试需要）。
#
# ## 为什么要装
#
# I08/I09 此前的断言是**源码结构级**（正则匹配"函数是否存在"），
# 于是 I09 判据 2 的「1.5s/0.8s 可配置且**修改后生效**」**无从核验** —— "生效"不是文本性质。
#
# 装上 `quickjs`（pure-wheel，无需编译）后，测试可以**加载并执行真实 JS 文件**，
# 从而对「改 CSS 变量 → JS 读到新值」做端到端核验。
#
# ## 为什么装在仓库内而非 venv
#
# 装进 `.tools/jsengine/`（**已 gitignore**）：
#   · 不污染项目 venv（13 MB 二进制不该混进依赖树）；
#   · 路径稳定，测试可直接引用，不依赖 `/tmp` 是否被清；
#   · 缺它时测试**跳过并说明**，不是静默通过。
#
# 用法：bash scripts/setup_js_engine.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO/.tools/jsengine"

if [ -d "$TARGET/quickjs" ]; then
  echo "[js-engine] already present: $TARGET"
  exit 0
fi

echo "[js-engine] installing quickjs into $TARGET"
mkdir -p "$TARGET"
python3 -m pip install --quiet --target "$TARGET" quickjs
echo "[js-engine] done"
python3 -c "
import sys; sys.path.insert(0, '$TARGET')
import quickjs
c = quickjs.Context()
assert c.eval('1+1') == 2
print('[js-engine] verified: quickjs evaluates JS')
"
