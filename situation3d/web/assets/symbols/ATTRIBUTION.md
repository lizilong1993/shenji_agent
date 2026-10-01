# 符号图形署名

本目录下的 SVG 由 `tools/gen_symbols.py` 生成，包含两部分来源：

1. **帧形**：milsymbol 3.0.4（MIT）生成，实现 MIL-STD-2525D/E 的符号结构。
2. **帧内平台图形**：来自 [game-icons.net](https://game-icons.net/)，
   经 `@iconify-json/game-icons` 1.2.4 分发，许可 **CC-BY-3.0**，
   **要求署名** —— 即下表。

| 平台（本项目） | game-icons 键 | 作者 |
|---|---|---|
| `destroyer` | `battleship` | game-icons.net contributors |
| `frigate` | `iron-hulled-warship` | game-icons.net contributors |
| `replenishment` | `cargo-ship` | game-icons.net contributors |
| `corvette` | `interceptor-ship` | game-icons.net contributors |
| `coastal_missile` | `missile-launcher` | game-icons.net contributors |
| `sam_battalion` | `missile-pod` | game-icons.net contributors |
| `armor` | `tank` | game-icons.net contributors |
| `artillery` | `field-gun` | game-icons.net contributors |
| `fighter` | `jet-fighter` | game-icons.net contributors |
| `aew` | `radar-dish` | game-icons.net contributors |

> ⚠ **注意**：Iconify 分发的 `game-icons` 数据里**没有携带逐图标的作者字段**（`author.name` 为空），所以上表只能归到 `game-icons.net contributors`。
> game-icons.net 的每个图标页各自标注了作者。若需要**严格满足** CC-BY-3.0 的署名要求（点名到具体作者），需要按 `game-icons 键` 到 game-icons.net 逐个人工核对并补进上表。
> 当前做法对内部/学术用途足够；对外发布前应补齐。

CC-BY-3.0 全文：<https://creativecommons.org/licenses/by/3.0/>
