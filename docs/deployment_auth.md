# 部署鉴权核对清单（MAC 绑定）

> **本文与 `docs/sdk_auth_runbook.md` 的关系**：runbook 是**操作手册**（怎么跑、跑哪条命令、什么算通过）；
> 本文是**核对清单**（部署前逐项确认什么、失败现象对照什么、哪些底线不能碰）。
> **两者若冲突，以 runbook 的「Legal Resolution Path」为准**——它记录的是实测过的合法路径。

---

## 1. 一句话结论 + 前置条件（本机现状，实测）

**本机鉴权不通过，原因是 MAC 不匹配**（`failed_mac_mismatch`），不是代码或仓库策略问题。

| 项 | 值 |
| --- | --- |
| 配置绑定 MAC | `<见 runbook>`（形如 `xx:xx:xx:xx:xx:xx`） |
| 当前容器 MAC | `<见 runbook>` |
| 结论 | 不匹配 → 引擎报 `did not pass authentication` |

> **注意**：runbook 里记录的那两个 MAC 是**容器虚拟网卡地址**，不是宿主真实网卡，也**不是机密**——
> 但它们仍**不写进本文**：本文沿用占位符，具体值请对照 runbook 与 `docker-compose.auth-mac.yml`。
> 但**`.engine_config` 本体、任何授权文件、密钥，一律不得入库**（见 §5）。

---

## 2. 前置条件：部署前逐项核对（按顺序过一遍）

| # | 核对项 | 怎么查 | 通过标准 |
| --- | --- | --- | --- |
| 1 | SDK 是否装在**当前**解释器下 | `python3.10 -c "import train_env; print(train_env.__file__)"` | 能导入，且路径属于本运行环境 |
| 2 | 配置文件在不在 | `ls -l /usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config` | 存在且可读 |
| 3 | 配置绑定的是哪个 MAC | `python3.10 scripts/check_sdk_auth_environment.py` 的报告字段 `bound_mac` | 与第 4 项一致 |
| 4 | 本运行环境的实际 MAC | `cat /sys/class/net/*/address` | 其中之一等于第 3 项 |
| 5 | 鉴权总判定 | 同上检查脚本 | 报告 `auth_environment_ready=true` |

**记住 runbook 给出的判定口径**：只有同时满足下列四条，一次运行才算**可用的晋升证据**：

1. `auth_status` 不再是 `failed_mac_mismatch`；
2. `battle_phase_entered` 为 `true`；
3. 选中的 `Move` / `Shoot` / `Occupy` 动作数 **> 0**；
4. `promotion.json` 不再因 `battle_phase_not_entered` / `no_move_shoot_occupy_selected_actions` 失败。

---

## 3. 失败表现：现象对照表

| 现象 | 最可能的原因 | 先看哪里 |
| --- | --- | --- |
| 引擎打印 `did not pass authentication` | MAC 不匹配（§1） | §2 第 3、4 项 |
| `auth_status=failed_mac_mismatch` | 同上 | runbook「Current Evidence」 |
| `battle_phase_entered=false` 且动作数为 0 | **鉴权的下游后果**，不是独立缺陷 | runbook「Expected current result」 |
| `promotion.json` 报 `battle_phase_not_entered` | 同上 | 不要改 promotion 逻辑去「让它过」 |
| 配置缺失（`ls` 找不到） | 容器重建后系统目录未随镜像恢复 | §4 场景 A |
| 脚本报 `blocker_reason=engine_config_missing` | 同上 | 检查脚本的 `next_action` |

**共同点**：这些都是**环境前置未满足**，**不是**可以在代码侧修好的缺陷。
把它们当代码 bug 去「修」（改校验、改 promotion 判定、伪造证据）**是明确禁止的**（§5）。

---

## 4. 排查步骤：按场景处置

### 场景 A：配置缺失（换机或容器重建后）

1. 确认第 2 项的路径**存在与否**；
2. **按 runbook 的合法路径解决**：
   - 让容器以**被授权的 MAC** 运行（runbook 给了 `docker-compose.auth-mac.yml` 的用法）；
   - 或**向 SDK/平台方申请针对实际运行 MAC 重新生成的 `.engine_config`**；
   - 或在**官方授权的宿主/容器镜像**上运行。
3. **不要**自己造一个 `.engine_config`。

### 场景 B：MAC 不匹配（**本机现状**）

同上——runbook 的 Legal Resolution Path 是唯一路径。要点：

```bash
# 按 runbook：用带授权 MAC 覆盖的 Compose 服务起容器
docker compose -f docker-compose.yml -f docker-compose.auth-mac.yml up -d --build dev
```

**前提**：该容器必须在非回环接口上报出 runbook 记录的那个**授权 MAC**。
**若平台不允许服务级 `mac_address`**（本远程环境即无容器运行时，见 §6），
则应直接走「申请新配置」或「换到授权环境」。

### 场景 C：已按授权 MAC 运行，仍报 `did not pass authentication`

runbook 已给出结论——**不要 patch SDK**，改走：

- 申请按实际 MAC 重新生成的 `.engine_config`；
- 请 SDK/平台方支持当前宿主/容器环境；
- 换用平台方提供的已授权宿主/容器镜像。

---

## 5. 底线（不可越过）

1. **不得 patch、替换或绕过** SDK 的鉴权扩展（`authenticator*.so`）；
2. **不得编辑 `.engine_config` 来「让它过」**——runbook 明文禁止；
3. **不得伪造** battle-phase 证据或 promotion 结果；
4. **不得把 `.engine_config`、授权文件、密钥、真实主机路径提交入库**；
5. 把「鉴权未满足」写成「已完成」「已达标」属**虚高判定**——本项目对判定一律从严，
   任务清单里 130 行明细「已满足 0 行」即此口径。

> ⚠ **一条需要说明的既有工具**：仓库里有 `scripts/rebind_engine_config.py`，它能**把配置重绑到当前 MAC**。
> 它与 runbook 的「Legal Resolution Path」**方向相反**。
> **本清单不把它列为解决路径**，也不为其提供 preflight 编排：
> 重绑只是改本地文件，**不改变授权本身**；若授权限定机器，重绑可能违约。
> 仅当**已从 SDK 方获得「可换机」的明确许可**时，才可将它作为**执行手段**；否则视为禁止操作。

---

## 6. 前置条件相关：本运行环境的一个硬事实（影响可验证性）

**远程 situ155 容器内没有任何容器运行时**：`docker`/`podman`/`nerdctl`/`ctr`/`crictl`/`buildah`/`runc`
**全部不存在**，也没有 `/var/run/docker.sock`；该远程**自身就是一个 Docker 容器**
（`/.dockerenv` 存在，cgroup 路径含 `.../docker/465a18f93ac6...`）。

**直接后果**：runbook 里那条「用 Compose 以授权 MAC 起容器」的路径**在本环境无法执行**。
故在本环境中，场景 B 只能走**申请重新生成的 `.engine_config`** 或**换到授权环境**。

这也解释了为什么 `experiments/shenji_agent/` 下会有 `*blocked*.md` 一类报告：
**阻塞是环境性的，且在容器内无法自解**。

---

## 7. 与项目验收口径的衔接

引擎相关的量化指标（如 31005 4.3.2 的规则引擎处理延迟、章 7 的引擎吞吐）
在本项目中的立场是：**项目侧不实施，由平台方或黑盒验收**（任务清单 §10.7 边界声明 6）。

**因此：以「鉴权没通」为由去替换或估算引擎指标，是不可接受的。**
鉴权问题**不改变**该边界，也不得用本项目自研组件（如 AI 策略推理耗时）顶替引擎指标。

---

## 8. 本文的边界

- 本文**不含** `.engine_config` 内容、密钥或授权文件；§1 的两个 MAC 沿用 runbook 的公开记录（容器虚拟网卡）。
- 机制描述基于**本会话实读仓库源码/文档**（`docs/sdk_auth_runbook.md`、`scripts/check_sdk_auth_environment.py`、
  `scripts/rebind_engine_config.py`、`docker-compose.auth-mac.yml`、`Dockerfile`），
  **未**对闭源 `authenticator*.so` 作逆向。
- **本会话未修改**上述任何鉴权相关文件；本文为**新增**文档。
- ⚠ **C2 原计划中的 `scripts/prepare_engine_config.py` 未实现**，原因见 §5 的说明：
  其设计的核心能力（重绑 `.engine_config`）与 runbook 明文禁止项冲突，
  **在获得 SDK 方明确许可前不应提供该工具**。本文以核对清单替代「preflight 脚本」，
  并把可执行的检查指向**已有的只读脚本** `scripts/check_sdk_auth_environment.py`。
