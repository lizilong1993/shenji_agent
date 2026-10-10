# P4 部署评估：原生数据库路径与 Docker 路径（P0-2）

> `P0-2`【目标】「把 P4 演示系统部署到远程 situ155 容器，使 **postgis/redis/backend/frontend 四服务**可运行、
> 数据库可验证，解除全部 P4 任务的前置门禁。」
>
> **本文所有结论均为远程实测**，不是推断。远程 = `situ155`（`server-10-254-1-155`）。

---

## 0. 一句话结论

**Docker 路径不可行；但"数据库可验证"这一半已经用原生 PostgreSQL 跑通了。**
`P0-2` 的四项验收判据里，**判据 2 所需的数据库能力已具备**（且过程中发现原脚本自身有缺陷）；
**判据 1/3 不能达成**（判据 1 要四服务都在，判据 3 要前端可访问）。

**故 `P0-2` 不能标完成**，但它**不再是"全部 13 项都无法推进"** ——
其中**只依赖数据库**的部分现在有真实环境了。

---

## 1. Docker 路径：**不可行**（实测）

| 检查 | 结果 |
| --- | --- |
| `docker` / `podman` / `nerdctl` / `ctr` / `containerd` / `runc` / `buildah` | **全部不存在** |
| `/var/run/docker.sock` | **不存在** |
| `/proc/1/cgroup` | `…:/docker/465a18f93ac6…` —— **远程自身就是一个容器** |

在容器内再起 dockerd 需要特权（`--privileged`），**不是可靠路径**；
`I03`/`I05` 已两次证明"先假定不可行"会误判，故此处给的是**实测清单**而非推测。

**→ `P0-2`【具体步骤】3（`docker compose up -d --build`）、5（`docker volume ls`）无法执行。**

---

## 2. 原生数据库路径：**已跑通**（实测）

### 2.1 安装

`apt-get install postgresql-12 postgresql-12-postgis-3` → **exit=0**。

| 项 | 实测 |
| --- | --- |
| 系统 | Ubuntu 20.04.6 LTS |
| PostgreSQL | **12.22**（apt 只有 `postgresql-12`，**无 16**） |
| PostGIS | **3.0.0 r17983**（GEOS 3.8.0 / PROJ 6.3.1） |
| 集群 | `12 main` @ **5432 online** |
| 资源 | 376 GB 内存；`/remote-home` 317 GB 可用 |

> ⚠ **与 compose 要求的版本不一致**：compose 用 `postgis/postgis:16-3.4`，
> 原生路径是 **PG 12 + PostGIS 3.0.0**。后端 `requirements.txt` 含 `geoalchemy2==0.16.0`，
> **该组合是否兼容须实测**（本文未做端到端后端连通）。

### 2.2 P4 的 `000-init.sh` **未经修改即可运行**

```
su postgres -c "cd /tmp && bash p4_init.sh"   → EXIT=0
输出以「✓ 数据库初始化完成!」结束
```

**为什么能直接跑**：该脚本用的是 `psql --username postgres`（**无 `-h`，走 peer 认证**），
`docker exec` 那一层不在它里面 —— 它本来就与 Docker 无关。

**六张表全部建出（判据 2 的核心）：**

| 表 | 存在 |
| --- | --- |
| `target` | ✓ |
| `target_trajectory` | ✓ |
| `region` | ✓ |
| `target_relation` | ✓ |
| `situation_snapshot` | ✓ |
| `situation_config` | ✓ |

另有 `target_attribute` / `target_track` / `spatial_ref_sys`（PostGIS 自带）。

**PostGIS 几何列可用：**

```
situation_config   | region   | POLYGON    | 4326
target_trajectory  | position | POINT      | 4326
target_track       | track_line | LINESTRING | 4326
region             | geom     | POLYGON    | 4326
situation_snapshot | region   | POLYGON    | 4326
```

**初始数据：** `situation_config` 3 行、`target` 10 行、`region` 0 行。

### 2.3 原生验证器：`scripts/verify_db_native.py`

`backend/verify-db.sh` 的 6 项检查**全部**写成
`docker exec situational-awareness-postgis psql … -c "<SQL>"` —— **只有那一层是 Docker 特有的**。

本脚本**从原脚本正则抽取 SQL**（`docker exec … psql … -c "<SQL>"`），
**不重写 SQL**；只替换"在哪里执行"。故原脚本改 SQL，重新抽取即同步。

**实测结果（6 项）：**

| # | 检查 | 结果 |
| --- | --- | --- |
| 1 | `situation_config` 架构与数据 | **ok**（3 行） |
| 2 | `situation_id` / `situation_code` 唯一性 | **ok** |
| 3 | `target` 外键关联 | **ok**（10 行） |
| 4 | 每个想定下单位数 | **ok**（3 行） |
| 5 | `target` 唯一性约束 | **ok** |
| 6 | 约束与索引清单 | **✗ 失败**（见 §3） |

---

## 3. 顺带发现：`verify-db.sh` 的判据 2 是**空过**的

### 3.1 第 6 段 SQL 本身有歧义

```sql
SELECT
  constraint_name, table_name, column_name, constraint_type
FROM information_schema.constraint_column_usage ccu
JOIN information_schema.table_constraints tc ON ccu.table_name = tc.table_name
WHERE ccu.table_name IN ('situation_config', 'target')
ORDER BY table_name, constraint_name;   -- ← 未加表别名
```

实测报错：

```
ERROR:  column reference "constraint_name" is ambiguous
```

**原因**：`constraint_column_usage` 与 `table_constraints` **两个视图都有** `constraint_name`
（`table_name` / `column_name` 同理），未加 `ccu.` / `tc.` 前缀即歧义。

### 3.2 原脚本**没有** `ON_ERROR_STOP`，所以它仍然 `exit 0`

`verify-db.sh` 是 `set -e` 也没有、`ON_ERROR_STOP` 也没有的一串 `psql -c`。
**psql 报错只写 stderr，脚本继续，最后 `exit 0`。**

**→ `P0-2`【验收判据】2「`bash backend/verify-db.sh` 退出码 0」是空过的**：
**即使第 6 项检查一直失败，退出码照样是 0。**
（第 1/2/3 项检查若失败同样被吞。）

**判据 2 现在有了真实判据**：本文的原生验证器对每项检查**取 psql 退出码**，
并以「核心表非空」为附加断言 —— 第 6 项会**如实报失败**。

### 3.3 处置

**未修改 P4 的 `verify-db.sh`**（属 P4 代码，且与 `P0-2` 的部署目标不同）。
本文把它记为**待修缺陷**：需给 SQL 加表别名，并给脚本加 `set -e` / `ON_ERROR_STOP`。

---

## 4. 四项验收判据的实测状态

| 判据 | 要求 | 状态 |
| --- | --- | --- |
| 1 | `docker compose ps` 四项非 `unhealthy`/`exited` | **✗ 无法达成**（无容器运行时） |
| 2 | `bash backend/verify-db.sh` 退出码 0 | **⚠ 原脚本恒 0（空过）**；**原生替代已实现**：6 项里 **5 项 ok、1 项真实失败** |
| 3 | `curl …/api/health` 返回 200 | **✗ 未启动后端**（需 `fastapi/uvicorn` + Redis，见 §5） |
| — | `docs/deploy_situ155.md` 记录地址与端口 | **本文即该文档的替代**，记录原生路径的地址/端口 |

---

## 5. 还需什么才能把判据 3 也做出来（**未做，仅评估**）

后端与前端**不一定非得用 Docker**，但需要：

| 组件 | 需要什么 | 本地可行性 |
| --- | --- | --- |
| backend（FastAPI） | `pip install fastapi uvicorn pydantic sqlalchemy asyncpg geoalchemy2 orjson` | apt 无 pip；`python3 -m pip` 实测**无** → 需 `apt install python3-pip` 或 `ensurepip` |
| redis | `redis-server`（apt 有） | 未验 |
| frontend | 构建产物或静态服务（compose 里是 `build:` 一个 nginx） | 未验；`config.json` 需改成远程地址 |
| `SECRET_KEY`/口令 | `.env`（不入库） | `P0-2` 步骤 2 要求替换默认弱口令 |

**这是下一步的自然工作**，但已超出"评估"范围，且会引入**与 compose 定义不同的运行形态** ——
须先由你裁定是否接受"原生进程替代 Docker Compose"这个部署形态变化。

---

## 6. 对 13 项 `阻塞-P0-2` 的影响（**是否解锁**）

**13 项的阻塞点可以分成三类**（按是否真的需要 Docker）：

| 类别 | 需要什么 | 是否已解锁 |
| --- | --- | --- |
| **A. 只依赖数据库 schema** | 原生 PG + PostGIS | **✓ 已解锁**（本文已让 schema 跑起来） |
| **B. 依赖后端 API 运行** | FastAPI + Redis 起起来 | **未解锁**（§5，需 pip） |
| **C. 依赖前端/四服务齐全** | 完整 compose | **✗ 不可达**（无容器运行时） |

**`I10`/`D3` 属 A 类**（元数据表与版本表都在数据库侧）：其步骤 2/3/4/6
**现在具备了可行的数据库环境** —— 需按 §5 补 pip 与后端，但**不再需要 Docker**。

**`C6`（安全合规）/`P2`/`P4`/`I13` 等**若要求"四服务齐全"的证据，仍属 C 类，**仍阻塞**。

---

## 7. 未做 / 待裁定

| 项 | 状态 |
| --- | --- |
| Docker Compose 四服务 | **不可行**（无容器运行时） |
| 原生 PG 12 + PostGIS 3.0.0 | **✓ 已安装并跑通 P4 schema** |
| 原生验证器（复用原 SQL） | **✓ 已实现并运行** |
| 后端/Redis/前端原生启动 | **未做**（需 pip、需裁定形态变化） |
| 修 `verify-db.sh` 的歧义 SQL 与缺失的 `ON_ERROR_STOP` | **未做**（属 P4 代码） |
| **是否接受"原生进程替代 Docker Compose"** | **待你裁定** |
