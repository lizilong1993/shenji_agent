# 资源库 SQLite + 管理面板（P8，2026-10-27）

## 1. 数据库

`situation3d/var/resources/resources.db`：

- `resource_kinds`：kind、描述、默认 criticality、TTL、schema、providers；
- `resources`：entry_id、kind、path、content_hash、size、version、schema、provider、
  source、fetched_at、as_of、bbox、license、quality、pinned、notes_json、created_at、
  updated_at；
- `resource_meta`：schema 版本与 `index.json` 迁移标记。

旧 `var/resources/index.json` 在首次打开时导入，随后保持只读备份。所有 CLI/服务/分辨率
代码继续使用 `ResourceStore`，无需感知存储介质。写入使用 SQLite 事务 + 资源根目录
advisory 文件锁；读取不加文件锁。删除默认只删记录，不动物理文件。

## 2. HTTP API

只在 `--mode test` 开放写操作；demo 模式返回 403。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/resources?kind=&q=` | 类型目录、统计、条目列表 |
| GET | `/api/resources/<entry_id>` | 单条详情 + 校验结果 |
| POST | `/api/resources` | 登记服务器已有绝对路径（JSON） |
| POST | `/api/resources/<entry_id>/update` | 编辑元数据 / 换路径重算哈希 |
| POST | `/api/resources/<entry_id>/pin` | `{pinned: bool}` |
| POST | `/api/resources/<entry_id>/verify` | 校验 sha256/size |
| POST | `/api/resources/<entry_id>/delete` | `{force: bool}`，只删记录 |
| POST | `/api/resources/upload?kind=&name=&filename=` | 原始字节上传，落 `uploads/<kind>/` |

路径登记只允许项目根、资源根及 `SITUATION_RESOURCE_ROOTS` 内的路径；上传目标固定为
资源根下的 `uploads/`。网页上传适合单文件；envpack/瓦片集等大目录请用路径登记。

## 3. 前端

顶栏「数据管理 → 资源管理」打开 `web/src/resourceManager.js`：

- 类型筛选、ID/路径/来源搜索、资源计数与总大小；
- 新增（路径登记）、上传文件、编辑、固定/取消固定、校验、删除；
- 表单错误就地提示，不清空已填内容；
- 删除二次确认；固定资源删除需再次确认强制删除记录。

「影像数据源」仍是只读目录，不参与 CRUD。
