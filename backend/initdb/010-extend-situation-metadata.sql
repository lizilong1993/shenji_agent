-- ============================================================================
-- 想定元数据扩展：31002 表2 的 19 项（I10 / D3 步骤 3 + 6）
-- ============================================================================
--
-- 标准依据
--   31002 5.2.2：「定义必选和可选元数据项，其内容**至少应包括表 2 中所列条目**」
--   31002 5.4.5：版本控制「记录每一次修改的内容、时间、修改者及修改原因」
--   原文：`.scratch/std/s31002.txt`，表2 见 `:198-219`
--
-- 与 P4 原 `000-init.sh` 的关系
--   **纯加法，不改既有列、不删既有数据。** 原脚本建出的 10 列
--   （situation_id / situation_code / situation_name / description / region /
--     center_lon / center_lat / center_alt / created_time / last_update_time）
--   **一个都不动**，否则 P4 既有读取方会坏。
--
-- 可重复执行
--   全部用 `ADD COLUMN IF NOT EXISTS` / `CREATE TABLE IF NOT EXISTS`，
--   故 `000-init.sh` → 本脚本 可按任意顺序重复跑，结论一致（P0-2【预期结果】要求）。
--
-- 列名口径
--   原表2 的「类型」列实为**字段名**（标准原文如此排版），且是中文。
--   此处按 **ascii 列名 + COMMENT 写明原文字段名** 处理 ——
--   ASCII 列名便于后端 ORM 与查询；原文字段名进 `COMMENT`，
--   使「逐字对应」可被 `psql \d+` 直接核对，不必依赖本文档。
--
-- 必填/选填
--   表2 中**只有第 19 项**原文自带「（可选）」→ 允许 NULL 且有默认值。
--   其余 18 项按 5.2.2 默认**必填**；但既有 3 行数据已存在，
--   故**不能**直接加 `NOT NULL`（会失败）。
--   处置：先加列（可空）→ 回填默认值 → 再置 `NOT NULL`（幂等）。
-- ============================================================================

BEGIN;

-- ── 1) situation_config：补 18 项必填 + 1 项可选 ──────────────────────────

-- #1 想定唯一标识符：既有 situation_code 已承担，加生成列式的解析视图不必要，
--     故本项**不新增列**，而是在注释里指明它由 situation_code 承载。
COMMENT ON COLUMN situation_config.situation_code IS
  '31002 表2 第 1 项「想定唯一标识符」的承载列（原文字段名即此）';

-- #2 想定标题 —— 既有 situation_name 已承担
COMMENT ON COLUMN situation_config.situation_name IS
  '31002 表2 第 2 项「想定标题」的承载列';

-- #10 想定目标/训练目的 —— 既有 description 承载（**注意：语义为近似**）
COMMENT ON COLUMN situation_config.description IS
  '31002 表2 第 10 项「想定目标/训练目的」（近似：原列语义为"描述"，非"目的"）';

-- 其余 16 项新增
ALTER TABLE situation_config
  ADD COLUMN IF NOT EXISTS scenario_type        VARCHAR(64),
  ADD COLUMN IF NOT EXISTS applicable_level     VARCHAR(32),
  ADD COLUMN IF NOT EXISTS service_domain       VARCHAR(128),
  ADD COLUMN IF NOT EXISTS environment_type     VARCHAR(128),
  ADD COLUMN IF NOT EXISTS estimated_duration_min INTEGER,
  ADD COLUMN IF NOT EXISTS opposing_force_desc  TEXT,
  ADD COLUMN IF NOT EXISTS friendly_force_desc  TEXT,
  ADD COLUMN IF NOT EXISTS complexity_level     SMALLINT,
  ADD COLUMN IF NOT EXISTS developed_date       DATE,
  ADD COLUMN IF NOT EXISTS revised_date         DATE,
  ADD COLUMN IF NOT EXISTS version_no           VARCHAR(32),
  ADD COLUMN IF NOT EXISTS author_org           VARCHAR(200),
  ADD COLUMN IF NOT EXISTS classification       VARCHAR(32),
  ADD COLUMN IF NOT EXISTS keywords             TEXT[],
  ADD COLUMN IF NOT EXISTS related_scenarios    TEXT[],
  ADD COLUMN IF NOT EXISTS applicable_platforms TEXT[],
  ADD COLUMN IF NOT EXISTS evaluation_metrics   JSONB;   -- #19 可选

COMMENT ON COLUMN situation_config.scenario_type IS '31002 表2 第 3 项「想定类型」';
COMMENT ON COLUMN situation_config.applicable_level IS '31002 表2 第 4 项「适用层级（战略/战役/战术/技术）」';
COMMENT ON COLUMN situation_config.service_domain IS '31002 表2 第 5 项「适用军兵种/任务领域」';
COMMENT ON COLUMN situation_config.environment_type IS '31002 表2 第 6 项「作战地域/环境类型」';
COMMENT ON COLUMN situation_config.estimated_duration_min IS '31002 表2 第 7 项「预估推演时长」（分钟）';
COMMENT ON COLUMN situation_config.opposing_force_desc IS '31002 表2 第 8 项「主要威胁/对手（蓝方）描述」';
COMMENT ON COLUMN situation_config.friendly_force_desc IS '31002 表2 第 9 项「主要己方（红方）角色/力量」';
COMMENT ON COLUMN situation_config.complexity_level IS '31002 表2 第 11 项「想定复杂度等级」（1-5，**须为著录值或显式标注的推算值**）';
COMMENT ON COLUMN situation_config.developed_date IS '31002 表2 第 12 项「开发/修订日期」之开发日期（**不得用 created_time 代替**）';
COMMENT ON COLUMN situation_config.revised_date IS '31002 表2 第 12 项「开发/修订日期」之修订日期';
COMMENT ON COLUMN situation_config.version_no IS '31002 表2 第 13 项「版本号」（与 situation_version 表配合，见 5.4.5）';
COMMENT ON COLUMN situation_config.author_org IS '31002 表2 第 14 项「作者/来源单位」';
COMMENT ON COLUMN situation_config.classification IS '31002 表2 第 15 项「密级」';
COMMENT ON COLUMN situation_config.keywords IS '31002 表2 第 16 项「关键词/标签」';
COMMENT ON COLUMN situation_config.related_scenarios IS '31002 表2 第 17 项「关联想定/素材」（situation_code 列表）';
COMMENT ON COLUMN situation_config.applicable_platforms IS '31002 表2 第 18 项「适用模拟系统/平台」';
COMMENT ON COLUMN situation_config.evaluation_metrics IS '31002 表2 第 19 项「评估指标（可选）」——原文自带"（可选）"';

-- ── 2) 回填既有行（使 NOT NULL 可加，且不丢既有数据）────────────────────
-- 既有 3 行是 P4 原脚本插入的示例数据。回填值**必须可被识别为占位**，
-- 不能填一个看起来像真实著录的值 —— 否则元数据会说谎。
UPDATE situation_config SET
  scenario_type        = COALESCE(scenario_type, 'UNSPECIFIED'),
  applicable_level     = COALESCE(applicable_level, 'UNSPECIFIED'),
  service_domain       = COALESCE(service_domain, 'UNSPECIFIED'),
  environment_type     = COALESCE(environment_type, 'UNSPECIFIED'),
  estimated_duration_min = COALESCE(estimated_duration_min, 0),
  opposing_force_desc  = COALESCE(opposing_force_desc, '（待著录）'),
  friendly_force_desc  = COALESCE(friendly_force_desc, '（待著录）'),
  complexity_level     = COALESCE(complexity_level, 0),
  developed_date       = COALESCE(developed_date, DATE '1970-01-01'),
  version_no           = COALESCE(version_no, '0.0.0'),
  author_org           = COALESCE(author_org, '（待著录）'),
  classification       = COALESCE(classification, 'UNSPECIFIED'),
  keywords             = COALESCE(keywords, ARRAY[]::TEXT[]),
  related_scenarios    = COALESCE(related_scenarios, ARRAY[]::TEXT[]),
  applicable_platforms = COALESCE(applicable_platforms, ARRAY[]::TEXT[])
WHERE scenario_type IS NULL
   OR applicable_level IS NULL
   OR service_domain IS NULL
   OR environment_type IS NULL
   OR estimated_duration_min IS NULL
   OR opposing_force_desc IS NULL
   OR friendly_force_desc IS NULL
   OR complexity_level IS NULL
   OR developed_date IS NULL
   OR version_no IS NULL
   OR author_org IS NULL
   OR classification IS NULL
   OR keywords IS NULL
   OR related_scenarios IS NULL
   OR applicable_platforms IS NULL;

-- ── 3) 加 NOT NULL（18 项必填；第 19 项 evaluation_metrics 保持可空）────
-- 幂等：列已是 NOT NULL 时重复执行不报错。
-- 先回填可能仍为 NULL 的列（幂等：只在列**仍可空**时执行）。
--
-- ⚠ 我第一版用一个通用 CASE 处理所有列，结果 `date` 与 `text[]` 分支
-- **类型无法统一** → `ERROR: CASE types date and text[] cannot be matched`。
-- 故改为**按列类型分组**：每组一条 UPDATE，类型各自明确。
DO $$
BEGIN
  -- 文本列
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='scenario_type' AND is_nullable='YES') THEN
    UPDATE situation_config SET scenario_type = COALESCE(scenario_type, 'UNSPECIFIED');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='applicable_level' AND is_nullable='YES') THEN
    UPDATE situation_config SET applicable_level = COALESCE(applicable_level, 'UNSPECIFIED');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='service_domain' AND is_nullable='YES') THEN
    UPDATE situation_config SET service_domain = COALESCE(service_domain, 'UNSPECIFIED');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='environment_type' AND is_nullable='YES') THEN
    UPDATE situation_config SET environment_type = COALESCE(environment_type, 'UNSPECIFIED');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='opposing_force_desc' AND is_nullable='YES') THEN
    UPDATE situation_config SET opposing_force_desc = COALESCE(opposing_force_desc, '（待著录）');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='friendly_force_desc' AND is_nullable='YES') THEN
    UPDATE situation_config SET friendly_force_desc = COALESCE(friendly_force_desc, '（待著录）');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='version_no' AND is_nullable='YES') THEN
    UPDATE situation_config SET version_no = COALESCE(version_no, '0.0.0');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='author_org' AND is_nullable='YES') THEN
    UPDATE situation_config SET author_org = COALESCE(author_org, '（待著录）');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='classification' AND is_nullable='YES') THEN
    UPDATE situation_config SET classification = COALESCE(classification, 'UNSPECIFIED');
  END IF;

  -- 日期列（第 12 项拆成的两列）
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='developed_date' AND is_nullable='YES') THEN
    UPDATE situation_config SET developed_date = COALESCE(developed_date, DATE '1970-01-01');
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='revised_date' AND is_nullable='YES') THEN
    UPDATE situation_config SET revised_date = COALESCE(revised_date, DATE '1970-01-01');
  END IF;

  -- 数值列
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='estimated_duration_min' AND is_nullable='YES') THEN
    UPDATE situation_config SET estimated_duration_min = COALESCE(estimated_duration_min, 0);
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='complexity_level' AND is_nullable='YES') THEN
    UPDATE situation_config SET complexity_level = COALESCE(complexity_level, 0);
  END IF;

  -- 数组列
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='keywords' AND is_nullable='YES') THEN
    UPDATE situation_config SET keywords = COALESCE(keywords, ARRAY[]::TEXT[]);
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='related_scenarios' AND is_nullable='YES') THEN
    UPDATE situation_config SET related_scenarios = COALESCE(related_scenarios, ARRAY[]::TEXT[]);
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='situation_config'
             AND column_name='applicable_platforms' AND is_nullable='YES') THEN
    UPDATE situation_config SET applicable_platforms = COALESCE(applicable_platforms, ARRAY[]::TEXT[]);
  END IF;
END $$;

DO $$
DECLARE
  col TEXT;
  required_cols TEXT[] := ARRAY[
    'scenario_type','applicable_level','service_domain','environment_type',
    'estimated_duration_min','opposing_force_desc','friendly_force_desc',
    'complexity_level','developed_date','revised_date','version_no','author_org',
    'classification','keywords','related_scenarios','applicable_platforms'];
BEGIN
  FOREACH col IN ARRAY required_cols LOOP
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name='situation_config' AND column_name=col
                 AND is_nullable='YES') THEN
      EXECUTE format('ALTER TABLE situation_config ALTER COLUMN %I SET NOT NULL', col);
    END IF;
  END LOOP;
END $$;

-- ── 4) 取值词表约束（31002 5.2.2 c) 要求"取值词表"）──────────────────────
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='ck_applicable_level') THEN
    ALTER TABLE situation_config ADD CONSTRAINT ck_applicable_level
      CHECK (applicable_level IN ('战略','战役','战术','技术','UNSPECIFIED'));
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='ck_complexity_level') THEN
    ALTER TABLE situation_config ADD CONSTRAINT ck_complexity_level
      CHECK (complexity_level BETWEEN 0 AND 5);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='ck_classification') THEN
    ALTER TABLE situation_config ADD CONSTRAINT ck_classification
      CHECK (classification IN ('公开','内部','秘密','机密','UNSPECIFIED'));
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_config_scenario_type ON situation_config(scenario_type);
CREATE INDEX IF NOT EXISTS idx_config_classification ON situation_config(classification);
CREATE INDEX IF NOT EXISTS idx_config_keywords ON situation_config USING GIN(keywords);

-- ── 5) situation_version：版本表（31002 5.4.5）───────────────────────────
CREATE TABLE IF NOT EXISTS situation_version (
  version_id      BIGSERIAL PRIMARY KEY,
  situation_id    UUID NOT NULL REFERENCES situation_config(situation_id) ON DELETE CASCADE,
  version_no      VARCHAR(32) NOT NULL,
  -- 内容快照：31002 5.4.5 要求"记录每一次修改的内容"
  content_snapshot JSONB      NOT NULL,
  changed_fields  TEXT[]      NOT NULL DEFAULT ARRAY[]::TEXT[],
  change_reason   TEXT        NOT NULL,
  changed_by      VARCHAR(200) NOT NULL,
  changed_time    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE situation_version IS
  '31002 5.4.5 版本控制：记录每一次修改的内容、时间、修改者及修改原因（I10/D3 步骤 3）';
COMMENT ON COLUMN situation_version.content_snapshot IS '修改后的完整元数据快照（结构化）';
COMMENT ON COLUMN situation_version.changed_fields IS '本次修改涉及的字段名列表';
COMMENT ON COLUMN situation_version.change_reason IS '31002 5.4.5 要求的"修改原因"';
COMMENT ON COLUMN situation_version.changed_by IS '31002 5.4.5 要求的"修改者"';

CREATE INDEX IF NOT EXISTS idx_version_situation ON situation_version(situation_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_version_situation_no
  ON situation_version(situation_id, version_no);

-- ── 6) 版本自动记录（使"任一修改产生版本行"，无需应用层配合）────────────
CREATE OR REPLACE FUNCTION situation_config_version_trigger()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'UPDATE' AND to_jsonb(NEW) = to_jsonb(OLD) THEN
    RETURN NEW;                      -- 无实际变化则不记版本
  END IF;
  INSERT INTO situation_version
    (situation_id, version_no, content_snapshot, changed_fields, change_reason, changed_by)
  VALUES (
    NEW.situation_id,
    NEW.version_no,
    to_jsonb(NEW) - 'region',        -- 几何列不可直接入 JSONB
    CASE WHEN TG_OP = 'INSERT' THEN ARRAY['(created)']
         ELSE ARRAY(SELECT key FROM jsonb_each(to_jsonb(NEW))
                    WHERE to_jsonb(NEW)->key IS DISTINCT FROM to_jsonb(OLD)->key) END,
    CASE WHEN TG_OP = 'INSERT' THEN 'initial creation'
         ELSE COALESCE(current_setting('situation.change_reason', TRUE), 'updated') END,
    COALESCE(current_setting('situation.changed_by', TRUE), CURRENT_USER)
  )
  ON CONFLICT (situation_id, version_no) DO NOTHING;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_situation_config_version ON situation_config;
CREATE TRIGGER trg_situation_config_version
  AFTER INSERT OR UPDATE ON situation_config
  FOR EACH ROW EXECUTE FUNCTION situation_config_version_trigger();

-- ── 7) 为既有 3 行补一条初始版本行 ───────────────────────────────────────
INSERT INTO situation_version
  (situation_id, version_no, content_snapshot, changed_fields, change_reason, changed_by)
SELECT situation_id, version_no, to_jsonb(sc) - 'region',
       ARRAY['(backfilled)'], 'backfilled for pre-existing rows', 'migration'
FROM situation_config sc
ON CONFLICT (situation_id, version_no) DO NOTHING;

COMMIT;
