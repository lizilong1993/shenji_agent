/**
 * 角色卡读取与展示辅助（P1，只读）。
 *
 * 数据契约：dataset.targets[*].role_card 由 engine/rolecards/legacy.py 生成，
 * 含 class_id / subclass_id / confidence / legacy / projection / effective / summary。
 * 本模块不推导能力，只把 summary 格式化为右栏文字；缺失时返回 null，
 * 调用方保持旧界面不变（无卡数据零影响）。
 */
(function (global) {
  'use strict';

  function summary(def) {
    if (!def || !def.role_card) return null;
    var rc = def.role_card;
    if (rc.summary) return rc.summary;
    var eff = rc.effective || {};
    var motion = eff.motion || {};
    var protection = eff.protection || {};
    return {
      class_label: rc.class_id || '',
      subclass_label: rc.subclass_id || '',
      confidence: rc.confidence || '',
      unknown: rc.confidence === 'unknown',
      max_speed_ms: motion.max_speed_ms,
      cruise_speed_ms: motion.cruise_speed_ms,
      max_speed_display: '',
      cruise_speed_display: '',
      armor_class: protection.armor_class,
      max_blood: protection.max_blood,
      size_class: protection.size_class,
      sensors: [],
      weapons: [],
      capacity: eff.capacity || {}
    };
  }

  function typeText(rc) {
    if (!rc) return '—';
    if (rc.subclass_label && rc.class_label &&
        rc.subclass_label.indexOf(rc.class_label) >= 0) {
      return rc.subclass_label;
    }
    if (rc.subclass_label && rc.class_label &&
        rc.subclass_label !== rc.class_label) {
      return rc.subclass_label + '（' + rc.class_label + '）';
    }
    return rc.subclass_label || rc.class_label || '—';
  }

  function speedText(rc) {
    if (!rc) return '—';
    var cruise = rc.cruise_speed_display || '';
    var max = rc.max_speed_display || '';
    if (!cruise && !max) return '—';
    if (!cruise) return '最大 ' + max;
    if (!max || cruise === max) return '巡航 ' + cruise;
    return '巡航 ' + cruise + ' / 最大 ' + max;
  }

  function armorText(rc) {
    if (!rc) return '—';
    var levels = ['无装甲', '轻型', '中型', '重型', '复合'];
    var armor = (typeof rc.armor_class === 'number')
      ? (levels[rc.armor_class] || ('等级 ' + rc.armor_class)) : '—';
    var blood = (typeof rc.max_blood === 'number') ? ('编制 ' + rc.max_blood) : '';
    return blood ? armor + ' · ' + blood : armor;
  }

  function sensorText(rc) {
    if (!rc || !rc.sensors || !rc.sensors.length) return '';
    return rc.sensors.slice(0, 3).map(function (s) {
      var range = (typeof s.range_km === 'number') ? (' ' + Math.round(s.range_km) + ' km') : '';
      return (s.label || s.kind || '传感器') + range;
    }).join('；');
  }

  function weaponText(rc) {
    if (!rc || !rc.weapons || !rc.weapons.length) return '';
    return rc.weapons.slice(0, 4).map(function (w) {
      var range = (typeof w.range_km === 'number') ? (' ' + Math.round(w.range_km) + ' km') : '';
      var ammo = (typeof w.ammo_max === 'number') ? (' ×' + w.ammo_max) : '';
      return (w.label || '武器') + range + ammo;
    }).join('；');
  }

  function capacityText(rc) {
    if (!rc || !rc.capacity) return '';
    var c = rc.capacity;
    var parts = [];
    if (c.personnel) parts.push(c.personnel + ' 人');
    if (c.cargo_t) parts.push(c.cargo_t + ' t');
    if (c.vehicles) parts.push(c.vehicles + ' 车');
    if (c.aircraft) parts.push(c.aircraft + ' 架');
    if (c.lcac) parts.push(c.lcac + ' 气垫艇');
    return parts.join(' / ');
  }

  /**
   * 「目标属性」面板「基本」分区要追加的行（2026-10-22 从右栏目标意图迁入）。
   * 返回 [{label, value}]；无 role_card 时返回空数组，面板保持原样。
   * 展示上不再出现"角色卡"字样。
   */
  function basicRows(def) {
    var rc = summary(def);
    if (!rc) return [];
    var rows = [
      { label: '类型', value: typeText(rc) + (rc.unknown ? ' · 未识别小类' : '') },
      { label: '速度', value: speedText(rc) },
      { label: '防护', value: armorText(rc) },
    ];
    var sensors = sensorText(rc);
    var weapons = weaponText(rc);
    var capacity = capacityText(rc);
    if (sensors) rows.push({ label: '传感器', value: sensors });
    if (weapons) rows.push({ label: '武器', value: weapons });
    if (capacity) rows.push({ label: '运载', value: capacity });
    if (rc.confidence === 'legacy') rows.push({ label: '数据', value: '旧数据迁移快照' });
    return rows;
  }

  global.RoleCard = {
    summary: summary,
    typeText: typeText,
    speedText: speedText,
    armorText: armorText,
    sensorText: sensorText,
    weaponText: weaponText,
    capacityText: capacityText,
    basicRows: basicRows
  };
}(typeof window !== 'undefined' ? window : this));
