"""Military symbology catalogue for situation3d.

This module is the single source of truth for the icon keys that the engine
emits on every target and that the front end renders. Keeping the catalogue in
the engine rather than hard-coding it in JavaScript means a target's symbol can
never disagree with the scenario definition it came from.

Symbol construction follows MIL-STD-2525D. **The frame shape encodes both
affiliation and domain** -- it is not a rectangle with a colour difference:

* affiliation: friendly = a filled frame in the friendly palette, hostile = a
  **pointed diamond**. Colour alone is not the encoding;
* domain determines the *frame shape*: land unit (friendly) = square,
  sea surface (friendly) = circle, air (friendly) = arch/dome, and hostile air
  is a pentagon/"house" rather than a diamond with an arc over it;
* there is **no domain modifier arc for surface units**. A below-frame arc is the
  *subsurface* (submerged) modifier, not surface. Land units carry no modifier
  either. Only air gets an arc above the frame;
* the glyph inside the frame encodes the platform family.

The rendered symbols are **pre-generated** by ``tools/gen_symbols.py`` using
milsymbol (MIT, MIL-STD-2525D/APP-6) from the SIDC table in that tool, and
committed as ``web/assets/symbols/*.svg``. ``web/src/symbology.js`` prefers those
static assets and falls back to a built-in approximation only if they are
missing. The icon keys below remain the contract between engine and front end;
the SIDC mapping lives with the generator that consumes it.

NOTE: MIL-STD-2525D has **no dedicated coastal anti-ship missile battery
entity**. ``coastal_missile`` is drawn from Land Unit -> Fires -> Missile
(130700), which is an explicit approximation -- see PLATFORMS in
``tools/gen_symbols.py``.
"""

from __future__ import annotations

from typing import Dict, List

# Domains understood across the engine and the front end.
DOMAINS: Dict[str, str] = {
    "sea": "水面",
    "land": "陆上",
    "air": "空中",
}

# Affiliation -> display colour. Chosen to stay legible over satellite imagery.
AFFILIATIONS: Dict[str, dict] = {
    "red": {"label": "红方", "color": "#d0342c", "ink": "#7d1611"},
    "blue": {"label": "蓝方", "color": "#1f5fbf", "ink": "#123a78"},
}

# Icon key -> symbol metadata.
#
#   label    Chinese name shown in the UI
#   domain   which domain modifier the frame gets
#   family   coarser platform family, used for filtering and for the glyph
#   abbrev   short text drawn under the frame when the glyph is ambiguous
ICONS: Dict[str, dict] = {
    # -- surface ---------------------------------------------------------
    "destroyer": {"label": "驱逐舰", "domain": "sea", "family": "水面舰艇", "abbrev": "DDG"},
    "frigate": {"label": "护卫舰", "domain": "sea", "family": "水面舰艇", "abbrev": "FFG"},
    "replenishment": {"label": "补给舰", "domain": "sea", "family": "辅助舰船", "abbrev": "AOE"},
    "corvette": {"label": "巡逻舰", "domain": "sea", "family": "水面舰艇", "abbrev": "PC"},
    # -- land ------------------------------------------------------------
    "coastal_missile": {"label": "岸舰导弹", "domain": "land", "family": "岸防兵力", "abbrev": "ASCM"},
    "sam_battalion": {"label": "防空导弹", "domain": "land", "family": "地面防空", "abbrev": "SAM"},
    "armor": {"label": "装甲分队", "domain": "land", "family": "地面机动", "abbrev": "ARM"},
    "artillery": {"label": "炮兵分队", "domain": "land", "family": "地面火力", "abbrev": "ARTY"},
    # -- air -------------------------------------------------------------
    "fighter": {"label": "战斗机", "domain": "air", "family": "作战飞机", "abbrev": "FTR"},
    "aew": {"label": "预警机", "domain": "air", "family": "支援飞机", "abbrev": "AEW"},
    # -- campaign additions ----------------------------------------------
    # The campaign demo needs a few more distinct silhouettes: a UAV that is
    # not an AEW dish, a helicopter that is not a jet, an amphibious vehicle
    # that is not a tank, and a landing craft that is not a cargo ship.
    "uav": {"label": "无人机", "domain": "air", "family": "支援飞机", "abbrev": "UAV"},
    "helicopter": {"label": "直升机", "domain": "air", "family": "作战飞机", "abbrev": "HEL"},
    "amphibious": {"label": "两栖车辆", "domain": "land", "family": "地面机动", "abbrev": "AAV"},
    "landing_craft": {"label": "登陆艇", "domain": "sea", "family": "两栖舰船", "abbrev": "LCU"},
    "infantry": {"label": "步兵", "domain": "land", "family": "地面机动", "abbrev": "INF"},
    "transport": {"label": "运输分队", "domain": "land", "family": "后勤保障", "abbrev": "TRN"},
    "engineering": {"label": "工兵", "domain": "land", "family": "工程保障", "abbrev": "ENG"},
}


def catalog() -> dict:
    """Return the full symbology catalogue for embedding in the dataset."""
    return {
        "standard": "MIL-STD-2525 风格（红蓝双方对抗着色）",
        "domains": DOMAINS,
        "affiliations": AFFILIATIONS,
        "icons": ICONS,
    }


def describe(icon: str) -> dict:
    """Metadata for one icon key, falling back to an unknown-symbol entry."""
    return ICONS.get(
        icon,
        {"label": "未知目标", "domain": "land", "family": "未识别", "abbrev": "UNK"},
    )


def validate(icon_keys: List[str]) -> List[str]:
    """Return the icon keys that are not in the catalogue (empty list is good)."""
    return sorted({k for k in icon_keys if k not in ICONS})
