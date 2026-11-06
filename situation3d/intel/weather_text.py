"""WMO 天气代码 → 中文描述。Open-Meteo 用 WMO 4677 表，直接显示数字对指挥员没有意义。"""

WMO = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "强阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def describe(code) -> str:
    try:
        value = int(code)
    except (TypeError, ValueError):
        return ""
    return WMO.get(value, "天气代码 %d" % value)


def wind_direction(deg) -> str:
    """风向角度 → 八方位中文，指挥员读"西北风"比读 315° 快。"""
    try:
        value = float(deg) % 360.0
    except (TypeError, ValueError):
        return ""
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return names[int((value + 22.5) // 45.0) % 8] + "风"
