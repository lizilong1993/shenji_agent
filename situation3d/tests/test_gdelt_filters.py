"""GDELT 检索条件的扩展。

GDELT 此前只按案例关键词查：非英语地区的案例拿不到本地报道（台海案例
只看英文源会漏掉大量中文报道）。这里覆盖语言与附加词两个收窄条件，并
锁住"不配就等于旧查询"这条性质。
"""

from __future__ import annotations

import os
import sys
import unittest
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from intel.case import CaseIntelProfile  # noqa: E402
from intel.connectors.gdelt import LANGUAGES, build_url  # noqa: E402

START = "2026-11-06T00:00:00Z"
END = "2026-11-06T12:00:00Z"


def query_of(url):
    return parse_qs(urlparse(url).query)["query"][0]


class BaselineTest(unittest.TestCase):
    def test_default_query_is_unchanged(self):
        # 不传收窄条件时必须与旧实现逐字相同
        url = build_url(["taiwan", "台海"], START, END)
        self.assertEqual(query_of(url), '"taiwan" OR "台海"')

    def test_keywords_are_required(self):
        with self.assertRaises(ValueError):
            build_url([], START, END)

    def test_empty_keywords_strings_ignored(self):
        with self.assertRaises(ValueError):
            build_url(["", "  "], START, END)

    def test_maxrecords_is_clamped(self):
        url = build_url(["x"], START, END, maxrecords=999)
        self.assertEqual(parse_qs(urlparse(url).query)["maxrecords"][0], "250")


class LanguageTest(unittest.TestCase):
    def test_language_filter_is_appended(self):
        url = build_url(["taiwan"], START, END, languages=["chi", "eng"])
        query = query_of(url)
        self.assertIn("sourcelang:chi", query)
        self.assertIn("sourcelang:eng", query)

    def test_unknown_language_is_dropped(self):
        # 未登记的代码不能塞进查询——GDELT 会因为无法识别而整条查询失败
        url = build_url(["taiwan"], START, END, languages=["klingon", "chi"])
        query = query_of(url)
        self.assertNotIn("klingon", query)
        self.assertIn("sourcelang:chi", query)

    def test_all_unknown_languages_falls_back(self):
        url = build_url(["taiwan"], START, END, languages=["xx"])
        self.assertEqual(query_of(url), '"taiwan"')

    def test_language_list_is_bounded(self):
        url = build_url(["taiwan"], START, END,
                        languages=list(LANGUAGES))
        self.assertLessEqual(query_of(url).count("sourcelang:"), 5)


class ExtraTermsTest(unittest.TestCase):
    def test_extra_terms_are_and_joined(self):
        url = build_url(["台海"], START, END, extra_terms=["演习", "封锁"])
        query = query_of(url)
        # 主关键词是 OR；附加词是 AND（这里用空格表示）
        self.assertIn('("台海")', query)
        self.assertIn('("演习")', query)
        self.assertIn('("封锁")', query)

    def test_extra_terms_do_not_rejoin_with_or(self):
        url = build_url(["a", "b"], START, END, extra_terms=["c"])
        query = query_of(url)
        self.assertLess(query.index('"c"'), query.index('"a"') + 1000)
        # 顺序是 主关键词组 → 附加词，不要颠倒
        self.assertLess(query.index('"b"'), query.index('("c")'))

    def test_quotes_are_stripped(self):
        url = build_url(['a"b'], START, END, extra_terms=['c"d'])
        query = query_of(url)
        self.assertNotIn('""', query)


class ProfileTest(unittest.TestCase):
    def test_languages_and_extra_terms_round_trip(self):
        profile = CaseIntelProfile(
            case_id="c1", name="c1", bbox=(119.0, 22.0, 121.0, 25.0),
            keywords=["台海"], languages=["CHI", "eng"],
            extra_terms=["演习"])
        self.assertEqual(profile.languages, ("chi", "eng"))
        self.assertEqual(profile.extra_terms, ("演习",))
        again = CaseIntelProfile.from_dict(profile.to_dict())
        self.assertEqual(again.languages, ("chi", "eng"))
        self.assertEqual(again.extra_terms, ("演习",))

    def test_absent_fields_are_empty(self):
        profile = CaseIntelProfile(case_id="c1", name="c1",
                                   bbox=(0, 0, 1, 1))
        self.assertEqual(profile.languages, ())
        self.assertEqual(profile.extra_terms, ())


if __name__ == "__main__":
    unittest.main()
