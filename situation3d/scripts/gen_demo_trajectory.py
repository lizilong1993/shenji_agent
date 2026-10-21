#!/usr/bin/env python3
"""Generate the demonstration situation datasets consumed by the web front end.

One dataset is written per registered demo case, into ``web/data/situations/``
(the server scans that directory to build the case list; there is no index file)
that drives the case menu.

Usage (from the repository root):

    python3 situation3d/scripts/gen_demo_trajectory.py            # every case
    python3 situation3d/scripts/gen_demo_trajectory.py sea_convoy # one case

Writes ``situation3d/web/data/situations/<case_id>.json`` -- one file per case.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "situation3d"))

from engine import cases, scenario, symbology  # noqa: E402


def write_json(path: str, payload) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


def write_base(out_dir: str) -> int:
    """写出与案例无关的基础数据。

    空态（不加载任何TS）下，前端仍需要城区 AOI 来生成 WSF3D 建筑体块，也需要
    军标目录来渲染符号——这两样原本只随案例数据集下发。把它们提到一个启动即加载
    的 base.json，前端就不必为了底图资产先加载一个案例。

    放进 cases.json 也可以，但 cases.index() 目前零依赖，而 scenario.build_dataset
    正是靠函数内 import 才避开循环依赖；让菜单索引反向依赖 scenario / symbology
    是把这条边固化错。
    """
    payload = {
        "schema": "situation3d.base.v1",
        "region": scenario.REGION,
        "scenario_id": scenario.SCENARIO_ID,
        "scenario_name": scenario.SCENARIO_NAME,
        "camera": dict(cases.DEFAULT_CAMERA),
        "urban_aoi": scenario.URBAN_AOI,
        "symbology": symbology.catalog(),
    }
    path = os.path.join(out_dir, "base.json")
    size = write_json(path, payload)
    print("base        : %s (%.1f KB)  urban_aoi=%d  icons=%d"
          % (path, size / 1024.0, len(payload["urban_aoi"]), len(payload["symbology"]["icons"])))
    return 0


def main(argv) -> int:
    wanted = argv[1:] or [c.case_id for c in cases.CASES
                         if getattr(c, "generator", "scenario") == "scenario"]
    unknown = [c for c in wanted if c not in cases.CASES_BY_ID]
    if unknown:
        print("ERROR: unknown case(s): %s" % ", ".join(unknown), file=sys.stderr)
        return 2

    # `base.json` 仍在 web/data 下（启动即加载的基础资产），
    # 而**每一份 TS 写进单独的目录** —— 服务端扫描该目录生成案例清单
    # （`/api/situations`），所以新增/删除案例只需增删文件，
    # 不再需要往这里写一份 `cases.json` 索引。
    out_dir = os.path.join(REPO, "situation3d", "web", "data")
    os.makedirs(out_dir, exist_ok=True)
    sit_dir = os.path.join(out_dir, "situations")
    os.makedirs(sit_dir, exist_ok=True)

    for case_id in wanted:
        case = cases.get_case(case_id)
        if getattr(case, "generator", "scenario") == "campaign":
            from engine.campaigns import taiwan_dense
            path, geo = taiwan_dense.write_case(sit_dir)
            print("case        : %s  (%s) [campaign generator]" % (case_id, case.name))
            print("  written   : %s" % path)
            print("  geo       : %s" % geo)
            continue
        dataset = scenario.build_dataset(case)
        out_path = os.path.join(sit_dir, case_id + ".json")
        size = write_json(out_path, dataset)

        summary = dataset["inference_summary"]
        print("case        : %s  (%s)" % (case_id, case.name))
        print("  targets   : %d  (sea/land/air = %s)"
              % (len(dataset["targets"]), _domain_counts(dataset["targets"])))
        print("  hotspots  : %d" % len(dataset["hotspots"]))
        print("  frames    : %d  (tick=%ss, total=%sh)"
              % (len(dataset["frames"]), dataset["tick_seconds"],
                 len(dataset["frames"]) * dataset["tick_seconds"] / 3600.0))
        print("  inference : peak=%s (tick %d), mean threat=%.1f, histogram=%s"
              % (summary["peak_level_label"], summary["peak_tick"],
                 summary["mean_threat"], summary["level_histogram"]))
        print("  written   : %s (%.1f KB)" % (out_path, size / 1024.0))

    # 不再写 cases.json：索引改由服务端扫描 situations 目录得出。
    # 留着它只会变成一份会过期的第二真相来源。
    write_base(out_dir)
    return 0


def _domain_counts(targets) -> str:
    counts = {}
    for t in targets:
        counts[t["domain"]] = counts.get(t["domain"], 0) + 1
    return "/".join("%s=%d" % (d, counts.get(d, 0)) for d in ("sea", "land", "air"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
