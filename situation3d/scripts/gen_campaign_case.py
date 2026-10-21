#!/usr/bin/env python3
"""Generate a campaign case into web/data/situations."""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from engine.campaigns.taiwan_dense import write_case
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="taiwan_dense_transfer_v1")
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "data", "situations"))
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    if a.case != "taiwan_dense_transfer_v1":
        raise SystemExit("unknown campaign case: %s" % a.case)
    t0 = time.time()
    p, g = write_case(a.out, a.seed)
    print("wrote", p, g, "in %.1fs" % (time.time() - t0))
