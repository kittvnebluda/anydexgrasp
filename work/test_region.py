"""Planner on a saved cloud in a region, with collision thresholds: test_region.py CLOUD X Y R."""

import sys

import numpy as np

import adg_server

d = np.load(sys.argv[1])
x, y, r = map(float, sys.argv[2:5])
planner = adg_server.Planner()
base = d["base_from_cam"]
for thresh in (None, 0, 20, 60):
    if thresh is not None:
        adg_server.COLLISION_THRESH = thresh
    out = planner.plan({"points": d["points"], "base_from_cam": base, "max_tilt": 55.0, "top": 10,
                        "workspace": ((x - r, x + r), (y - r, y + r), (-0.2, 0.2)), "no_collision": thresh is None})
    print("thresh", thresh, len(out))
    for g in out[:4]:
        c = base[:3, :3] @ g["center"] + base[:3, 3]
        print("  ", g["name"], round(g["score"], 2), round(g["width"], 3), np.round(c, 3), np.round(base[:3, :3] @ g["approach"], 2))
