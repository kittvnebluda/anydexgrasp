"""Run the planner on a saved cloud: test_offline.py CLOUD.npz [no_collision]."""

import sys

import numpy as np

import adg_server

d = np.load(sys.argv[1])
planner = adg_server.Planner()
for _ in range(2):
    out = planner.plan({"points": d["points"], "base_from_cam": d["base_from_cam"], "max_tilt": 55.0, "top": 20,
                        "workspace": ((0.25, 0.70), (-0.55, 0.35), (-0.2, 0.2)), "no_collision": len(sys.argv) > 2})
base = d["base_from_cam"]
for g in out[:15]:
    c = base[:3, :3] @ g["center"] + base[:3, 3]
    print(g["name"], round(g["score"], 2), round(g["width"], 3), np.round(c, 3), np.round(base[:3, :3] @ g["approach"], 2), g["angle"].astype(int))
