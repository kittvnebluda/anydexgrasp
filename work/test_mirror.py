"""Survivor counts for a cloud as is and mirrored (optical x): test_mirror.py CLOUD."""

import sys

import numpy as np

import adg_server

d = np.load(sys.argv[1])
planner = adg_server.Planner()
F = np.diag([-1.0, 1.0, 1.0])
M = np.diag([1.0, -1.0, 1.0])
b = d["base_from_cam"]
bm = b.copy()
bm[:3, :3] = M @ b[:3, :3] @ F
bm[:3, 3] = M @ b[:3, 3]
pm = d["points"] @ F
print("check: mirrored world points == M * world points:",
      np.allclose(pm[:5] @ bm[:3, :3].T + bm[:3, 3], (d["points"][:5] @ b[:3, :3].T + b[:3, 3]) @ M, atol=1e-5))
for name, pts, base, ws in (("plain", d["points"], b, ((0.25, 0.55), (0.25, 0.55), (-0.2, 0.2))),
                            ("mirror", pm, bm, ((0.25, 0.55), (-0.55, -0.25), (-0.2, 0.2)))):
    out = planner.plan({"points": pts, "base_from_cam": base, "max_tilt": 55.0, "top": 100, "workspace": ws})
    print(name, len(out))
