"""Plan on the recorded scene in-process and print the timings: python -m tests.bench [near]

`near` plans around one object (as the solution asks), else on the whole table region.
"""

import os
import sys
import time

import numpy as np

from adg_server import compat
from adg_server.planner import Planner

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_sort2_pool1.npz")
TABLE_TOP = -0.16
WORKSPACE = ((0.22, 0.80), (-0.70, 0.70), (TABLE_TOP - 0.03, TABLE_TOP + 0.17))  # the solution's cloud cut
REGION = ((0.25, 0.70), (-0.48, 0.20), (TABLE_TOP + 0.015, 0.20))  # the right hand's grasp centres
OBJECT = (0.40, -0.42)  # an object of the recorded layout on the right half


def request(near: bool) -> dict[str, np.ndarray]:
    d = np.load(SCENE)
    points, base_from_cam = d["points"], d["base_from_cam"]
    base = points @ base_from_cam[:3, :3].T + base_from_cam[:3, 3]
    keep = np.ones(len(base), bool)
    for axis, (lo, hi) in enumerate(WORKSPACE):
        keep &= (base[:, axis] > lo) & (base[:, axis] < hi)
    req = {
        "points": points[keep],
        "base_from_cam": base_from_cam,
        "up": np.array([0.0, 0.0, 1.0]),
        "max_tilt": np.array(55.0),
        "top": np.array(60),
        "workspace": np.array(REGION),
    }
    if near:
        b = base[keep]
        obj = b[(np.linalg.norm(b[:, :2] - OBJECT, axis=1) < 0.06) & (b[:, 2] > TABLE_TOP + 0.01)]
        req |= {"near": obj[:, :2], "radius": np.array(0.02)}
    return req


if __name__ == "__main__":
    planner = Planner(compat.load())
    req = request("near" in sys.argv[1:])
    for _ in range(2):  # the first run warms up CUDA
        t = time.time()
        out = planner.plan(req)
        print(f"total {time.time() - t:.1f} s")
    for name, score, centre in zip(out["name"][:10], out["score"][:10], out["center"][:10]):
        print(
            f"{name:20s} {score:.2f} {np.round(req['base_from_cam'][:3, :3] @ centre + req['base_from_cam'][:3, 3], 3)}"
        )
