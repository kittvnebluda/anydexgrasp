"""Import paths for the AnyDexGrasp checkout without installing ur_toolbox (it pulls in the UR driver)."""

import collections
import collections.abc
import importlib.util
import os
import sys
import types

# the 2023 code base predates Python 3.10
collections.Sequence = collections.abc.Sequence

ROOT = os.path.expanduser("~/Projects/anydexgrasp/AnyDexGrasp")
for sub in ("", "models", "dataset", "utils", "pointnet2", "knn", "generate_mesh_and_pointcloud"):
    sys.path.insert(0, os.path.join(ROOT, sub))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for pkg in ("ur_toolbox", "ur_toolbox.robot"):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))
inspire_grasp = _load(
    "ur_toolbox.robot.InspireHandR_grasp",
    os.path.join(ROOT, "ur_toolbox/ur_toolbox/robot/Inspire/InspireHandR_grasp.py"),
)
