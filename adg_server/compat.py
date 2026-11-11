"""AnyDexGrasp's research code imported as it is, without installing it or its UR driver."""

import collections
import collections.abc
import importlib.util
import os
import sys
import time
import types
from types import SimpleNamespace

DEFAULT_ROOT = os.environ.get("ADG_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "AnyDexGrasp"))


def _load(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load(root: str = DEFAULT_ROOT) -> SimpleNamespace:
    """The AnyDexGrasp modules the planner uses; the process's working directory becomes `root`."""
    collections.Sequence = collections.abc.Sequence  # pyright: ignore[reportAttributeAccessIssue]  # pre-3.10 code
    for sub in ("", "models", "dataset", "utils", "pointnet2", "knn", "generate_mesh_and_pointcloud"):
        sys.path.insert(0, os.path.join(root, sub))
    for pkg in ("ur_toolbox", "ur_toolbox.robot"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    inspire = _load(
        "ur_toolbox.robot.InspireHandR_grasp",
        os.path.join(root, "ur_toolbox/ur_toolbox/robot/Inspire/InspireHandR_grasp.py"),
    )
    # robot_inspire.py parses its args and imports the UR driver at import time: give it both
    robot = sys.modules["ur_toolbox.robot"]
    robot.UR_Camera_Gripper = None  # pyright: ignore[reportAttributeAccessIssue]
    robot.InspireHandR_grasp = inspire  # pyright: ignore[reportAttributeAccessIssue]
    ckpt = os.path.join(root, "logs/model/checkpoint.tar.18")
    argv, sys.argv = sys.argv, [sys.argv[0], "--checkpoint_path", ckpt, "--robot_ip", "none", "--use_graspnet_v2"]
    os.chdir(root)
    try:
        import collision_detector
        import minkowski_graspnet
        import robot_inspire
        from graspnetAPI import GraspGroup
    finally:
        sys.argv = argv
    collision_detector.time = time  # used there, never imported
    robot_inspire.RANDOM_GRASP = False  # it ships in data-collection mode

    import torch

    # the decision models are pickled modules: allow exactly the classes they hold, no arbitrary code
    torch.serialization.add_safe_globals(
        [minkowski_graspnet.MinkowskiGraspNetMultifingerType1, torch.nn.BatchNorm1d, torch.nn.Conv1d, torch.nn.Dropout]
    )
    return SimpleNamespace(
        root=root,
        inspire=inspire,
        ri=robot_inspire,
        GraspGroup=GraspGroup,
        InspireHandRGraspGroup=inspire.InspireHandRGraspGroup,
        grasp_types=inspire.grasp_types,
        CollisionDetector=collision_detector.ModelFreeCollisionDetectorMultifinger,
    )
