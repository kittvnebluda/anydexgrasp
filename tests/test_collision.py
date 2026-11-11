"""The GPU collision check against AnyDexGrasp's own, on the recorded scene (needs the GPU and the weights)."""

import copy
import time

import numpy as np
import pytest

from adg_server import compat
from adg_server.planner import Planner

from .bench import request


@pytest.fixture(scope="module")
def planner():
    return Planner(compat.load())


def test_gpu_check_agrees_with_the_original(planner):
    np.random.seed(0)  # the network's augmentations
    req = request(near=True)
    hand, two = planner.candidates(req)
    points = np.asarray(req["points"], np.float32)
    t = time.time()
    h_old, _, old = planner.collide(points, copy.deepcopy(hand), copy.deepcopy(two), legacy=True)
    t_old = time.time() - t
    t = time.time()
    h_new, _, new = planner.collide(points, copy.deepcopy(hand), copy.deepcopy(two))
    t_new = time.time() - t
    print(
        f"\n{len(old)} grasps: original {t_old:.1f} s, {old.sum()} free; gpu {t_new:.2f} s, {new.sum()} free; "
        f"{(old != new).sum()} differ"
    )
    np.testing.assert_allclose(h_new.translations, h_old.translations, atol=1e-5)  # the same centre adjustment
    # the voxels are aligned differently (camera axes there, hand axes here): grasps that graze the scene by a voxel
    # can go either way
    assert (old != new).sum() <= 0.05 * len(old)
    assert t_new < 2.0


def test_hand_map_is_the_original_mapping(planner):
    np.random.seed(0)
    hand, two = planner.candidates(request(near=True))
    original = copy.deepcopy(hand)
    original.graspgroupTR_2_TR(two, planner.mesh_dir)
    planner.hand_map.apply(hand, two)
    for field in ("widths", "translations", "rotation_matrices", "angles"):
        np.testing.assert_allclose(getattr(hand, field), getattr(original, field))
