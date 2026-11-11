"""AnyDexGrasp's hand collision check (collision_detector.ModelFreeCollisionDetectorMultifinger) on the GPU.

The original moves every hand cloud to its grasp, sweeps it back along the approach, voxelises it and tests every
scene point against the voxels, one grasp at a time in open3d: ~35 s for 720 grasps on a 110k-point cloud. Here
each hand cloud is swept and voxelised once, in its own frame, at load; a check moves the scene points into all the
grasps' hand frames at once and looks them up in those voxels.

The same test up to the voxel grid: the original's voxels are aligned with the camera axes, these with the hand's.
"""

import json
import os

import numpy as np
import torch

VOXEL = 0.003  # m, robot_inspire.INSPIREHANDR_VOXElGRID
# m back along the approach the hand is also checked at: the original adds a 2 cm copy of the hand, then a 5 cm copy
# of both (approach_dist 0.06)
SWEEP = (0.0, 0.02, 0.05, 0.07)
FINGER_LENGTH = 0.06  # the original's two-finger box, for the centre adjustment
MIN_WIDTH = 0.025  # m, the adjusted two-finger width's floor
WIDENING = 1.7  # the original widens the adjusted width by this, twice
KEY_SHIFT = 40  # hand volume codes: key index << KEY_SHIFT | voxel index in that hand's box
CHUNK = 64  # grasps checked at once: CHUNK x scene points x 3 floats on the GPU


class HandVolumes:
    """The swept hand of every grasp type and width as occupied voxels in the hand frame."""

    def __init__(self, meshes: dict, mesh_dir: str, device: str = "cuda"):
        """`meshes`: open3d hand clouds by "<type name>_<width cm>" (robot_inspire.load_meshes_pointcloud)."""
        with open(os.path.join(mesh_dir, "width_12Dangle_6Dangle.json"), encoding="utf-8") as f:
            mapping = json.load(f)
        self.device = device
        self.keys = {key: i for i, key in enumerate(sorted(meshes))}
        lows, dims, codes = [], [], []
        for key, i in self.keys.items():
            name, width = key.rsplit("_", 1)
            # the two-finger approach (its x) in the hand frame: hand = two-finger * inv(this pose)
            approach = np.asarray(mapping[name][width]["rotation"]) @ (1.0, 0.0, 0.0)
            points = np.asarray(meshes[key].points)
            swept = np.concatenate([points - s * approach for s in SWEEP])
            cells = np.floor(swept / VOXEL).astype(np.int64)
            low = cells.min(0)
            dim = cells.max(0) - low + 1
            lows.append(low)
            dims.append(dim)
            codes.append((i << KEY_SHIFT) | np.unique(_linear(cells - low, dim)))
        self.lows = torch.tensor(np.array(lows), device=device)
        self.dims = torch.tensor(np.array(dims), device=device)
        self.codes = torch.sort(torch.tensor(np.concatenate(codes), device=device)).values

    def count(self, points: torch.Tensor, keys: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """Scene points (N, 3) inside each hand: hand `keys` (G,) at poses `rot` (G, 3, 3), `trans` (G, 3)."""
        out = []
        for s in range(0, len(keys), CHUNK):
            k, r, t = keys[s : s + CHUNK], rot[s : s + CHUNK], trans[s : s + CHUNK]
            local = torch.matmul(points[None] - t[:, None], r)  # (g, N, 3): R^T (p - t)
            cells = torch.floor(local / VOXEL).long() - self.lows[k][:, None]
            dim = self.dims[k][:, None]
            inside = ((cells >= 0) & (cells < dim)).all(-1)
            code = (k[:, None] << KEY_SHIFT) | _linear(cells.clamp(min=0), dim)
            at = torch.searchsorted(self.codes, code).clamp(max=len(self.codes) - 1)
            out.append((inside & (self.codes[at] == code)).sum(1))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=self.device)


def _linear(cells, dim):
    return (cells[..., 0] * dim[..., 1] + cells[..., 1]) * dim[..., 2] + cells[..., 2]


def adjust_centres(two, points: torch.Tensor, min_width: float) -> np.ndarray:
    """The original's adjust_gripper_centers, then its width filter: each two-finger grasp is moved across to the
    middle of the scene points between its fingers and its width set from them. Changes `two`; the kept rows."""
    t = torch.tensor(two.translations, dtype=torch.float32, device=points.device)
    r = torch.tensor(two.rotation_matrices, dtype=torch.float32, device=points.device)
    h, d, w = (
        torch.tensor(a, dtype=torch.float32, device=points.device)[:, None]
        for a in (two.heights, two.depths, two.widths)
    )
    ymin, ymax = [], []
    for s in range(0, len(t), CHUNK):
        local = torch.matmul(points[None] - t[s : s + CHUNK, None], r[s : s + CHUNK])  # (g, N, 3)
        hs, ds, ws = h[s : s + CHUNK], d[s : s + CHUNK], w[s : s + CHUNK]
        inner = (
            (local[..., 2] > -hs / 2)
            & (local[..., 2] < hs / 2)
            & (local[..., 0] > ds - FINGER_LENGTH)
            & (local[..., 0] < ds)
            & (local[..., 1] >= -ws / 2)
            & (local[..., 1] <= ws / 2)
        )
        y = torch.where(inner, local[..., 1], torch.zeros_like(local[..., 1]))  # 0 for the rest, as the original
        ymin.append(y.min(1).values)
        ymax.append(y.max(1).values)
    lo, hi = torch.cat(ymin).cpu().numpy(), torch.cat(ymax).cpu().numpy()
    offset = np.zeros((len(lo), 3))
    offset[:, 1] = (lo + hi) / 2
    two.translations = two.translations + np.einsum("gij,gj->gi", two.rotation_matrices, offset)
    two.widths = np.maximum(MIN_WIDTH, WIDENING * (hi - lo)) * WIDENING
    return two.widths > min_width


def check(volumes: HandVolumes, hand_map, points: np.ndarray, hand, two, min_width: float, thresh: int):
    """The original's detect(): (hand, two, free) of the grasps left after the width filter, `free` True where at
    most `thresh` scene points are inside the swept hand."""
    scene = torch.tensor(points, dtype=torch.float32, device=volumes.device)
    keep = adjust_centres(two, scene, min_width)
    hand, two = hand[keep], two[keep]
    if not len(hand):
        return hand, two, np.zeros(0, bool)
    hand_map.apply(hand, two)
    names = hand_map.names(hand.grasp_types)
    keys = torch.tensor(
        [volumes.keys[f"{n}_{round(w * 100, 1)}"] for n, w in zip(names, hand.widths)], device=volumes.device
    )
    # the hand cloud at the hand pose, then pushed in by its depth along the two-finger approach
    trans = hand.translations + hand.depths[:, None] * two.rotation_matrices[:, :, 0]
    rot = torch.tensor(hand.rotation_matrices, dtype=torch.float32, device=volumes.device)
    counts = volumes.count(scene, keys, rot, torch.tensor(trans, dtype=torch.float32, device=volumes.device))
    return hand, two, (counts <= thresh).cpu().numpy()
