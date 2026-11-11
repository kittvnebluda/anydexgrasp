"""AnyDexGrasp's planning (robot_inspire.py) on a given cloud, for the Inspire right hand.

Changes to the original: the cloud comes in, not a RealSense frame (no camera-specific depth cut or view filters);
grasp centres are kept in a workspace box and near target points, approaches within a tilt of straight down; both
thumb sides of every two-finger grasp are kept (the original drops one for its UR); select_grasp_type without its
off-by-one on type 8.
"""

import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial import cKDTree

from .collision import HandVolumes, check
from .hand import HandMap

AUGMENTS = 10  # rotated (every other one also flipped) copies of the cloud the network sees, besides the cloud
SCORE_MIN = 0.5  # decision score a grasp needs
PER_TYPE = 100  # best grasps per hand type into the collision check (AnyDexGrasp: 50)
CANDIDATES = 1000  # best two-finger grasps into the decision models
APPROACH_DIST = 0.06  # m the hand is swept back along its approach in the collision check
COLLISION_THRESH = 0  # scene points allowed inside the swept hand
PUSH_EXTRA = 0.014  # m past the depth the hand goes in, as AnyDexGrasp's UR executes it (ur.py)
MESH_DIR = "generate_mesh_and_pointcloud/inspire_urdf"


def net_grasps(adg: SimpleNamespace, net, points: np.ndarray, augment: np.ndarray, flip: bool):
    """ri.get_grasp on a cloud: two-finger grasps (graspnet array layout, rows) and their features, in the cloud's
    frame; (None, None) if none."""
    import MinkowskiEngine as ME

    ri = adg.ri
    pts = torch.from_numpy(ri.transform_point_cloud(points, augment).astype(np.float32))
    coords = np.ascontiguousarray(pts / 0.005, dtype=int)
    _, idxs = ME.utils.sparse_quantize(coords, return_index=True)
    coords, pts = coords[idxs], pts[idxs]
    coords_batch, points_batch = ME.utils.sparse_collate([coords], [pts])
    sinput = ME.SparseTensor(points_batch, coords_batch, device="cuda")
    with torch.no_grad():
        end_points = net({"sinput": sinput, "point_clouds": [sinput.F]})
        preds, features = ri.parse_preds(end_points, use_v2=True)
    if not preds:
        return None, None
    preds = preds[0]
    augment = augment.copy()
    if flip:
        augment[:, 0] = -augment[:, 0]
    inv = torch.tensor(np.linalg.inv(augment).astype(np.float32), device="cuda")
    rot = inv[:3, :3].expand(len(preds), 3, 3)
    preds[:, 12:15] = torch.matmul(rot, preds[:, 12:15].view(-1, 3, 1)).view(-1, 3) + inv[:3, 3]
    pose_rot = torch.matmul(rot, preds[:, 3:12].view(-1, 3, 3))
    if flip:
        preds[:, 12] = -preds[:, 12]
        pose_rot[:, 0, :] = -pose_rot[:, 0, :]
        pose_rot[:, :, 1] = -pose_rot[:, :, 1]
    preds[:, 3:12] = pose_rot.reshape(-1, 9)
    keep = (preds[:, 1] < ri.MAX_GRASP_WIDTH) & (preds[:, 1] > ri.MIN_GRASP_WIDTH)
    preds, features = preds[keep], features[0][keep]
    n = len(preds)
    ones = torch.ones((n, 1), device="cuda")
    return torch.cat([preds[:, 0:2], 0.03 * ones, preds[:, 2:15], -ones], 1), features


def select_grasp_type(hand) -> list[int]:
    """The best PER_TYPE per hand type (a fifth of that for rings), in score order."""
    kept: dict[int, list[int]] = {}
    for i, grasp_type in enumerate(hand.grasp_types):
        ids = kept.setdefault(int(grasp_type), [])
        if len(ids) < (PER_TYPE // 5 if int(grasp_type) == 1 else PER_TYPE):
            ids.append(i)
    return sorted(i for ids in kept.values() for i in ids)


class Planner:
    def __init__(self, adg: SimpleNamespace):
        self.adg = adg
        ri = adg.ri
        self.mesh_dir = os.path.join(adg.root, MESH_DIR)
        t = time.time()
        self.net = ri.get_net(ri.cfgs.checkpoint_path, use_v2=True)
        self.inspire = ri.get_inspire_model(os.path.join(adg.root, "logs/model/inspire_model/obj140"))
        self.meshes = ri.load_meshes_pointcloud(os.path.join(self.mesh_dir, "meshes"))
        self.volumes = HandVolumes(self.meshes, self.mesh_dir)
        self.hand_map = HandMap(adg, self.mesh_dir)
        print(f"models loaded in {time.time() - t:.1f} s", flush=True)

    def plan(self, req: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """A protocol request -> the reply arrays (see protocol.py), without the version."""
        t0 = time.time()
        found = self.candidates(req)
        if found is None:
            return empty()
        hand, two = found
        t1 = time.time()
        n_checked = len(hand)
        hand, two, free = self.collide(np.asarray(req["points"], np.float32), hand, two)
        out = empty() if not np.any(free) else reply(self.adg, hand[free], two[free], int(req.get("top", 20)))
        print(
            f"{len(req['points'])} points: candidates {t1 - t0:.1f} s, collision {time.time() - t1:.1f} s "
            f"({n_checked} checked), {len(out['score'])} grasps",
            flush=True,
        )
        return out

    def collide(self, points: np.ndarray, hand, two, legacy: bool = False):
        """(hand, two, free): the grasps the width filter keeps, free True where the swept hand is clear of the
        scene; `legacy` with AnyDexGrasp's own (slow) check."""
        ri = self.adg.ri
        if not legacy:
            return check(self.volumes, self.hand_map, points, hand, two, ri.MIN_GRASP_WIDTH, COLLISION_THRESH)
        hand, two, free, _ = self.adg.CollisionDetector(points, voxel_size=0.001).detect(
            hand,
            two,
            self.mesh_dir,
            self.meshes,
            min_grasp_width=ri.MIN_GRASP_WIDTH,
            VoxelGrid=ri.INSPIREHANDR_VOXElGRID,
            approach_dist=APPROACH_DIST,
            collision_thresh=COLLISION_THRESH,
            adjust_gripper_centers=True,
        )
        return hand, two, np.asarray(free, bool)

    def candidates(self, req: dict[str, np.ndarray]):
        """(hand, two): the hand grasps and their two-finger grasps into the collision check, best first; None if
        there are none."""
        ri = self.adg.ri
        points = np.asarray(req["points"], np.float32)
        base_from_cam = np.asarray(req["base_from_cam"], float)
        up = np.asarray(req.get("up", (0.0, 0.0, 1.0)), float)
        max_tilt = float(req.get("max_tilt", 50.0))
        ggs, feats = [], []
        for i in range(AUGMENTS + 1):
            flip = i % 2 == 0 and i > 0
            augment = np.eye(4) if i == 0 else ri.augment_data(flip=flip)
            gg, f = net_grasps(self.adg, self.net, points, augment.astype(np.float32), flip)
            if gg is not None and len(gg):
                ggs.append(gg)
                feats.append(f)
        if not ggs:
            return None
        gg = torch.cat(ggs).cpu().numpy()
        feat = torch.cat(feats).cpu().numpy()

        # both hand orientations about the approach axis: the thumb on either side
        flipped = gg.copy()
        rots = flipped[:, 4:13].reshape(-1, 3, 3)
        rots[:, :, 1:3] = -rots[:, :, 1:3]  # half a turn about the approach axis
        flipped[:, 4:13] = rots.reshape(-1, 9)
        gg = np.concatenate([gg, flipped])
        feat = np.concatenate([np.c_[feat, np.zeros(len(feat))], np.c_[feat, np.ones(len(feat))]])

        rot_base = base_from_cam[:3, :3] @ gg[:, 4:13].reshape(-1, 3, 3)
        ok = rot_base[:, :, 0] @ -up > np.cos(np.radians(max_tilt))
        centre = gg[:, 13:16] @ base_from_cam[:3, :3].T + base_from_cam[:3, 3]
        if "workspace" in req:
            for axis, (lo, hi) in enumerate(np.asarray(req["workspace"], float)):
                ok &= (centre[:, axis] > lo) & (centre[:, axis] < hi)
        if "near" in req and len(req["near"]):
            radius = float(req["radius"])
            ok &= cKDTree(np.asarray(req["near"])[:, :2]).query(centre[:, :2], distance_upper_bound=radius)[0] < radius
        gg, feat = gg[ok], feat[ok]
        order = gg[:, 0].argsort()[::-1][:CANDIDATES]
        gg, feat = gg[order], feat[order]
        if not len(gg):
            return None

        features = ri.get_graspgroup_features(feat, None)
        depth, htype, scores, gg, feat = ri.get_inspire_depth_type(self.inspire, features, gg, feat)
        keep = scores > SCORE_MIN
        depth, htype, scores, gg = depth[keep], htype[keep], scores[keep], gg[keep]
        if not len(gg):
            return None
        two = self.adg.GraspGroup(gg)
        hand = self.adg.InspireHandRGraspGroup()  # from_graspgroup, with the hand poses from HandMap
        hand.grasp_group_array = np.zeros((len(two), self.adg.inspire.INSPIREHANDR_ARRAY_LEN))
        hand.depths = two.depths
        hand.grasp_types = htype
        hand.object_ids = two.object_ids
        self.hand_map.apply(hand, two)
        hand.scores = scores
        hand.depths = hand.depths + depth
        order = np.argsort(hand.scores)[::-1]
        hand, two = hand[order], two[order]
        index = select_grasp_type(hand)
        return hand[index], two[index]


def reply(adg: SimpleNamespace, hand, two, top: int) -> dict[str, np.ndarray]:
    rows = [(hand[int(i)], two[int(i)]) for i in np.argsort(hand.scores)[::-1][:top]]
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    for k, (h, _) in enumerate(rows):
        poses[k, :3, :3], poses[k, :3, 3] = h.rotation_matrix, h.translation
    return {
        "hand": poses,
        "approach": np.array([np.asarray(g.rotation_matrix)[:, 0] for _, g in rows]).reshape(-1, 3),
        "push": np.array([float(h.depth) + PUSH_EXTRA for h, _ in rows]),
        "angle": np.array([np.asarray(h.angle, float) for h, _ in rows]).reshape(-1, 6),
        "type": np.array([int(h.grasp_type) for h, _ in rows], int),
        "name": np.array([adg.grasp_types[str(int(h.grasp_type))]["name"] for h, _ in rows], dtype=str),
        "score": np.array([float(h.score) for h, _ in rows]),
        "width": np.array([float(h.width) for h, _ in rows]),
        "center": np.array([np.asarray(g.translation) for _, g in rows]).reshape(-1, 3),
    }


def empty() -> dict[str, np.ndarray]:
    return {
        "hand": np.zeros((0, 4, 4)),
        "approach": np.zeros((0, 3)),
        "push": np.zeros(0),
        "angle": np.zeros((0, 6)),
        "type": np.zeros(0, int),
        "name": np.zeros(0, dtype=str),
        "score": np.zeros(0),
        "width": np.zeros(0),
        "center": np.zeros((0, 3)),
    }
