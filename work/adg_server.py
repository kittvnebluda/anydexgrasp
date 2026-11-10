"""AnyDexGrasp inference for the Inspire hand, served over a local TCP socket.

Request (pickle): {"points": (N,3) float32 camera optical frame, "base_from_cam": (4,4), "up": (3,) unit vector in the
base frame, "max_tilt": deg, "top": int, "workspace": ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in base or None,
"near": ((M,2) xy points in base, radius m) or None: only grasps centred within radius of one of the points}.
Reply (pickle): list of dicts per grasp, best first, poses in the camera frame:
  hand (4,4)   InspireHandR mesh frame (the urdf-five3 Link111 shifted to the wrist centre)
  approach (3) two-finger approach axis; the hand goes in along it by `push` from `hand`
  push         m, depth + 0.014 as in AnyDexGrasp's grasp_and_throw
  angle (6)    preshape, Inspire units 0..1000 (little, ring, middle, index, thumb bend, thumb rotation)
  type, name, score, width
"""

import os
import pickle
import socket
import struct
import sys
import time

import numpy as np
import torch

import adg_paths

# robot_inspire.py parses its args and imports the UR driver at import time: give it both
sys.modules["ur_toolbox.robot"].UR_Camera_Gripper = None
sys.modules["ur_toolbox.robot"].InspireHandR_grasp = adg_paths.inspire_grasp
ckpt = os.path.join(adg_paths.ROOT, "logs/model/checkpoint.tar.18")
argv, sys.argv = (
    sys.argv,
    [sys.argv[0], "--checkpoint_path", ckpt, "--robot_ip", "none", "--use_graspnet_v2"],
)
os.chdir(adg_paths.ROOT)
import MinkowskiEngine as ME  # noqa: E402
import robot_inspire as ri  # noqa: E402
import collision_detector  # noqa: E402
from collision_detector import ModelFreeCollisionDetectorMultifinger  # noqa: E402

collision_detector.time = time  # used there, never imported
from graspnetAPI import GraspGroup  # noqa: E402

sys.argv = argv

InspireHandRGraspGroup = adg_paths.inspire_grasp.InspireHandRGraspGroup
grasp_types = adg_paths.inspire_grasp.grasp_types

ri.RANDOM_GRASP = False

# the decision models are pickled modules: allow exactly the classes they hold, no arbitrary code
import minkowski_graspnet  # noqa: E402

torch.serialization.add_safe_globals(
    [
        minkowski_graspnet.MinkowskiGraspNetMultifingerType1,
        torch.nn.BatchNorm1d,
        torch.nn.Conv1d,
        torch.nn.Dropout,
    ]
)
MESH_PATH = os.path.join(adg_paths.ROOT, "generate_mesh_and_pointcloud/inspire_urdf")
PORT = 47011
AUGMENTS = 10
APPROACH_DIST = 0.06
SCORE_MIN = 0.5
COLLISION_THRESH = 0  # scene points allowed inside the swept hand
PER_TYPE = 100  # best grasps per hand type sent to the collision check (AnyDexGrasp: 50)


def net_grasps(net, points: np.ndarray, augment: np.ndarray, flip: bool):
    """ri.get_grasp from the cloud instead of the RealSense depth image, without its camera-specific filters."""
    pts = ri.transform_point_cloud(points, augment).astype(np.float32)
    pts = torch.from_numpy(pts)
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
    gg = torch.cat(
        [
            preds[:, 0:2],
            torch.full((n, 1), 0.03, device="cuda"),
            preds[:, 2:15],
            -torch.ones((n, 1), device="cuda"),
        ],
        1,
    )
    return gg, features


def select_grasp_type(hand) -> list[int]:
    """ri.select_grasp_type without its off-by-one on type 8: the best PER_TYPE per type (a fifth of that for rings), in score order."""
    kept: dict[int, list[int]] = {}
    for i, grasp_type in enumerate(hand.grasp_types):
        ids = kept.setdefault(int(grasp_type), [])
        if len(ids) < (PER_TYPE // 5 if int(grasp_type) == 1 else PER_TYPE):
            ids.append(i)
    return sorted(i for ids in kept.values() for i in ids)


class Planner:
    def __init__(self):
        t = time.time()
        self.net = ri.get_net(ri.cfgs.checkpoint_path, use_v2=True)
        self.inspire = ri.get_inspire_model(os.path.join(adg_paths.ROOT, "logs/model/inspire_model/obj140"))
        self.meshes = ri.load_meshes_pointcloud(os.path.join(MESH_PATH, "meshes"))
        print(f"models loaded in {time.time() - t:.1f} s", flush=True)

    def plan(self, req: dict) -> list[dict]:
        t0 = time.time()
        points = np.asarray(req["points"], np.float32)
        base_from_cam = np.asarray(req["base_from_cam"])
        up = np.asarray(req.get("up", (0, 0, 1)), float)
        max_tilt = float(req.get("max_tilt", 50.0))
        ggs, feats = [], []
        for i in range(AUGMENTS + 1):
            flip = i % 2 == 0 and i > 0
            augment = np.eye(4) if i == 0 else ri.augment_data(flip=flip)
            gg, f = net_grasps(self.net, points, augment.astype(np.float32), flip)
            if gg is not None and len(gg):
                ggs.append(gg)
                feats.append(f)
        if not ggs:
            return []
        gg = torch.cat(ggs).cpu().numpy()
        feat = torch.cat(feats).cpu().numpy()
        t1 = time.time()

        # both hand orientations about the approach axis: the thumb on either side (AnyDexGrasp keeps one, for a UR)
        flipped = gg.copy()
        rots = flipped[:, 4:13].reshape(-1, 3, 3)
        rots[:, :, 1:3] = -rots[:, :, 1:3]  # half a turn about the approach axis
        flipped[:, 4:13] = rots.reshape(-1, 9)
        gg = np.concatenate([gg, flipped])
        feat = np.concatenate([np.c_[feat, np.zeros(len(feat))], np.c_[feat, np.ones(len(feat))]])

        # approach roughly down in the base frame, centre inside the workspace
        rot_base = base_from_cam[:3, :3] @ gg[:, 4:13].reshape(-1, 3, 3)
        approach = rot_base[:, :, 0]
        ok = approach @ -up > np.cos(np.radians(max_tilt))
        if req.get("workspace") is not None:
            centre = gg[:, 13:16] @ base_from_cam[:3, :3].T + base_from_cam[:3, 3]
            for axis, (lo, hi) in enumerate(req["workspace"]):
                ok &= (centre[:, axis] > lo) & (centre[:, axis] < hi)
        if req.get("near") is not None:
            from scipy.spatial import cKDTree

            near, radius = req["near"]
            centre = gg[:, 13:16] @ base_from_cam[:3, :3].T + base_from_cam[:3, 3]
            ok &= cKDTree(np.asarray(near)[:, :2]).query(centre[:, :2], distance_upper_bound=radius)[0] < radius
        gg, feat = gg[ok], feat[ok]
        order = gg[:, 0].argsort()[::-1][:1000]
        gg, feat = gg[order], feat[order]
        if not len(gg):
            return []

        features = ri.get_graspgroup_features(feat, None)
        depth, htype, scores, gg, feat = ri.get_inspire_depth_type(self.inspire, features, gg, feat)
        keep = scores > SCORE_MIN
        depth, htype, scores, gg, feat = (
            depth[keep],
            htype[keep],
            scores[keep],
            gg[keep],
            feat[keep],
        )
        if not len(gg):
            return []
        two = GraspGroup(gg)
        hand = InspireHandRGraspGroup()
        hand.set_grasp_min_width(ri.MIN_GRASP_WIDTH)
        hand.from_graspgroup(two, htype, MESH_PATH)
        hand.scores = scores
        hand.depths = hand.depths + depth
        order = np.argsort(hand.scores)[::-1]
        hand, two = hand[order], two[order]
        index = select_grasp_type(hand)
        hand, two = hand[index], two[index]
        t2 = time.time()

        print(
            f"{len(hand)} grasps before the collision check: "
            + str(np.bincount(hand.grasp_types.astype(int), minlength=9)[1:].tolist()),
            flush=True,
        )
        if req.get("no_collision"):
            empty = np.ones(len(hand), bool)
        else:
            detector = ModelFreeCollisionDetectorMultifinger(points, voxel_size=0.001)
            hand, two, empty, _ = detector.detect(
                hand,
                two,
                MESH_PATH,
                self.meshes,
                min_grasp_width=ri.MIN_GRASP_WIDTH,
                VoxelGrid=ri.INSPIREHANDR_VOXElGRID,
                approach_dist=APPROACH_DIST,
                collision_thresh=0,
                adjust_gripper_centers=True,
            )
        if len(empty) == 0:
            return []
        hand, two = hand[empty], two[empty]
        order = np.argsort(hand.scores)[::-1][: int(req.get("top", 20))]
        out = []
        for i in order:
            h, g = hand[int(i)], two[int(i)]
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = h.rotation_matrix, h.translation
            out.append(
                {
                    "hand": pose,
                    "approach": np.array(g.rotation_matrix)[:, 0],
                    "push": float(h.depth) + 0.014,
                    "angle": np.array(h.angle, float),
                    "type": int(h.grasp_type),
                    "name": grasp_types[str(int(h.grasp_type))]["name"],
                    "score": float(h.score),
                    "width": float(h.width),
                    "center": np.array(g.translation),
                }
            )
        print(
            f"{len(points)} points: net {t1 - t0:.1f} s, decision {t2 - t1:.1f} s, collision {time.time() - t2:.1f} s, "
            f"{len(out)} grasps",
            flush=True,
        )
        return out


def _recv(conn) -> bytes:
    size = struct.unpack("!Q", conn.recv(8, socket.MSG_WAITALL))[0]
    return conn.recv(size, socket.MSG_WAITALL)


def _send(conn, data: bytes) -> None:
    conn.sendall(struct.pack("!Q", len(data)) + data)


def serve():
    planner = Planner()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", PORT))
        s.listen(1)
        print(f"listening on {PORT}", flush=True)
        while True:
            conn, _ = s.accept()
            with conn:
                try:
                    req = pickle.loads(_recv(conn))
                    if req.get("save"):
                        np.savez(
                            req["save"],
                            points=req["points"],
                            base_from_cam=req["base_from_cam"],
                        )
                    _send(conn, pickle.dumps(planner.plan(req)))
                except Exception as e:  # keep serving; the client sees an empty answer
                    import traceback

                    traceback.print_exc()
                    _send(conn, pickle.dumps({"error": repr(e)}))


if __name__ == "__main__":
    serve()
