"""AnyDexGrasp grasps executed by the TienKung right arm in the sim, one command at a time.

Needs adg_server.py running on the host (venv with MinkowskiEngine). Run inside the ROS distrobox, workspace sourced:
    python3 adg_client.py           commands from $ADG_DIR/cmd.fifo (default /tmp/adg), each ends with <done>

Commands:
    label [off]      plans keep only grasps on YOLO-detected objects (off: all)
    plan [r|l] [X Y R]  for the right (default) or left arm: head-camera cloud of the table (around X, Y within R m if given) -> grasps, reachable ones
                     listed best first and saved
    go [I]           execute grasp I (default 0): ready, preshape, pre-grasp, straight in, close, lift
    auto [N]         sort: look at the trays, then up to N picks; YOLO names the object under each grasp, it goes to its
                     category's tray with the arm on that tray's side; heavy objects are skipped; /sim/task/done at the end
    place X Y        carry the held object over (X, Y) at its height and open the hand
    open | home      open the hand / open and fold the arm
    hand c1 .. c6    one hand command, 1 open .. 0 closed (little, ring, middle, index, thumb bend, thumb rotation)
    drop             open the hand where it is
    quit
"""

import json
import os
import pickle
import socket
import struct
import sys
import threading
import time

import numpy as np
import pinocchio as pin
import rclpy
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from bodyctrl_msgs.srv import SetForce
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from sol_application_py.motion.control import Arm
from sol_application_py.motion.poses import TABLE_TOP
from sol_application_py.perception.yolo import YoloPerception
from sol_application_py.task import parse_task
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener

PORT = 47011
RUN_DIR = os.environ.get("ADG_DIR", "/tmp/adg")
CAMERA = "cam_head"
HEAD_TILT = 0.44  # rad, head pitch that shows the table
# table region the cloud is cut to, `pelvis`, both sides (the table top stays in: the network saw tables); below the
# folded (homed) hands, so they are not grasped
WORKSPACE = ((0.22, 0.80), (-0.70, 0.70), (TABLE_TOP - 0.03, TABLE_TOP + 0.17))
GRASP_REGION = (
    (0.25, 0.70),
    (-0.48, 0.20),  # not into the tray on that side; over the middle, the other arm's side is out of reach
    (TABLE_TOP + 0.015, 0.20),  # not the table top itself
)  # where a grasp centre may be
MAX_TILT = 55.0  # deg, approach off straight down
APPROACH = 0.08  # m, pre-grasp back along the approach
LIFTS = (0.25, 0.18, 0.12)  # m, straight up after closing, the highest that reaches: clear of the other objects
DESCENT_FORCE = 8.0  # N
HELD = 0.3  # mean closing finger position after the lift above which the hand holds something
TRAY_SIDE = {"tray_right": "r", "tray_left": "l"}
TRAY_LOOK = {"tray_right": -0.9, "tray_left": 0.9}  # rad, head yaw (+ left) that shows the whole tray
TRAY_PITCH = 0.6  # rad
TRAY_DEFAULT = {"tray_right": (0.36, -0.69), "tray_left": (0.36, 0.69)}  # pelvis, if the survey misses one
TRAY_HALF = (0.11 + 0.03, 0.16 + 0.03)  # m, half the tray (0.22 x 0.32) and a margin: objects inside are done
# objects one Inspire hand does not hold (hackathon/config/objects.yaml masses); others over 0.4 kg in other pools:
# Deskstar_Desk_Top_Hard_Drive_1_TB 0.65 (pool 2), 006_mustard_bottle 0.43 (pool 3)
HEAVY = {"004_sugar_box", "JBL_Charge_Speaker_portable_wireless_wired_Green"}  # 0.51, 0.60 kg
MATCH = 0.02  # m, grasp centre to the nearest point of an object's cloud in xy (the centre is inside the object)
MAX_MISSES = 3  # attempts per object before it is left
# an object on the far half from its tray is put down on the other half for the other arm, at the free spot of this
# area (x, |y| on that half) farthest from every other object
RELAY_AREA = ((0.30, 0.52), (0.05, 0.22))
TRAY_DROP = 0.10  # m above the table: the held object's bottom when the hand opens over a tray (rim 0.07)
# AnyDexGrasp's hand frame: the Inspire urdf-five3 Link111 shifted to the wrist centre (recover_inspire_hand_to_stl)
HAND_FROM_LINK111 = pin.SE3(np.eye(3), np.array([0.04123 + 0.0078, 0.00804, -0.01796]))
# the same palm in our URDF's R_base_link: fingers +x -> +z, index side -z -> +y, palm side -y -> +x; offset from the
# four finger base joints (they agree within 2 mm; the thumb base within 3 mm)
RBASE_FROM_LINK111 = pin.SE3(
    np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
    np.array([-0.008, 0.0167, 0.0553]),
)


def _quat_matrix(q) -> np.ndarray:
    return pin.Quaternion(q.w, q.x, q.y, q.z).toRotationMatrix()


class Client(Node):
    def __init__(self):
        super().__init__("adg_client")
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.depth: Image | None = None
        self.k: np.ndarray | None = None
        self.create_subscription(Image, f"/{CAMERA}/depth/image_raw", self._on_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo,
            f"/{CAMERA}/depth/camera_info",
            self._on_info,
            qos_profile_sensor_data,
        )
        self.color: dict[str, Image] = {}
        for cam in ("cam_head", "cam_torso"):
            self.create_subscription(
                Image, f"/{cam}/color/image_raw", lambda m, c=cam: self.color.__setitem__(c, m), qos_profile_sensor_data
            )
        self.head = self.create_publisher(CmdSetMotorPosition, "/head/cmd_pos", 10)
        self.task: dict = {}
        self.create_subscription(
            String,
            "/sim/task",
            lambda m: self.task.update(json.loads(m.data)),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.done = self.create_publisher(Empty, "/sim/task/done", 1)
        self.yolo = YoloPerception(self)
        self.force = {
            s: self.create_client(SetForce, f"/inspire_hand/set_force/{n}_hand")
            for s, n in (("r", "right"), ("l", "left"))
        }

    def _on_depth(self, msg: Image) -> None:
        self.depth = msg

    def _on_info(self, msg: CameraInfo) -> None:
        self.k = np.array(msg.k).reshape(3, 3)

    def snap(self, tag: str) -> None:
        """Both cameras' latest color images to RUN_DIR/<tag>_<camera>.png."""
        import cv2

        time.sleep(0.3)
        for cam, msg in list(self.color.items()):
            rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
            img = cv2.cvtColor(rows.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(RUN_DIR, f"{tag}_{cam}.png"), img)

    def tilt_head(self, yaw: float = 0.0, pitch: float = HEAD_TILT) -> None:
        for _ in range(3):
            self.head.publish(
                CmdSetMotorPosition(
                    cmds=[
                        SetMotorPosition(name=3, pos=yaw, spd=1.0, cur=1.0),
                        SetMotorPosition(name=2, pos=pitch, spd=1.0, cur=1.0),
                    ]
                )
            )
            time.sleep(0.1)

    def set_force(self, side: str, ratio: float) -> None:
        client = self.force[side]
        if not client.wait_for_service(timeout_sec=2.0):
            print("no set_force service", flush=True)
            return
        req = SetForce.Request()
        for i in range(6):
            setattr(req, f"force{i}_ratio", ratio)
        client.call_async(req)

    def cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Latest depth frame -> points in the optical frame, cut to WORKSPACE; and pelvis <- camera."""
        since = self.get_clock().now()
        while self.depth is None or self.k is None or Time.from_msg(self.depth.header.stamp) < since:
            time.sleep(0.05)
        msg = self.depth
        tf = self.tf.lookup_transform(
            "pelvis",
            msg.header.frame_id,
            Time.from_msg(msg.header.stamp),
            Duration(seconds=1.0),
        )
        base_from_cam = np.eye(4)
        base_from_cam[:3, :3] = _quat_matrix(tf.transform.rotation)
        t = tf.transform.translation
        base_from_cam[:3, 3] = (t.x, t.y, t.z)
        depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width] / 1000.0
        v, u = np.nonzero(depth > 0)
        z = depth[v, u]
        k = self.k
        cam = np.column_stack(((u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z)).astype(np.float32)
        base_from_cam = level(cam @ base_from_cam[:3, :3].T + base_from_cam[:3, 3]) @ base_from_cam
        base = cam @ base_from_cam[:3, :3].T + base_from_cam[:3, 3]
        keep = np.ones(len(cam), bool)
        for axis, (lo, hi) in enumerate(WORKSPACE):
            keep &= (base[:, axis] > lo) & (base[:, axis] < hi)
        return cam[keep], base_from_cam


def level(base: np.ndarray) -> np.ndarray:
    """Correction (4,4) that puts the table plane seen in `base` (pelvis points) level at TABLE_TOP.

    The head camera's cloud has the table rising ~3 cm over 0.4 m away from the robot; grasps are planned on it.
    """
    pts = base[(base[:, 0] > 0.25) & (base[:, 0] < 0.85) & (np.abs(base[:, 1]) < 0.7)]
    pts = pts[np.abs(pts[:, 2] - TABLE_TOP) < 0.05]
    inliers = pts
    for tol in (0.03, 0.01, 0.005):
        a = np.c_[inliers[:, :2], np.ones(len(inliers))]
        coef = np.linalg.lstsq(a, inliers[:, 2], rcond=None)[0]  # z = cx x + cy y + c0
        inliers = pts[np.abs(pts[:, 2] - np.c_[pts[:, :2], np.ones(len(pts))] @ coef) < tol]
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    angle = np.arcsin(min(1.0, np.linalg.norm(axis)))
    rot = pin.exp3(axis / max(np.linalg.norm(axis), 1e-12) * angle)
    centre = np.array([0.45, 0.0, coef[0] * 0.45 + coef[2]])  # a table point near the objects
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.array([0.0, 0.0, TABLE_TOP]) - rot @ centre + np.array([centre[0], centre[1], 0.0])
    print(
        f"table plane: slope x {coef[0]:.3f} y {coef[1]:.3f}, {len(inliers)} points, tilt {np.degrees(angle):.1f} deg",
        flush=True,
    )
    return out


def request(req: dict):
    with socket.create_connection(("127.0.0.1", PORT)) as s:
        data = pickle.dumps(req)
        s.sendall(struct.pack("!Q", len(data)) + data)
        size = struct.unpack("!Q", s.recv(8, socket.MSG_WAITALL))[0]
        return pickle.loads(s.recv(size, socket.MSG_WAITALL))


MIRROR = np.diag([1.0, -1.0, 1.0])  # pelvis y; the left arm and hand are the right ones mirrored in it (TCP y too)
FLIP_X = np.diag([-1.0, 1.0, 1.0])  # optical x: a mirrored camera image


class Executor:
    """Plans for the right hand (AnyDexGrasp's); for the left arm the scene is mirrored first and the grasp back."""

    def __init__(self, node: Client, arms: dict[str, Arm]):
        self.node, self.arms = node, arms
        ik = arms["r"].ik
        pin.framesForwardKinematics(ik.model, ik.data, pin.neutral(ik.model))
        tcp = ik.data.oMf[ik.model.getFrameId("right_tcp_link")]
        rbase = ik.data.oMf[ik.model.getFrameId("R_base_link")]
        self.tcp_from_hand = tcp.actInv(rbase) * RBASE_FROM_LINK111 * HAND_FROM_LINK111.inverse()
        self.grasps: list[dict] = []
        self.side = "r"
        self.targets: dict | None = None  # name -> Detection: auto keeps only grasps on these objects
        self.trays: dict[str, np.ndarray] = {}
        self.bottom = 0.0  # set by go()

    def plan(self, args: list[str]) -> None:
        side = args.pop(0) if args and args[0] in ("r", "l") else "r"
        arm = self.arms[side]
        self.node.tilt_head()
        time.sleep(1.0)
        points, base_from_cam = self.node.cloud()
        region = GRASP_REGION
        if args:
            x, y, r = map(float, args[:3])
            region = ((x - r, x + r), (y - r, y + r), GRASP_REGION[2])
        near = None if self.targets is None else np.concatenate([d.points[:, :2] for d in self.targets.values()])
        if side == "l":  # the right hand on the mirrored scene; the camera image mirrored to stay a camera view
            near = None if near is None else near * (1.0, -1.0)
            points = points @ FLIP_X
            base_from_cam = base_from_cam.copy()
            base_from_cam[:3, :3] = MIRROR @ base_from_cam[:3, :3] @ FLIP_X
            base_from_cam[:3, 3] = MIRROR @ base_from_cam[:3, 3]
            if args:  # given in the real scene; GRASP_REGION is the right arm's side, in the mirror the left one's
                region = (region[0], (-region[1][1], -region[1][0]), region[2])
        t = time.time()
        os.makedirs(RUN_DIR, exist_ok=True)
        out = request(
            {
                "points": points,
                "base_from_cam": base_from_cam,
                "up": (0.0, 0.0, 1.0),
                "max_tilt": MAX_TILT,
                "top": 60,
                "workspace": region,
                "near": None if near is None else (near, MATCH),
                "save": os.path.join(RUN_DIR, "cloud.npz"),
            }
        )
        if isinstance(out, dict):
            print("server error:", out["error"], flush=True)
            return
        print(
            f"{len(points)} points, {len(out)} grasps in {time.time() - t:.1f} s",
            flush=True,
        )
        base = pin.SE3(base_from_cam[:3, :3], base_from_cam[:3, 3])
        q = arm.q()
        self.grasps, self.side = [], side
        unlabelled: list[np.ndarray] = []
        for g in out:
            hand = base * pin.SE3(g["hand"][:3, :3], g["hand"][:3, 3])
            approach = base.rotation @ g["approach"]
            tcp = hand * self.tcp_from_hand.inverse()
            centre = base.act(g["center"])
            if side == "l":
                tcp = pin.SE3(MIRROR @ tcp.rotation @ MIRROR, MIRROR @ tcp.translation)
                approach, centre = MIRROR @ approach, MIRROR @ centre
            if self.targets is not None:
                g["object"] = label(centre, self.targets)
                if g["object"] is None:
                    unlabelled.append(centre)
                    continue
            grasp = pin.SE3(tcp.rotation, tcp.translation + g["push"] * approach)
            pre = pin.SE3(tcp.rotation, grasp.translation - APPROACH * approach)
            q_pre = arm.solve(pre, q)
            q_grasp = None if q_pre is None else arm.solve(grasp, q_pre)
            reach = q_grasp is not None and np.abs(q_grasp - q_pre).max() < 0.6
            if not reach:
                why = "pre IK" if q_pre is None else "grasp IK" if q_grasp is None else "joint jump"
                print(
                    f"   unreachable ({why}): {g['name']} tcp {np.round(grasp.translation, 3).tolist()} "
                    f"fingers {np.round(-tcp.rotation[:, 2], 2).tolist()} closing {np.round(tcp.rotation[:, 1], 2).tolist()}",
                    flush=True,
                )
            g.update(
                pre=pre,
                grasp=grasp,
                approach_base=approach,
                centre_base=centre,
                reach=reach,
            )
            if reach:
                self.grasps.append(g)
        for i, g in enumerate(self.grasps[:12]):
            print(
                f"{i:2d} {g['name']:18s} {g.get('object', '')[:24]:24s} score {g['score']:.2f} width {g['width']:.3f} centre "
                f"{np.round(g['centre_base'], 3).tolist()} approach {np.round(g['approach_base'], 2).tolist()} "
                f"angle {g['angle'].astype(int).tolist()}",
                flush=True,
            )
        if unlabelled:
            print(f"   {len(unlabelled)} grasps on no target object", flush=True)
            with open(os.path.join(RUN_DIR, "unlabelled.pkl"), "wb") as f:
                pickle.dump((unlabelled, {n: (d.position, d.points) for n, d in self.targets.items()}), f)
        print(f"{side}: {len(self.grasps)}/{len(out)} reachable", flush=True)
        with open(os.path.join(RUN_DIR, "grasps.pkl"), "wb") as f:
            pickle.dump((side, self.grasps), f)

    def survey(self) -> None:
        """Each tray seen with the head turned to it (straight ahead only its near wall is in view)."""
        task = parse_task(self.node.task)
        for tray in task.containers:
            self.node.tilt_head(TRAY_LOOK.get(tray, 0.0), TRAY_PITCH)
            time.sleep(2.5)  # the head turns at 1 rad/s and lags its command
            for _ in range(3):
                world = self.node.yolo.observe(task)
                if tray in world.containers:
                    self.trays[tray] = np.array(world.containers[tray])
                    print(f"tray {tray} at {np.round(self.trays[tray], 3).tolist()}", flush=True)
                    break
                time.sleep(1.0)
            else:
                self.trays[tray] = np.array([*TRAY_DEFAULT[tray], TABLE_TOP])
                print(f"tray {tray} not seen, assumed at {TRAY_DEFAULT[tray]}", flush=True)
            self.node.tilt_head(0.0, TRAY_PITCH)
            time.sleep(1.5)
        self.node.tilt_head()
        time.sleep(1.5)

    def objects(self) -> dict:
        """Detected objects not yet in a tray."""
        world = self.node.yolo.observe(parse_task(self.node.task))
        return {
            n: d
            for n, d in world.objects.items()
            if not any(
                abs(d.position[0] - t[0]) < TRAY_HALF[0] and abs(d.position[1] - t[1]) < TRAY_HALF[1]
                for t in self.trays.values()
            )
        }

    def go(self, args: list[str]) -> bool:
        """Execute a planned grasp; True when the hand still holds something after the lift."""
        g = self.grasps[int(args[0]) if args else 0]
        arm = self.arms[self.side]
        print(
            f"grasp: {g.get('object', '?')[:30]}, {g['name']}, score {g['score']:.2f}, "
            f"centre {np.round(g['centre_base'], 3).tolist()}",
            flush=True,
        )
        preshape = list(np.clip(g["angle"] / 1000.0, 0.0, 1.0))
        r = arm.ready()
        print("ready:", r.reason, flush=True)
        arm.hand(preshape)
        r = arm.move_tcp(g["pre"])
        print("pre:", r.reason, flush=True)
        self.node.snap("0_pre")
        if not r.ok:
            return False
        r = arm.move_linear(g["grasp"], max_force=DESCENT_FORCE)
        print("in:", r.reason, flush=True)
        self.node.snap("1_in")
        if r.reason not in ("ok", "contact", "blocked", "residual"):
            return False
        self.node.set_force(self.side, 1.0)
        # the fingers the type uses (part bent: unused ones are folded at 0 or open at 1) and the thumb bend close;
        # the thumb rotation stays
        close = [0.0 if 0.01 < p < 0.99 else p for p in preshape[:4]] + [
            0.0,
            preshape[5],
        ]
        state = arm.hand(close)
        print(
            "closed:",
            None if state is None else np.round(state.position, 2).tolist(),
            flush=True,
        )
        self.node.snap("2_closed")
        tcp = arm.tcp()
        self.bottom = (
            tcp.translation[2] - TABLE_TOP
        )  # TCP height above the held object's bottom (it stood on the table)
        for lift in LIFTS:
            up = pin.SE3(tcp.rotation, tcp.translation + np.array([0.0, 0.0, lift]))
            if arm.solve(up) is not None:
                break
        r = arm.move_linear(up)
        state = arm.driver.hand_state()
        print(
            "lift:",
            r.reason,
            "hand",
            None if state is None else np.round(state.position, 2).tolist(),
            flush=True,
        )
        self.node.snap("3_lift")
        # closing fingers that stopped well short of closed are held open by the object
        used = [i for i in range(5) if close[i] < 0.5]
        held = state is not None and float(np.mean(state.position[used])) > HELD
        print("holding" if held else "empty", flush=True)
        return held

    def auto(self, args: list[str]) -> None:
        """Sort: each object into its category's tray with the arm on that tray's side, until N attempts are used.

        An object out of that arm's reach (on the other half of the table) is first moved by the other arm to the middle.
        """
        attempts = int(args[0]) if args else 10
        task = parse_task(self.node.task)
        self.survey()
        done, misses = 0, {}
        for i in range(attempts):
            objects = self.objects()
            for n in objects:
                if n in HEAVY:
                    print(f"skip {n[:30]}: too heavy for one hand", flush=True)
            todo = {
                n: d
                for n, d in objects.items()
                if n in task.targets and n not in HEAVY and misses.get(n, 0) < MAX_MISSES
            }
            per_side: dict[str, dict] = {"r": {}, "l": {}}
            for n, d in todo.items():
                side = TRAY_SIDE.get(next(iter(task.targets[n])), "r")
                reach = d.position[1] < GRASP_REGION[1][1] if side == "r" else d.position[1] > -GRASP_REGION[1][1]
                per_side[side if reach else other(side)][n] = d
            sides = sorted((s for s in "rl" if per_side[s]), key=lambda s: -len(per_side[s]))
            for side in sides:
                print(f"targets {side}: {', '.join(n[:20] for n in per_side[side])}", flush=True)
                self.targets = per_side[side]
                self.plan([side])
                if self.grasps:
                    break
            else:
                print("auto: nothing left to pick", flush=True)
                break
            self.targets = None
            # after a miss on an object, its next best grasp
            g = self.grasps[0]
            same = [j for j, h in enumerate(self.grasps) if h["object"] == g["object"]]
            idx = same[min(misses.get(g["object"], 0), len(same) - 1)]
            name = self.grasps[idx]["object"]
            tray = next(iter(task.targets[name]))
            relay = TRAY_SIDE.get(tray) != self.side
            arm = self.arms[self.side]
            held = self.go([str(idx)])
            if held and relay:
                x, y = relay_spot(
                    [d.points for n, d in objects.items() if n != name], -1.0 if self.side == "l" else 1.0
                )
                place(arm, x, y, TABLE_TOP + self.bottom - 0.10, contact=True)  # it may have stood on something
                outcome = f"put down at ({x:.2f}, {y:.2f}) for the other arm"
            elif held and tray in self.trays:
                x, y = self.trays[tray][:2]
                place(arm, float(x), float(y), TABLE_TOP + self.bottom + TRAY_DROP)
                self.node.snap(f"auto{i}_placed")
                done += 1
                outcome = f"placed in {tray}"
            else:
                misses[name] = misses.get(name, 0) + 1
                arm.hand([1.0] * 6)
                outcome = "missed"
            for a in self.arms.values():
                a.home()
            print(f"auto {i}: {name[:30]} {outcome} ({done} placed)", flush=True)
        self.node.done.publish(Empty())
        print(f"auto: {done} placed, task done sent", flush=True)


def relay_spot(clouds: list, sign: float) -> tuple[float, float]:
    """The point of RELAY_AREA (y times sign) farthest in xy from all the clouds."""
    xs = np.linspace(*RELAY_AREA[0], 12)
    ys = sign * np.linspace(*RELAY_AREA[1], 9)
    grid = np.array([(x, y) for x in xs for y in ys])
    pts = np.concatenate([c[:, :2] for c in clouds if c is not None and len(c)]) if clouds else np.empty((0, 2))
    if not len(pts):
        return float(grid[len(grid) // 2][0]), float(grid[len(grid) // 2][1])
    clear = np.min(np.linalg.norm(grid[:, None, :] - pts[None, :, :], axis=2), axis=1)
    x, y = grid[int(np.argmax(clear))]
    print(f"relay spot ({x:.2f}, {y:.2f}), {clear.max():.2f} m clear", flush=True)
    return float(x), float(y)


def other(side: str) -> str:
    return "l" if side == "r" else "r"


def label(centre: np.ndarray, objects: dict) -> str | None:
    """The object whose cloud is nearest to a grasp centre in xy, within MATCH, and not above its top."""
    best, dist = None, MATCH
    for name, det in objects.items():
        if det.points is None or not len(det.points) or centre[2] > det.position[2] + MATCH:
            continue
        d = float(np.min(np.linalg.norm(det.points[:, :2] - centre[:2], axis=1)))
        if d < dist:
            best, dist = name, d
    return best


def place(arm: Arm, x: float, y: float, tcp_z: float | None = None, contact: bool = False) -> None:
    """Carry the held object over (x, y) at the current height, hand orientation kept (else turned about z), lower the
    TCP to `tcp_z` if given (stopping on contact if `contact`), open."""
    tcp = arm.tcp()
    for yaw in (0, 30, -30, 60, -60, 90, -90):
        rot = pin.utils.rotate("z", np.radians(yaw)) @ tcp.rotation
        goal = pin.SE3(rot, np.array([x, y, tcp.translation[2]]))
        if arm.solve(goal) is not None:
            # a straight line where it reaches: the joint-space move swings the hand and shakes the object out
            r = arm.move_linear(goal)
            if r.reason == "unreachable":
                r = arm.move_tcp(goal)
            print(f"over ({x:.3f}, {y:.3f}), yaw {yaw}: {r.reason}", flush=True)
            break
    else:
        print("place unreachable", flush=True)
        return
    if tcp_z is not None and tcp_z < goal.translation[2]:
        down = pin.SE3(goal.rotation, np.array([x, y, tcp_z]))
        r = arm.move_linear(down, max_force=DESCENT_FORCE) if contact else arm.move_linear(down)
        print(f"down to {tcp_z:.3f}: {r.reason}", flush=True)
    arm.hand([1.0] * 6)
    time.sleep(1.0)
    tcp = arm.tcp()
    arm.move_linear(pin.SE3(tcp.rotation, tcp.translation + np.array([0.0, 0.0, 0.08])))  # out of the tray first


def main():
    rclpy.init()
    node = Client()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    arms = {s: Arm(node, s) for s in ("r", "l")}
    for a in arms.values():
        a.ready_to_move()
    run = Executor(node, arms)
    os.makedirs(RUN_DIR, exist_ok=True)
    fifo = os.path.join(RUN_DIR, "cmd.fifo")
    if not os.path.exists(fifo):
        os.mkfifo(fifo)
    print("ready for commands", flush=True)
    while True:
        with open(fifo) as f:
            lines = [text.split() for text in f.read().splitlines() if text.strip()]
        for line in lines:
            cmd, args = line[0], line[1:]
            if cmd == "quit":
                ex.shutdown()
                sys.exit(0)
            print(">", " ".join(line), flush=True)
            try:
                if cmd == "plan":
                    run.plan(args)
                elif cmd == "label":  # following plans keep only grasps on detected objects ("label off": all)
                    run.targets = None if args == ["off"] else run.objects()
                    print("targets:", None if run.targets is None else sorted(run.targets), flush=True)
                elif cmd == "go":
                    run.go(args)
                elif cmd == "auto":
                    run.auto(args)
                elif cmd == "place":
                    place(arms[run.side], float(args[0]), float(args[1]))
                    run.node.snap("4_place")
                elif cmd == "open" or cmd == "drop":
                    arms[run.side].hand([1.0] * 6)
                elif cmd == "home":
                    for a in arms.values():
                        a.hand([1.0] * 6)
                        print("home:", a.home().reason, flush=True)
                elif cmd == "hand":
                    state = arms[run.side].hand([float(a) for a in args])
                    print(
                        None if state is None else np.round(state.position, 2).tolist(),
                        flush=True,
                    )
                else:
                    print("unknown command", flush=True)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print("error:", e, flush=True)
            print("<done>", flush=True)


if __name__ == "__main__":
    main()
