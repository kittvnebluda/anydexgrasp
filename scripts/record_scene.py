"""Save the head camera's next depth frame as a scene: record_scene.py OUT.npz (ROS 2, the sim running).

The head is tilted to the table first. The archive holds `points` (N, 3) float32 in the optical frame and
`base_from_cam` (4, 4) pelvis <- optical frame; tests/ plan on it.
"""

import sys
import time

import numpy as np
import rclpy
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

HEAD_PITCH = 0.44  # rad, the solution's head pose over the table


def main() -> None:
    rclpy.init()
    node = Node("record_scene", parameter_overrides=[Parameter("use_sim_time", value=True)])
    tf = Buffer()
    TransformListener(tf, node)
    got: dict = {}
    node.create_subscription(
        Image, "/cam_head/depth/image_raw", lambda m: got.__setitem__("depth", m), qos_profile_sensor_data
    )
    node.create_subscription(
        CameraInfo, "/cam_head/depth/camera_info", lambda m: got.__setitem__("info", m), qos_profile_sensor_data
    )
    head = node.create_publisher(CmdSetMotorPosition, "/head/cmd_pos", 10)
    end = time.monotonic() + 4.0
    while time.monotonic() < end:
        head.publish(
            CmdSetMotorPosition(
                cmds=[
                    SetMotorPosition(name=3, pos=0.0, spd=0.8, cur=1.0),
                    SetMotorPosition(name=2, pos=HEAD_PITCH, spd=0.8, cur=1.0),
                ]
            )
        )
        rclpy.spin_once(node, timeout_sec=0.1)
    got.clear()
    while "depth" not in got or "info" not in got:
        rclpy.spin_once(node, timeout_sec=0.1)
    msg, k = got["depth"], np.array(got["info"].k).reshape(3, 3)
    t = tf.lookup_transform("pelvis", msg.header.frame_id, Time.from_msg(msg.header.stamp), Duration(seconds=2.0))
    q, p = t.transform.rotation, t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    base_from_cam = np.eye(4)
    base_from_cam[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    base_from_cam[:3, 3] = (p.x, p.y, p.z)
    depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width] / 1000.0
    v, u = np.nonzero(depth > 0)
    d = depth[v, u]
    points = np.column_stack(((u - k[0, 2]) * d / k[0, 0], (v - k[1, 2]) * d / k[1, 1], d)).astype(np.float32)
    np.savez_compressed(sys.argv[1], points=points, base_from_cam=base_from_cam)
    print(f"{len(points)} points -> {sys.argv[1]}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
