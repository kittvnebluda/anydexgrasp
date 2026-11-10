"""Look at each tray with the head and locate it with the solution's YOLO: test_trays.py YAW PITCH."""
import json, sys, threading, time
import cv2, numpy as np, rclpy
from bodyctrl_msgs.msg import CmdSetMotorPosition, SetMotorPosition
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from sol_application_py.perception.yolo import YoloPerception, _decode_rgb
from sol_application_py.task import parse_task

rclpy.init()
node = Node("tray_test")
node.set_parameters([Parameter("use_sim_time", value=True)])
task = {}
node.create_subscription(String, "/sim/task", lambda m: task.update(json.loads(m.data)),
                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
head = node.create_publisher(CmdSetMotorPosition, "/head/cmd_pos", 10)
yolo = YoloPerception(node)
ex = MultiThreadedExecutor(); ex.add_node(node); threading.Thread(target=ex.spin, daemon=True).start()
while not task: time.sleep(0.1)
t = parse_task(task); print(json.dumps({k: task[k] for k in ("categories", "trays", "objects")}), flush=True)
for yaw in map(float, sys.argv[1].split(",")):
    pitch = float(sys.argv[2])
    for _ in range(3):
        head.publish(CmdSetMotorPosition(cmds=[SetMotorPosition(name=3, pos=yaw, spd=1.0, cur=1.0),
                                               SetMotorPosition(name=2, pos=pitch, spd=1.0, cur=1.0)]))
        time.sleep(0.1)
    time.sleep(3)
    w = yolo.observe(t)
    print("yaw", yaw, "trays", {k: np.round(v, 3).tolist() for k, v in w.containers.items()},
          "objects", {k[:20]: np.round(d.position, 3).tolist() for k, d in w.objects.items()}, flush=True)
    cam = next(c for c in yolo.cameras if c.name == "cam_head")
    p = cam.pairs(0)
    if p:
        cv2.imwrite(f"/tmp/adg/trays_{yaw}.png", _decode_rgb(p[max(p)][0])[:, :, ::-1])
rclpy.shutdown()
