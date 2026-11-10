"""Tilt the head, save the cut head-camera cloud and pelvis <- camera: grab_cloud.py OUT.npz."""

import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from adg_client import Client

rclpy.init()
node = Client()
ex = MultiThreadedExecutor()
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()
node.tilt_head()
time.sleep(2.0)
points, base_from_cam = node.cloud()
np.savez(sys.argv[1], points=points, base_from_cam=base_from_cam)
print(len(points), "points")
ex.shutdown()
