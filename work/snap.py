"""Save the latest cam_head color image as PNG: snap.py OUT.png [camera]."""

import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

rclpy.init()
node = rclpy.create_node("snap")
cam = sys.argv[2] if len(sys.argv) > 2 else "cam_head"
got = []
node.create_subscription(Image, f"/{cam}/color/image_raw", got.append, qos_profile_sensor_data)
end = time.time() + 10
while not got and time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.2)
msg = got[-1]
rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
cv2.imwrite(sys.argv[1], cv2.cvtColor(rows, cv2.COLOR_RGB2BGR))
print("saved", sys.argv[1])
