"""Record the head and torso cameras side by side, in sim time, with the client's current step as a caption.

record.py OUT.mp4 TITLE [CLIENT_LOG [LIMIT]]; stops on SIGINT/SIGTERM. Frames are taken every 1/FPS s of sim time (the
camera stamps), so the video plays at sim speed whatever the real-time factor. The task clock (LIMIT s of sim time,
default 270, sort2) starts at the client's `auto` command; the score comes from /sim/task/check -> /sim/task/result.
"""

import os
import re
import json
import signal
import sys
import threading

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String

FPS = 15
CHECK_EVERY = 2.0  # s sim time between score requests
CAPTION = re.compile(
    r"^(> |grasp:|ready:|pre:|in:|closed:|lift:|holding|empty|over |place |auto|\d+ points|[rl]: \d+/|no |skip |targets |tray )"
)


def _rgb(msg: Image) -> np.ndarray:
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
    return cv2.cvtColor(rows.reshape(msg.height, msg.width, 3), cv2.COLOR_RGB2BGR)


class Recorder(Node):
    def __init__(self, out: str, title: str, log: str | None, limit: float):
        super().__init__("adg_recorder")
        self.title, self.log, self.limit = title, log, limit
        self.task_start: float | None = None
        self.score, self.at_limit, self.next_check = "", "", 0.0
        self.torso: Image | None = None
        self.writer: cv2.VideoWriter | None = None
        self.out = out
        self.next_t = self.start = None
        self.caption, self.log_pos = "", 0
        self.lock = threading.Lock()
        self.create_subscription(Image, "/cam_head/color/image_raw", self._on_head, qos_profile_sensor_data)
        self.create_subscription(
            Image, "/cam_torso/color/image_raw", lambda m: setattr(self, "torso", m), qos_profile_sensor_data
        )
        self.check = self.create_publisher(Int32, "/sim/task/check", 1)
        self.create_subscription(String, "/sim/task/result", self._on_result, 1)

    def _on_result(self, msg: String) -> None:
        r = json.loads(msg.data)
        if "in_right_tray" in r:
            self.score = f"in the right tray {r['in_right_tray']}/{r['total']}, wrong tray {r['in_wrong_tray']}"

    def _read_log(self) -> None:
        if not self.log or not os.path.exists(self.log):
            return
        with open(self.log) as f:
            f.seek(self.log_pos)
            for line in f.read().splitlines():
                if CAPTION.match(line):
                    self.caption = line.strip()[:110]
                if line.startswith("> auto"):
                    self.task_start = -1.0  # set from this frame's stamp
            self.log_pos = f.tell()

    def _on_head(self, msg: Image) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.next_t is None:
            self.next_t = self.start = t
        if t < self.next_t or self.torso is None:
            return
        self.next_t += 1.0 / FPS * max(1, int((t - self.next_t) * FPS))
        self._read_log()
        if self.task_start == -1.0:
            self.task_start = t
        if t >= self.next_check:
            self.check.publish(Int32(data=1))
            self.next_check = t + CHECK_EVERY
        head, torso = _rgb(msg), _rgb(self.torso)
        frame = np.vstack(
            [
                np.hstack([head, cv2.resize(torso, (head.shape[1], head.shape[0]))]),
                np.zeros((100, 2 * head.shape[1], 3), np.uint8),
            ]
        )
        h = head.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for x, label in ((10, "head camera"), (head.shape[1] + 10, "torso camera")):
            cv2.putText(frame, label, (x, 22), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{self.title}   sim t {t - self.start:6.1f} s",
            (10, h + 26),
            font,
            0.65,
            (120, 220, 255),
            2,
            cv2.LINE_AA,
        )
        if self.task_start is not None:
            elapsed = t - self.task_start
            over = elapsed > self.limit
            if over and not self.at_limit:
                self.at_limit = f"at the limit: {self.score or '?'}"
            clock = f"task time {elapsed:5.1f} / {self.limit:.0f} s" + ("  LIMIT EXCEEDED" if over else "")
            cv2.putText(
                frame,
                clock,
                (2 * head.shape[1] - 470, h + 26),
                font,
                0.65,
                (60, 60, 255) if over else (120, 255, 120),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            "   ".join(x for x in (self.score, self.at_limit) if x),
            (10, h + 56),
            font,
            0.6,
            (120, 255, 120),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(frame, self.caption, (10, h + 86), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        with self.lock:
            if self.writer is None:
                self.writer = cv2.VideoWriter(
                    self.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (frame.shape[1], frame.shape[0])
                )
            self.writer.write(frame)

    def close(self) -> None:
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None


def main():
    rclpy.init()
    node = Recorder(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3] if len(sys.argv) > 3 else None,
        float(sys.argv[4]) if len(sys.argv) > 4 else 270.0,
    )
    node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.is_set():
        ex.spin_once(timeout_sec=0.1)
    node.close()
    print(f"wrote {sys.argv[1]}", flush=True)


if __name__ == "__main__":
    main()
