"""Send the recorded scene to a running server and check grasps come back: python -m tests.smoke [PORT]."""

import socket
import sys
import time

import numpy as np

from adg_server import protocol

from .bench import request

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 47011
    req = {"version": np.array(protocol.VERSION), **request(near=True)}
    for _ in range(2):
        t = time.time()
        with socket.create_connection(("127.0.0.1", port), timeout=120) as s:
            protocol.send(s, req)
            out = protocol.recv(s)
        print(f"{len(out.get('score', []))} grasps in {time.time() - t:.1f} s", out.get("error", ""))
    assert "error" not in out and len(out["score"]) > 0
    print("smoke ok")
