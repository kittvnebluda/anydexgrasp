"""python -m adg_server [--root DIR] [--port PORT]: load the models, then answer plan requests one at a time."""

import argparse
import os
import socket
import time
import traceback

import numpy as np

from . import compat, protocol
from .planner import Planner


def main() -> None:
    parser = argparse.ArgumentParser(prog="adg_server", description=__doc__)
    parser.add_argument("--root", default=compat.DEFAULT_ROOT, help="the AnyDexGrasp checkout with its weights")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47011)
    parser.add_argument("--dump", metavar="DIR", help="save every request to DIR/request_<n>.npz (to replay one)")
    args = parser.parse_args()

    planner = Planner(compat.load(args.root))
    t = time.time()
    planner.plan(warm_up_request())  # the first plan sets CUDA up (~15 s): before the first client, not during it
    print(f"warmed up in {time.time() - t:.1f} s", flush=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host, args.port))
        s.listen(1)
        print(f"ready on {args.host}:{args.port}", flush=True)
        n = 0
        while True:
            conn, _ = s.accept()
            with conn:
                try:
                    req = protocol.recv(conn)
                    if args.dump:
                        os.makedirs(args.dump, exist_ok=True)
                        np.savez_compressed(os.path.join(args.dump, f"request_{n}.npz"), **req)
                        n += 1
                    if int(req.get("version", -1)) != protocol.VERSION:
                        out = {"error": np.array(f"protocol {req.get('version')}, server speaks {protocol.VERSION}")}
                    else:
                        out = planner.plan(req)
                except Exception as e:  # noqa: BLE001  # keep serving; the client gets the error
                    traceback.print_exc()
                    out = {"error": np.array(repr(e))}
                try:
                    protocol.send(conn, {"version": np.array(protocol.VERSION), **out})
                except OSError:
                    traceback.print_exc()


def warm_up_request() -> dict[str, np.ndarray]:
    """A camera 0.6 m above a table with a 6 cm box on it, looking straight down."""
    rng = np.random.default_rng(0)
    table = np.c_[rng.uniform(-0.3, 0.3, (20000, 2)), np.full(20000, 0.6)]
    box = np.c_[rng.uniform(-0.03, 0.03, (3000, 2)), np.full(3000, 0.54)]
    base_from_cam = np.diag([1.0, -1.0, -1.0, 1.0])  # optical z down
    return {"points": np.vstack((table, box)).astype(np.float32), "base_from_cam": base_from_cam, "top": np.array(5)}


if __name__ == "__main__":
    main()
