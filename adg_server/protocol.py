"""Messages between the server and its clients: an 8-byte big-endian length, then an .npz archive (no pickles).

Request: version, points (N, 3) float32 in the camera's optical frame, base_from_cam (4, 4) base <- camera, up (3,)
in the base frame, max_tilt deg (approach off -up), top (grasps wanted), workspace (3, 2) base-frame box for the grasp
centres, near (M, 2) base-frame xy and radius m (only grasp centres that close to one of them).

Reply, best first, poses in the optical frame: version, hand (K, 4, 4) AnyDexGrasp's hand frame (urdf-five3 Link111
shifted to the wrist centre), approach (K, 3) the two-finger approach axis, push (K,) m the hand goes in along it from
`hand` before closing, angle (K, 6) preshape in Inspire units 0..1000 (little, ring, middle, index, thumb bend, thumb
rotation), type (K,), name (K,), score (K,), width (K,) m, center (K, 3) the two-finger grasp centre; or version and
error.
"""

import io
import socket
import struct

import numpy as np

VERSION = 1


def send(sock: socket.socket, arrays: dict[str, np.ndarray]) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    sock.sendall(struct.pack("!Q", len(data)) + data)


def recv(sock: socket.socket) -> dict[str, np.ndarray]:
    def exact(n: int) -> bytes:
        chunks = []
        while n:
            chunk = sock.recv(min(n, 1 << 20))
            if not chunk:
                raise ConnectionError("peer closed the connection")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    size = struct.unpack("!Q", exact(8))[0]
    with np.load(io.BytesIO(exact(size)), allow_pickle=False) as npz:
        return dict(npz)
