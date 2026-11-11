# adg_server

[AnyDexGrasp](https://github.com/graspnet/AnyDexGrasp) for the Inspire hand, served over a local socket. A client
sends a depth cloud and gets ranked Inspire-hand grasps back. The robot side lives in the client: the
[nssr-sol-team-071](https://github.com/kittvnebluda/nssr-sol-team-071) solution's `adg_pick.py`.

| Path | What |
| --- | --- |
| `AnyDexGrasp/` | upstream, unchanged (submodule) |
| `MinkowskiEngine/` | fork patched for CUDA 12.8 / 13, CCCL 3, torch 2.11+ (submodule, branch `cuda13`) |
| `adg_server/` | the server: `planner.py` AnyDexGrasp's planning on a given cloud, `collision.py` its hand collision check on the GPU, `hand.py` its two-finger to hand mapping without the quadratic loop, `protocol.py` the messages |
| `install.sh` | venv with torch and the prebuilt extensions, weights; from the GitHub release |
| `scripts/` | `build_ext.sh` / `pack_assets.sh` make the release files, `record_scene.py` saves a head-camera scene from the sim |
| `tests/` | on a recorded scene (GPU and weights): the GPU collision check against the original, `bench.py` timings |

## Run

```bash
git clone --recursive git@github.com:kittvnebluda/anydexgrasp.git && cd anydexgrasp && ./install.sh
venv/bin/python -m adg_server          # loads the models (~20 s), then "ready on 127.0.0.1:47011"
```

`--dump DIR` saves every request, to replay one: `planner.plan(dict(np.load(...)))`.

## Protocol

Each message is an 8-byte big-endian length and an `.npz` archive (no pickles); fields in `adg_server/protocol.py`.
Request: the cloud in the camera's optical frame, the base <- camera transform, a workspace box and target points
(grasp centres near them). Reply, best first: the hand pose (AnyDexGrasp's hand frame, in the camera frame), the
approach axis and how far to push in along it, the 6 finger preshape, type, score.

## What it does per request (~3 s, RTX 5060 Ti)

1. The two-finger network on the cloud and 10 rotated/flipped copies (1.1 s).
2. Kept: approach within `max_tilt` of down, centre in the workspace and near the targets; both thumb sides.
3. Decision models: hand type, extra depth, score; mapped to Inspire poses and preshapes (`hand.py`).
4. 100 best per type (720) into the collision check: the hand swept 7 cm back along its approach must miss the
   cloud (0.3 s; AnyDexGrasp's open3d loop took 35 s and agrees within a voxel: ~4 % of grasps differ, both ways).

## Release files (judges' container: Ubuntu 22.04, Python 3.10, no CUDA toolkit)

```bash
scripts/build_ext.sh      # docker, CUDA 12.8 devel: MinkowskiEngine wheel, pointnet2 + knn modules -> dist/
scripts/pack_assets.sh    # weights + voxel_size_3 hand clouds -> dist/adg-assets.tar
gh release create v0.1.0 dist/*      # the tag install.sh fetches is scripts/versions.sh RELEASE
```

The host dev venv (`.venv`, Python 3.12, torch 2.14 cu130) builds MinkowskiEngine in place:
`CC=gcc-15 CXX=g++-15 TORCH_CUDA_ARCH_LIST=12.0 python setup.py build_ext --inplace --force_cuda --blas=blas` in
`MinkowskiEngine/` (copy the .so from `build/`), the same for `AnyDexGrasp/pointnet2` and `AnyDexGrasp/knn`.

AnyDexGrasp is licensed CC BY-NC 4.0 (non-commercial).
