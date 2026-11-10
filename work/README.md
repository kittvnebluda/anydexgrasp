# AnyDexGrasp on the TienKung / Inspire hands (sim)

AnyDexGrasp (github.com/graspnet/AnyDexGrasp) plans Inspire-hand grasps from a depth cloud: a GraspNet-style
two-finger network (MinkowskiEngine) proposes grasps, per-hand-type decision MLPs pick hand type + depth, the
two-finger pose is mapped to a hand pose (`width_12Dangle_6Dangle.json`) and checked for collisions with the hand mesh.

## Layout (`~/Projects/anydexgrasp`)

| Path | What |
| --- | --- |
| `AnyDexGrasp/` | upstream clone; weights in `logs/model/` (checkpoint.tar.18, inspire_model/obj140), hand meshes generated into `generate_mesh_and_pointcloud/inspire_urdf/meshes/` |
| `MinkowskiEngine/` | ME 0.5.4 patched for CUDA 13 / CCCL 3 / torch 2.14 (see below) |
| `.venv/` | Python 3.12, torch 2.14.1+cu130, ME, pointnet2, knn, open3d, graspnetAPI |
| `work/adg_server.py` | inference server (host GPU, venv), TCP 127.0.0.1:47011 |
| `work/adg_client.py` | ROS 2 client (distrobox `ubuntu24`, solution workspace): camera cloud -> server -> arm |
| `work/cmd.sh` | send one command to the running client and print its output |
| `work/record.py` | head + torso cameras side by side to mp4 in sim time, with the client's step, task clock and score |
| `work/episode.sh` | one recorded run: restart the sim (sort2, headless, quality render), client, recorder, `auto N`, mp4 |
| `work/test_trays.py` | turn the head and locate the trays with the solution's YOLO (tuning TRAY_LOOK) |

## Run

```bash
# host, venv: the networks
cd ~/Projects/anydexgrasp/work && source ../.venv/bin/activate && OMP_NUM_THREADS=8 python -u adg_server.py
# distrobox, with the sim up: the robot side (ros.sh sources the solution's .envrc)
cd ~/Projects/anydexgrasp/work && ./ros.sh python3 -u adg_client.py > client.log 2>&1 &
./cmd.sh plan r            # grasps for the right arm (l: left, the scene is mirrored and planned for the right hand)
./cmd.sh go 0              # execute grasp 0: ready, preshape, pre-grasp 8 cm back, straight in, close, lift 15 cm
./cmd.sh place 0.35 -0.6   # carry over (x, y) in a straight line and open
./cmd.sh label             # following plans keep only grasps on YOLO-detected objects (label off: all)
./cmd.sh auto 12           # sort, see below; publishes /sim/task/done at the end
```

Recorded run (needs the server; restarts the sim): `./episode.sh POOL SEED ATTEMPTS OUT_DIR` writes
`OUT_DIR/pool<POOL>.mp4` and the client/sim logs. The video plays in sim time; the task clock (270 s, sort2) starts at
`auto` and the score comes from `/sim/task/check` -> `/sim/task/result`.

## Sorting loop (`auto`)

1. Survey: the head turns to each tray (yaw +-0.9, pitch 0.6) and the solution's YOLO (`YoloPerception`, head + torso)
   locates it. Straight ahead only the near tray wall is in view; the trays are at y ~ +-0.69..0.71, not 0.60.
2. Each attempt: YOLO detects the objects not yet in a tray; heavy ones (`HEAVY`: sugar box 0.51 kg, JBL speaker
   0.60 kg) are skipped. An object goes to the arm on its category's tray side if it lies on that arm's half
   (`GRASP_REGION`), else to the other arm, which puts it down mid-table on the free spot of `RELAY_AREA` for a later
   pick (relay).
3. The server gets the target objects' xy points (`near`) and keeps only grasps centred within 2 cm of them; the client
   labels each grasp with its object (xy footprint, `label()`).
4. Grasp, lift 25 cm (18/12 if out of reach), carry in a straight line, put the held object's centre (not the TCP) over
   the tray, lower to 10 cm above the table, open, back up 8 cm. After a miss the object's next grasp; 3 misses and it
   is left.

Stage images go to `/tmp/adg/*_cam_head.png` / `*_cam_torso.png`.

## Hand frame mapping

AnyDexGrasp's hand pose is the urdf-five3 `Link111` frame shifted by (0.04903, 0.00804, -0.01796) ("wrist
centre", recover_inspire_hand_to_stl.py). Our `R_base_link` is the same palm: fingers +x -> +z, index side -z -> +y,
palm side -y -> +x, offset (-0.008, 0.0167, 0.0553) fitted on the four finger base joints (agree within 2 mm, the
thumb base within 3 mm). Check: the two-finger grasp centre lands at ~(0.01, 0.07, -0.16) in `right_tcp_link`, our
measured pad midpoint is (0.018, 0.060, -0.151). Hand commands: Inspire 0..1000 / 1000 = sim command, same order
(little, ring, middle, index, thumb bend, thumb rotation; 0 rotation = opposed). Unused fingers are folded (0) or
open (1000); the rest and the thumb bend close to 0.

## Changes vs upstream (all outside the clone except the ME patches)

- ME: no `pip uninstall` in setup.py, `blas`+`cblas`, CCCL include dir; dropped `thrust::unary_function` base and the
  bundled nvtx3 (clashes with CUDA's); added thrust includes; `cudaMemPrefetchAsync` CUDA 13 signature. Build:
  `CC=gcc-15 CXX=g++-15 TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/opt/cuda python setup.py build_ext --inplace --force_cuda --blas=blas`
  then copy the .so from build/ (in-place copy fails) — same for pointnet2 and knn.
- Decision models are pickled modules: loaded with `weights_only` and an allow-list of their 4 classes.
- `collections.Sequence` shim (py3.10+); `time` injected into collision_detector; `select_grasp_type` off-by-one
  for type 8 rewritten; `robot_inspire.py` imported with stub args, RANDOM_GRASP off (it ships in data-collection mode).
- Planning: their RealSense depth cut and camera-axis filters replaced by a pelvis workspace and "approach within
  55 deg of down"; both thumb sides of every two-finger grasp are kept (they drop one for their UR); 10 augmentations,
  100 candidates per hand type into the collision check.
- The head-camera cloud has the table tilted ~3.8 deg (z rises ~3 cm over 0.4 m): the client levels it on a fitted
  table plane before planning.

## Known issues

- In some runs both cameras' images get horizontal black bands (rows of zeros) after ~145 s of sim time, with the
  depth degraded too; later attempts then plan on a bad cloud. Not from the ZMQ camera bridge; quality render mode
  does not prevent it. Cause not found.
- Grasp success in the sim is ~35 % of attempts (lifted and held); flat (gelatin, mouse) and tall narrow standing
  boxes miss most.
- Planning takes ~50 s per attempt (the hand-mesh collision check), so 6 objects do not fit the 270 s limit.
