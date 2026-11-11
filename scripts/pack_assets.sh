#!/usr/bin/env bash
# Pack what the server needs at runtime besides the code, for a release: the network and decision weights and the
# hand clouds the collision check uses (voxel_size_3 only: the full generated set is 19 GB). -> dist/adg-assets.tar
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/dist"
tar cf "$ROOT/dist/adg-assets.tar" -C "$ROOT/AnyDexGrasp" \
    logs/model/checkpoint.tar.18 \
    logs/model/inspire_model/obj140 \
    generate_mesh_and_pointcloud/inspire_urdf/meshes/source_pointclouds/voxel_size_3
ls -la "$ROOT/dist/adg-assets.tar"
