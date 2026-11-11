#!/usr/bin/env bash
# install.sh in a clean Ubuntu 22.04 (the judges' OS and Python) with the GPU, from this working tree and dist/
# (the release files, so nothing is downloaded but pip packages); then the server plans on the recorded scene.
# Usage: scripts/check_install.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
docker run --rm --gpus all -v "$ROOT:/src:ro" ubuntu:22.04 bash -euc '
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    python3 python3-venv git curl ca-certificates libopenblas0 libgl1 libegl1 libusb-1.0-0 libglib2.0-0 libgomp1 >/dev/null
mkdir -p /opt/adg && cd /src
tar cf - --exclude=./.venv --exclude=./venv --exclude=./videos --exclude=./work --exclude=./MinkowskiEngine \
    --exclude=./AnyDexGrasp/logs --exclude=./AnyDexGrasp/generate_mesh_and_pointcloud/inspire_urdf/meshes \
    --exclude="*.so" --exclude="*/build" --exclude="__pycache__" . | tar xf - -C /opt/adg
cd /opt/adg && t=$(date +%s) && bash install.sh && echo "install took $(( $(date +%s) - t )) s"
du -sh /opt/adg
venv/bin/python -m adg_server > /tmp/server.log 2>&1 &
until grep -q "ready on" /tmp/server.log; do sleep 2; kill -0 $! 2>/dev/null || { cat /tmp/server.log; exit 1; }; done
grep "models loaded" /tmp/server.log
venv/bin/python -m tests.smoke
grep "points:" /tmp/server.log
nvidia-smi --query-compute-apps=used_memory --format=csv
'
