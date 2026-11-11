#!/usr/bin/env bash
# Install the server into this checkout (cloned with --recursive): a venv with torch and the prebuilt CUDA extensions,
# and the weights, both from the GitHub release. No CUDA toolkit needed. Python 3.10 (the judges' Ubuntu 22.04).
# Usage: install.sh        then: venv/bin/python -m adg_server
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/scripts/versions.sh"
cd "$ROOT"
python3 -m venv venv
venv/bin/pip install -q --upgrade pip
venv/bin/pip install -q "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
# the release files: from dist/ if built here (scripts/build_ext.sh, pack_assets.sh), else downloaded
files=$(mktemp -d)
trap 'rm -rf "$files"' EXIT
for f in "$ME_WHEEL" "adg-ext-$EXT_TAG.tar.gz" adg-assets.tar; do
    if [ -f "dist/$f" ]; then ln -s "$ROOT/dist/$f" "$files/$f"; else curl -fsSL -o "$files/$f" "$RELEASE_URL/$f"; fi
done
venv/bin/pip install -q "$files/$ME_WHEEL"
tar xzf "$files/adg-ext-$EXT_TAG.tar.gz" -C AnyDexGrasp
tar xf "$files/adg-assets.tar" -C AnyDexGrasp
# graspnetAPI's own requirements include the dead `sklearn` package: its imports are installed with this package
venv/bin/pip install -q --no-deps graspnetAPI==1.2.11
venv/bin/pip install -q -e .
echo "adg_server installed: $ROOT/venv/bin/python -m adg_server"
