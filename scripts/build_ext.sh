#!/usr/bin/env bash
# Build the CUDA extensions for the judges' container (Ubuntu 22.04, Python 3.10, no nvcc) in a CUDA devel image:
# a MinkowskiEngine wheel and the in-place pointnet2 / knn modules, packed into dist/.
# Usage: [MAX_JOBS=4] scripts/build_ext.sh   (docker with the NVIDIA runtime; knn builds its CUDA kernels only if it
# sees a GPU; each nvcc job for three architectures takes ~2 GB of RAM)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/scripts/versions.sh"
mkdir -p "$ROOT/dist"
docker run --rm --gpus all -v "$ROOT:/adg" -w /adg -e TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS" -e MAX_JOBS=${MAX_JOBS:-4} \
    "$CUDA_DEVEL_IMAGE" bash -euc "
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    python3-venv python3-dev git libopenblas-dev build-essential gcc-12 g++-12 >/dev/null
# GCC 11's shared_ptr headers trip nvcc in MinkowskiEngine (two std::__to_address); 12 is the 22.04 runtime anyway
ln -sf /usr/bin/gcc-12 /usr/bin/gcc && ln -sf /usr/bin/g++-12 /usr/bin/g++ && ln -sf /usr/bin/g++-12 /usr/bin/c++
python3 -m venv /venv && . /venv/bin/activate
pip install -q --upgrade pip 'setuptools<80' wheel ninja
pip install -q torch==$TORCH_VERSION --index-url $TORCH_INDEX
pip install -q 'numpy<2'
rm -rf /tmp/me && cp -r MinkowskiEngine /tmp/me && cd /tmp/me
python setup.py -q bdist_wheel --blas=openblas --force_cuda
cp dist/*.whl /adg/dist/
cd /tmp && rm -rf ext && mkdir ext && cp -r /adg/AnyDexGrasp/pointnet2 /adg/AnyDexGrasp/knn ext/
# where AnyDexGrasp imports them from: pointnet2/_ext*.so, knn/knn_pytorch/knn_pytorch*.so
(cd ext/pointnet2 && rm -rf build *.so && python setup.py -q build_ext && cp build/lib*/pointnet2/_ext*.so .)
(cd ext/knn && rm -rf build knn_pytorch/*.so && python setup.py -q build_ext && cp build/lib*/knn_pytorch/*.so knn_pytorch/)
cd ext && ls pointnet2/_ext*.so knn/knn_pytorch/*.so | tee /dev/stderr | tar czf /adg/dist/adg-ext-$EXT_TAG.tar.gz -T -
chown -R $(id -u):$(id -g) /adg/dist
"
ls -la "$ROOT/dist"
