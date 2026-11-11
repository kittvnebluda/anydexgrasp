# What the judges' container gets: torch for its Python 3.10, the CUDA the extensions are built with, and the release
# assets install.sh downloads. Sourced by build_ext.sh, pack_assets.sh and install.sh.
TORCH_VERSION=2.11.0
TORCH_INDEX=https://download.pytorch.org/whl/cu128
CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
CUDA_ARCHS="8.6;8.9;12.0"
EXT_TAG=cp310-cu128
RELEASE=v0.1.0
RELEASE_URL=https://github.com/kittvnebluda/anydexgrasp/releases/download/$RELEASE
ME_WHEEL=minkowskiengine-0.5.4-cp310-cp310-linux_x86_64.whl  # what build_ext.sh produces (built against CUDA 12.8)
