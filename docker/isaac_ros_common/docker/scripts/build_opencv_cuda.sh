#!/usr/bin/env bash
#
# Build OpenCV with opencv_contrib + CUDA support inside the project_mpcc
# Docker image. Adapted from setup_tools/scripts/04_opencv_cuda_x86.bash.
#
# Differences vs the original:
#   - Designed to run as root during `docker build` (no sudo).
#   - CUDA_ARCH is taken from the OPENCV_CUDA_ARCH_BIN build arg/env, not
#     from `nvidia-smi` (no GPU is visible during image build). Default
#     covers the typical desktop GPUs in our lab: Turing 7.5 (RTX 20xx /
#     T4), Ampere 8.0/8.6 (A100 / RTX 30xx), Ada 8.9 (RTX 40xx).
#   - Drops X11/GTK and GStreamer modules: the project_mpcc pipeline uses
#     OpenCV for solvePnP/Rodrigues + cudawarping (consumed by
#     yolo_inference_cpp + as2_gates_localization). No imshow, no
#     video capture from gst.
#   - Removes /tmp/opencv_build at the end to keep the image small.

set -euo pipefail

if [ "$(uname -m)" != "x86_64" ]; then
    echo "[build_opencv_cuda] Architecture $(uname -m) not supported. Skipping."
    exit 0
fi

OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
CUDA_ARCH_BIN="${OPENCV_CUDA_ARCH_BIN:-7.5;8.0;8.6;8.9}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
NUM_JOBS="${OPENCV_BUILD_JOBS:-$(nproc)}"
BUILD_DIR="/tmp/opencv_build"

echo "=========================================="
echo "Building OpenCV ${OPENCV_VERSION} with CUDA"
echo "  CUDA_ARCH_BIN  = ${CUDA_ARCH_BIN}"
echo "  INSTALL_PREFIX = ${INSTALL_PREFIX}"
echo "  PARALLEL JOBS  = ${NUM_JOBS}"
echo "=========================================="

apt-get update --allow-releaseinfo-change || apt-get update -y
apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libjpeg-dev libtiff-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libatlas-base-dev gfortran \
    python3-dev python3-numpy \
    libtbb-dev libdc1394-dev

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Shallow clones of the official mirror — we only need the source tree.
if [ ! -d opencv ]; then
    git clone --depth 1 --branch "${OPENCV_VERSION}" https://github.com/opencv/opencv.git
fi
if [ ! -d opencv_contrib ]; then
    git clone --depth 1 --branch "${OPENCV_VERSION}" https://github.com/opencv/opencv_contrib.git
fi

mkdir -p opencv/build
cd opencv/build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
    -DOPENCV_EXTRA_MODULES_PATH="${BUILD_DIR}/opencv_contrib/modules" \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DOPENCV_DNN_CUDA=ON \
    -DENABLE_FAST_MATH=ON \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
    -DCUDA_ARCH_PTX="" \
    -DBUILD_CUDA_STUBS=ON \
    -DWITH_TBB=ON \
    -DWITH_V4L=ON \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GTK=OFF \
    -DWITH_OPENGL=ON \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE="$(which python3)" \
    -DBUILD_opencv_sfm=OFF \
    -DWITH_CERES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=ON

make -j"${NUM_JOBS}"
make install
ldconfig

# Drop any pip-installed cv2 so it doesn't shadow the freshly built lib.
pip3 uninstall -y \
    opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true

# Free disk space (source + build tree).
rm -rf "${BUILD_DIR}"

echo "=========================================="
echo "OpenCV ${OPENCV_VERSION} with CUDA installed."
echo "Verify (runtime): python3 -c 'import cv2; print(cv2.getBuildInformation())' | grep -i cuda"
echo "=========================================="
