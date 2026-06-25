#!/usr/bin/env bash
#
# Rebuild ros-humble-cv_bridge from source against the CUDA-enabled
# OpenCV installed at /usr/local/lib/cmake/opencv4.
#
# Rationale: the apt package `ros-humble-cv-bridge` is linked against
# the system `libopencv-4.5d` shipped by Ubuntu jammy. Once we install
# OpenCV 4.10 in /usr/local (via build_opencv_cuda.sh) and remove the
# apt libopencv-dev, the apt cv_bridge stops resolving and any package
# downstream (notably as2_gates_localization) fails to link.
#
# This script clones vision_opencv@humble, builds only the cv_bridge
# package against /usr/local OpenCV, and installs it into
# /opt/cv_bridge_ws/install/. The Dockerfile then appends `source
# /opt/cv_bridge_ws/install/setup.bash` to the user's ~/.bashrc so the
# rebuilt cv_bridge is picked up before the apt one (which has been
# removed at that point).
#
# Designed to run as root during `docker build`.

set -euo pipefail

if [ "$(uname -m)" != "x86_64" ]; then
    echo "[rebuild_cv_bridge] Architecture $(uname -m) not supported. Skipping."
    exit 0
fi

CV_BRIDGE_WS="${CV_BRIDGE_WS:-/opt/cv_bridge_ws}"
OPENCV_DIR="${OPENCV_DIR:-/usr/local/lib/cmake/opencv4}"

if [ ! -d "${OPENCV_DIR}" ]; then
    echo "[rebuild_cv_bridge] ERROR: ${OPENCV_DIR} not found — run build_opencv_cuda.sh first." >&2
    exit 1
fi

# Do NOT `apt remove libopencv*` — that cascade-removes ros-humble-desktop
# (and hence ros2cli, perception, image_transport, etc.) because those
# packages declare a runtime dep on libopencv4.5d. The newly-built
# cv_bridge in /opt/cv_bridge_ws/install/ has RPATH pointing at
# /usr/local/lib, so it loads the OpenCV-CUDA libs from there
# regardless of what apt keeps installed. The two coexist peacefully:
# apt's libopencv-4.5d stays in /usr/lib/x86_64-linux-gnu for its
# legacy consumers; our overlay's cv_bridge resolves to /usr/local.

mkdir -p "${CV_BRIDGE_WS}/src"
cd "${CV_BRIDGE_WS}/src"
if [ ! -d vision_opencv ]; then
    git clone --depth 1 -b humble https://github.com/ros-perception/vision_opencv.git
fi

# Build only cv_bridge — image_geometry needs the rest of the cluster
# and we don't use it in the project_mpcc pipeline today.
cd "${CV_BRIDGE_WS}"
# /opt/ros/humble/setup.bash references AMENT_TRACE_SETUP_FILES without
# declaring it, which aborts under `set -u`. Disable nounset just for
# the source line.
set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u
colcon build --packages-select cv_bridge \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
                 "-DOpenCV_DIR=${OPENCV_DIR}"

# Strip the build/log directories to keep the image small. The install
# tree is what subsequent layers source.
rm -rf "${CV_BRIDGE_WS}/build" "${CV_BRIDGE_WS}/log"

echo "[rebuild_cv_bridge] cv_bridge installed at ${CV_BRIDGE_WS}/install"
echo "[rebuild_cv_bridge] Source via: source ${CV_BRIDGE_WS}/install/setup.bash"
