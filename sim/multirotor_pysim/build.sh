#!/usr/bin/env bash
# Build the multirotor_pysim pybind11 module. Produces multirotor_pysim*.so in
# this directory; add this dir to PYTHONPATH (or it's auto-added by the env).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
cmake -S "$HERE" -B "$HERE/build" \
  -DPYTHON_EXECUTABLE="$($PY -c 'import sys; print(sys.executable)')" \
  -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$HERE/build" -j"$(nproc)"
echo "built: $(ls "$HERE"/multirotor_pysim*.so)"
