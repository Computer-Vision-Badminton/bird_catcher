#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

if command -v hailortcli >/dev/null; then
  hailortcli --version
  hailortcli fw-control identify || echo "Hailo runtime is installed but the AI HAT+ 2 did not answer"
else
  echo "hailortcli not found: install the Hailo runtime from the Raspberry Pi AI HAT+ 2 guide (CPU tests still work)"
fi

python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r vision/deploy/requirements-pi.txt
if command -v c++ >/dev/null; then
  bash vision/flight_cpp/build.sh "$repo_root/.venv/bin/python"
else
  echo "no C++ compiler: run 'sudo apt install g++' and re-run this script (the flight fit falls back to slow Python until then)"
fi
.venv/bin/python -c "import sys; sys.path.insert(0, 'vision'); import flight_model; print('flight fit engine:', flight_model.FLIGHT_ENGINE)"
.venv/bin/python -c "import cv2, numpy, scipy; print('opencv', cv2.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"
.venv/bin/python -c "import hailo_platform; print('hailo_platform', hailo_platform.__version__)" \
  || echo "hailo_platform is not visible to Python: Hailo tests will not run until the Hailo runtime is installed"
