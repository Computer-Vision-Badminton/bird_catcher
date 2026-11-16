#!/usr/bin/env bash
set -euo pipefail

vision_folder="$(cd "$(dirname "$0")/.." && pwd)"
python="${1:-$vision_folder/../.venv/bin/python}"
output="$vision_folder/flight_core$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
platform_flags=()
if [ "$(uname)" = "Darwin" ]; then
  platform_flags=(-undefined dynamic_lookup)
fi

c++ -O3 -std=c++17 -shared -fPIC $("$python" -m pybind11 --includes) "${platform_flags[@]}" \
  "$vision_folder/flight_cpp/flight_core.cpp" -o "$output"
echo "built $output"
