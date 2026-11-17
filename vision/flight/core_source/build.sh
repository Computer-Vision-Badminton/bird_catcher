#!/usr/bin/env bash
set -euo pipefail

flight_folder="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "$flight_folder/../.." && pwd)"
python="${1:-$repo_root/.venv/bin/python}"
output="$flight_folder/flight_core$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
platform_flags=()
if [ "$(uname)" = "Darwin" ]; then
  platform_flags=(-undefined dynamic_lookup)
fi

c++ -O3 -std=c++17 -shared -fPIC $("$python" -m pybind11 --includes) "${platform_flags[@]}" \
  "$flight_folder/core_source/flight_core.cpp" -o "$output"
echo "built $output"
