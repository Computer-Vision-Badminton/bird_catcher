#!/usr/bin/env bash
set -euo pipefail

pi_target="${1:-catcher@catcher.local}"
remote_folder="${2:-bird_catcher}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

to_copy=(vision)
if [ -d test/pi_test_package ]; then
  to_copy+=(test/pi_test_package)
else
  echo "no test/pi_test_package/: run vision/experiment/make_pi_test_package.py first"
fi
if [ -d test/pi_hit_test_package ]; then
  to_copy+=(test/pi_hit_test_package)
else
  echo "no test/pi_hit_test_package/: run vision/experiment/hit_from_rising_chain.py package --package test/pi_hit_test_package first"
fi
if [ -d test/pi_live_test_package ]; then
  to_copy+=(test/pi_live_test_package)
else
  echo "no test/pi_live_test_package/: run vision/experiment/make_live_test_package.py make first"
fi
for model in models/*_640_ncnn_model models/*.hef; do
  if [ -e "$model" ]; then
    to_copy+=("$model")
  fi
done

ssh "$pi_target" "mkdir -p '$remote_folder'"
rsync -avR --exclude 'helper/' --exclude '__pycache__/' --exclude '.DS_Store' --exclude '*.so' \
  --exclude 'test/pi_hit_test_package/hit_detection_cache_*/' --exclude 'test/pi_hit_test_package/hit_results_*.json' \
  --exclude 'test/pi_live_test_package/live_results_*.json' --exclude 'test/pi_live_test_package/live_log_*.json' --exclude 'test/pi_live_test_package/live_score_*.json' \
  --exclude 'test/pi_live_test_package/live_cache_*/' --exclude '*.writing' \
  "${to_copy[@]}" "$pi_target:$remote_folder/"
echo "copied ${to_copy[*]} to $pi_target:$remote_folder"
