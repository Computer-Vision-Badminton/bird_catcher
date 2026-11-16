#!/usr/bin/env bash
set -euo pipefail

pi_target="${1:-catcher@catcher.local}"
remote_folder="${2:-bird_catcher}"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

to_copy=(vision)
if [ -d pi_test_package ]; then
  to_copy+=(pi_test_package)
else
  echo "no pi_test_package/: run vision/experiment/make_pi_test_package.py first"
fi
for model in models/*_640_ncnn_model models/*.hef; do
  if [ -e "$model" ]; then
    to_copy+=("$model")
  fi
done

ssh "$pi_target" "mkdir -p '$remote_folder'"
rsync -avR --exclude 'helper/' --exclude '__pycache__/' --exclude '.DS_Store' --exclude '*.so' \
  "${to_copy[@]}" "$pi_target:$remote_folder/"
echo "copied ${to_copy[*]} to $pi_target:$remote_folder"
