#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python train.py \
  --exp_name "dbt_bleed_mby140_N300" \
  --csv_dir "${REPO_ROOT}/dataset/mby140_N=300_Stride=200" \
  --output_dir "${REPO_ROOT}/outputs" \
  --save_model
