#!/usr/bin/env bash
set -euo pipefail

# Reproducible author example for the ALPX3 evaluation year 2000.
# Public users should replace these three filesystem paths with their own data.

PGD="/archive/globc/quenum/Results/Data_UHI_Paper/PGD_ALPX3_selected_alpine_cities_buffer_1p0deg.nc"
TAS="/archive/globc/quenum/model_output/AROME/ALPX3/Evaluation/1hr/tas/tas_ALPX-3_ERA5_evaluation_r1i1p1f1_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_200001010100-200101010000.nc"
OUT="/archive/globc/quenum/Results/Data_UHI_Paper/UHI_MOD_MASK_V2"

python -u grid_uhi_mask/scripts/run_UHI_process_parallel.py \
  --pgd "$PGD" \
  --tas "$TAS" \
  --output "$OUT" \
  --min-values 70 \
  --sea-water-thresholds 0.30 \
  --urban-thresholds 0.20 \
  --rural-thresholds 0.60 \
  --nO 2 \
  --initial-nbg 4 \
  --max-iterations 26 \
  --min-ratio-floor 50 \
  --ratio-step 5 \
  --height-limits 100,200,300,500 \
  --lapse-rate 0.0065 \
  --nproc 1 \
  2>&1 | tee MOD_MASK_V2_ALPX3_2000.log
