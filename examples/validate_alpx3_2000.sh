#!/usr/bin/env bash
set -euo pipefail

PGD="/archive/globc/quenum/Results/Data_UHI_Paper/PGD_ALPX3_selected_alpine_cities_buffer_1p0deg.nc"
TAS="/archive/globc/quenum/model_output/AROME/ALPX3/Evaluation/1hr/tas/tas_ALPX-3_ERA5_evaluation_r1i1p1f1_CNRM-MF_CNRM-AROME46t1_v1-r1_1hr_200001010100-200101010000.nc"
OUT="/archive/globc/quenum/Results/Data_UHI_Paper/UHI_MOD_MASK_V2_0_1"

python -u grid_uhi_mask/scripts/run_UHI_process_parallel.py \
  --pgd "$PGD" \
  --tas "$TAS" \
  --output "$OUT" \
  --coord-tolerance 1e-4 \
  --validate-only
