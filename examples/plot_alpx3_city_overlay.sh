#!/usr/bin/env bash
set -euo pipefail

PGD="/archive/globc/quenum/Results/Data_UHI_Paper/PGD_ALPX3_selected_alpine_cities_buffer_1p0deg.nc"
UHI_DIR="/archive/globc/quenum/Results/Data_UHI_Paper/UHI_MOD_MASK_V2_0_1/Min70_sea0p30_urb0p20_rur0p60"
CITY_GPKG="clim_city_mask/data/GHS_UCDB_REGION_EUROPE_R2024A.gpkg"

python -u clim_city_mask/scripts/standalone_plot_uhi_city_rural_overlay_RAW_v2.py \
  --uhi-dir "$UHI_DIR" \
  --pgd "$PGD" \
  --city-gpkg "$CITY_GPKG" \
  --cities Grenoble Chambery Geneva \
  --rural-mode exact \
  --output ./output_city_uhi_figures_RAW_v2_0_1
