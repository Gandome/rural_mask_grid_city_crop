"""MOD_Mask / gridded UHI package, version 2."""

__version__ = "2.0.1"

from .urban_mask import build_masks, classify_grid_points, to_yx
from .rural_reference import (
    RuralReference,
    diagnostics_from_references,
    find_rural_reference_once,
    precompute_rural_references,
)
from .calculation import compute_uhi_timeseries
from .calculation_process import process_file
from .grid_alignment import GridAlignment, determine_grid_alignment
from .parallel_process import process_files_parallel, process_folder_parallel

__all__ = [
    "RuralReference",
    "build_masks",
    "classify_grid_points",
    "compute_uhi_timeseries",
    "diagnostics_from_references",
    "find_rural_reference_once",
    "precompute_rural_references",
    "process_file",
    "process_files_parallel",
    "process_folder_parallel",
    "GridAlignment",
    "determine_grid_alignment",
    "to_yx",
]
