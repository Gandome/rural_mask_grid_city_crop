"""Multiprocessing orchestration for MOD_Mask version 2."""
from __future__ import annotations

import glob
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .calculation_process import process_file

_WORKER_CONTEXT = None


def _init_worker(context):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _worker(file_path):
    return process_file(file_path=file_path, **_WORKER_CONTEXT)


def process_files_parallel(
    files,
    elevation,
    output_path,
    sea_water_mask,
    references,
    height_limits=(100, 200, 300, 500),
    lapse_rate=0.0065,
    urban_threshold=0.20,
    rural_threshold=0.60,
    sea_water_threshold=0.30,
    min_value_requested=70.0,
    nO=2,
    initial_nbg=4,
    max_iterations=26,
    min_ratio_floor=50.0,
    ratio_step=5.0,
    nproc=1,
    spatial_slice=None,
):
    """Process an explicit sequence of NetCDF files with process-level parallelism."""
    files = [str(Path(f)) for f in files]
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise FileNotFoundError(f"Input NetCDF file(s) not found: {missing[:5]}")
    if not files:
        print("No NetCDF input files supplied", flush=True)
        return []

    Path(output_path).mkdir(parents=True, exist_ok=True)
    nproc = max(1, min(int(nproc), len(files)))
    print(f"Found {len(files)} input file(s)", flush=True)
    print(f"Using nproc = {nproc}", flush=True)

    context = {
        "elevation": np.asarray(elevation, dtype=np.float32),
        "output_path": str(output_path),
        "sea_water_mask": np.asarray(sea_water_mask, dtype=bool),
        "references": references,
        "height_limits": tuple(height_limits),
        "lapse_rate": float(lapse_rate),
        "urban_threshold": float(urban_threshold),
        "rural_threshold": float(rural_threshold),
        "sea_water_threshold": float(sea_water_threshold),
        "min_value_requested": float(min_value_requested),
        "nO": int(nO),
        "initial_nbg": int(initial_nbg),
        "max_iterations": int(max_iterations),
        "min_ratio_floor": float(min_ratio_floor),
        "ratio_step": float(ratio_step),
        "spatial_slice": spatial_slice,
    }

    if nproc == 1:
        _init_worker(context)
        return [_worker(f) for f in files]

    with Pool(processes=nproc, initializer=_init_worker, initargs=(context,)) as pool:
        return pool.map(_worker, files)


def process_folder_parallel(folder_path, **kwargs):
    """Backward-compatible wrapper that processes ``*.nc`` in a directory."""
    files = sorted(glob.glob(os.path.join(str(folder_path), "*.nc")))
    if not files:
        print(f"No NetCDF files found in {folder_path}", flush=True)
        return []
    return process_files_parallel(files=files, **kwargs)
