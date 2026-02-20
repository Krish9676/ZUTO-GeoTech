"""Utility helpers for raster/array processing."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class DataProcessor:
    """Common array processing helpers used by satellite ingestion."""

    @staticmethod
    def calculate_array_stats(arr: np.ndarray) -> Dict[str, float]:
        """Return summary stats for a numeric array."""
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {'mean': None, 'std': None, 'min': None, 'max': None, 'p10': None, 'p50': None, 'p90': None}
        return {
            'mean': float(np.nanmean(finite)),
            'std': float(np.nanstd(finite)),
            'min': float(np.nanmin(finite)),
            'max': float(np.nanmax(finite)),
            'p10': float(np.nanpercentile(finite, 10)),
            'p50': float(np.nanpercentile(finite, 50)),
            'p90': float(np.nanpercentile(finite, 90)),
        }

    @staticmethod
    def resample_to_resolution(data: np.ndarray, native_resolution: int, target_resolution: int) -> Tuple[int, int]:
        """Compute output shape when resampling from native to target resolution."""
        h, w = data.shape[:2]
        scale = max(float(native_resolution) / float(target_resolution), 1e-6)
        out_h = max(1, int(round(h * scale)))
        out_w = max(1, int(round(w * scale)))
        return out_h, out_w

    @staticmethod
    def clean_satellite_data(
        data: np.ndarray,
        nodata_value: float | None = None,
        min_value: float = 0.0,
        max_value: float = 10000.0,
    ) -> np.ndarray:
        """Replace nodata and clip input data to valid range."""
        arr = np.asarray(data, dtype=np.float32).copy()
        if nodata_value is not None and np.isfinite(nodata_value):
            arr[arr == nodata_value] = np.nan
        arr[(arr < min_value) | (arr > max_value)] = np.nan
        return arr

    @staticmethod
    def validate_band_data(data: np.ndarray, min_valid_ratio: float = 0.2) -> bool:
        """Validate that enough finite pixels are available."""
        arr = np.asarray(data)
        if arr.size == 0:
            return False
        valid_ratio = float(np.isfinite(arr).sum()) / float(arr.size)
        return valid_ratio >= min_valid_ratio
