"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Module: Spectral Index Engine

Computes ALL 28 spectral indices on EVERY satellite scene timestamp.
No sub-sampling. Every valid scene gets the full index suite.

Index Categories:
  Tier 1 — Core vegetation:   NDVI, RENDVI, EVI, GNDVI, NDMI, OSAVI, SAVI
  Tier 2 — Soil nutrients:    CIre, TGI, MSR, SI, NDSI, CMR, CAI, IOR, NDTI, BSI
  Tier 3 — Moisture & stress: NDWI, MNDWI, NMDI, MSI_STRESS, NDDI, PSRI
  Tier 4 — Canopy & growth:   SR1, SR2, CRI, LSWI

Band mapping (Sentinel-2):
  B02=Blue, B03=Green, B04=Red, B05=RedEdge1, B06=RedEdge2,
  B07=RedEdge3, B08=NIR, B8A=NarrowNIR, B11=SWIR1, B12=SWIR2
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SpectralIndexEngine:
    """
    Computes the full 28-index ZUTO spectral suite from Sentinel-2 band arrays.

    Usage:
        engine = SpectralIndexEngine()
        results = engine.compute_all(band_data)
        # results is a dict of {index_name: {'mean', 'std', 'p10', 'p50', 'p90', 'map'}}
    """

    # Sentinel-2 band scale factor (L2A products are in 0–10000 range)
    SCALE_FACTOR = 10000.0

    def __init__(self, scale_bands: bool = True):
        """
        Args:
            scale_bands: If True, divide raw DN values by 10000 to get
                         surface reflectance (0–1 range). Set False if
                         bands are already in reflectance.
        """
        self.scale_bands = scale_bands

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def compute_all(
        self,
        band_data: Dict[str, np.ndarray],
        scene_date: Optional[str] = None,
        return_maps: bool = False,
    ) -> Dict[str, Dict]:
        """
        Compute all available indices from provided band data.
        Automatically skips indices whose required bands are missing.

        Args:
            band_data:   Dict of {band_name: np.ndarray}, e.g. {'B08': arr, ...}
            scene_date:  Optional date string for logging context
            return_maps: If True, include full 2D index maps in output.
                         Warning: memory-heavy for large scenes.

        Returns:
            Dict of:
              {
                'NDVI': {'mean': 0.62, 'std': 0.08, 'p10': 0.45, 'p50': 0.64,
                          'p90': 0.79, 'valid_pixels': 1240, 'map': array (if requested)},
                'RENDVI': {...},
                ...
              }
        """
        label = f"[{scene_date}] " if scene_date else ""
        logger.debug(f"{label}Computing spectral indices. Available bands: {list(band_data.keys())}")

        # Optionally scale bands to reflectance
        bands = self._prepare_bands(band_data)

        results = {}
        computed = []
        skipped  = []

        # ── TIER 1: Core Vegetation ───────────────────────────────────────
        self._try_compute(results, computed, skipped, 'NDVI',    self._ndvi,    bands, return_maps)
        self._try_compute(results, computed, skipped, 'RENDVI',  self._rendvi,  bands, return_maps)
        self._try_compute(results, computed, skipped, 'EVI',     self._evi,     bands, return_maps)
        self._try_compute(results, computed, skipped, 'GNDVI',   self._gndvi,   bands, return_maps)
        self._try_compute(results, computed, skipped, 'NDMI',    self._ndmi,    bands, return_maps)
        self._try_compute(results, computed, skipped, 'OSAVI',   self._osavi,   bands, return_maps)
        self._try_compute(results, computed, skipped, 'SAVI',    self._savi,    bands, return_maps)

        # ── TIER 2: Soil Nutrient Indices ─────────────────────────────────
        self._try_compute(results, computed, skipped, 'CI_REDEDGE', self._ci_rededge, bands, return_maps)
        self._try_compute(results, computed, skipped, 'TGI',     self._tgi,     bands, return_maps)
        self._try_compute(results, computed, skipped, 'MSR',     self._msr,     bands, return_maps)
        self._try_compute(results, computed, skipped, 'SI',      self._si,      bands, return_maps)
        self._try_compute(results, computed, skipped, 'NDSI',    self._ndsi,    bands, return_maps)
        self._try_compute(results, computed, skipped, 'CMR',     self._cmr,     bands, return_maps)
        self._try_compute(results, computed, skipped, 'CAI',     self._cai,     bands, return_maps)
        self._try_compute(results, computed, skipped, 'IOR',     self._ior,     bands, return_maps)
        self._try_compute(results, computed, skipped, 'NDTI',    self._ndti,    bands, return_maps)
        self._try_compute(results, computed, skipped, 'BSI',     self._bsi,     bands, return_maps)

        # ── TIER 3: Moisture & Stress ─────────────────────────────────────
        self._try_compute(results, computed, skipped, 'NDWI',       self._ndwi,       bands, return_maps)
        self._try_compute(results, computed, skipped, 'MNDWI',      self._mndwi,      bands, return_maps)
        self._try_compute(results, computed, skipped, 'NMDI',       self._nmdi,       bands, return_maps)
        self._try_compute(results, computed, skipped, 'MSI_STRESS', self._msi_stress, bands, return_maps)
        self._try_compute(results, computed, skipped, 'NDDI',       self._nddi,       bands, return_maps)
        self._try_compute(results, computed, skipped, 'PSRI',       self._psri,       bands, return_maps)

        # ── TIER 4: Canopy & Growth Stage ────────────────────────────────
        self._try_compute(results, computed, skipped, 'SR1',  self._sr1,  bands, return_maps)
        self._try_compute(results, computed, skipped, 'SR2',  self._sr2,  bands, return_maps)
        self._try_compute(results, computed, skipped, 'CRI',  self._cri,  bands, return_maps)
        self._try_compute(results, computed, skipped, 'LSWI', self._lswi, bands, return_maps)

        logger.debug(f"{label}Computed {len(computed)} indices. "
                     f"Skipped {len(skipped)} (missing bands): {skipped}")

        return results

    def get_index_summary(self, band_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Returns a flat dict of {index_mean: value} for all computed indices.
        Convenience method for ML feature extraction.
        """
        full = self.compute_all(band_data, return_maps=False)
        summary = {}
        for idx_name, stats in full.items():
            summary[f"{idx_name}_mean"] = stats.get('mean', np.nan)
            summary[f"{idx_name}_std"]  = stats.get('std',  np.nan)
            summary[f"{idx_name}_p90"]  = stats.get('p90',  np.nan)
        return summary

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    def _prepare_bands(self, band_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Scale band DN values to reflectance (0–1) if needed."""
        if not self.scale_bands:
            return band_data
        scaled = {}
        for k, v in band_data.items():
            scaled[k] = v.astype(np.float32) / self.SCALE_FACTOR
        return scaled

    def _try_compute(
        self,
        results:     Dict,
        computed:    list,
        skipped:     list,
        name:        str,
        func,
        bands:       Dict[str, np.ndarray],
        return_maps: bool,
    ):
        """Attempt to compute one index; silently skip if bands missing."""
        try:
            arr = func(bands)
            if arr is None:
                skipped.append(name)
                return
            stats = self._array_stats(arr)
            if return_maps:
                stats['map'] = arr
            results[name] = stats
            computed.append(name)
        except (KeyError, TypeError):
            skipped.append(name)
        except Exception as e:
            logger.warning(f"Index {name} failed: {e}")
            skipped.append(name)

    @staticmethod
    def _array_stats(arr: np.ndarray) -> Dict:
        """Compute statistics for a 2D index array."""
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return {'mean': np.nan, 'std': np.nan, 'min': np.nan,
                    'max': np.nan, 'p10': np.nan, 'p50': np.nan,
                    'p90': np.nan, 'valid_pixels': 0}
        return {
            'mean':         float(np.mean(valid)),
            'std':          float(np.std(valid)),
            'min':          float(np.min(valid)),
            'max':          float(np.max(valid)),
            'p10':          float(np.percentile(valid, 10)),
            'p50':          float(np.percentile(valid, 50)),
            'p90':          float(np.percentile(valid, 90)),
            'valid_pixels': int(len(valid)),
        }

    @staticmethod
    def _safe_ratio(num: np.ndarray, den: np.ndarray,
                    clip_min: float = -1.0, clip_max: float = 1.0) -> np.ndarray:
        """Compute num/den safely, clipping to [clip_min, clip_max]."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                np.abs(den) > 1e-10,
                num / den,
                np.nan
            )
        return np.clip(result, clip_min, clip_max).astype(np.float32)

    @staticmethod
    def _align(*arrays) -> Tuple[np.ndarray, ...]:
        """Align multiple arrays to the minimum shared shape."""
        min_h = min(a.shape[0] for a in arrays)
        min_w = min(a.shape[1] for a in arrays)
        return tuple(a[:min_h, :min_w] for a in arrays)

    # =========================================================================
    # TIER 1 — CORE VEGETATION INDICES
    # =========================================================================

    def _ndvi(self, b: Dict) -> Optional[np.ndarray]:
        """
        NDVI — Normalized Difference Vegetation Index
        Formula:  (B08 - B04) / (B08 + B04)
        Use:      Baseline vegetation health, crop monitoring
        Range:    -1 to 1  (healthy crop: 0.4–0.9)
        """
        nir, red = self._align(b['B08'], b['B04'])
        return self._safe_ratio(nir - red, nir + red)

    def _rendvi(self, b: Dict) -> Optional[np.ndarray]:
        """
        RENDVI — Red Edge NDVI (MOST IMPORTANT for Nitrogen)
        Formula:  (B08 - B05) / (B08 + B05)
        Use:      Direct nitrogen deficiency detection (78–89% accuracy)
        Range:    -1 to 1  (healthy: 0.6–1.0)
        Priority: HIGHEST — replace NDVI for nutrient work
        """
        nir, re1 = self._align(b['B08'], b['B05'])
        return self._safe_ratio(nir - re1, nir + re1)

    def _evi(self, b: Dict) -> Optional[np.ndarray]:
        """
        EVI — Enhanced Vegetation Index
        Formula:  2.5 × (B08 - B04) / (B08 + 6×B04 - 7.5×B02 + 1)
        Use:      Dense vegetation (P/K proxy), atmospheric correction
        Range:    -1 to 3  (healthy: 0.2–0.8)
        """
        nir, red, blue = self._align(b['B08'], b['B04'], b['B02'])
        with np.errstate(divide='ignore', invalid='ignore'):
            den = nir + 6.0 * red - 7.5 * blue + 1.0
            evi = np.where(np.abs(den) > 1e-10, 2.5 * (nir - red) / den, np.nan)
        return np.clip(evi, -1.0, 3.0).astype(np.float32)

    def _gndvi(self, b: Dict) -> Optional[np.ndarray]:
        """
        GNDVI — Green NDVI
        Formula:  (B08 - B03) / (B08 + B03)
        Use:      Chlorophyll content, nitrogen stress (all growth stages)
        Range:    -1 to 1
        """
        nir, grn = self._align(b['B08'], b['B03'])
        return self._safe_ratio(nir - grn, nir + grn)

    def _ndmi(self, b: Dict) -> Optional[np.ndarray]:
        """
        NDMI — Normalized Difference Moisture Index
        Formula:  (B08 - B11) / (B08 + B11)
        Use:      Vegetation water content, drought stress, irrigation mgmt
        Range:    -1 to 1  (no drought: 0.4–1.0; severe drought: -1 to 0)
        """
        nir, swir1 = self._align(b['B08'], b['B11'])
        return self._safe_ratio(nir - swir1, nir + swir1)

    def _osavi(self, b: Dict) -> Optional[np.ndarray]:
        """
        OSAVI — Optimized Soil Adjusted Vegetation Index
        Formula:  (B08 - B04) / (B08 + B04 + 0.16)
        Use:      Sparse/early vegetation, nitrogen assessment (early stages)
        Range:    -1 to 1
        """
        nir, red = self._align(b['B08'], b['B04'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (nir - red) / (nir + red + 0.16)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    def _savi(self, b: Dict, L: float = 0.5) -> Optional[np.ndarray]:
        """
        SAVI — Soil Adjusted Vegetation Index
        Formula:  ((B08 - B04) / (B08 + B04 + L)) × (1 + L)
        L=0.5 for moderate vegetation cover
        Use:      Early crop season, arid regions with exposed soil
        Range:    -1 to 1
        """
        nir, red = self._align(b['B08'], b['B04'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = ((nir - red) / (nir + red + L)) * (1 + L)
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    # =========================================================================
    # TIER 2 — SOIL NUTRIENT INDICES
    # =========================================================================

    def _ci_rededge(self, b: Dict) -> Optional[np.ndarray]:
        """
        CIred-edge — Chlorophyll Index Red Edge
        Formula:  (B08 / B05) - 1
        Use:      DIRECT chlorophyll estimation, Nitrogen deficiency assessment
        Range:    0 to ~10  (higher = more chlorophyll)
        Priority: HIGHEST for N management — companion to RENDVI
        """
        nir, re1 = self._align(b['B08'], b['B05'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(re1 > 1e-10, (nir / re1) - 1.0, np.nan)
        return np.clip(result, 0.0, 20.0).astype(np.float32)

    def _tgi(self, b: Dict) -> Optional[np.ndarray]:
        """
        TGI — Triangle Greenness Index
        Formula:  B03 - 0.39×B04 - 0.61×B02
        Use:      EARLY nitrogen stress detection (pre-symptom warning)
        Range:    Unconstrained, centered ~0
        """
        grn, red, blue = self._align(b['B03'], b['B04'], b['B02'])
        result = grn - 0.39 * red - 0.61 * blue
        return result.astype(np.float32)

    def _msr(self, b: Dict) -> Optional[np.ndarray]:
        """
        MSR — Modified Simple Ratio
        Formula:  (B08/B04 - 1) / sqrt(B08/B04 + 1)
        Use:      P/K status correlation, overall plant health, LAI
        Range:    -1 to ~10
        """
        nir, red = self._align(b['B08'], b['B04'])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(red > 1e-10, nir / red, np.nan)
            result = np.where(
                np.isfinite(ratio) & (ratio + 1 > 0),
                (ratio - 1) / np.sqrt(ratio + 1),
                np.nan
            )
        return np.clip(result, -2.0, 15.0).astype(np.float32)

    def _si(self, b: Dict) -> Optional[np.ndarray]:
        """
        SI — Salinity Index
        Formula:  sqrt(B02 × B04)
        Use:      Soil salinity (EC) assessment — VERY HIGH correlation
        Range:    0 to ~0.5 (higher = more saline)
        Priority: ESSENTIAL for saline soil areas
        """
        blue, red = self._align(b['B02'], b['B04'])
        with np.errstate(invalid='ignore'):
            result = np.sqrt(np.maximum(blue * red, 0.0))
        return result.astype(np.float32)

    def _ndsi(self, b: Dict) -> Optional[np.ndarray]:
        """
        NDSI — Normalized Difference Salinity Index
        Formula:  (B04 - B08) / (B04 + B08)
        Use:      Salt deposit identification, saline soil mapping
        Range:    -1 to 1  (positive = salt deposits / bare saline soil)
        Note:     Different from Snow NDSI — this is the soil salinity version
        """
        red, nir = self._align(b['B04'], b['B08'])
        return self._safe_ratio(red - nir, red + nir)

    def _cmr(self, b: Dict) -> Optional[np.ndarray]:
        """
        CMR — Clay Minerals Ratio
        Formula:  B11 / B12
        Use:      Clay content estimation, CEC (Cation Exchange Capacity),
                  nutrient retention capacity mapping
        Range:    0.5 to 3.0  (higher = more clay)
        Priority: HIGH — critical for understanding nutrient retention
        """
        swir1, swir2 = self._align(b['B11'], b['B12'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(swir2 > 1e-10, swir1 / swir2, np.nan)
        return np.clip(result, 0.1, 5.0).astype(np.float32)

    def _cai(self, b: Dict) -> Optional[np.ndarray]:
        """
        CAI — Cellulose Absorption Index
        Formula:  0.5 × (B11 + B12) - B12  →  simplified: 0.5×B11 - 0.5×B12
        Use:      Organic matter (SOM) estimation, nutrient cycling,
                  carbon sequestration studies
        Range:    Unconstrained, small values
        """
        swir1, swir2 = self._align(b['B11'], b['B12'])
        result = 0.5 * (swir1 + swir2) - swir2
        return result.astype(np.float32)

    def _ior(self, b: Dict) -> Optional[np.ndarray]:
        """
        IOR — Iron Oxide Ratio
        Formula:  B04 / B02
        Use:      Iron content, soil pH estimation (proxy), nutrient solubility
        Range:    0.5 to 5.0  (higher = more iron / lower pH)
        """
        red, blue = self._align(b['B04'], b['B02'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(blue > 1e-10, red / blue, np.nan)
        return np.clip(result, 0.1, 10.0).astype(np.float32)

    def _ndti(self, b: Dict) -> Optional[np.ndarray]:
        """
        NDTI — Normalized Difference Tillage Index
        Formula:  (B11 - B12) / (B11 + B12)
        Use:      Crop residue cover, tillage practice monitoring,
                  soil conservation, organic matter input tracking
        Range:    -1 to 1
        """
        swir1, swir2 = self._align(b['B11'], b['B12'])
        return self._safe_ratio(swir1 - swir2, swir1 + swir2)

    def _bsi(self, b: Dict) -> Optional[np.ndarray]:
        """
        BSI — Bare Soil Index
        Formula:  ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
        Use:      Bare soil identification, soil erosion risk,
                  field preparation tracking, LULC support
        Range:    -1 to 1  (positive = bare soil; negative = vegetation)
        """
        swir1, red, nir, blue = self._align(b['B11'], b['B04'], b['B08'], b['B02'])
        num = (swir1 + red) - (nir + blue)
        den = (swir1 + red) + (nir + blue)
        return self._safe_ratio(num, den)

    # =========================================================================
    # TIER 3 — MOISTURE & STRESS INDICES
    # =========================================================================

    def _ndwi(self, b: Dict) -> Optional[np.ndarray]:
        """
        NDWI — Normalized Difference Water Index (McFeeters 1996)
        Formula:  (B03 - B08) / (B03 + B08)
        Use:      Open water body detection, wetland monitoring,
                  flood extent mapping
        Range:    -1 to 1  (water: >0.0; vegetation: <0)
        """
        grn, nir = self._align(b['B03'], b['B08'])
        return self._safe_ratio(grn - nir, grn + nir)

    def _mndwi(self, b: Dict) -> Optional[np.ndarray]:
        """
        MNDWI — Modified NDWI
        Formula:  (B03 - B11) / (B03 + B11)
        Use:      Enhanced water body detection, better urban suppression,
                  soil moisture in arid regions
        Range:    -1 to 1
        """
        grn, swir1 = self._align(b['B03'], b['B11'])
        return self._safe_ratio(grn - swir1, grn + swir1)

    def _nmdi(self, b: Dict) -> Optional[np.ndarray]:
        """
        NMDI — Normalized Multi-band Drought Index
        Formula:  (B08 - (B11 - B12)) / (B08 + (B11 - B12))
        Use:      Drought monitoring, water stress assessment,
                  agricultural drought early warning
        Range:    -1 to 1
        """
        nir, swir1, swir2 = self._align(b['B08'], b['B11'], b['B12'])
        diff = swir1 - swir2
        return self._safe_ratio(nir - diff, nir + diff)

    def _msi_stress(self, b: Dict) -> Optional[np.ndarray]:
        """
        MSI_STRESS — Moisture Stress Index
        Formula:  B11 / B08
        Use:      Vegetation water stress detection, irrigation management
        Range:    0 to ~3  (higher = more stressed)
        Note:     Inverse of water content — higher = drier
        """
        swir1, nir = self._align(b['B11'], b['B08'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(nir > 1e-10, swir1 / nir, np.nan)
        return np.clip(result, 0.0, 5.0).astype(np.float32)

    def _nddi(self, b: Dict) -> Optional[np.ndarray]:
        """
        NDDI — Normalized Difference Drought Index
        Formula:  (NDVI - NDWI) / (NDVI + NDWI)
        Use:      Drought severity, agricultural drought monitoring,
                  combined vegetation-water stress
        Range:    -1 to 1  (higher = more drought stress)
        """
        nir, red, grn = self._align(b['B08'], b['B04'], b['B03'])
        ndvi = (nir - red) / (nir + red + 1e-10)
        ndwi = (grn - nir) / (grn + nir + 1e-10)
        ndvi = np.clip(ndvi, -1, 1)
        ndwi = np.clip(ndwi, -1, 1)
        return self._safe_ratio(ndvi - ndwi, ndvi + ndwi)

    def _psri(self, b: Dict) -> Optional[np.ndarray]:
        """
        PSRI — Plant Senescence Reflectance Index
        Formula:  (B04 - B03) / B08
        Use:      Crop maturity assessment, harvest timing optimization,
                  senescence / fruit ripening detection
        Range:    -0.1 (healthy) to >0.4 (senescent)
        """
        red, grn, nir = self._align(b['B04'], b['B03'], b['B08'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(nir > 1e-10, (red - grn) / nir, np.nan)
        return np.clip(result, -1.0, 2.0).astype(np.float32)

    # =========================================================================
    # TIER 4 — CANOPY & GROWTH STAGE INDICES
    # =========================================================================

    def _sr1(self, b: Dict) -> Optional[np.ndarray]:
        """
        SR1 — Simple Ratio 1 (NIR/Red)
        Formula:  B08 / B04
        Use:      Biomass estimation, nitrogen correlation in dense veg
        Range:    0 to ~20
        """
        nir, red = self._align(b['B08'], b['B04'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(red > 1e-10, nir / red, np.nan)
        return np.clip(result, 0.0, 30.0).astype(np.float32)

    def _sr2(self, b: Dict) -> Optional[np.ndarray]:
        """
        SR2 — Simple Ratio 2 (Blue/Green)
        Formula:  B02 / B03
        Use:      Early stress detection, general plant health (early season)
        Range:    0 to ~2
        """
        blue, grn = self._align(b['B02'], b['B03'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(grn > 1e-10, blue / grn, np.nan)
        return np.clip(result, 0.0, 5.0).astype(np.float32)

    def _cri(self, b: Dict) -> Optional[np.ndarray]:
        """
        CRI — Carotenoid Reflectance Index (Sentinel-2 adapted)
        Formula:  (1/B03) - (1/B05)
        Use:      Crop stress detection, carotenoid content estimation,
                  early stress indicator before NDVI changes
        Range:    Unconstrained (typically small negative to positive)
        """
        grn, re1 = self._align(b['B03'], b['B05'])
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                (grn > 1e-10) & (re1 > 1e-10),
                (1.0 / grn) - (1.0 / re1),
                np.nan
            )
        return np.clip(result, -50.0, 50.0).astype(np.float32)

    def _lswi(self, b: Dict) -> Optional[np.ndarray]:
        """
        LSWI — Land Surface Water Index
        Formula:  (B8A - B11) / (B8A + B11)
        Use:      Leaf and canopy water content, wetland/paddy detection,
                  soil moisture mapping
        Range:    -1 to 1  (paddy fields: high positive values)
        """
        nir_narrow, swir1 = self._align(b['B8A'], b['B11'])
        return self._safe_ratio(nir_narrow - swir1, nir_narrow + swir1)

    # =========================================================================
    # NUTRIENT INTERPRETATION
    # =========================================================================

    @staticmethod
    def interpret_nutrient_levels(index_results: Dict) -> Dict:
        """
        Interpret spectral indices as nutrient stress indicators.

        Args:
            index_results: Output from compute_all()

        Returns:
            Dict with nutrient stress assessments
        """
        interpretation = {}

        # Nitrogen assessment
        rendvi = index_results.get('RENDVI', {}).get('mean', np.nan)
        ci_re  = index_results.get('CI_REDEDGE', {}).get('mean', np.nan)
        if not np.isnan(rendvi):
            if rendvi > 0.60:
                n_status = 'Adequate'
            elif rendvi > 0.40:
                n_status = 'Moderate Deficiency'
            elif rendvi > 0.25:
                n_status = 'Significant Deficiency'
            else:
                n_status = 'Severe Deficiency'
            interpretation['Nitrogen'] = {
                'status': n_status, 'RENDVI': round(rendvi, 4),
                'CI_RedEdge': round(ci_re, 4) if not np.isnan(ci_re) else None,
                'action': 'Apply N fertilizer' if 'Deficiency' in n_status else 'Monitor'
            }

        # Salinity assessment
        si = index_results.get('SI', {}).get('mean', np.nan)
        if not np.isnan(si):
            if si < 0.05:
                ec_status = 'Low Salinity'
            elif si < 0.10:
                ec_status = 'Moderate Salinity'
            elif si < 0.20:
                ec_status = 'High Salinity'
            else:
                ec_status = 'Very High Salinity — Reclamation Needed'
            interpretation['Salinity_EC'] = {
                'status': ec_status, 'SI': round(si, 4),
                'action': 'Leaching / amendment' if 'High' in ec_status else 'Monitor'
            }

        # Organic matter (OC)
        cai = index_results.get('CAI', {}).get('mean', np.nan)
        if not np.isnan(cai):
            oc_status = 'Adequate' if cai > 0.01 else 'Low — Add Organic Matter'
            interpretation['Organic_Carbon'] = {
                'status': oc_status, 'CAI': round(cai, 4),
                'action': 'Add compost/FYM' if 'Low' in oc_status else 'Monitor'
            }

        # Moisture stress
        ndmi = index_results.get('NDMI', {}).get('mean', np.nan)
        if not np.isnan(ndmi):
            if ndmi > 0.4:
                m_status = 'No Stress'
            elif ndmi > 0.2:
                m_status = 'Mild Stress'
            elif ndmi > 0.0:
                m_status = 'Moderate Stress'
            else:
                m_status = 'Severe Drought Stress'
            interpretation['Moisture_Stress'] = {
                'status': m_status, 'NDMI': round(ndmi, 4),
                'action': 'Irrigate' if 'Stress' in m_status else 'No action'
            }

        return interpretation
