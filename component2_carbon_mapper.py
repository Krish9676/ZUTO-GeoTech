"""
ZUTO Geotech Solutions — Environmental Analysis Platform
=========================================================
Module A: Carbon Footprint Mapper

Computes grid-level carbon stock estimates and tracks carbon gains/losses
over time using LULC classification from Sentinel-2 + Landsat archive.

Workflow:
  1. Classify LULC from spectral indices (NDVI, NDBI, MNDWI, BSI)
  2. Assign IPCC Tier-1 carbon stock factors per land class
  3. Run bi-temporal change detection (dNDVI) to find transitions
  4. Compute emission and sequestration estimates per grid cell
  5. Aggregate to field/company/district polygon for ESG reporting
  6. Export as GeoJSON + downloadable PDF report summary

Target clients: Industries (ESG compliance), NGOs, carbon credit platforms,
                state forest departments, climate finance institutions.
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from env_config import ZutoEnvConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LULCPixel:
    """Represents a classified land-use pixel or polygon"""
    pixel_id:       str
    latitude:       float
    longitude:      float
    area_ha:        float
    lulc_class:     int
    lulc_name:      str
    ndvi:           float
    ndbi:           float
    mndwi:          float
    bsi:            float
    carbon_stock_t: float       # tonnes of carbon
    co2e_stock:     float       # tCO2e
    timestamp:      str


@dataclass
class CarbonChange:
    """Carbon change between two time periods"""
    pixel_id:           str
    from_class:         str
    to_class:           str
    transition_type:    str
    area_ha:            float
    carbon_stock_t1:    float   # baseline carbon
    carbon_stock_t2:    float   # current carbon
    carbon_change_t:    float   # negative = loss
    co2e_emitted:       float   # positive = emissions
    co2e_sequestered:   float   # positive = removal
    period_years:       float
    annual_rate_tco2e:  float


@dataclass
class CarbonReport:
    """Full carbon assessment for an area of interest"""
    aoi_id:                 str
    aoi_name:               str
    report_date:            str
    baseline_year:          int
    current_year:           int
    total_area_ha:          float
    lulc_summary:           Dict        # class → area breakdown
    total_carbon_stock_t:   float
    total_co2e_stock:       float
    carbon_changes:         List[Dict]
    total_co2e_emitted:     float
    total_co2e_sequestered: float
    net_co2e_balance:       float
    annual_net_co2e:        float
    emission_hotspots:      List[Dict]
    sequestration_zones:    List[Dict]
    esg_score:              float       # 0-100 (100 = best)
    recommendations:        List[str]
    methodology:            str


# =============================================================================
# CARBON MAPPER — MAIN CLASS
# =============================================================================

class CarbonMapper:
    """
    Maps carbon stocks and tracks emission/sequestration from LULC change.

    Uses Sentinel-2 derived spectral indices for LULC classification and
    applies IPCC Tier-1 emission factors to quantify carbon dynamics.

    Example:
        mapper = CarbonMapper()
        report = mapper.generate_carbon_report(
            aoi_geojson    = my_area,
            scenes_t1      = baseline_scenes,
            scenes_t2      = current_scenes,
            baseline_year  = 2020,
            current_year   = 2024,
            aoi_name       = "Koyna Industrial Zone"
        )
    """

    def __init__(self):
        self.cfg       = ZutoEnvConfig
        self.lulc_cls  = self.cfg.LULC_CLASSES
        self.ipcc      = self.cfg.IPCC_EMISSION_FACTORS
        self.thresh    = self.cfg.LULC_THRESHOLDS

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_carbon_report(
        self,
        aoi_geojson:    Dict,
        scenes_t1:      List[Dict],
        scenes_t2:      List[Dict],
        baseline_year:  int,
        current_year:   int,
        aoi_name:       str = "Area of Interest",
        aoi_id:         str = "AOI_001",
    ) -> CarbonReport:
        """
        Full carbon assessment: classify LULC for two time periods,
        compute change, estimate emissions and sequestration.

        Args:
            aoi_geojson:   GeoJSON polygon of area of interest
            scenes_t1:     List of satellite scenes for baseline period
            scenes_t2:     List of satellite scenes for current period
            baseline_year: Year of baseline (T1)
            current_year:  Year of current assessment (T2)
            aoi_name:      Human-readable name for report
            aoi_id:        Unique ID for this AOI

        Returns:
            CarbonReport dataclass with full assessment
        """
        logger.info(f"\nCARBON MAPPING — {aoi_name} | {baseline_year} → {current_year}")

        # — Compute composite indices for T1 and T2
        indices_t1 = self._compute_composite_indices(scenes_t1)
        indices_t2 = self._compute_composite_indices(scenes_t2)

        # — Classify LULC for both periods
        lulc_t1 = self._classify_lulc(indices_t1)
        lulc_t2 = self._classify_lulc(indices_t2)

        # — Estimate area per class (simulated grid approach)
        total_area_ha = self._estimate_aoi_area(aoi_geojson)
        area_t1       = self._lulc_area_breakdown(lulc_t1, total_area_ha)
        area_t2       = self._lulc_area_breakdown(lulc_t2, total_area_ha)

        # — Compute carbon stocks
        stock_t1 = self._compute_carbon_stock(area_t1)
        stock_t2 = self._compute_carbon_stock(area_t2)

        # — Change detection
        period_years  = current_year - baseline_year
        changes       = self._detect_lulc_changes(lulc_t1, lulc_t2, total_area_ha, period_years)

        # — Emissions & sequestration
        co2e_emitted      = sum(c['co2e_emitted'] for c in changes)
        co2e_sequestered  = sum(c['co2e_sequestered'] for c in changes)
        net_balance       = co2e_sequestered - co2e_emitted
        annual_net        = net_balance / period_years if period_years > 0 else 0.0

        # — Hotspots
        hotspots    = self._identify_emission_hotspots(changes)
        seq_zones   = self._identify_sequestration_zones(changes)

        # — ESG score and recommendations
        esg_score   = self._compute_esg_score(lulc_t2, net_balance, total_area_ha)
        recoms      = self._generate_recommendations(lulc_t1, lulc_t2, changes, esg_score)

        report = CarbonReport(
            aoi_id                 = aoi_id,
            aoi_name               = aoi_name,
            report_date            = date.today().isoformat(),
            baseline_year          = baseline_year,
            current_year           = current_year,
            total_area_ha          = total_area_ha,
            lulc_summary           = {'baseline': area_t1, 'current': area_t2},
            total_carbon_stock_t   = stock_t2['total_carbon_t'],
            total_co2e_stock       = stock_t2['total_co2e'],
            carbon_changes         = changes,
            total_co2e_emitted     = co2e_emitted,
            total_co2e_sequestered = co2e_sequestered,
            net_co2e_balance       = net_balance,
            annual_net_co2e        = annual_net,
            emission_hotspots      = hotspots,
            sequestration_zones    = seq_zones,
            esg_score              = esg_score,
            recommendations        = recoms,
            methodology            = self._methodology_note(),
        )

        self._log_report_summary(report)
        return report

    def classify_single_scene(self, scene: Dict) -> Dict:
        """
        Classify LULC for a single satellite scene.
        Returns pixel-level classification dict.
        """
        indices = self._scene_to_indices(scene)
        lulc    = self._classify_lulc(indices)
        return {
            'timestamp': scene.get('timestamp', ''),
            'lulc_map':  lulc,
            'stats':     self._lulc_stats(lulc),
        }

    def compute_dndvi_change(
        self,
        scenes_t1: List[Dict],
        scenes_t2: List[Dict],
    ) -> Dict:
        """
        Compute dNDVI (delta NDVI) between two periods.
        Positive = vegetation gain, Negative = vegetation loss.
        Used as a fast carbon gain/loss proxy.
        """
        ndvi_t1 = np.mean([s['indices'].get('ndvi', 0.0) for s in scenes_t1 if 'indices' in s])
        ndvi_t2 = np.mean([s['indices'].get('ndvi', 0.0) for s in scenes_t2 if 'indices' in s])
        dndvi   = float(ndvi_t2 - ndvi_t1)

        return {
            'ndvi_t1':       float(ndvi_t1),
            'ndvi_t2':       float(ndvi_t2),
            'dndvi':         dndvi,
            'change_type':   'gain' if dndvi > 0.05 else ('loss' if dndvi < -0.05 else 'stable'),
            'magnitude':     'significant' if abs(dndvi) > 0.15 else ('moderate' if abs(dndvi) > 0.05 else 'minor'),
        }

    def export_geojson(self, report: CarbonReport, output_path: str) -> str:
        """
        Export carbon report as GeoJSON FeatureCollection.
        Each change polygon becomes a Feature with carbon properties.
        """
        features = []
        for i, change in enumerate(report.carbon_changes):
            feature = {
                'type':       'Feature',
                'geometry':   change.get('geometry', None),
                'properties': {
                    'change_id':        i + 1,
                    'from_class':       change['from_class'],
                    'to_class':         change['to_class'],
                    'area_ha':          round(change['area_ha'], 2),
                    'co2e_emitted':     round(change['co2e_emitted'], 2),
                    'co2e_sequestered': round(change['co2e_sequestered'], 2),
                    'net_co2e':         round(change['co2e_sequestered'] - change['co2e_emitted'], 2),
                },
            }
            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'properties': {
                'aoi_name':     report.aoi_name,
                'period':       f"{report.baseline_year}–{report.current_year}",
                'esg_score':    report.esg_score,
                'net_co2e':     round(report.net_co2e_balance, 2),
                'generated_by': 'ZUTO Geotech Solutions',
            },
            'features': features,
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"GeoJSON exported → {output_path}")
        return output_path

    # =========================================================================
    # INTERNAL — INDEX COMPUTATION
    # =========================================================================

    def _compute_composite_indices(self, scenes: List[Dict]) -> Dict:
        """Average spectral indices across all scenes in a period."""
        if not scenes:
            return self._zero_indices()

        index_keys = ['ndvi', 'ndbi', 'mndwi', 'bsi', 'evi', 'b02', 'b03', 'b04', 'b08', 'b11']
        composites = {k: [] for k in index_keys}

        for scene in scenes:
            idx = scene.get('indices', {})
            for k in index_keys:
                val = idx.get(k, None)
                if val is not None and not np.isnan(val):
                    composites[k].append(val)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in composites.items()}

    def _scene_to_indices(self, scene: Dict) -> Dict:
        """Extract indices dict from a single scene."""
        return scene.get('indices', self._zero_indices())

    def _zero_indices(self) -> Dict:
        return {'ndvi': 0.0, 'ndbi': 0.0, 'mndwi': 0.0, 'bsi': 0.0, 'evi': 0.0}

    # =========================================================================
    # INTERNAL — LULC CLASSIFICATION
    # =========================================================================

    def _classify_lulc(self, indices: Dict) -> Dict:
        """
        Rule-based LULC classification from spectral indices.
        Priority order: Water > Urban > Forest > Cropland > Grassland > Barren

        Returns dict with 'class_id', 'class_name', 'confidence'.
        """
        ndvi  = indices.get('ndvi',  0.0)
        ndbi  = indices.get('ndbi',  0.0)
        mndwi = indices.get('mndwi', 0.0)
        bsi   = indices.get('bsi',   0.0)
        t     = self.thresh

        # — Water
        if mndwi > t['mndwi_water_min'] and ndvi < 0.1:
            return self._lulc_result(5, 'Water', 0.90)

        # — Urban / Built-up
        if ndbi > t['ndbi_urban_min'] and ndvi < 0.20:
            return self._lulc_result(4, 'Urban', 0.85)

        # — Forest
        if ndvi >= t['ndvi_forest_min']:
            return self._lulc_result(0, 'Forest', 0.88)

        # — Cropland
        if t['ndvi_cropland_min'] <= ndvi < t['ndvi_forest_min']:
            return self._lulc_result(1, 'Cropland', 0.80)

        # — Grassland
        if t['ndvi_grassland_min'] <= ndvi < t['ndvi_cropland_min']:
            return self._lulc_result(2, 'Grassland', 0.75)

        # — Barren
        if bsi > t['bsi_barren_min'] or ndvi < t['ndvi_grassland_min']:
            return self._lulc_result(6, 'Barren', 0.78)

        # — Default: Shrubland
        return self._lulc_result(7, 'Shrubland', 0.65)

    def _lulc_result(self, class_id: int, name: str, confidence: float) -> Dict:
        return {
            'class_id':   class_id,
            'class_name': name,
            'confidence': confidence,
            'carbon_factor': self.lulc_cls[class_id]['carbon_factor'],
        }

    # =========================================================================
    # INTERNAL — AREA & CARBON COMPUTATIONS
    # =========================================================================

    def _estimate_aoi_area(self, aoi_geojson: Dict) -> float:
        """
        Estimate AOI area in hectares from GeoJSON polygon.
        Uses shoelace formula for lat/lon coordinates (approximate).
        """
        try:
            geom  = aoi_geojson.get('geometry', aoi_geojson)
            coords = geom.get('coordinates', [[]])[0]
            if len(coords) < 3:
                return 100.0   # fallback 100ha

            # Shoelace in degrees then convert
            n    = len(coords)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += coords[i][0] * coords[j][1]
                area -= coords[j][0] * coords[i][1]
            area = abs(area) / 2.0

            # Approximate: 1 degree² ≈ 12,308 km² at lat 20°N
            area_km2 = area * 12308
            return area_km2 * 100   # km² → ha
        except Exception:
            return 100.0

    def _lulc_area_breakdown(self, lulc: Dict, total_ha: float) -> Dict:
        """
        From a single classified pixel/composite, derive area breakdown.
        In production this would be per-pixel; here we model as dominant class
        with confidence-weighted fraction.
        """
        dominant = lulc['class_name']
        conf     = lulc['confidence']

        # Assign confidence fraction to dominant class, distribute rest
        breakdown = {cls['name']: 0.0 for cls in self.lulc_cls.values()}
        breakdown[dominant] = round(conf * total_ha, 2)

        remaining_ha = (1.0 - conf) * total_ha
        others = [c['name'] for c in self.lulc_cls.values() if c['name'] != dominant]
        if others:
            share = remaining_ha / len(others)
            for c in others:
                breakdown[c] = round(share, 2)

        return breakdown

    def _compute_carbon_stock(self, area_breakdown: Dict) -> Dict:
        """Compute total carbon stock from LULC area breakdown."""
        total_carbon = 0.0
        per_class    = {}

        factor_map = {cls['name']: cls['carbon_factor'] for cls in self.lulc_cls.values()}

        for class_name, area_ha in area_breakdown.items():
            factor = factor_map.get(class_name, 0.0)
            carbon = area_ha * factor
            total_carbon += carbon
            per_class[class_name] = {
                'area_ha':    area_ha,
                'factor':     factor,
                'carbon_t':   round(carbon, 2),
                'co2e':       round(carbon * self.cfg.CO2_FACTOR, 2),
            }

        return {
            'per_class':     per_class,
            'total_carbon_t': round(total_carbon, 2),
            'total_co2e':    round(total_carbon * self.cfg.CO2_FACTOR, 2),
        }

    # =========================================================================
    # INTERNAL — CHANGE DETECTION
    # =========================================================================

    def _detect_lulc_changes(
        self,
        lulc_t1:      Dict,
        lulc_t2:      Dict,
        total_area_ha: float,
        period_years:  float,
    ) -> List[Dict]:
        """
        Detect LULC transitions between T1 and T2 and compute
        carbon emissions / sequestration per transition.
        """
        changes = []

        t1_name = lulc_t1['class_name']
        t2_name = lulc_t2['class_name']

        if t1_name == t2_name:
            # No change — natural carbon dynamics only
            changes.append(self._stable_change_record(t1_name, total_area_ha, period_years))
            return changes

        # Transition detected
        trans_key  = f"{t1_name.lower()}_to_{t2_name.lower()}"
        emis_factor = self.ipcc.get(trans_key, 0.0)   # tCO2e/ha

        co2e_emitted     = max(emis_factor * total_area_ha, 0.0)
        co2e_sequestered = 0.0

        # Check if new class is sequestering (forest gain etc.)
        seq_key  = f"{t2_name.lower()}_afforestation"
        seq_rate = self.cfg.CARBON_SEQUESTRATION.get(seq_key, 0.0)
        co2e_sequestered = seq_rate * total_area_ha * period_years

        t1_factor = self.lulc_cls.get(lulc_t1['class_id'], {}).get('carbon_factor', 0.0)
        t2_factor = self.lulc_cls.get(lulc_t2['class_id'], {}).get('carbon_factor', 0.0)

        annual_rate = (co2e_sequestered - co2e_emitted) / period_years if period_years > 0 else 0.0

        changes.append({
            'from_class':         t1_name,
            'to_class':           t2_name,
            'transition_type':    trans_key,
            'area_ha':            round(total_area_ha, 2),
            'carbon_stock_t1':    round(t1_factor * total_area_ha, 2),
            'carbon_stock_t2':    round(t2_factor * total_area_ha, 2),
            'carbon_change_t':    round((t2_factor - t1_factor) * total_area_ha, 2),
            'co2e_emitted':       round(co2e_emitted, 2),
            'co2e_sequestered':   round(co2e_sequestered, 2),
            'period_years':       period_years,
            'annual_rate_tco2e':  round(annual_rate, 2),
            'geometry':           None,   # populated by GIS layer in production
        })

        return changes

    def _stable_change_record(self, class_name: str, area_ha: float, period_years: float) -> Dict:
        """Record for areas with no LULC change — still track natural flux."""
        factor   = next((c['carbon_factor'] for c in self.lulc_cls.values() if c['name'] == class_name), 0.0)
        seq_rate = self.cfg.CARBON_SEQUESTRATION.get(f"{class_name.lower()}_natural_regen", 0.0)
        co2e_seq = seq_rate * area_ha * period_years
        return {
            'from_class': class_name, 'to_class': class_name,
            'transition_type': 'stable',
            'area_ha': round(area_ha, 2),
            'carbon_stock_t1': round(factor * area_ha, 2),
            'carbon_stock_t2': round(factor * area_ha, 2),
            'carbon_change_t': 0.0,
            'co2e_emitted': 0.0,
            'co2e_sequestered': round(co2e_seq, 2),
            'period_years': period_years,
            'annual_rate_tco2e': round(co2e_seq / period_years if period_years > 0 else 0.0, 2),
            'geometry': None,
        }

    # =========================================================================
    # INTERNAL — HOTSPOTS & ESG
    # =========================================================================

    def _identify_emission_hotspots(self, changes: List[Dict]) -> List[Dict]:
        """Flag changes with significant emissions as hotspots."""
        hotspots = [
            c for c in changes
            if c['co2e_emitted'] > 100   # > 100 tCO2e threshold
        ]
        return sorted(hotspots, key=lambda x: x['co2e_emitted'], reverse=True)[:10]

    def _identify_sequestration_zones(self, changes: List[Dict]) -> List[Dict]:
        """Flag zones with active carbon sequestration."""
        zones = [
            c for c in changes
            if c['co2e_sequestered'] > 10
        ]
        return sorted(zones, key=lambda x: x['co2e_sequestered'], reverse=True)[:10]

    def _compute_esg_score(
        self,
        lulc_t2:      Dict,
        net_co2e:     float,
        total_area_ha: float,
    ) -> float:
        """
        ESG score 0-100. Higher = better environmental performance.
        Based on: dominant land class quality + net carbon balance per ha.
        """
        # Class quality scores
        class_scores = {
            'Forest': 100, 'Wetland': 95, 'Grassland': 70, 'Cropland': 55,
            'Shrubland': 60, 'Water': 80, 'Barren': 20, 'Urban': 25,
        }
        base_score = class_scores.get(lulc_t2['class_name'], 50)

        # Adjust for carbon balance (net per ha, normalized)
        net_per_ha     = net_co2e / total_area_ha if total_area_ha > 0 else 0.0
        carbon_adj     = np.clip(net_per_ha / 10.0, -20, 20)   # ±20 point adjustment
        esg_score      = np.clip(base_score + carbon_adj, 0, 100)
        return round(float(esg_score), 1)

    def _generate_recommendations(
        self,
        lulc_t1:  Dict,
        lulc_t2:  Dict,
        changes:  List[Dict],
        esg_score: float,
    ) -> List[str]:
        """Generate actionable ESG recommendations based on assessment."""
        recs = []
        t1, t2 = lulc_t1['class_name'], lulc_t2['class_name']

        if t1 == 'Forest' and t2 != 'Forest':
            recs.append(f"CRITICAL: Forest loss detected ({t1}→{t2}). Initiate compensatory afforestation of ≥1.5× lost area per MoEFCC guidelines.")

        if t2 == 'Urban':
            recs.append("Install urban greening corridors (min. 33% green cover per Master Plan norms) to offset impervious surface heat and carbon impact.")

        total_emitted = sum(c['co2e_emitted'] for c in changes)
        if total_emitted > 500:
            recs.append(f"Carbon emissions of {total_emitted:.0f} tCO2e detected. Consider VCS/Gold Standard carbon offset projects to reach net-zero.")

        if esg_score < 40:
            recs.append("Low ESG score. Prioritize land restoration, wetland conservation, and renewable energy transition to improve score.")
        elif esg_score < 65:
            recs.append("Moderate ESG performance. Incremental agroforestry and soil carbon enhancement can improve score by 15–20 points.")
        else:
            recs.append("Good ESG baseline. Maintain forest cover, explore carbon credit monetization via REDD+ or voluntary carbon markets.")

        if not recs:
            recs.append("Land use appears stable. Continue annual monitoring to detect early-stage change.")

        return recs

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _lulc_stats(self, lulc: Dict) -> Dict:
        return {
            'dominant_class': lulc['class_name'],
            'class_id':       lulc['class_id'],
            'confidence':     lulc['confidence'],
            'carbon_factor':  lulc['carbon_factor'],
        }

    def _methodology_note(self) -> str:
        return (
            "LULC classified using Sentinel-2 spectral indices (NDVI, NDBI, MNDWI, BSI) "
            "via rule-based thresholds validated on Indian land cover. Carbon stocks estimated "
            "using IPCC Tier-1 emission factors (IPCC 2006 GL, Vol 4). Biomass-to-carbon "
            "conversion: CF=0.47, CO2e factor=3.667. Change detection via bi-temporal "
            "composite analysis. Results indicative; field validation recommended for "
            "carbon credit certification."
        )

    def _log_report_summary(self, report: CarbonReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  CARBON REPORT — {report.aoi_name}\n"
            f"  Period      : {report.baseline_year} → {report.current_year}\n"
            f"  Total Area  : {report.total_area_ha:,.1f} ha\n"
            f"  Carbon Stock: {report.total_carbon_stock_t:,.1f} tC  "
            f"({report.total_co2e_stock:,.1f} tCO2e)\n"
            f"  CO2e Emitted   : {report.total_co2e_emitted:,.1f} tCO2e\n"
            f"  CO2e Sequestered: {report.total_co2e_sequestered:,.1f} tCO2e\n"
            f"  Net Balance    : {report.net_co2e_balance:+,.1f} tCO2e\n"
            f"  Annual Net     : {report.annual_net_co2e:+,.1f} tCO2e/yr\n"
            f"  ESG Score   : {report.esg_score}/100\n"
            f"  Recommendations: {len(report.recommendations)}\n"
            f"{'='*60}"
        )
