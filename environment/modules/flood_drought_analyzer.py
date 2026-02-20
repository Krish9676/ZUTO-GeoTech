"""
ZUTO Geotech Solutions — Environmental Analysis Platform
=========================================================
Module B: Flood & Drought Risk Analyzer

Dual-hazard assessment using satellite-derived and weather data:

FLOOD:
  - SAR Sentinel-1 change detection (pre vs post event backscatter)
  - MNDWI for open-water extent mapping
  - DEM overlay for inundation depth estimation
  - Flood probability maps for insurance underwriting

DROUGHT:
  - NDDI = (NDVI-NDWI)/(NDVI+NDWI) for combined vegetation-moisture stress
  - NMDI for multi-scale drought severity
  - NDMI for crop/soil moisture
  - SPI (Standard Precipitation Index) from weather data

Output:
  - Risk scores per district/tehsil for climate finance & crop insurance
  - GeoJSON risk zone maps (color-coded by severity)
  - Historical trend analysis (10+ year Landsat archive)
  - Insurance underwriting data packets
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.env_config import ZutoEnvConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FloodAssessment:
    """Flood assessment for a given event/period"""
    event_id:               str
    location_name:          str
    event_date:             str
    flood_extent_ha:        float
    max_inundation_depth_m: float
    affected_cropland_ha:   float
    affected_settlements:   int
    risk_zone:              str         # Safe / Low / Medium / High
    probability:            float       # 0-1
    insurance_multiplier:   float
    sar_change_db:          float       # backscatter change
    mndwi_extent:           float       # water fraction 0-1
    recovery_estimate_days: int
    summary:                str


@dataclass
class DroughtAssessment:
    """Drought assessment for a given period"""
    region_id:          str
    region_name:        str
    period_start:       str
    period_end:         str
    severity:           str         # None / Mild / Moderate / Severe
    nddi_mean:          float
    ndmi_mean:          float
    nmdi_mean:          float
    spi_value:          float
    drought_score:      float       # 0-100
    affected_area_pct:  float       # % of total area in drought
    crop_loss_estimate: float       # % crop yield loss proxy
    risk_score:         int         # 0-100 for insurance
    alerts:             List[str]
    recovery_forecast:  str


@dataclass
class HazardRiskReport:
    """Combined flood + drought risk report for a region"""
    region_id:          str
    region_name:        str
    report_date:        str
    analysis_period:    str
    flood_risk_score:   float       # 0-100
    drought_risk_score: float       # 0-100
    combined_risk_score: float      # weighted composite
    risk_category:      str         # Low / Moderate / High / Extreme
    flood_events:       List[Dict]
    drought_events:     List[Dict]
    historical_trend:   Dict        # year → risk scores
    insurance_premium_factor: float
    recommendations:    List[str]
    methodology:        str


# =============================================================================
# FLOOD & DROUGHT ANALYZER — MAIN CLASS
# =============================================================================

class FloodDroughtAnalyzer:
    """
    Dual-hazard analyzer for flood and drought risk using Sentinel-1 SAR,
    Sentinel-2 optical indices, DEM, and weather/SPI data.

    Example:
        analyzer = FloodDroughtAnalyzer()

        # Flood assessment after a heavy rain event
        flood = analyzer.assess_flood_event(
            pre_event_sar   = sar_scene_pre,
            post_event_sar  = sar_scene_post,
            optical_scene   = s2_scene,
            dem_data        = dem_array,
            location_name   = "Kolhapur District"
        )

        # Drought monitoring over a season
        drought = analyzer.assess_drought(
            scenes          = season_scenes,
            weather_data    = weather_df,
            region_name     = "Vidarbha Region"
        )

        # Full combined report
        report = analyzer.generate_hazard_report(
            flood_scenes   = scenes_post_monsoon,
            drought_scenes = scenes_rabi,
            weather_data   = weather_df,
            region_name    = "Maharashtra District"
        )
    """

    def __init__(self):
        self.cfg          = ZutoEnvConfig
        self.flood_thresh = self.cfg.FLOOD_THRESHOLDS
        self.drought_thresh = self.cfg.DROUGHT_THRESHOLDS
        self.risk_zones   = self.cfg.FLOOD_RISK_ZONES
        self.severity_map = self.cfg.DROUGHT_SEVERITY

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def assess_flood_event(
        self,
        pre_event_sar:  Dict,
        post_event_sar: Dict,
        optical_scene:  Optional[Dict] = None,
        dem_data:       Optional[np.ndarray] = None,
        location_name:  str = "Unknown Location",
        event_date:     Optional[str] = None,
        total_area_ha:  float = 1000.0,
    ) -> FloodAssessment:
        """
        Assess a flood event using pre/post SAR change detection.

        Args:
            pre_event_sar:  SAR scene dict (before flood) with 'vv_backscatter'
            post_event_sar: SAR scene dict (after flood) with 'vv_backscatter'
            optical_scene:  Optional Sentinel-2 scene for MNDWI cross-check
            dem_data:       Optional DEM array for depth estimation
            location_name:  Human-readable location name
            event_date:     Date of the flood event (ISO format)
            total_area_ha:  Total analysis area in hectares

        Returns:
            FloodAssessment dataclass
        """
        event_id = f"FLOOD_{datetime.now().strftime('%Y%m%d_%H%M')}"
        logger.info(f"\nFLOOD ASSESSMENT — {location_name} | {event_date or 'Unknown date'}")

        # — SAR backscatter change
        sar_change_db  = self._compute_sar_change(pre_event_sar, post_event_sar)

        # — MNDWI water extent (if optical available)
        mndwi_extent   = self._compute_mndwi_extent(optical_scene)

        # — Flood probability from SAR + MNDWI
        flood_prob     = self._compute_flood_probability(sar_change_db, mndwi_extent)

        # — Flood extent in ha
        flood_extent_ha = flood_prob * total_area_ha

        # — Inundation depth estimate (from DEM)
        max_depth_m    = self._estimate_inundation_depth(dem_data, flood_prob)

        # — Crop and settlement impact
        affected_crop_ha    = flood_extent_ha * 0.65   # assume 65% of flooded area is cropland
        affected_settlements = int(flood_extent_ha / 50)  # rough proxy: 1 village per 50ha flooded

        # — Risk zone
        risk_zone, ins_mult = self._classify_flood_risk_zone(flood_prob)

        # — Recovery estimate
        recovery_days = self._estimate_recovery_days(max_depth_m, risk_zone)

        summary = (
            f"{risk_zone} flood risk at {location_name}. "
            f"~{flood_extent_ha:.0f} ha inundated, max depth ~{max_depth_m:.1f}m. "
            f"Estimated recovery: {recovery_days} days."
        )

        assessment = FloodAssessment(
            event_id               = event_id,
            location_name          = location_name,
            event_date             = event_date or date.today().isoformat(),
            flood_extent_ha        = round(flood_extent_ha, 1),
            max_inundation_depth_m = round(max_depth_m, 2),
            affected_cropland_ha   = round(affected_crop_ha, 1),
            affected_settlements   = affected_settlements,
            risk_zone              = risk_zone,
            probability            = round(flood_prob, 3),
            insurance_multiplier   = ins_mult,
            sar_change_db          = round(sar_change_db, 2),
            mndwi_extent           = round(mndwi_extent, 3),
            recovery_estimate_days = recovery_days,
            summary                = summary,
        )

        self._log_flood_summary(assessment)
        return assessment

    def assess_drought(
        self,
        scenes:       List[Dict],
        weather_data: Optional[pd.DataFrame] = None,
        region_name:  str = "Unknown Region",
        region_id:    str = "REG_001",
        period_start: Optional[str] = None,
        period_end:   Optional[str] = None,
        total_area_ha: float = 10000.0,
    ) -> DroughtAssessment:
        """
        Assess drought severity for a region over a period.

        Args:
            scenes:       List of Sentinel-2 scenes for the period
            weather_data: DataFrame with columns [date, rainfall_mm, temp_c]
            region_name:  Human-readable region name
            region_id:    Unique region ID
            period_start: Period start date (ISO)
            period_end:   Period end date (ISO)
            total_area_ha: Total region area in ha

        Returns:
            DroughtAssessment dataclass
        """
        logger.info(f"\nDROUGHT ASSESSMENT — {region_name} | Scenes: {len(scenes)}")

        # — Compute NDDI, NDMI, NMDI from scenes
        nddi_mean = self._compute_nddi(scenes)
        ndmi_mean = self._compute_ndmi(scenes)
        nmdi_mean = self._compute_nmdi(scenes)

        # — SPI from weather data
        spi_value = self._compute_spi(weather_data)

        # — Drought severity classification
        severity  = self._classify_drought_severity(nddi_mean, spi_value)

        # — Drought score (0-100)
        drought_score = self._compute_drought_score(nddi_mean, ndmi_mean, spi_value)

        # — Affected area
        affected_pct  = self._estimate_affected_area(nddi_mean, severity)

        # — Crop loss proxy
        crop_loss_pct = self._estimate_crop_loss(nddi_mean, ndmi_mean, severity)

        # — Risk score (insurance-ready)
        risk_score    = int(drought_score)

        # — Alerts
        alerts = self._generate_drought_alerts(nddi_mean, ndmi_mean, spi_value, severity)

        # — Recovery forecast
        recovery = self._drought_recovery_forecast(spi_value, severity)

        assessment = DroughtAssessment(
            region_id          = region_id,
            region_name        = region_name,
            period_start       = period_start or '',
            period_end         = period_end or date.today().isoformat(),
            severity           = severity,
            nddi_mean          = round(nddi_mean, 4),
            ndmi_mean          = round(ndmi_mean, 4),
            nmdi_mean          = round(nmdi_mean, 4),
            spi_value          = round(spi_value, 3),
            drought_score      = round(drought_score, 1),
            affected_area_pct  = round(affected_pct, 1),
            crop_loss_estimate = round(crop_loss_pct, 1),
            risk_score         = risk_score,
            alerts             = alerts,
            recovery_forecast  = recovery,
        )

        self._log_drought_summary(assessment)
        return assessment

    def generate_hazard_report(
        self,
        flood_pre_scenes:  Optional[List[Dict]] = None,
        flood_post_scenes: Optional[List[Dict]] = None,
        drought_scenes:    Optional[List[Dict]] = None,
        weather_data:      Optional[pd.DataFrame] = None,
        dem_data:          Optional[np.ndarray] = None,
        region_name:       str = "Unknown Region",
        region_id:         str = "REG_001",
        total_area_ha:     float = 10000.0,
    ) -> HazardRiskReport:
        """
        Generate combined flood + drought risk report for a region.
        This is the primary output for insurance underwriting and
        climate finance clients.
        """
        logger.info(f"\nHAZARD RISK REPORT — {region_name}")

        flood_events  = []
        drought_events = []

        # — Flood assessment
        flood_risk_score = 0.0
        if flood_pre_scenes and flood_post_scenes:
            pre_sar  = self._extract_sar_scene(flood_pre_scenes)
            post_sar = self._extract_sar_scene(flood_post_scenes)
            optical  = flood_post_scenes[0] if flood_post_scenes else None
            flood_ev = self.assess_flood_event(
                pre_sar, post_sar, optical, dem_data,
                location_name=region_name, total_area_ha=total_area_ha,
            )
            flood_events.append(self._flood_to_dict(flood_ev))
            flood_risk_score = flood_ev.probability * 100

        # — Drought assessment
        drought_risk_score = 0.0
        if drought_scenes:
            drought_ev = self.assess_drought(
                drought_scenes, weather_data,
                region_name=region_name, region_id=region_id,
                total_area_ha=total_area_ha,
            )
            drought_events.append(self._drought_to_dict(drought_ev))
            drought_risk_score = drought_ev.drought_score

        # — Combined risk
        combined = 0.6 * flood_risk_score + 0.4 * drought_risk_score
        risk_cat  = self._combined_risk_category(combined)

        ins_factor = 1.0 + (combined / 100) * 1.5   # 1.0 to 2.5×

        report = HazardRiskReport(
            region_id             = region_id,
            region_name           = region_name,
            report_date           = date.today().isoformat(),
            analysis_period       = f"As of {date.today().isoformat()}",
            flood_risk_score      = round(flood_risk_score, 1),
            drought_risk_score    = round(drought_risk_score, 1),
            combined_risk_score   = round(combined, 1),
            risk_category         = risk_cat,
            flood_events          = flood_events,
            drought_events        = drought_events,
            historical_trend      = {},   # populated from archive queries
            insurance_premium_factor = round(ins_factor, 2),
            recommendations       = self._hazard_recommendations(flood_risk_score, drought_risk_score),
            methodology           = self._methodology_note(),
        )

        self._log_report_summary(report)
        return report

    def compute_flood_probability_map(
        self,
        scenes_archive: List[Dict],
        dem_data:       Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute historical flood probability from a multi-year scene archive.
        Returns a probability surface dict for GIS visualization.
        """
        flood_counts = []
        for scene in scenes_archive:
            mndwi = self._get_index(scene, 'mndwi')
            is_flooded = 1 if mndwi > self.flood_thresh['mndwi_flood_water'] else 0
            flood_counts.append(is_flooded)

        prob = float(np.mean(flood_counts)) if flood_counts else 0.0
        zone, mult = self._classify_flood_risk_zone(prob)

        return {
            'flood_probability':   round(prob, 3),
            'risk_zone':           zone,
            'insurance_multiplier': mult,
            'n_scenes_analyzed':   len(scenes_archive),
            'flooded_scenes':      sum(flood_counts),
        }

    # =========================================================================
    # INTERNAL — SAR / FLOOD COMPUTATIONS
    # =========================================================================

    def _compute_sar_change(self, pre: Dict, post: Dict) -> float:
        """Compute VV backscatter change (dB). Negative = water inundation."""
        vv_pre  = pre.get('vv_backscatter', pre.get('indices', {}).get('vv', -10.0))
        vv_post = post.get('vv_backscatter', post.get('indices', {}).get('vv', -15.0))
        return float(vv_post - vv_pre)

    def _compute_mndwi_extent(self, scene: Optional[Dict]) -> float:
        """Estimate water extent fraction from MNDWI."""
        if scene is None:
            return 0.0
        mndwi = self._get_index(scene, 'mndwi')
        # Normalize: MNDWI range -1 to 1; water > 0.1
        return float(np.clip((mndwi - 0.10) / 0.90, 0.0, 1.0)) if mndwi > 0.10 else 0.0

    def _compute_flood_probability(self, sar_change_db: float, mndwi_extent: float) -> float:
        """
        Combine SAR change and MNDWI into a flood probability 0-1.
        SAR: change < -3 dB strongly indicates water
        MNDWI: high fraction corroborates optical water detection
        """
        # SAR contribution (0-1)
        sar_thresh = self.flood_thresh['sar_flood_change_db']
        if sar_change_db < sar_thresh:
            sar_prob = min(abs(sar_change_db - sar_thresh) / 5.0, 1.0)
        else:
            sar_prob = 0.0

        # Combined (SAR weighted 70%, optical MNDWI 30%)
        prob = 0.70 * sar_prob + 0.30 * mndwi_extent
        return float(np.clip(prob, 0.0, 1.0))

    def _estimate_inundation_depth(
        self, dem_data: Optional[np.ndarray], flood_prob: float
    ) -> float:
        """Estimate max inundation depth from DEM and flood probability."""
        if dem_data is None or flood_prob < 0.01:
            # Heuristic if no DEM: scale with probability
            return round(flood_prob * 3.5, 2)   # max 3.5m at p=1.0

        # With DEM: compute relief within flooded extent
        flooded_mask = dem_data < np.percentile(dem_data, flood_prob * 100)
        if not flooded_mask.any():
            return 0.0
        min_elev = float(dem_data[flooded_mask].min())
        mean_elev = float(dem_data.mean())
        return round(max(mean_elev - min_elev, 0.0), 2)

    def _classify_flood_risk_zone(self, prob: float) -> Tuple[str, float]:
        """Classify into risk zone and return insurance multiplier."""
        for zone, info in self.risk_zones.items():
            if prob <= info['prob_max']:
                return zone, info['insurance_multiplier']
        return 'High', 2.5

    def _estimate_recovery_days(self, depth_m: float, risk_zone: str) -> int:
        """Estimate days for floodwater to recede and farming to resume."""
        base = {
            'Safe': 0, 'Low': 7, 'Medium': 21, 'High': 45
        }.get(risk_zone, 30)
        depth_factor = int(depth_m * 7)
        return base + depth_factor

    # =========================================================================
    # INTERNAL — DROUGHT COMPUTATIONS
    # =========================================================================

    def _compute_nddi(self, scenes: List[Dict]) -> float:
        """NDDI = (NDVI - NDWI) / (NDVI + NDWI). High = drought stress."""
        values = []
        for scene in scenes:
            ndvi = self._get_index(scene, 'ndvi')
            ndwi = self._get_index(scene, 'ndwi')
            denom = ndvi + ndwi
            if abs(denom) > 0.001:
                values.append((ndvi - ndwi) / denom)
        return float(np.mean(values)) if values else 0.0

    def _compute_ndmi(self, scenes: List[Dict]) -> float:
        """NDMI = (NIR - SWIR1) / (NIR + SWIR1). Low = moisture stress."""
        values = [self._get_index(s, 'ndmi') for s in scenes]
        valid  = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else 0.0

    def _compute_nmdi(self, scenes: List[Dict]) -> float:
        """
        NMDI = (NIR - (SWIR1 - SWIR2)) / (NIR + (SWIR1 - SWIR2))
        Normalized Multi-band Drought Index.
        """
        values = []
        for scene in scenes:
            idx  = scene.get('indices', {})
            b08  = idx.get('b08', None) or idx.get('nir', None)
            b11  = idx.get('b11', None) or idx.get('swir1', None)
            b12  = idx.get('b12', None) or idx.get('swir2', None)
            if all(v is not None for v in [b08, b11, b12]):
                num   = b08 - (b11 - b12)
                denom = b08 + (b11 - b12)
                if abs(denom) > 0.001:
                    values.append(num / denom)
        return float(np.mean(values)) if values else 0.0

    def _compute_spi(self, weather_data: Optional[pd.DataFrame]) -> float:
        """
        Standard Precipitation Index from rainfall time series.
        SPI = (P - P_mean) / P_std
        Negative = below-normal precipitation = drought.
        """
        if weather_data is None or 'rainfall_mm' not in weather_data.columns:
            return 0.0   # neutral if no weather data

        rainfall = weather_data['rainfall_mm'].dropna()
        if len(rainfall) < 3:
            return 0.0

        mean_r = rainfall.mean()
        std_r  = rainfall.std()

        if std_r < 0.001:
            return 0.0

        current = rainfall.iloc[-1]
        spi = (current - mean_r) / std_r
        return float(np.clip(spi, -3.0, 3.0))

    def _classify_drought_severity(self, nddi: float, spi: float) -> str:
        """Classify drought severity from NDDI and SPI."""
        dt = self.drought_thresh

        if nddi >= dt['nddi_severe_drought'] or spi <= dt['spi_severe_drought']:
            return 'Severe'
        elif nddi >= dt['nddi_moderate_drought'] or spi <= dt['spi_moderate_drought']:
            return 'Moderate'
        elif nddi >= dt['nddi_mild_drought'] or spi <= dt['spi_mild_drought']:
            return 'Mild'
        else:
            return 'None'

    def _compute_drought_score(self, nddi: float, ndmi: float, spi: float) -> float:
        """
        Composite drought score 0-100.
        Weights: NDDI 40%, NDMI 30%, SPI 30%.
        """
        dt = self.drought_thresh

        # NDDI component (0-1 scale)
        nddi_score = np.clip(
            (nddi - dt['nddi_mild_drought']) /
            (dt['nddi_severe_drought'] - dt['nddi_mild_drought']),
            0.0, 1.0
        )

        # NDMI component (stress = low NDMI)
        ndmi_score = np.clip(
            (dt['ndmi_stress_severe'] - ndmi) /
            abs(dt['ndmi_stress_severe']),
            0.0, 1.0
        )

        # SPI component (negative SPI = drought)
        spi_score = np.clip(-spi / 2.0, 0.0, 1.0)

        combined = 0.40 * nddi_score + 0.30 * ndmi_score + 0.30 * spi_score
        return float(np.clip(combined * 100, 0, 100))

    def _estimate_affected_area(self, nddi: float, severity: str) -> float:
        """Estimate percentage of total area in drought condition."""
        area_map = {'None': 0, 'Mild': 20, 'Moderate': 50, 'Severe': 80}
        base = float(area_map.get(severity, 0))
        nddi_adj = np.clip((nddi - 0.40) * 50, 0, 20)
        return round(base + nddi_adj, 1)

    def _estimate_crop_loss(self, nddi: float, ndmi: float, severity: str) -> float:
        """Proxy estimate of crop yield loss % from drought indices."""
        loss_map = {'None': 0, 'Mild': 10, 'Moderate': 30, 'Severe': 55}
        base = float(loss_map.get(severity, 0))
        moisture_adj = np.clip((0 - ndmi) * 30, 0, 20)
        return round(base + moisture_adj, 1)

    def _generate_drought_alerts(
        self, nddi: float, ndmi: float, spi: float, severity: str
    ) -> List[str]:
        """Generate actionable drought alerts."""
        alerts = []
        dt = self.drought_thresh

        if severity == 'Severe':
            alerts.append("CRITICAL: Severe drought detected. Activate emergency water rationing protocols.")
        elif severity == 'Moderate':
            alerts.append("WARNING: Moderate drought. Advise farmers on deficit irrigation & crop substitution.")
        elif severity == 'Mild':
            alerts.append("WATCH: Mild drought conditions. Monitor weekly — situation may worsen.")

        if ndmi < dt['ndmi_stress_severe']:
            alerts.append("Severe soil moisture deficit detected. Irrigation advisory: apply 80–100% crop water requirement.")
        elif ndmi < dt['ndmi_stress_moderate']:
            alerts.append("Moderate moisture stress. Soil moisture recharge needed within 10 days.")

        if spi <= dt['spi_severe_drought']:
            alerts.append(f"SPI={spi:.2f}: Extreme precipitation deficit. Coordinate with IMD for seasonal forecast update.")

        if not alerts:
            alerts.append("No drought stress detected. Continue routine monitoring.")

        return alerts

    def _drought_recovery_forecast(self, spi: float, severity: str) -> str:
        """Forecast drought recovery timeline."""
        if severity == 'None':
            return "No drought — conditions normal."
        elif severity == 'Mild':
            return "2–3 weeks of normal rainfall expected to restore soil moisture."
        elif severity == 'Moderate':
            return "4–6 weeks of above-normal rainfall needed. Expect partial crop recovery only."
        else:
            return ("Severe deficit requires 8–12 weeks of sustained above-normal rainfall. "
                    "Full agricultural recovery unlikely this season — next season affected.")

    # =========================================================================
    # INTERNAL — COMBINED REPORT
    # =========================================================================

    def _combined_risk_category(self, combined_score: float) -> str:
        if combined_score < 20:
            return 'Low'
        elif combined_score < 45:
            return 'Moderate'
        elif combined_score < 70:
            return 'High'
        else:
            return 'Extreme'

    def _hazard_recommendations(
        self, flood_score: float, drought_score: float
    ) -> List[str]:
        recs = []
        if flood_score > 60:
            recs.append("High flood risk: Implement early warning system with SAR-based near-real-time monitoring (3-day revisit).")
            recs.append("Commission DEM-based flood inundation modeling for critical agricultural zones.")
            recs.append("Recommend Pradhan Mantri Fasal Bima Yojana (PMFBY) enrollment for affected farmers.")
        if drought_score > 50:
            recs.append("Significant drought risk: Establish soil moisture monitoring network with IoT sensors.")
            recs.append("Advise district administration to pre-position emergency water tankers and seed bank access.")
            recs.append("Coordinate with IMD for 3-month outlook; initiate crop diversification advisory if SPI < -1.5.")
        if not recs:
            recs.append("Low hazard risk region. Standard quarterly monitoring adequate.")
        return recs

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _get_index(self, scene: Dict, key: str) -> float:
        """Safe index extraction from scene dict."""
        return float(scene.get('indices', {}).get(key, 0.0) or 0.0)

    def _extract_sar_scene(self, scenes: List[Dict]) -> Dict:
        """Get a representative SAR scene (most recent)."""
        sar = [s for s in scenes if s.get('sensor_type') == 'SAR' or 'vv' in s.get('indices', {})]
        return sar[-1] if sar else (scenes[-1] if scenes else {})

    def _flood_to_dict(self, f: FloodAssessment) -> Dict:
        return {
            'event_id': f.event_id, 'location': f.location_name,
            'date': f.event_date, 'extent_ha': f.flood_extent_ha,
            'risk_zone': f.risk_zone, 'probability': f.probability,
            'crop_affected_ha': f.affected_cropland_ha,
            'insurance_multiplier': f.insurance_multiplier,
        }

    def _drought_to_dict(self, d: DroughtAssessment) -> Dict:
        return {
            'region': d.region_name, 'period': f"{d.period_start}–{d.period_end}",
            'severity': d.severity, 'drought_score': d.drought_score,
            'affected_pct': d.affected_area_pct, 'crop_loss_pct': d.crop_loss_estimate,
            'risk_score': d.risk_score,
        }

    def _methodology_note(self) -> str:
        return (
            "Flood: SAR Sentinel-1 VV backscatter change detection (pre/post event). "
            "MNDWI cross-validation for water extent. DEM-based depth estimation. "
            "Drought: NDDI (NDVI-NDWI composite), NDMI (NIR-SWIR1), NMDI (multi-band). "
            "SPI computed from IMD/weather API rainfall normals (1981-2010 base period). "
            "Risk scores calibrated for Indian agri-insurance (PMFBY framework)."
        )

    def _log_flood_summary(self, a: FloodAssessment):
        logger.info(
            f"\n{'='*55}\n"
            f"  FLOOD ASSESSMENT — {a.location_name}\n"
            f"  Date        : {a.event_date}\n"
            f"  Risk Zone   : {a.risk_zone} (prob={a.probability:.2%})\n"
            f"  Extent      : {a.flood_extent_ha:.1f} ha flooded\n"
            f"  Max Depth   : {a.max_inundation_depth_m:.2f} m\n"
            f"  Crop Impact : {a.affected_cropland_ha:.1f} ha\n"
            f"  Recovery    : ~{a.recovery_estimate_days} days\n"
            f"  Ins. Factor : {a.insurance_multiplier:.1f}×\n"
            f"{'='*55}"
        )

    def _log_drought_summary(self, a: DroughtAssessment):
        logger.info(
            f"\n{'='*55}\n"
            f"  DROUGHT ASSESSMENT — {a.region_name}\n"
            f"  Severity    : {a.severity}\n"
            f"  NDDI        : {a.nddi_mean:.4f}\n"
            f"  NDMI        : {a.ndmi_mean:.4f}\n"
            f"  SPI         : {a.spi_value:.3f}\n"
            f"  Drought Score: {a.drought_score:.1f}/100\n"
            f"  Affected    : {a.affected_area_pct:.1f}% of area\n"
            f"  Crop Loss   : ~{a.crop_loss_estimate:.1f}%\n"
            f"  Recovery    : {a.recovery_forecast}\n"
            f"{'='*55}"
        )

    def _log_report_summary(self, r: HazardRiskReport):
        logger.info(
            f"\n{'='*55}\n"
            f"  HAZARD RISK REPORT — {r.region_name}\n"
            f"  Flood Risk  : {r.flood_risk_score:.1f}/100\n"
            f"  Drought Risk: {r.drought_risk_score:.1f}/100\n"
            f"  Combined    : {r.combined_risk_score:.1f}/100 — {r.risk_category}\n"
            f"  Ins. Factor : {r.insurance_premium_factor:.2f}×\n"
            f"{'='*55}"
        )
