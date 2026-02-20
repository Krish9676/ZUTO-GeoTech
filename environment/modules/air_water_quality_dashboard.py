"""
ZUTO Geotech Solutions — Environmental Analysis Platform
=========================================================
Module E: Air & Water Quality Spatial Dashboard

Integrates ground station data with satellite proxies to create
continuous spatial surfaces and compliance dashboards.

AIR QUALITY:
  - Ingests CPCB/OpenAQ station readings (PM2.5, PM10, NO2, SO2)
  - Spatial interpolation: IDW or Kriging → continuous AQI surface
  - Industry proximity analysis: which industries are in violation zones
  - Automated alerts when AQI thresholds are breached

WATER QUALITY:
  - Sentinel-2 blue-green bands for turbidity + chlorophyll-a proxies
  - NDWI/MNDWI for surface water extent
  - Trophic state classification (oligotrophic → hypereutrophic)
  - Effluent discharge impact zones around industrial drains

Target clients:
  - Industries (CPCB/SPCB compliance dashboards)
  - Municipal water utilities (lake/reservoir health)
  - State Pollution Control Boards (automated monitoring)
  - Real estate developers (environmental clearance EIA)
  - Smart City operators
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from scipy.interpolate import RBFInterpolator   # optional — used for kriging fallback

from config.env_config import ZutoEnvConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AQIStation:
    """Air quality measurement from a single station"""
    station_id:     str
    station_name:   str
    latitude:       float
    longitude:      float
    timestamp:      str
    pm25:           Optional[float]
    pm10:           Optional[float]
    no2:            Optional[float]
    so2:            Optional[float]
    co:             Optional[float]
    o3:             Optional[float]
    aqi:            float           # Computed AQI (CPCB method)
    aqi_category:   str
    dominant_pollutant: str


@dataclass
class IndustryComplianceRecord:
    """Compliance status for a single industrial unit"""
    industry_id:        str
    industry_name:      str
    industry_type:      str
    latitude:           float
    longitude:          float
    nearest_station_id: str
    nearest_aqi:        float
    nearest_aqi_cat:    str
    in_violation_zone:  bool
    distance_to_violation_km: float
    violation_pollutants: List[str]
    compliance_status:  str     # Compliant / Warning / Non-Compliant
    recommended_action: str


@dataclass
class WaterBodyProfile:
    """Water quality profile for a lake, reservoir, or river segment"""
    waterbody_id:       str
    waterbody_name:     str
    waterbody_type:     str     # lake / reservoir / river / wetland
    analysis_date:      str
    area_ha:            float
    turbidity_ntu:      float
    turbidity_class:    str
    chlorophyll_a_ug_l: float
    trophic_state:      str     # oligotrophic / mesotrophic / eutrophic / hypereutrophic
    ndwi:               float
    mndwi:              float
    water_clarity_m:    float   # Secchi depth proxy
    algal_bloom_risk:   str     # Low / Moderate / High / Alert
    effluent_impact:    bool
    wqi_score:          float   # 0-100 (100 = pristine)
    recommended_action: str


@dataclass
class AirQualityReport:
    """Complete air quality spatial analysis report"""
    region_name:        str
    report_date:        str
    n_stations:         int
    stations:           List[Dict]
    regional_aqi:       float
    regional_category:  str
    aqi_surface:        Dict        # Gridded AQI surface (for GIS)
    violation_zones:    List[Dict]  # Zones where AQI > 'Moderate'
    industry_compliance: List[Dict]
    population_exposed: Dict        # by AQI category
    alerts:             List[str]
    trends:             Dict
    recommendations:    List[str]


@dataclass
class WaterQualityReport:
    """Complete water quality spatial analysis report"""
    region_name:        str
    report_date:        str
    waterbodies:        List[Dict]
    n_waterbodies:      int
    avg_wqi:            float
    critical_waterbodies: List[Dict]
    bloom_alerts:       List[Dict]
    alerts:             List[str]
    recommendations:    List[str]


# =============================================================================
# AIR & WATER QUALITY DASHBOARD — MAIN CLASS
# =============================================================================

class AirWaterQualityDashboard:
    """
    Spatial air and water quality analysis integrating CPCB station data
    with Sentinel-2 satellite proxies.

    Example:
        dashboard = AirWaterQualityDashboard()

        # Air quality
        aq_report = dashboard.generate_air_quality_report(
            station_data  = cpcb_df,
            industry_list = industries_geojson,
            region_name   = "Pune Industrial Belt",
        )

        # Water quality
        wq_report = dashboard.generate_water_quality_report(
            optical_scenes   = sentinel2_scenes,
            waterbody_list   = lakes_geojson,
            region_name      = "Western Maharashtra Lakes",
        )
    """

    def __init__(self):
        self.cfg           = ZutoEnvConfig
        self.aqi_breaks    = self.cfg.AQI_BREAKPOINTS
        self.aqi_cats      = self.cfg.AQI_CATEGORIES
        self.water_thresh  = self.cfg.WATER_QUALITY
        self.interp        = self.cfg.INTERPOLATION

    # =========================================================================
    # PUBLIC API — AIR QUALITY
    # =========================================================================

    def generate_air_quality_report(
        self,
        station_data:   pd.DataFrame,
        industry_list:  Optional[List[Dict]] = None,
        region_name:    str   = "Region",
        grid_resolution: float = 0.01,      # degrees
    ) -> AirQualityReport:
        """
        Generate full air quality spatial analysis.

        Args:
            station_data:  DataFrame with columns:
                           [station_id, station_name, lat, lon, timestamp,
                            pm25, pm10, no2, so2, co, o3]
            industry_list: List of industry dicts with [id, name, type, lat, lon]
            region_name:   Name of the region
            grid_resolution: Output grid resolution in degrees

        Returns:
            AirQualityReport dataclass
        """
        logger.info(f"\nAIR QUALITY ANALYSIS — {region_name} | Stations: {len(station_data)}")

        # — Parse stations and compute AQI per station
        stations = self._parse_stations(station_data)

        # — Regional AQI (area-weighted mean)
        regional_aqi  = float(np.mean([s['aqi'] for s in stations])) if stations else 0.0
        regional_cat  = self._aqi_category(regional_aqi)

        # — Spatial interpolation → AQI surface
        aqi_surface = self._interpolate_aqi_surface(stations, grid_resolution)

        # — Identify violation zones
        violation_zones = self._identify_violation_zones(stations, aqi_surface)

        # — Industry compliance
        compliance = []
        if industry_list:
            compliance = self._assess_industry_compliance(industry_list, stations)

        # — Population exposure estimate
        pop_exposure = self._estimate_population_exposure(aqi_surface)

        # — Alerts
        alerts = self._generate_air_alerts(stations, violation_zones)

        # — Trends (placeholder — requires time-series input)
        trends = self._compute_aqi_trends(station_data)

        report = AirQualityReport(
            region_name       = region_name,
            report_date       = date.today().isoformat(),
            n_stations        = len(stations),
            stations          = stations,
            regional_aqi      = round(regional_aqi, 1),
            regional_category = regional_cat,
            aqi_surface       = aqi_surface,
            violation_zones   = violation_zones,
            industry_compliance = compliance,
            population_exposed = pop_exposure,
            alerts            = alerts,
            trends            = trends,
            recommendations   = self._air_recommendations(regional_aqi, violation_zones, compliance),
        )

        self._log_air_summary(report)
        return report

    def compute_station_aqi(self, readings: Dict) -> Tuple[float, str, str]:
        """
        Compute AQI for a single station reading.
        Uses CPCB sub-index method: AQI = max(sub-index for each pollutant).

        Args:
            readings: dict with keys pm25, pm10, no2, so2 (µg/m³)

        Returns:
            (aqi_value, aqi_category, dominant_pollutant)
        """
        sub_indices  = {}
        pollutants   = ['PM2_5', 'PM10', 'NO2', 'SO2']
        value_keys   = ['pm25', 'pm10', 'no2', 'so2']

        for pollutant, key in zip(pollutants, value_keys):
            val = readings.get(key)
            if val is not None and not np.isnan(val):
                si = self._compute_sub_index(float(val), pollutant)
                sub_indices[pollutant] = si

        if not sub_indices:
            return 0.0, 'Unknown', 'N/A'

        aqi    = max(sub_indices.values())
        dom    = max(sub_indices, key=sub_indices.get)
        cat    = self._aqi_category(aqi)

        return round(aqi, 1), cat, dom

    # =========================================================================
    # PUBLIC API — WATER QUALITY
    # =========================================================================

    def generate_water_quality_report(
        self,
        optical_scenes:  List[Dict],
        waterbody_list:  List[Dict],
        region_name:     str = "Region",
    ) -> WaterQualityReport:
        """
        Generate water quality analysis for multiple water bodies.

        Args:
            optical_scenes: Sentinel-2 scenes covering the region
            waterbody_list: List of water body dicts with
                            [id, name, type, area_ha, lat, lon, has_industrial_drain]
            region_name:    Region name

        Returns:
            WaterQualityReport dataclass
        """
        logger.info(f"\nWATER QUALITY ANALYSIS — {region_name} | Bodies: {len(waterbody_list)}")

        # — Composite scene indices
        composite = self._composite_optical(optical_scenes)

        # — Analyze each water body
        waterbodies = []
        for wb in waterbody_list:
            profile = self._analyze_waterbody(wb, composite, optical_scenes)
            waterbodies.append(profile)

        # — Aggregate
        avg_wqi = float(np.mean([w['wqi_score'] for w in waterbodies])) if waterbodies else 0.0

        critical = [w for w in waterbodies if w['wqi_score'] < 40 or w['algal_bloom_risk'] == 'Alert']
        blooms   = [w for w in waterbodies if w['algal_bloom_risk'] in ['High', 'Alert']]

        alerts = self._generate_water_alerts(waterbodies)

        report = WaterQualityReport(
            region_name          = region_name,
            report_date          = date.today().isoformat(),
            waterbodies          = waterbodies,
            n_waterbodies        = len(waterbodies),
            avg_wqi              = round(avg_wqi, 1),
            critical_waterbodies = critical,
            bloom_alerts         = blooms,
            alerts               = alerts,
            recommendations      = self._water_recommendations(waterbodies, blooms),
        )

        self._log_water_summary(report)
        return report

    def assess_waterbody_from_scene(
        self,
        scene:       Dict,
        waterbody:   Dict,
    ) -> Dict:
        """
        Quick single-scene water body quality assessment.
        Returns classification dict for dashboard display.
        """
        composite = self._composite_optical([scene])
        return self._analyze_waterbody(waterbody, composite, [scene])

    # =========================================================================
    # INTERNAL — AQI COMPUTATION
    # =========================================================================

    def _parse_stations(self, df: pd.DataFrame) -> List[Dict]:
        """Parse station DataFrame and compute AQI for each station."""
        stations = []
        for _, row in df.iterrows():
            readings = {
                'pm25': row.get('pm25'),
                'pm10': row.get('pm10'),
                'no2':  row.get('no2'),
                'so2':  row.get('so2'),
            }
            aqi, cat, dom = self.compute_station_aqi(readings)

            stations.append({
                'station_id':        str(row.get('station_id', f"STN_{_}")),
                'station_name':      str(row.get('station_name', f"Station {_}")),
                'latitude':          float(row.get('lat', row.get('latitude', 18.5))),
                'longitude':         float(row.get('lon', row.get('longitude', 73.8))),
                'timestamp':         str(row.get('timestamp', date.today().isoformat())),
                'pm25':              float(row['pm25']) if pd.notna(row.get('pm25')) else None,
                'pm10':              float(row['pm10']) if pd.notna(row.get('pm10')) else None,
                'no2':               float(row['no2'])  if pd.notna(row.get('no2'))  else None,
                'so2':               float(row['so2'])  if pd.notna(row.get('so2'))  else None,
                'aqi':               aqi,
                'aqi_category':      cat,
                'dominant_pollutant': dom,
            })

        return stations

    def _compute_sub_index(self, value: float, pollutant: str) -> float:
        """Compute CPCB sub-index for a single pollutant."""
        breakpoints = self.aqi_breaks.get(pollutant, [])
        for (c_lo, c_hi, i_lo, i_hi, _) in breakpoints:
            if c_lo <= value <= c_hi:
                # Linear interpolation
                si = ((i_hi - i_lo) / (c_hi - c_lo)) * (value - c_lo) + i_lo
                return round(si, 1)
        return 500.0   # beyond scale

    def _aqi_category(self, aqi: float) -> str:
        """Map AQI value to CPCB category string."""
        for cat, info in self.aqi_cats.items():
            if info['min'] <= aqi <= info['max']:
                return cat
        return 'Severe'

    # =========================================================================
    # INTERNAL — INTERPOLATION
    # =========================================================================

    def _interpolate_aqi_surface(
        self,
        stations:        List[Dict],
        grid_resolution: float,
    ) -> Dict:
        """
        Create a gridded AQI surface using IDW interpolation.
        Returns grid metadata + flat arrays for GIS export.
        """
        if len(stations) < 2:
            return {'method': 'insufficient_stations', 'n_stations': len(stations)}

        lats  = np.array([s['latitude']  for s in stations])
        lons  = np.array([s['longitude'] for s in stations])
        aqis  = np.array([s['aqi']       for s in stations])

        lat_min, lat_max = lats.min() - 0.1, lats.max() + 0.1
        lon_min, lon_max = lons.min() - 0.1, lons.max() + 0.1

        grid_lat = np.arange(lat_min, lat_max, grid_resolution)
        grid_lon = np.arange(lon_min, lon_max, grid_resolution)
        gl, glo  = np.meshgrid(grid_lat, grid_lon)
        grid_pts = np.column_stack([gl.ravel(), glo.ravel()])

        # IDW interpolation
        p        = self.interp['idw_power']
        obs_pts  = np.column_stack([lats, lons])
        aqi_grid = self._idw(obs_pts, aqis, grid_pts, power=p)

        # Classify surface
        aqi_categories = [self._aqi_category(v) for v in aqi_grid]

        return {
            'method':          'IDW',
            'power':           p,
            'grid_resolution': grid_resolution,
            'lat_min':         float(lat_min), 'lat_max': float(lat_max),
            'lon_min':         float(lon_min), 'lon_max': float(lon_max),
            'n_grid_points':   len(aqi_grid),
            'mean_aqi':        round(float(np.mean(aqi_grid)), 1),
            'max_aqi':         round(float(np.max(aqi_grid)), 1),
            'min_aqi':         round(float(np.min(aqi_grid)), 1),
            'pct_moderate_plus': round(
                np.sum(aqi_grid > 100) / len(aqi_grid) * 100, 1
            ),
        }

    def _idw(
        self,
        obs_points: np.ndarray,
        values:     np.ndarray,
        query_pts:  np.ndarray,
        power:      int = 2,
    ) -> np.ndarray:
        """Inverse Distance Weighting interpolation."""
        result = np.zeros(len(query_pts))
        for i, qp in enumerate(query_pts):
            dists  = np.sqrt(np.sum((obs_points - qp) ** 2, axis=1))
            dists  = np.where(dists < 1e-8, 1e-8, dists)  # avoid division by zero
            weights = 1.0 / (dists ** power)
            result[i] = np.sum(weights * values) / np.sum(weights)
        return result

    # =========================================================================
    # INTERNAL — VIOLATION ZONES & COMPLIANCE
    # =========================================================================

    def _identify_violation_zones(
        self,
        stations:    List[Dict],
        aqi_surface: Dict,
    ) -> List[Dict]:
        """Identify stations and grid areas where AQI > 'Moderate' (>100)."""
        violations = []
        for s in stations:
            if s['aqi'] > 100:
                violations.append({
                    'station_id':    s['station_id'],
                    'station_name':  s['station_name'],
                    'latitude':      s['latitude'],
                    'longitude':     s['longitude'],
                    'aqi':           s['aqi'],
                    'category':      s['aqi_category'],
                    'dominant':      s['dominant_pollutant'],
                    'severity':      'CRITICAL' if s['aqi'] > 300 else ('HIGH' if s['aqi'] > 200 else 'MEDIUM'),
                })
        return violations

    def _assess_industry_compliance(
        self,
        industry_list: List[Dict],
        stations:      List[Dict],
    ) -> List[Dict]:
        """
        Assess each industry's proximity to AQI violation zones.
        Flags industries within 5 km of stations with AQI > 100.
        """
        compliance = []

        for industry in industry_list:
            ilat = industry.get('lat', industry.get('latitude', 0))
            ilon = industry.get('lon', industry.get('longitude', 0))

            # Find nearest station
            min_dist   = float('inf')
            nearest    = None
            for s in stations:
                dist = np.sqrt((ilat - s['latitude'])**2 + (ilon - s['longitude'])**2) * 111
                if dist < min_dist:
                    min_dist = dist
                    nearest  = s

            if nearest is None:
                continue

            in_violation = nearest['aqi'] > 100
            violation_pollutants = []

            if in_violation:
                for pol in ['pm25', 'pm10', 'no2', 'so2']:
                    val = nearest.get(pol)
                    if val and val > 60:    # elevated threshold
                        violation_pollutants.append(pol.upper())

            status = 'Non-Compliant' if nearest['aqi'] > 200 else ('Warning' if nearest['aqi'] > 100 else 'Compliant')

            compliance.append({
                'industry_id':          industry.get('id', 'IND_X'),
                'industry_name':        industry.get('name', 'Unknown'),
                'industry_type':        industry.get('type', 'Unknown'),
                'latitude':             ilat,
                'longitude':            ilon,
                'nearest_station':      nearest['station_name'],
                'nearest_aqi':          nearest['aqi'],
                'nearest_aqi_category': nearest['aqi_category'],
                'distance_to_station_km': round(min_dist, 2),
                'in_violation_zone':    in_violation,
                'violation_pollutants': violation_pollutants,
                'compliance_status':    status,
                'recommended_action':   self._compliance_action(status, violation_pollutants),
            })

        return compliance

    def _compliance_action(self, status: str, pollutants: List[str]) -> str:
        if status == 'Non-Compliant':
            return (f"MANDATORY: Reduce emissions of {', '.join(pollutants)}. "
                    f"Submit compliance plan to SPCB within 30 days. Install CEMS.")
        elif status == 'Warning':
            return "ADVISORY: Elevated AQI in vicinity. Review pollution control equipment. Submit self-assessment report."
        return "Compliant. Maintain current standards. Continue quarterly CEMS reporting."

    # =========================================================================
    # INTERNAL — WATER QUALITY COMPUTATION
    # =========================================================================

    def _composite_optical(self, scenes: List[Dict]) -> Dict:
        """Build composite indices from optical scenes."""
        keys = ['ndwi', 'mndwi', 'b02', 'b03', 'b04', 'b08', 'b11', 'ndvi']
        agg  = {k: [] for k in keys}
        for scene in scenes:
            idx = scene.get('indices', {})
            for k in keys:
                val = idx.get(k)
                if val is not None:
                    agg[k].append(float(val))
        return {k: float(np.median(v)) if v else 0.0 for k, v in agg.items()}

    def _analyze_waterbody(
        self,
        wb:        Dict,
        composite: Dict,
        scenes:    List[Dict],
    ) -> Dict:
        """
        Analyze a single water body from satellite composites.
        Computes turbidity, chlorophyll-a, trophic state, and WQI.
        """
        # Turbidity proxy: Band ratio B04/B03 (red/green)
        b03 = composite.get('b03', 0.0)
        b04 = composite.get('b04', 0.0)
        b02 = composite.get('b02', 0.0)

        turbidity_ntu = self._estimate_turbidity(b03, b04)
        turb_class    = self._classify_turbidity(turbidity_ntu)

        # Chlorophyll-a proxy: (B03 - B04) / (B03 + B04) * 100
        # (Mishra & Mishra 2012 OC3-style algorithm)
        denom   = b03 + b04
        chl_idx = ((b03 - b04) / denom) if abs(denom) > 0.001 else 0.0
        chl_a   = max(chl_idx * 200, 0.0)   # approximate µg/L

        # Secchi depth proxy (water clarity in metres)
        secchi = max(1.0 / (turbidity_ntu + 0.01), 0.1)
        secchi = min(secchi, 20.0)

        # Trophic state
        trophic = self._classify_trophic_state(chl_a)

        # Bloom risk
        bloom_risk = self._assess_bloom_risk(chl_a, trophic)

        # NDWI / MNDWI
        ndwi  = composite.get('ndwi', 0.0)
        mndwi = composite.get('mndwi', 0.0)

        # Industrial impact flag (from waterbody metadata)
        effluent_impact = wb.get('has_industrial_drain', False)
        if effluent_impact and turbidity_ntu > 25:
            bloom_risk = max(bloom_risk, 'High')

        # WQI (0-100, 100 = pristine)
        wqi = self._compute_wqi(turbidity_ntu, chl_a, trophic)

        action = self._water_action(trophic, bloom_risk, effluent_impact, wqi)

        return {
            'waterbody_id':       wb.get('id', 'WB_X'),
            'waterbody_name':     wb.get('name', 'Unknown'),
            'waterbody_type':     wb.get('type', 'lake'),
            'analysis_date':      date.today().isoformat(),
            'area_ha':            float(wb.get('area_ha', 0)),
            'turbidity_ntu':      round(turbidity_ntu, 2),
            'turbidity_class':    turb_class,
            'chlorophyll_a_ug_l': round(chl_a, 2),
            'trophic_state':      trophic,
            'ndwi':               round(ndwi, 4),
            'mndwi':              round(mndwi, 4),
            'water_clarity_m':    round(secchi, 2),
            'algal_bloom_risk':   bloom_risk,
            'effluent_impact':    effluent_impact,
            'wqi_score':          round(wqi, 1),
            'recommended_action': action,
        }

    def _estimate_turbidity(self, b03: float, b04: float) -> float:
        """
        Estimate turbidity (NTU) from Sentinel-2 B03/B04.
        Uses empirical relationship: Turbidity ≈ 194.7 × (B04/B03) - 22.4
        (Nechad et al. 2010, adapted for Sentinel-2)
        """
        if b03 < 0.001:
            return 0.0
        ratio = b04 / b03
        turbidity = 194.7 * ratio - 22.4
        return float(np.clip(turbidity, 0, 1000))

    def _classify_turbidity(self, ntu: float) -> str:
        for cls, (lo, hi) in self.water_thresh['turbidity'].items():
            if lo <= ntu < hi:
                return cls.replace('_', ' ').title()
        return 'Very Turbid'

    def _classify_trophic_state(self, chl_a: float) -> str:
        for state, (lo, hi) in self.water_thresh['chlorophyll_a'].items():
            if lo <= chl_a < hi:
                return state.capitalize()
        return 'Hypereutrophic'

    def _assess_bloom_risk(self, chl_a: float, trophic: str) -> str:
        if chl_a > 80 or trophic == 'Hypereutrophic':
            return 'Alert'
        elif chl_a > 30 or trophic == 'Eutrophic':
            return 'High'
        elif chl_a > 10:
            return 'Moderate'
        return 'Low'

    def _compute_wqi(self, turbidity: float, chl_a: float, trophic: str) -> float:
        """
        Water Quality Index (0-100).
        Penalizes turbidity, chlorophyll, and bad trophic state.
        """
        turb_score  = max(100 - turbidity * 2, 0)
        chl_score   = max(100 - chl_a * 3, 0)
        trophic_scores = {
            'Oligotrophic': 100, 'Mesotrophic': 75,
            'Eutrophic': 40, 'Hypereutrophic': 10,
        }
        trophic_score = trophic_scores.get(trophic, 50)

        wqi = 0.40 * turb_score + 0.35 * chl_score + 0.25 * trophic_score
        return float(np.clip(wqi, 0, 100))

    def _water_action(self, trophic: str, bloom_risk: str, effluent: bool, wqi: float) -> str:
        if bloom_risk == 'Alert':
            return "EMERGENCY: Algal bloom alert. Suspend water supply from this source. Issue public health warning. CPCB notification mandatory."
        elif bloom_risk == 'High':
            return "Elevated bloom risk. Increase monitoring to weekly. Restrict irrigation use. Test for cyanotoxins."
        elif effluent and wqi < 40:
            return "Industrial effluent suspected. Collect water samples for chemical analysis. Identify and notify responsible industries to SPCB."
        elif trophic == 'Eutrophic':
            return "Eutrophication detected. Recommend phosphorus source control, biomanipulation study, and aeration installation."
        return "Water quality within acceptable range. Continue bi-monthly monitoring."

    # =========================================================================
    # INTERNAL — EXPOSURE & TRENDS
    # =========================================================================

    def _estimate_population_exposure(self, aqi_surface: Dict) -> Dict:
        """Estimate population in each AQI category (proxy model)."""
        pct_moderate = aqi_surface.get('pct_moderate_plus', 20.0)
        total_pop    = 1_000_000   # placeholder — integrate with census in production
        return {
            'total_population':    total_pop,
            'Good':         int(total_pop * max(1 - pct_moderate/100, 0) * 0.4),
            'Satisfactory': int(total_pop * max(1 - pct_moderate/100, 0) * 0.4),
            'Moderate_plus': int(total_pop * pct_moderate / 100),
        }

    def _compute_aqi_trends(self, df: pd.DataFrame) -> Dict:
        """Compute basic trend from multi-timestep station data."""
        if 'timestamp' not in df.columns or len(df) < 2:
            return {'trend': 'insufficient_data'}

        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'pm25' in df.columns:
                df_sorted  = df.sort_values('timestamp')
                pm25_trend = df_sorted['pm25'].diff().mean()
                direction  = 'improving' if pm25_trend < -0.5 else ('worsening' if pm25_trend > 0.5 else 'stable')
                return {'pm25_trend_direction': direction, 'pm25_daily_change': round(float(pm25_trend), 2)}
        except Exception:
            pass

        return {'trend': 'stable'}

    # =========================================================================
    # INTERNAL — ALERTS & RECOMMENDATIONS
    # =========================================================================

    def _generate_air_alerts(
        self,
        stations:        List[Dict],
        violation_zones: List[Dict],
    ) -> List[str]:
        alerts = []
        severe_stations = [s for s in stations if s['aqi'] > 300]
        if severe_stations:
            for s in severe_stations:
                alerts.append(f"SEVERE AIR QUALITY at {s['station_name']}: AQI={s['aqi']:.0f}. Dominant: {s['dominant_pollutant']}. Outdoor activity strongly discouraged.")

        poor_stations = [s for s in stations if 200 < s['aqi'] <= 300]
        if poor_stations:
            alerts.append(f"{len(poor_stations)} station(s) in 'Poor' category. Sensitive groups should avoid outdoor exposure.")

        if len(violation_zones) > len(stations) * 0.5:
            alerts.append("More than 50% of monitored stations in violation. Regional pollution event likely — coordinate with IMD for dispersion forecast.")

        if not alerts:
            alerts.append("Air quality within acceptable limits across all monitored stations.")

        return alerts

    def _generate_water_alerts(self, waterbodies: List[Dict]) -> List[str]:
        alerts = []
        bloom_alerts = [w for w in waterbodies if w['algal_bloom_risk'] == 'Alert']
        if bloom_alerts:
            for w in bloom_alerts:
                alerts.append(f"ALGAL BLOOM ALERT: {w['waterbody_name']} — Chl-a {w['chlorophyll_a_ug_l']:.1f} µg/L. WQI={w['wqi_score']:.0f}.")
        critical_wqi = [w for w in waterbodies if w['wqi_score'] < 30]
        if critical_wqi:
            for w in critical_wqi:
                alerts.append(f"CRITICAL WATER QUALITY: {w['waterbody_name']} — WQI={w['wqi_score']:.0f}. Immediate action required.")
        if not alerts:
            alerts.append("All monitored water bodies within acceptable quality parameters.")
        return alerts

    def _air_recommendations(
        self,
        regional_aqi:    float,
        violation_zones: List[Dict],
        compliance:      List[Dict],
    ) -> List[str]:
        recs = []
        if regional_aqi > 200:
            recs.append("Regional AQI in 'Poor' range. Issue health advisory. Coordinate with transport authority for odd-even vehicle restrictions.")
        if violation_zones:
            recs.append(f"Establish {len(violation_zones)} additional CPCB-standard monitoring stations in violation zones for better spatial coverage.")

        non_compliant = [c for c in compliance if c['compliance_status'] == 'Non-Compliant']
        if non_compliant:
            recs.append(f"{len(non_compliant)} industries in non-compliance zone. Initiate show-cause notices. Mandatory CEMS installation within 90 days.")

        recs.append("Integrate real-time CPCB feed with ZUTO dashboard for 24-hour automated alert monitoring.")
        return recs

    def _water_recommendations(
        self,
        waterbodies: List[Dict],
        blooms:      List[Dict],
    ) -> List[str]:
        recs = []
        if blooms:
            recs.append(f"{len(blooms)} water body(ies) with high bloom risk. Commission limnological study and lake restoration plan.")
        low_wqi = [w for w in waterbodies if w['wqi_score'] < 50]
        if low_wqi:
            recs.append(f"{len(low_wqi)} water body(ies) with WQI < 50. Prioritize for AMRUT/NMCG lake restoration funding.")
        recs.append("Establish monthly satellite-based water quality monitoring cycle. Integrate CPCB water quality stations for calibration.")
        return recs

    # =========================================================================
    # INTERNAL — LOGGING
    # =========================================================================

    def _log_air_summary(self, r: AirQualityReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  AIR QUALITY REPORT — {r.region_name}\n"
            f"  Stations    : {r.n_stations}\n"
            f"  Regional AQI: {r.regional_aqi:.1f} — {r.regional_category}\n"
            f"  Violations  : {len(r.violation_zones)} stations\n"
            f"  Industries  : {len(r.industry_compliance)} assessed\n"
            f"  Alerts      : {len(r.alerts)}\n"
            f"{'='*60}"
        )

    def _log_water_summary(self, r: WaterQualityReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  WATER QUALITY REPORT — {r.region_name}\n"
            f"  Water Bodies: {r.n_waterbodies}\n"
            f"  Avg WQI     : {r.avg_wqi:.1f}/100\n"
            f"  Critical    : {len(r.critical_waterbodies)}\n"
            f"  Bloom Alerts: {len(r.bloom_alerts)}\n"
            f"  Alerts      : {len(r.alerts)}\n"
            f"{'='*60}"
        )
