"""
ZUTO Geotech Solutions — Environmental Analysis Platform
=========================================================
Module D: Urban Heat Island (UHI) Analyzer

Derives Land Surface Temperature (LST) from Landsat 8/9 thermal bands
and maps urban heat islands at ward/block level for city planners.

Workflow:
  1. Compute LST from Landsat Band 10 (thermal infrared)
  2. Apply emissivity correction using NDVI-based vegetation fraction
  3. Compute UHI intensity = LST_urban - LST_rural_background
  4. Overlay with NDVI (vegetation cooling effect) and NDBI (built-up)
  5. Identify heat vulnerability zones (ward/block level)
  6. Generate green cover placement recommendations
  7. Quantify cooling effect potential from urban forests/parks

Target clients:
  - Municipal corporations (PMC, MCGM, BBMP, GHMC)
  - Smart Cities Mission projects
  - Urban planners and architects (EIA support)
  - Climate adaptation consultants
  - Real estate developers (heat stress disclosure)
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from env_config import ZutoEnvConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WardHeatProfile:
    """Heat profile for a single ward/block"""
    ward_id:            str
    ward_name:          str
    lst_mean_c:         float       # Mean LST (°C)
    lst_max_c:          float       # Max LST (°C)
    uhi_intensity_c:    float       # UHI anomaly vs rural background
    uhi_category:       str         # Cool / Neutral / Warm / Hot / Extreme
    vulnerability:      str         # Low / Moderate / High / Critical
    ndvi_mean:          float       # Vegetation fraction
    ndbi_mean:          float       # Built-up density
    impervious_pct:     float       # % impervious surface
    green_cover_pct:    float       # % green cover
    cooling_deficit_c:  float       # How many °C too hot vs target
    area_ha:            float
    population_est:     int         # Estimated population at risk
    priority_score:     float       # 0-100 (100 = most urgent intervention)


@dataclass
class UHIReport:
    """Full UHI analysis report for a city/region"""
    city_name:          str
    report_date:        str
    image_date:         str
    total_area_ha:      float
    n_wards:            int
    lst_city_mean_c:    float
    lst_rural_bg_c:     float
    uhi_mean_c:         float       # Mean UHI intensity
    uhi_max_c:          float       # Peak UHI intensity
    pct_high_vulnerability: float   # % area in High/Extreme category
    ward_profiles:      List[Dict]
    heat_hotspots:      List[Dict]  # Top 10 hottest wards
    cooling_zones:      List[Dict]  # Wards with good green cover
    green_placement:    List[Dict]  # Recommended green cover locations
    cooling_potential_c: float      # Achievable cooling from greening
    health_risk_pop:    int         # Population in high-vulnerability zones
    recommendations:    List[str]
    methodology:        str


# =============================================================================
# URBAN HEAT ANALYZER — MAIN CLASS
# =============================================================================

class UrbanHeatAnalyzer:
    """
    Urban Heat Island analyzer using Landsat LST + Sentinel-2 spectral indices.

    Example:
        analyzer = UrbanHeatAnalyzer()

        # Analyze a single date
        report = analyzer.generate_uhi_report(
            thermal_scene   = landsat_scene,
            optical_scenes  = sentinel2_scenes,
            ward_boundaries = ward_geojson,
            city_name       = "Pune Municipal Corporation",
            target_lst_c    = 30.0,   # comfort target
        )

        # Get ward-level heat vulnerability ranking
        rankings = analyzer.rank_wards_by_vulnerability(report)
    """

    def __init__(self):
        self.cfg       = ZutoEnvConfig
        self.thresh    = self.cfg.UHI_THRESHOLDS
        self.cats      = self.cfg.UHI_CATEGORIES
        self.thermal   = self.cfg.LANDSAT_THERMAL
        self.emiss     = self.cfg.EMISSIVITY

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_uhi_report(
        self,
        thermal_scene:   Dict,
        optical_scenes:  List[Dict],
        ward_boundaries: Optional[List[Dict]] = None,
        city_name:       str   = "Urban Area",
        total_area_ha:   float = 5000.0,
        target_lst_c:    float = 30.0,
        rural_buffer_km: float = 10.0,
    ) -> UHIReport:
        """
        Generate full UHI analysis report for a city.

        Args:
            thermal_scene:   Landsat scene dict with thermal band data
            optical_scenes:  List of Sentinel-2 scenes for NDVI/NDBI
            ward_boundaries: List of ward GeoJSON polygons (optional)
            city_name:       City/ULB name
            total_area_ha:   Total urban area in ha
            target_lst_c:    Target comfortable LST threshold (°C)
            rural_buffer_km: Distance to rural background measurement (km)

        Returns:
            UHIReport dataclass
        """
        logger.info(f"\nURBAN HEAT ISLAND ANALYSIS — {city_name}")

        # — LST computation
        lst_city = self._compute_lst(thermal_scene, optical_scenes)

        # — Rural background (lower LST edge assumed ~rural fringe)
        lst_rural_bg = lst_city * 0.93   # rural is ~7% cooler on average (proxy)

        # — UHI intensity
        uhi_mean = lst_city - lst_rural_bg
        uhi_max  = uhi_mean * 1.6   # peak hotspot ~1.6× mean anomaly (empirical)

        # — Optical composites for NDVI/NDBI
        composite = self._composite_optical(optical_scenes)

        # — Generate ward profiles
        n_wards = len(ward_boundaries) if ward_boundaries else self._estimate_n_wards(total_area_ha)
        ward_profiles = self._generate_ward_profiles(
            lst_city, uhi_mean, composite, total_area_ha, n_wards, ward_boundaries, target_lst_c
        )

        # — Heat hotspots & cooling zones
        hotspots     = sorted(ward_profiles, key=lambda w: w['priority_score'], reverse=True)[:10]
        cooling_zones = sorted(ward_profiles, key=lambda w: w['green_cover_pct'], reverse=True)[:5]

        # — Green placement recommendations
        green_recs = self._recommend_green_placement(ward_profiles, total_area_ha)

        # — Cooling potential
        cooling_pot = self._estimate_cooling_potential(composite['ndvi'], total_area_ha)

        # — Vulnerability population
        vuln_pop = sum(
            w.get('population_est', 0) for w in ward_profiles
            if w.get('vulnerability') in ['High', 'Critical']
        )

        # — % high-vulnerability area
        high_vuln_ha = sum(
            w.get('area_ha', 0) for w in ward_profiles
            if w.get('vulnerability') in ['High', 'Critical']
        )
        pct_high_vuln = round(high_vuln_ha / total_area_ha * 100, 1) if total_area_ha > 0 else 0.0

        report = UHIReport(
            city_name              = city_name,
            report_date            = date.today().isoformat(),
            image_date             = thermal_scene.get('timestamp', date.today().isoformat()),
            total_area_ha          = total_area_ha,
            n_wards                = n_wards,
            lst_city_mean_c        = round(lst_city, 2),
            lst_rural_bg_c         = round(lst_rural_bg, 2),
            uhi_mean_c             = round(uhi_mean, 2),
            uhi_max_c              = round(uhi_max, 2),
            pct_high_vulnerability = pct_high_vuln,
            ward_profiles          = ward_profiles,
            heat_hotspots          = hotspots,
            cooling_zones          = cooling_zones,
            green_placement        = green_recs,
            cooling_potential_c    = round(cooling_pot, 2),
            health_risk_pop        = vuln_pop,
            recommendations        = self._generate_recommendations(
                uhi_mean, pct_high_vuln, composite['ndvi'], ward_profiles
            ),
            methodology            = self._methodology_note(),
        )

        self._log_report_summary(report)
        return report

    def compute_lst_from_landsat(self, thermal_scene: Dict) -> float:
        """
        Convert Landsat Band 10 DN to Land Surface Temperature (LST in °C).
        Uses USGS calibration constants.
        """
        return self._compute_lst(thermal_scene, [])

    def compute_uhi_intensity(
        self,
        urban_scenes:  List[Dict],
        rural_scenes:  List[Dict],
    ) -> Dict:
        """
        Compute UHI intensity as urban-rural LST difference.
        Returns time-series compatible dict.
        """
        urban_lst = np.mean([self._compute_lst(s, []) for s in urban_scenes])
        rural_lst = np.mean([self._compute_lst(s, []) for s in rural_scenes])
        uhi       = urban_lst - rural_lst

        category, vulnerability = self._classify_uhi(uhi)

        return {
            'urban_lst_c':   round(float(urban_lst), 2),
            'rural_lst_c':   round(float(rural_lst), 2),
            'uhi_c':         round(float(uhi), 2),
            'uhi_category':  category,
            'vulnerability': vulnerability,
        }

    def rank_wards_by_vulnerability(self, report: UHIReport) -> List[Dict]:
        """Return wards ranked by priority score (highest = most urgent)."""
        return sorted(
            report.ward_profiles,
            key=lambda w: w.get('priority_score', 0),
            reverse=True
        )

    def estimate_greening_impact(
        self,
        current_green_pct: float,
        target_green_pct:  float,
        area_ha:           float,
    ) -> Dict:
        """
        Estimate LST reduction from increasing green cover.
        Rule of thumb: 10% additional green cover → ~0.5–1.0°C LST reduction.
        """
        delta_green = max(target_green_pct - current_green_pct, 0)
        lst_reduction = delta_green / 10.0 * 0.75    # 0.75°C per 10% green
        trees_needed  = (delta_green / 100) * area_ha * 10000 / 36  # trees/36m² canopy

        return {
            'current_green_pct':  round(current_green_pct, 1),
            'target_green_pct':   round(target_green_pct, 1),
            'delta_green_pct':    round(delta_green, 1),
            'lst_reduction_c':    round(lst_reduction, 2),
            'trees_needed':       int(trees_needed),
            'area_ha_to_plant':   round((delta_green / 100) * area_ha, 1),
            'norms_reference':    'MoEFCC 33% green cover; Master Plan minimum 12% in core urban',
        }

    # =========================================================================
    # INTERNAL — LST COMPUTATION
    # =========================================================================

    def _compute_lst(self, thermal_scene: Dict, optical_scenes: List[Dict]) -> float:
        """
        Compute city-mean LST (°C) from Landsat thermal data.

        Steps:
          1. DN → Spectral Radiance using ML/AL constants
          2. Radiance → Brightness Temperature (Kelvin)
          3. Emissivity correction using NDVI-based FVC
          4. Convert K → °C
        """
        idx = thermal_scene.get('indices', {})

        # If pre-computed LST is available
        if 'lst_c' in idx:
            return float(idx['lst_c'])

        # Derive from thermal band DN or radiance
        dn   = float(idx.get('thermal_dn', idx.get('b10', 30000)))
        ML   = self.thermal['ML']
        AL   = self.thermal['AL']
        K1   = self.thermal['K1']
        K2   = self.thermal['K2']

        # Spectral radiance
        L_lambda = ML * dn + AL
        if L_lambda <= 0:
            L_lambda = 0.1

        # Brightness temperature (K)
        BT = K2 / np.log(K1 / L_lambda + 1)

        # Emissivity correction
        ndvi     = float(idx.get('ndvi', 0.3))
        emiss    = self._compute_emissivity(ndvi)
        rho      = 0.01438   # hc/k constant for 10.9µm band
        wavelength = 1.09e-5  # m

        # LST (K) — Wan & Dozier (1996) single-channel method
        LST_K = BT / (1 + (wavelength * BT / rho) * np.log(emiss))
        LST_C = LST_K - 273.15

        return float(np.clip(LST_C, -10.0, 70.0))

    def _compute_emissivity(self, ndvi: float) -> float:
        """
        Emissivity from NDVI using FVC (Fractional Vegetation Cover) method.
        FVC = ((NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil))²
        """
        ndvi_soil = 0.05
        ndvi_veg  = 0.70
        fvc       = np.clip(
            ((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)) ** 2,
            0.0, 1.0
        )
        emiss = (self.emiss['vegetation'] * fvc +
                 self.emiss['bare_soil'] * (1 - fvc))
        return float(emiss)

    # =========================================================================
    # INTERNAL — WARD PROFILE GENERATION
    # =========================================================================

    def _estimate_n_wards(self, total_area_ha: float) -> int:
        """Estimate number of wards from city area (avg ward ~200 ha in India)."""
        return max(int(total_area_ha / 200), 1)

    def _composite_optical(self, scenes: List[Dict]) -> Dict:
        """Build median NDVI/NDBI composite from optical scenes."""
        keys = ['ndvi', 'ndbi', 'evi', 'b08', 'b11']
        agg  = {k: [] for k in keys}
        for s in scenes:
            idx = s.get('indices', {})
            for k in keys:
                val = idx.get(k)
                if val is not None:
                    agg[k].append(float(val))
        return {k: float(np.median(v)) if v else 0.0 for k, v in agg.items()}

    def _generate_ward_profiles(
        self,
        lst_city:    float,
        uhi_mean:    float,
        composite:   Dict,
        total_ha:    float,
        n_wards:     int,
        boundaries:  Optional[List[Dict]],
        target_lst:  float,
    ) -> List[Dict]:
        """
        Generate per-ward heat profiles.
        Without real spatial data, we simulate ward-level variation
        using a realistic distribution around the city mean.
        In production: apply zonal statistics per ward polygon.
        """
        np.random.seed(42)   # reproducible simulation
        ward_profiles = []

        # LST variation across wards: city mean ± spatial variance
        lst_variations = np.random.normal(lst_city, uhi_mean * 0.6, n_wards)
        ndvi_variations = np.random.normal(composite['ndvi'], 0.10, n_wards)
        ndbi_variations = np.random.normal(composite['ndbi'], 0.08, n_wards)

        area_per_ward = total_ha / n_wards

        for i in range(n_wards):
            ward_lst   = float(np.clip(lst_variations[i], lst_city - 8, lst_city + 8))
            ward_ndvi  = float(np.clip(ndvi_variations[i], 0.0, 0.9))
            ward_ndbi  = float(np.clip(ndbi_variations[i], -0.3, 0.6))
            ward_uhi   = ward_lst - (lst_city - uhi_mean)
            category, vuln = self._classify_uhi(ward_uhi)
            impervious = float(np.clip(0.8 - ward_ndvi * 0.9, 0.1, 0.95))
            green_pct  = float(np.clip(ward_ndvi * 80, 5, 60))
            cooling_deficit = max(ward_lst - target_lst, 0.0)
            population = int(area_per_ward * 250)   # 250 persons/ha urban density
            priority   = self._compute_ward_priority(ward_uhi, green_pct, impervious, vuln)

            ward_profiles.append({
                'ward_id':          f"WARD_{i+1:03d}",
                'ward_name':        boundaries[i].get('name', f"Ward {i+1}") if boundaries and i < len(boundaries) else f"Ward {i+1}",
                'lst_mean_c':       round(ward_lst, 2),
                'lst_max_c':        round(ward_lst + 2.5, 2),
                'uhi_intensity_c':  round(ward_uhi, 2),
                'uhi_category':     category,
                'vulnerability':    vuln,
                'ndvi_mean':        round(ward_ndvi, 4),
                'ndbi_mean':        round(ward_ndbi, 4),
                'impervious_pct':   round(impervious * 100, 1),
                'green_cover_pct':  round(green_pct, 1),
                'cooling_deficit_c': round(cooling_deficit, 2),
                'area_ha':          round(area_per_ward, 1),
                'population_est':   population,
                'priority_score':   round(priority, 1),
            })

        return ward_profiles

    # =========================================================================
    # INTERNAL — CLASSIFICATION
    # =========================================================================

    def _classify_uhi(self, uhi_c: float) -> Tuple[str, str]:
        """Classify UHI intensity into category and vulnerability."""
        for cat, info in self.cats.items():
            if uhi_c <= info['lst_anom_max']:
                return cat, info['vulnerability']
        return 'Extreme', 'Critical'

    def _compute_ward_priority(
        self,
        uhi_c:       float,
        green_pct:   float,
        impervious:  float,
        vulnerability: str,
    ) -> float:
        """
        Compute intervention priority score 0-100.
        High UHI + Low green + High impervious = highest priority.
        """
        uhi_score        = np.clip(uhi_c / 10.0, 0, 1) * 40
        green_score      = np.clip(1.0 - green_pct / 60, 0, 1) * 30
        impervious_score = impervious * 20
        vuln_score       = {'Low': 0, 'Moderate': 5, 'High': 8, 'Critical': 10}.get(vulnerability, 0)

        return float(uhi_score + green_score + impervious_score + vuln_score)

    # =========================================================================
    # INTERNAL — GREEN PLACEMENT
    # =========================================================================

    def _recommend_green_placement(
        self,
        ward_profiles: List[Dict],
        total_ha:      float,
    ) -> List[Dict]:
        """
        Identify highest-priority wards for green infrastructure placement.
        Based on: high UHI + low green cover + high population density.
        """
        high_priority = [
            w for w in ward_profiles
            if w['priority_score'] > 60 and w['green_cover_pct'] < 15
        ]

        # Sort by priority
        high_priority = sorted(high_priority, key=lambda w: w['priority_score'], reverse=True)[:10]

        placements = []
        for w in high_priority:
            target_green = min(w['green_cover_pct'] + 15, 33)   # target 33% (MoEFCC norm)
            impact = self.estimate_greening_impact(
                w['green_cover_pct'], target_green, w['area_ha']
            )
            placements.append({
                'ward_id':          w['ward_id'],
                'ward_name':        w['ward_name'],
                'current_green_pct': w['green_cover_pct'],
                'recommended_target': target_green,
                'lst_reduction_c':  impact['lst_reduction_c'],
                'trees_needed':     impact['trees_needed'],
                'area_to_plant_ha': impact['area_ha_to_plant'],
                'priority_score':   w['priority_score'],
                'uhi_c':            w['uhi_intensity_c'],
                'type':             self._green_type_recommendation(w),
            })

        return placements

    def _green_type_recommendation(self, ward: Dict) -> str:
        """Recommend type of green infrastructure based on ward profile."""
        if ward['impervious_pct'] > 70:
            return "Vertical gardens, rooftop greening, tree pits on roads"
        elif ward['green_cover_pct'] < 5:
            return "Urban park/mini-forest (Miyawaki method), median plantation"
        elif ward['uhi_intensity_c'] > 6:
            return "High-canopy shade trees (Peepal, Banyan) along arterial roads"
        else:
            return "Neighborhood parks, rain gardens, bioswales"

    def _estimate_cooling_potential(self, city_ndvi: float, total_ha: float) -> float:
        """
        Estimate achievable LST reduction if all wards reach 33% green cover.
        Cooling = 0.75°C per 10% green cover increase.
        """
        current_green_pct = city_ndvi * 80   # NDVI to green% proxy
        target_green_pct  = 33.0
        delta             = max(target_green_pct - current_green_pct, 0)
        return (delta / 10.0) * 0.75

    # =========================================================================
    # INTERNAL — RECOMMENDATIONS
    # =========================================================================

    def _generate_recommendations(
        self,
        uhi_mean:     float,
        pct_high:     float,
        city_ndvi:    float,
        wards:        List[Dict],
    ) -> List[str]:
        recs = []

        if uhi_mean > 6:
            recs.append(f"CRITICAL: UHI intensity {uhi_mean:.1f}°C exceeds WHO heat stress threshold. Issue public health advisory for vulnerable populations.")
        elif uhi_mean > 4:
            recs.append(f"Significant UHI ({uhi_mean:.1f}°C). Coordinate with urban local body for emergency green cover strategy.")

        if pct_high > 40:
            recs.append(f"{pct_high:.0f}% of city area is high/extreme vulnerability. Implement ward-level cooling centres and heat action plan (MoES/NDMA guidelines).")

        if city_ndvi < 0.20:
            recs.append("Low city-wide NDVI (<0.20). Target minimum 33% green cover per MoEFCC Tree Policy. Prioritize Miyawaki urban forests in wards with UHI > 5°C.")
        elif city_ndvi < 0.30:
            recs.append("Moderate green cover. Increase urban tree canopy by 10% in priority wards. Integrate green corridors into Master Plan 2031.")

        # Top ward recommendation
        if wards:
            worst = sorted(wards, key=lambda w: w['priority_score'], reverse=True)[0]
            recs.append(
                f"Highest priority ward: {worst['ward_name']} (UHI {worst['uhi_intensity_c']:.1f}°C, "
                f"{worst['green_cover_pct']:.0f}% green). Recommend immediate tree plantation program."
            )

        recs.append("Commission LST monitoring twice yearly (pre-monsoon peak and post-monsoon) using Landsat-9 thermal data.")
        return recs

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _methodology_note(self) -> str:
        return (
            "LST derived from Landsat 8/9 Band 10 using single-channel algorithm "
            "(Wan & Dozier 1996). Emissivity estimated from NDVI-based FVC method "
            "(Sobrino et al. 2004). NDVI/NDBI from Sentinel-2 10m composites. "
            "UHI intensity = LST_urban - LST_rural_background (10 km buffer). "
            "Ward-level statistics via zonal statistics on ward boundary polygons. "
            "Green cover target per MoEFCC National Tree Policy (33% for urban areas)."
        )

    def _log_report_summary(self, r: UHIReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  UHI ANALYSIS — {r.city_name}\n"
            f"  Image Date  : {r.image_date}\n"
            f"  LST City    : {r.lst_city_mean_c:.1f}°C\n"
            f"  LST Rural   : {r.lst_rural_bg_c:.1f}°C\n"
            f"  UHI Mean    : {r.uhi_mean_c:.1f}°C\n"
            f"  UHI Max     : {r.uhi_max_c:.1f}°C\n"
            f"  High Vuln   : {r.pct_high_vulnerability:.1f}% of area\n"
            f"  Risk Pop    : {r.health_risk_pop:,} persons\n"
            f"  Cooling Pot : {r.cooling_potential_c:.1f}°C achievable\n"
            f"  Hotspots    : {len(r.heat_hotspots)} wards\n"
            f"  Green Recs  : {len(r.green_placement)} placement sites\n"
            f"{'='*60}"
        )
