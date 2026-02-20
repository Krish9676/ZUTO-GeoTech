"""
ZUTO Geotech Solutions — Logistics Intelligence Platform
=========================================================
Module C: Warehouse & Hub Placement Intelligence

Spatial optimization for warehouse, cold storage, dark store,
and aggregation center placement.

Inputs:
  - Demand heatmap (from LastMileOptimizer or order data)
  - Road network accessibility (from AgriLogisticsMapper)
  - Land availability (from LULC maps, Component 2)
  - Existing competitor/hub locations
  - Cost surfaces (land cost by zone)

Output:
  - Ranked list of optimal locations for warehouses, cold storage, dark stores
  - Site-level scoring: demand + accessibility + cost + competition
  - Consulting report format (for govt/corporate clients)
  - SaaS-ready API endpoint output

Target clients:
  - Food processing companies (ITC, Cargill, Reliance Retail)
  - State warehousing corporations (CWC, SWC)
  - Quick-commerce operators (Blinkit, Zepto, Swiggy Instamart)
  - NABARD / NCDC for rural cooperative warehouse network
  - Pharma distributors (Apollo, Medplus)
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.logistics_config import ZutoLogisticsConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CandidateSite:
    """A candidate location for a warehouse or hub"""
    site_id:            str
    name:               str
    latitude:           float
    longitude:          float
    site_type:          str         # primary_hub / cold_storage / aggregation_center / dark_store
    demand_score:       float       # 0-100
    accessibility_score: float      # 0-100
    land_availability:  float       # 0-100
    competition_score:  float       # 0-100 (100 = no competition nearby)
    cost_score:         float       # 0-100 (100 = lowest cost)
    composite_score:    float       # Weighted final score
    rank:               int
    estimated_catchment_km: float
    estimated_demand_mt: float
    land_cost_est_lakh:  float      # INR Lakhs
    construction_cost_lakh: float
    total_capex_lakh:   float
    payback_years:      float
    recommended:        bool
    notes:              str


@dataclass
class PlacementReport:
    """Full warehouse placement analysis"""
    project_name:       str
    hub_type:           str
    report_date:        str
    region_name:        str
    n_candidates:       int
    top_sites:          List[Dict]  # Top 5 recommended sites
    all_sites:          List[Dict]
    total_demand_mt:    float
    coverage_achieved_pct: float
    total_capex_lakh:   float
    roi_summary:        Dict
    recommendations:    List[str]
    methodology:        str


# =============================================================================
# WAREHOUSE PLANNER — MAIN CLASS
# =============================================================================

class WarehousePlanner:
    """
    Spatial optimization for warehouse and hub placement.

    Example:
        planner = WarehousePlanner()

        report = planner.find_optimal_locations(
            demand_heatmap   = optimizer.build_demand_heatmap(orders_df),
            candidate_areas  = district_centroids,
            existing_hubs    = existing_warehouses,
            hub_type         = 'cold_storage',
            region_name      = "Maharashtra",
            n_to_recommend   = 3,
        )
    """

    def __init__(self):
        self.cfg       = ZutoLogisticsConfig
        self.hub_types = self.cfg.WAREHOUSE_TYPES
        self.weights   = self.cfg.PLACEMENT_WEIGHTS
        self.costs     = self.cfg.COST_PARAMS

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def find_optimal_locations(
        self,
        demand_heatmap:  Dict,
        candidate_areas: List[Dict],
        existing_hubs:   Optional[List[Dict]] = None,
        lulc_map:        Optional[List[Dict]] = None,
        hub_type:        str   = 'aggregation_center',
        region_name:     str   = "Region",
        project_name:    str   = "Hub Placement Study",
        n_to_recommend:  int   = 5,
    ) -> PlacementReport:
        """
        Find optimal warehouse/hub locations from a list of candidate areas.

        Args:
            demand_heatmap:  Output from LastMileOptimizer.build_demand_heatmap()
            candidate_areas: List of dicts with [id, name, lat, lon, area_type]
                             (district centroids, taluka HQs, etc.)
            existing_hubs:   Already existing hub locations to avoid duplication
            lulc_map:        LULC data for land availability check
            hub_type:        Key from ZutoLogisticsConfig.WAREHOUSE_TYPES
            region_name:     Region name
            project_name:    Project/study name
            n_to_recommend:  Number of top sites to recommend

        Returns:
            PlacementReport dataclass
        """
        logger.info(f"\nWAREHOUSE PLACEMENT — {project_name} | Type: {hub_type} | Candidates: {len(candidate_areas)}")

        hub_cfg = self.hub_types.get(hub_type, self.hub_types['aggregation_center'])

        # — Score all candidate sites
        sites = []
        for area in candidate_areas:
            site = self._score_candidate_site(
                area, demand_heatmap, existing_hubs or [], lulc_map, hub_cfg, hub_type
            )
            sites.append(site)

        # — Sort by composite score
        sites = sorted(sites, key=lambda s: s['composite_score'], reverse=True)
        for i, s in enumerate(sites):
            s['rank'] = i + 1
            s['recommended'] = i < n_to_recommend

        top_sites = sites[:n_to_recommend]

        # — Coverage analysis
        total_demand   = demand_heatmap.get('total_demand_kg', 0) / 1000   # → MT
        covered_demand = sum(s['estimated_demand_mt'] for s in top_sites)
        coverage_pct   = min(covered_demand / total_demand * 100, 100) if total_demand > 0 else 0

        # — Financial summary
        total_capex = sum(s['total_capex_lakh'] for s in top_sites)
        roi_summary = self._roi_summary(top_sites, hub_cfg)

        report = PlacementReport(
            project_name       = project_name,
            hub_type           = hub_cfg['name'],
            report_date        = date.today().isoformat(),
            region_name        = region_name,
            n_candidates       = len(sites),
            top_sites          = top_sites,
            all_sites          = sites,
            total_demand_mt    = round(total_demand, 1),
            coverage_achieved_pct = round(coverage_pct, 1),
            total_capex_lakh   = round(total_capex, 2),
            roi_summary        = roi_summary,
            recommendations    = self._generate_recommendations(top_sites, hub_type, coverage_pct),
            methodology        = self._methodology_note(),
        )

        self._log_report_summary(report)
        return report

    def score_single_site(
        self,
        site:           Dict,
        hub_type:       str,
        demand_heatmap: Optional[Dict] = None,
        existing_hubs:  Optional[List[Dict]] = None,
    ) -> Dict:
        """Quick score a single site for a given hub type."""
        hub_cfg = self.hub_types.get(hub_type, self.hub_types['aggregation_center'])
        return self._score_candidate_site(site, demand_heatmap or {}, existing_hubs or [], None, hub_cfg, hub_type)

    # =========================================================================
    # INTERNAL — SCORING
    # =========================================================================

    def _score_candidate_site(
        self,
        area:          Dict,
        demand_hm:     Dict,
        existing_hubs: List[Dict],
        lulc_map:      Optional[List[Dict]],
        hub_cfg:       Dict,
        hub_type:      str,
    ) -> Dict:
        """Score a candidate area on all placement criteria."""
        lat = area.get('lat', area.get('latitude', 0))
        lon = area.get('lon', area.get('longitude', 0))
        name = area.get('name', area.get('district', 'Unknown'))

        # — Demand score: proximity to high-demand cells
        demand_score = self._score_demand(lat, lon, demand_hm, hub_cfg.get('catchment_km', 30))

        # — Accessibility: distance to nearest NH/SH
        access_score = self._score_accessibility(area, hub_cfg)

        # — Land availability: based on LULC
        land_score = self._score_land_availability(lat, lon, lulc_map)

        # — Competition: distance to existing hubs
        competition_score = self._score_competition(lat, lon, existing_hubs, hub_cfg.get('catchment_km', 30))

        # — Cost: land cost zone
        cost_score = self._score_cost(area)

        # — Weighted composite
        w = self.weights
        composite = (
            w['demand_score']       * demand_score +
            w['road_accessibility'] * access_score +
            w['land_availability']  * land_score +
            w['competition_proximity'] * competition_score +
            w['cost_surface']       * cost_score
        )

        # — Financial estimates
        area_type     = area.get('area_type', 'peri_urban')
        min_area      = hub_cfg.get('min_area_sqm', 1000)
        land_cost     = self.costs['land_cost_per_sqm'].get(area_type, 2000) * min_area / 1e5
        constr_cost   = self.costs['construction_per_sqm'].get(hub_type.split('_')[0], 4000) * min_area / 1e5
        total_capex   = land_cost + constr_cost
        demand_mt_yr  = demand_score / 100 * hub_cfg.get('min_capacity_t', 100) * 2   # throughput proxy
        revenue_est   = demand_mt_yr * 500   # ₹500/MT warehousing margin
        payback       = total_capex * 1e5 / revenue_est if revenue_est > 0 else 99.0

        return {
            'site_id':              area.get('id', f"SITE_{name[:4].upper()}"),
            'name':                 name,
            'latitude':             lat,
            'longitude':            lon,
            'site_type':            hub_type,
            'demand_score':         round(demand_score, 1),
            'accessibility_score':  round(access_score, 1),
            'land_availability':    round(land_score, 1),
            'competition_score':    round(competition_score, 1),
            'cost_score':           round(cost_score, 1),
            'composite_score':      round(composite, 1),
            'rank':                 0,
            'estimated_catchment_km': hub_cfg.get('catchment_km', 30),
            'estimated_demand_mt':  round(demand_mt_yr, 1),
            'land_cost_est_lakh':   round(land_cost, 2),
            'construction_cost_lakh': round(constr_cost, 2),
            'total_capex_lakh':     round(total_capex, 2),
            'payback_years':        round(payback, 1),
            'recommended':          False,
            'notes':                self._site_notes(demand_score, access_score, competition_score),
        }

    def _score_demand(self, lat: float, lon: float, heatmap: Dict, catchment_km: float) -> float:
        """Score demand based on proximity to high-demand heatmap cells."""
        cells = heatmap.get('cells', [])
        if not cells:
            return 50.0

        total_demand  = heatmap.get('total_demand_kg', 1)
        catchable_dem = 0.0

        for cell in cells:
            dist = self._haversine_km(lat, lon, cell['lat_center'], cell['lon_center'])
            if dist <= catchment_km:
                # Weight by inverse distance
                weight = 1 - (dist / catchment_km)
                catchable_dem += cell['total_demand_kg'] * weight

        return float(np.clip(catchable_dem / total_demand * 200, 0, 100))

    def _score_accessibility(self, area: Dict, hub_cfg: Dict) -> float:
        """Score road accessibility based on area metadata."""
        req_class = hub_cfg.get('road_access_req', 'MDR')
        area_type = area.get('area_type', 'rural')
        road_class = area.get('nearest_road_class', None)

        class_scores = {'NH': 100, 'SH': 85, 'MDR': 65, 'PMGSY': 45, 'ODR': 30, 'VR': 15}
        req_scores   = {'NH': 90, 'SH': 70, 'MDR': 50, 'ODR': 30}

        if road_class:
            return float(class_scores.get(road_class, 50))

        # Infer from area type
        type_scores = {'urban': 90, 'peri_urban': 70, 'rural': 40}
        return float(type_scores.get(area_type, 50))

    def _score_land_availability(
        self, lat: float, lon: float, lulc_map: Optional[List[Dict]]
    ) -> float:
        """Score land availability from LULC (barren/cropland = good; forest/urban = bad)."""
        if not lulc_map:
            return 60.0   # neutral default

        # Find nearest LULC pixel
        best = min(
            lulc_map,
            key=lambda p: abs(p.get('lat', 0) - lat) + abs(p.get('lon', 0) - lon)
        )
        lulc_scores = {
            'Barren': 95, 'Cropland': 70, 'Grassland': 75,
            'Urban': 40, 'Forest': 10, 'Wetland': 5, 'Water': 0,
        }
        return float(lulc_scores.get(best.get('lulc_name', 'Barren'), 60))

    def _score_competition(
        self, lat: float, lon: float, existing: List[Dict], catchment_km: float
    ) -> float:
        """Score competition: 100 = no existing hubs nearby; 0 = fully saturated."""
        if not existing:
            return 100.0

        min_dist = min(
            self._haversine_km(lat, lon, h.get('lat', 0), h.get('lon', 0))
            for h in existing
        )
        return float(np.clip((min_dist / catchment_km) * 100, 0, 100))

    def _score_cost(self, area: Dict) -> float:
        """Score cost: rural = 100 (lowest cost), urban = 20 (high cost)."""
        area_type = area.get('area_type', 'rural')
        return {'rural': 100, 'peri_urban': 60, 'urban': 20}.get(area_type, 60)

    def _site_notes(self, dem: float, acc: float, comp: float) -> str:
        notes = []
        if dem > 75:
            notes.append("High demand catchment.")
        if acc < 40:
            notes.append("Poor road access — requires infrastructure investment.")
        if comp < 30:
            notes.append("Competing hub in vicinity — evaluate market share risk.")
        return " ".join(notes) or "Balanced site — good overall candidate."

    def _roi_summary(self, sites: List[Dict], hub_cfg: Dict) -> Dict:
        if not sites:
            return {}
        total_capex = sum(s['total_capex_lakh'] for s in sites)
        total_demand = sum(s['estimated_demand_mt'] for s in sites)
        annual_rev   = total_demand * 500 / 1e5   # ₹500/MT → Lakhs
        payback_avg  = total_capex / annual_rev if annual_rev > 0 else 0
        return {
            'total_capex_lakh': round(total_capex, 2),
            'annual_revenue_lakh': round(annual_rev, 2),
            'avg_payback_years': round(payback_avg, 1),
            'irr_estimate_pct': round(max(25 - payback_avg * 2, 8), 1),
        }

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return float(2 * R * np.arcsin(np.sqrt(a)))

    def _generate_recommendations(
        self, top_sites: List[Dict], hub_type: str, coverage_pct: float
    ) -> List[str]:
        recs = []
        if top_sites:
            best = top_sites[0]
            recs.append(f"Top site: {best['name']} (score {best['composite_score']:.1f}/100, CapEx ₹{best['total_capex_lakh']:.1f}L). Recommend feasibility study within 60 days.")
        if coverage_pct < 60:
            recs.append(f"Current {len(top_sites)} sites cover only {coverage_pct:.0f}% of demand. Consider adding {int((100-coverage_pct)/15)} more sites.")
        recs.append(f"Prioritize NABARD RIDF / APEDA / MoFPI grant applications for {hub_type.replace('_', ' ')} funding (40–50% subsidy available).")
        return recs

    def _methodology_note(self) -> str:
        return (
            "Placement scores computed using weighted multi-criteria analysis: "
            "demand proximity (35%), road accessibility (25%), land availability (20%), "
            "competition distance (10%), cost surface (10%). "
            "Demand from order/transaction heatmap. Land availability from Sentinel-2 LULC. "
            "Road accessibility from OSM network classification. "
            "Financial estimates: land cost (India DLC rates), construction (CPWD DSR)."
        )

    def _log_report_summary(self, r: PlacementReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  WAREHOUSE PLACEMENT — {r.project_name}\n"
            f"  Type        : {r.hub_type}\n"
            f"  Candidates  : {r.n_candidates}\n"
            f"  Top sites   : {len(r.top_sites)}\n"
            f"  Coverage    : {r.coverage_achieved_pct:.1f}%\n"
            f"  Total CapEx : ₹{r.total_capex_lakh:.1f}L\n"
            f"  Payback     : {r.roi_summary.get('avg_payback_years', 'N/A')} years\n"
            f"{'='*60}"
        )
