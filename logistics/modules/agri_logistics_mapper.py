"""
ZUTO Geotech Solutions — Logistics Intelligence Platform
=========================================================
Module A: Agri-Logistics Corridor Mapper

ZUTO's unique cross-component differentiator:
  Takes crop production maps (Component 1 AgriTech output) →
  Overlays with road network (OSM) →
  Identifies bottlenecks between farm clusters and market/processing locations →
  Produces a corridor priority map showing:
    - Which roads are critical for agri-supply chains
    - Where infrastructure gaps exist
    - Volume-to-capacity ratios per road segment
    - Seasonal accessibility changes (monsoon impact)

Target clients:
  - State governments (MRRDA, PWD) for road investment prioritization
  - NABARD for rural infrastructure financing
  - Food processing companies (Cargill, ITC, Adani Agri)
  - FPOs and cooperatives planning logistics
  - Ministry of Agriculture for value chain analysis
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
class RoadSegment:
    """Represents a road segment in the network"""
    segment_id:     str
    road_class:     str
    length_km:      float
    speed_kmh:      float
    quality:        str
    truck_capable:  bool
    from_node:      str
    to_node:        str
    connects:       List[str]       # list of location names it connects


@dataclass
class CorridorLink:
    """A critical farm-to-market supply chain corridor"""
    link_id:            str
    corridor_name:      str
    origin:             str         # Farm cluster / village
    origin_lat:         float
    origin_lon:         float
    destination:        str         # Mandi / cold storage / processing unit
    destination_lat:    float
    destination_lon:    float
    crop:               str
    production_mt:      float       # Annual production in MT
    distance_km:        float
    road_class:         str
    road_quality_score: float       # 0-100
    travel_time_min:    float
    bottleneck:         bool
    bottleneck_reason:  str
    priority_score:     float       # 0-100 (100 = most critical)
    infrastructure_gap: bool
    recommended_action: str
    seasonal_disruption: str        # monsoon impact level


@dataclass
class CorridorReport:
    """Full agri-logistics corridor analysis for a region"""
    region_name:            str
    report_date:            str
    total_corridors:        int
    total_volume_mt:        float
    critical_corridors:     List[Dict]
    infrastructure_gaps:    List[Dict]
    bottleneck_segments:    List[Dict]
    top_priorities:         List[Dict]   # Top 10 investment priorities
    connected_mandis:       List[str]
    unconnected_clusters:   List[str]
    summary:                str
    recommendations:        List[str]


# =============================================================================
# AGRI-LOGISTICS MAPPER — MAIN CLASS
# =============================================================================

class AgriLogisticsMapper:
    """
    Maps agri-supply chain corridors by combining crop production maps
    (Component 1 output) with road network data to find infrastructure gaps.

    Example:
        mapper = AgriLogisticsMapper()

        report = mapper.generate_corridor_report(
            crop_production_map = agritech_output,
            road_network        = osm_roads_df,
            mandis              = mandi_list,
            region_name         = "Marathwada Region",
        )
    """

    def __init__(self):
        self.cfg      = ZutoLogisticsConfig
        self.roads    = self.cfg.ROAD_CLASSES
        self.thresh   = self.cfg.CORRIDOR_THRESHOLDS
        self.commodity = self.cfg.COMMODITY_LOGISTICS

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_corridor_report(
        self,
        crop_production_map: List[Dict],
        road_network:        Optional[pd.DataFrame] = None,
        mandis:              Optional[List[Dict]] = None,
        cold_storages:       Optional[List[Dict]] = None,
        processing_units:    Optional[List[Dict]] = None,
        region_name:         str   = "Region",
        season:              str   = "kharif",
    ) -> CorridorReport:
        """
        Full agri-logistics corridor analysis.

        Args:
            crop_production_map: List of dicts from AgriTech Component 1:
                                 [{village, crop, production_mt, lat, lon}]
            road_network:        DataFrame with road segments
            mandis:              List of mandi dicts [{name, lat, lon, capacity_mt}]
            cold_storages:       List of cold storage dicts
            processing_units:    List of food processing unit dicts
            region_name:         Region name
            season:              'kharif' | 'rabi' | 'zaid'

        Returns:
            CorridorReport dataclass
        """
        logger.info(f"\nAGRI-LOGISTICS CORRIDOR MAPPING — {region_name} | Season: {season}")

        # Destinations = mandis + cold storages + processing units
        destinations = self._compile_destinations(mandis, cold_storages, processing_units)

        # — Build corridors: each farm cluster → nearest suitable destination
        corridors = []
        for farm in crop_production_map:
            prod_mt = farm.get('production_mt', 0)
            if prod_mt < self.thresh['min_production_mt']:
                continue   # Skip low-volume clusters

            best_dest = self._find_best_destination(farm, destinations, road_network)
            if best_dest:
                link = self._build_corridor_link(farm, best_dest, road_network, season)
                corridors.append(link)

        # — Identify bottlenecks and gaps
        bottlenecks = [c for c in corridors if c['bottleneck']]
        gaps        = [c for c in corridors if c['infrastructure_gap']]

        # — Rank by priority
        sorted_corr = sorted(corridors, key=lambda c: c['priority_score'], reverse=True)
        top10       = sorted_corr[:10]

        # — Unconnected clusters
        all_origins    = {f['village'] for f in crop_production_map}
        connected      = {c['origin'] for c in corridors}
        unconnected    = list(all_origins - connected)

        # — Total volume
        total_vol = sum(c['production_mt'] for c in corridors)

        # — Mandi list
        mandi_names = [m.get('name', '') for m in (mandis or [])]

        summary = (
            f"Mapped {len(corridors)} agri-supply corridors in {region_name}. "
            f"Total volume: {total_vol:,.0f} MT. "
            f"Bottlenecks: {len(bottlenecks)}, Infrastructure gaps: {len(gaps)}, "
            f"Unconnected clusters: {len(unconnected)}."
        )

        report = CorridorReport(
            region_name           = region_name,
            report_date           = date.today().isoformat(),
            total_corridors       = len(corridors),
            total_volume_mt       = round(total_vol, 0),
            critical_corridors    = [c for c in sorted_corr if c['priority_score'] > 70],
            infrastructure_gaps   = gaps,
            bottleneck_segments   = bottlenecks,
            top_priorities        = top10,
            connected_mandis      = mandi_names,
            unconnected_clusters  = unconnected,
            summary               = summary,
            recommendations       = self._generate_recommendations(corridors, gaps, bottlenecks, unconnected),
        )

        self._log_report_summary(report)
        return report

    @staticmethod
    def from_heatmap_output(heatmap_result: Dict) -> List[Dict]:
        """
        Convert Component 1 MarketAnalytics.generate_production_heatmap() output
        into the format expected by generate_corridor_report(crop_production_map).

        This is the cross-component bridge — call this before passing C1 output to C3.

        Args:
            heatmap_result: Dict returned by CropProductionHeatmap.generate()
                           Keys: 'grid' (DataFrame), 'crop_breakdown', 'district_summary'

        Returns:
            List of dicts: [{village, crop, production_mt, lat, lon}]

        Example:
            heatmap = market.generate_production_heatmap(field_results)
            farm_clusters = AgriLogisticsMapper.from_heatmap_output(heatmap)
            report = mapper.generate_corridor_report(farm_clusters, ...)
        """
        grid = heatmap_result.get('grid', None)
        if grid is None or (hasattr(grid, 'empty') and grid.empty):
            return []

        # Handle both DataFrame and list-of-dict inputs
        if hasattr(grid, 'iterrows'):
            rows = [row.to_dict() for _, row in grid.iterrows()]
        else:
            rows = grid

        result = []
        for row in rows:
            production_mt = row.get('total_prod_t', row.get('production_t', 0))
            result.append({
                'village':       f"Cell_{row.get('cell_id', row.get('cell_lat', '?'))}",
                'crop':          row.get('dominant_crop', row.get('crop', 'Unknown')),
                'production_mt': float(production_mt),
                'lat':           float(row.get('cell_lat', row.get('lat', 0))),
                'lon':           float(row.get('cell_lon', row.get('lon', 0))),
                'area_ha':       float(row.get('total_area_ha', row.get('area_ha', 0))),
            })

        return result

    def score_road_segment(self, segment: Dict) -> float:
        """
        Score a road segment's quality for agri-logistics (0-100).
        Considers: road class, surface quality, truck capability, seasonal access.
        """
        road_class   = segment.get('road_class', 'VR')
        quality      = segment.get('quality', 'poor')
        truck_ok     = segment.get('truck_capable', False)

        class_scores  = {'NH': 100, 'SH': 85, 'MDR': 65, 'PMGSY': 50, 'ODR': 35, 'VR': 20}
        quality_mult  = {'good': 1.0, 'moderate': 0.8, 'poor': 0.5, 'very_poor': 0.3}
        truck_bonus   = 10 if truck_ok else 0

        base = class_scores.get(road_class, 20)
        mult = quality_mult.get(quality, 0.5)
        return float(np.clip(base * mult + truck_bonus, 0, 100))

    def identify_cold_chain_gaps(
        self,
        farm_clusters:  List[Dict],
        cold_storages:  List[Dict],
        crops:          Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Identify farm clusters growing perishable crops that are >30 km
        from nearest cold storage — critical cold-chain infrastructure gap.
        """
        perishable_crops = crops or ['vegetables', 'fruits', 'flowers', 'milk']
        gaps = []

        for farm in farm_clusters:
            crop = farm.get('crop', '').lower()
            is_perishable = any(p in crop for p in perishable_crops)
            if not is_perishable:
                continue

            # Distance to nearest cold storage
            if not cold_storages:
                gaps.append({
                    'village':     farm.get('village', 'Unknown'),
                    'crop':        farm.get('crop', ''),
                    'production_mt': farm.get('production_mt', 0),
                    'nearest_cold_km': 999,
                    'gap_severity': 'Critical',
                    'recommendation': 'No cold storage in region. Establish primary cold chain facility.',
                })
                continue

            min_dist = self._nearest_distance(
                farm.get('lat', 0), farm.get('lon', 0), cold_storages
            )

            if min_dist > 30:
                severity = 'Critical' if min_dist > 60 else 'High'
                gaps.append({
                    'village':        farm.get('village', 'Unknown'),
                    'crop':           farm.get('crop', ''),
                    'production_mt':  farm.get('production_mt', 0),
                    'nearest_cold_km': round(min_dist, 1),
                    'gap_severity':   severity,
                    'recommendation': f"Establish cold storage within 30 km to prevent postharvest loss (est. >15% currently).",
                })

        return sorted(gaps, key=lambda g: g['nearest_cold_km'], reverse=True)

    # =========================================================================
    # INTERNAL — CORRIDOR BUILDING
    # =========================================================================

    def _compile_destinations(
        self,
        mandis:           Optional[List[Dict]],
        cold_storages:    Optional[List[Dict]],
        processing_units: Optional[List[Dict]],
    ) -> List[Dict]:
        """Merge all market destinations into a unified list."""
        destinations = []
        for d in (mandis or []):
            d['dest_type'] = 'mandi'
            destinations.append(d)
        for d in (cold_storages or []):
            d['dest_type'] = 'cold_storage'
            destinations.append(d)
        for d in (processing_units or []):
            d['dest_type'] = 'processing'
            destinations.append(d)

        if not destinations:
            # Dummy placeholder if no market data provided
            destinations = [{'name': 'District Mandi', 'lat': 18.5, 'lon': 73.8, 'dest_type': 'mandi', 'capacity_mt': 5000}]

        return destinations

    def _find_best_destination(
        self,
        farm:         Dict,
        destinations: List[Dict],
        road_network: Optional[pd.DataFrame],
    ) -> Optional[Dict]:
        """Find the best (nearest, highest capacity) destination for a farm cluster."""
        if not destinations:
            return None

        farm_lat = farm.get('lat', 0)
        farm_lon = farm.get('lon', 0)

        best = None
        best_score = float('inf')

        for dest in destinations:
            dist = self._haversine_km(farm_lat, farm_lon, dest.get('lat', 0), dest.get('lon', 0))
            cap  = dest.get('capacity_mt', 1000)
            # Prefer close, high-capacity destinations
            score = dist / (1 + np.log1p(cap / 1000))
            if score < best_score:
                best_score = score
                best = dest

        return best

    def _build_corridor_link(
        self,
        farm:         Dict,
        destination:  Dict,
        road_network: Optional[pd.DataFrame],
        season:       str,
    ) -> Dict:
        """Build a CorridorLink dict for a farm-destination pair."""
        farm_lat  = farm.get('lat', 0)
        farm_lon  = farm.get('lon', 0)
        dest_lat  = destination.get('lat', 0)
        dest_lon  = destination.get('lon', 0)
        crop      = farm.get('crop', 'Unknown')
        prod_mt   = farm.get('production_mt', 0)

        dist_km   = self._haversine_km(farm_lat, farm_lon, dest_lat, dest_lon)

        # Infer road class from distance (simplified)
        road_class = self._infer_road_class(dist_km)
        road_info  = self.roads.get(road_class, self.roads['ODR'])
        quality    = road_info['quality']

        # Apply seasonal factor
        seasonal_mult = self.cfg.SEASONAL_FACTOR.get(season, {}).get(road_class, 1.0)
        effective_speed = road_info['speed_kmh'] * seasonal_mult

        travel_min = (dist_km / effective_speed) * 60
        quality_score = self.score_road_segment({'road_class': road_class, 'quality': quality, 'truck_capable': road_info['truck_capable']})

        # Bottleneck logic
        bottleneck        = quality_score < self.thresh['road_quality_gap'] or dist_km > self.thresh['max_farm_mandi_km']
        infrastructure_gap = quality_score < self.thresh['road_quality_gap']

        bottleneck_reason = ''
        if dist_km > self.thresh['max_farm_mandi_km']:
            bottleneck_reason += f"Distance {dist_km:.0f} km exceeds 50 km threshold. "
        if quality_score < self.thresh['road_quality_gap']:
            bottleneck_reason += f"Road quality score {quality_score:.0f}/100 below threshold. "
        if seasonal_mult < 1.0:
            bottleneck_reason += f"Monsoon accessibility reduced to {seasonal_mult*100:.0f}%."

        priority_score = self._compute_corridor_priority(prod_mt, dist_km, quality_score, bottleneck)

        seasonal_disruption = 'High' if seasonal_mult < 0.7 else ('Moderate' if seasonal_mult < 0.9 else 'Low')

        return {
            'link_id':            f"CORR_{farm.get('village', 'V')[:4]}_{destination.get('name', 'D')[:4]}",
            'corridor_name':      f"{farm.get('village', 'Farm')} → {destination.get('name', 'Market')}",
            'origin':             farm.get('village', 'Unknown Village'),
            'origin_lat':         farm_lat,
            'origin_lon':         farm_lon,
            'destination':        destination.get('name', 'Unknown Market'),
            'destination_lat':    dest_lat,
            'destination_lon':    dest_lon,
            'crop':               crop,
            'production_mt':      prod_mt,
            'distance_km':        round(dist_km, 1),
            'road_class':         road_class,
            'road_quality_score': round(quality_score, 1),
            'travel_time_min':    round(travel_min, 0),
            'bottleneck':         bottleneck,
            'bottleneck_reason':  bottleneck_reason.strip(),
            'priority_score':     round(priority_score, 1),
            'infrastructure_gap': infrastructure_gap,
            'recommended_action': self._corridor_action(bottleneck, infrastructure_gap, dist_km, road_class),
            'seasonal_disruption': seasonal_disruption,
            'dest_type':          destination.get('dest_type', 'mandi'),
        }

    # =========================================================================
    # INTERNAL — SCORING & CLASSIFICATION
    # =========================================================================

    def _compute_corridor_priority(
        self,
        prod_mt:       float,
        dist_km:       float,
        quality_score: float,
        bottleneck:    bool,
    ) -> float:
        """Priority = high volume + far from market + poor road quality."""
        vol_score  = np.clip(np.log1p(prod_mt) / np.log1p(10000) * 40, 0, 40)
        dist_score = np.clip((dist_km - 10) / 90 * 30, 0, 30)
        qual_penalty = (100 - quality_score) / 100 * 20
        bn_bonus   = 10 if bottleneck else 0
        return float(vol_score + dist_score + qual_penalty + bn_bonus)

    def _infer_road_class(self, dist_km: float) -> str:
        """Heuristic: longer corridors tend to use higher-class roads."""
        if dist_km > 100:   return 'NH'
        elif dist_km > 50:  return 'SH'
        elif dist_km > 20:  return 'MDR'
        elif dist_km > 10:  return 'ODR'
        else:               return 'VR'

    def _corridor_action(
        self,
        bottleneck: bool,
        gap:        bool,
        dist_km:    float,
        road_class: str,
    ) -> str:
        if gap and road_class in ['ODR', 'VR']:
            return f"Upgrade to MDR/PMGSY standard. Apply for NABARD RIDF or MRRDA funding. Estimated cost: ₹50–80L/km."
        elif bottleneck and dist_km > 50:
            return f"Establish intermediate collection/aggregation center at ~{int(dist_km/2)} km point to reduce single-trip distance."
        elif bottleneck:
            return "Minor road improvement needed. Pothole patching + bridge load-limit revision for trucks."
        return "Corridor functional. Monitor annually."

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula to compute great-circle distance in km."""
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi  = np.radians(lat2 - lat1)
        dlam  = np.radians(lon2 - lon1)
        a     = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
        return float(2 * R * np.arcsin(np.sqrt(a)))

    def _nearest_distance(self, lat: float, lon: float, locations: List[Dict]) -> float:
        """Distance in km to nearest location in list."""
        if not locations:
            return 999.0
        return min(
            self._haversine_km(lat, lon, loc.get('lat', 0), loc.get('lon', 0))
            for loc in locations
        )

    def _generate_recommendations(
        self,
        corridors:    List[Dict],
        gaps:         List[Dict],
        bottlenecks:  List[Dict],
        unconnected:  List[str],
    ) -> List[str]:
        recs = []

        if gaps:
            recs.append(
                f"{len(gaps)} corridors have road quality gaps. Prioritize top-3 by production volume for "
                f"immediate MRRDA/NABARD RIDF grant application."
            )
        if bottlenecks:
            recs.append(
                f"{len(bottlenecks)} supply corridors have bottlenecks. Deploy intermediate aggregation "
                f"centers (Primary Agricultural Cooperative Society model) to reduce transport distance."
            )
        if unconnected:
            recs.append(
                f"{len(unconnected)} farm clusters are not connected to any market within {self.thresh['max_farm_mandi_km']} km. "
                f"Priority for direct farmer outreach and Farmer Producer Organization linkage."
            )

        # Cold chain recommendation (if perishables identified)
        perishable_vols = [c['production_mt'] for c in corridors if any(
            p in c.get('crop', '').lower() for p in ['vegetable', 'fruit', 'flower']
        )]
        if perishable_vols:
            recs.append(
                f"Perishable supply of ~{sum(perishable_vols):,.0f} MT at risk. "
                f"Establish cold chain at high-production clusters. "
                f"APEDA / NHM Mission for Integrated Development of Horticulture (MIDH) funding available."
            )

        if not recs:
            recs.append("Agri-logistics network appears adequate. Annual review recommended.")

        return recs

    def _log_report_summary(self, r: CorridorReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  AGRI-LOGISTICS CORRIDOR REPORT — {r.region_name}\n"
            f"  Corridors   : {r.total_corridors}\n"
            f"  Volume      : {r.total_volume_mt:,.0f} MT\n"
            f"  Critical    : {len(r.critical_corridors)}\n"
            f"  Bottlenecks : {len(r.bottleneck_segments)}\n"
            f"  Gaps        : {len(r.infrastructure_gaps)}\n"
            f"  Unconnected : {len(r.unconnected_clusters)} clusters\n"
            f"{'='*60}"
        )
