"""
ZUTO Geotech Solutions — Logistics Intelligence Platform
=========================================================
Module E: Delivery Zone Manager

Interactive tool for defining and optimizing delivery zones / service areas.

Features:
  - Isochrone computation (reachability zones) from a depot
  - Multi-depot territory optimization (balanced zone boundaries)
  - Time-based, distance-based, and demand-based zone splitting
  - Overlap detection and gap identification between zones
  - Dynamic re-zoning when new depots or orders are added
  - REST API output for integration with LMS / dispatch systems

Use cases:
  - Quick-commerce (Blinkit, Zepto): 30-min delivery zones per dark store
  - Food delivery (Swiggy, Zomato): 45-min isochrones
  - Pharma distribution: 4-hour cold-chain zones
  - Agri last-mile: Same-day farm-to-doorstep service areas
  - FPO aggregation: Farmer membership service zones
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from logistics_config import ZutoLogisticsConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DeliveryZone:
    """A single delivery zone / service area"""
    zone_id:            str
    zone_name:          str
    depot_id:           str
    depot_name:         str
    depot_lat:          float
    depot_lon:          float
    delivery_mode:      str         # quick_commerce / food_delivery / agri_last_mile etc.
    time_limit_min:     int
    radius_km:          float       # Effective radius
    area_sq_km:         float
    estimated_demand:   float       # Total demand within zone (units/day)
    estimated_orders:   int
    vehicle_type:       str
    n_vehicles_required: int
    operational_cost_inr: float     # Daily operating cost
    coverage_pct:       float       # % of zone area reachable within time limit
    boundary_points:    List[Dict]  # List of {lat, lon} forming zone boundary
    color:              str         # HEX color for map display


@dataclass
class ZoneOptimizationResult:
    """Result of multi-depot zone optimization"""
    optimization_id:    str
    region_name:        str
    optimization_date:  str
    delivery_mode:      str
    n_depots:           int
    n_zones:            int
    total_area_sq_km:   float
    coverage_pct:       float       # % of demand covered
    zones:              List[Dict]
    gaps:               List[Dict]  # Uncovered demand hotspots
    overlaps:           List[Dict]  # Zone overlap areas
    total_daily_cost:   float
    avg_delivery_time:  float
    recommendations:    List[str]


# =============================================================================
# DELIVERY ZONE MANAGER — MAIN CLASS
# =============================================================================

class DeliveryZoneManager:
    """
    Computes and optimizes delivery zones for single or multi-depot networks.

    Example:
        manager = DeliveryZoneManager()

        # Single depot isochrone
        zone = manager.compute_isochrone(
            depot_lat     = 18.52,
            depot_lon     = 73.86,
            depot_name    = "Hadapsar Dark Store",
            delivery_mode = "quick_commerce",
        )

        # Multi-depot territory optimization
        result = manager.optimize_territories(
            depots        = depots_list,
            demand_heatmap = optimizer.build_demand_heatmap(orders_df),
            delivery_mode  = "food_delivery",
            region_name    = "Pune City",
        )
    """

    def __init__(self):
        self.cfg    = ZutoLogisticsConfig
        self.modes  = self.cfg.DELIVERY_ZONE_MODES
        self.vehicles = self.cfg.VEHICLE_TYPES
        self.max_zones = self.cfg.MAX_ZONES_PER_DEPOT

        # Zone colors for map display
        self._zone_colors = [
            '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
            '#ff7f00', '#a65628', '#f781bf', '#999999',
            '#1f78b4', '#33a02c', '#fb9a99', '#a6cee3',
        ]

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def compute_isochrone(
        self,
        depot_lat:      float,
        depot_lon:      float,
        depot_name:     str  = "Depot",
        depot_id:       str  = "DEPOT_001",
        delivery_mode:  str  = "quick_commerce",
        road_quality:   str  = "moderate",
    ) -> DeliveryZone:
        """
        Compute delivery isochrone for a single depot.

        An isochrone is the set of all points reachable within a given
        time limit from the depot, considering road speeds.

        Args:
            depot_lat:      Depot latitude
            depot_lon:      Depot longitude
            depot_name:     Human-readable depot name
            depot_id:       Unique depot ID
            delivery_mode:  Key from ZutoLogisticsConfig.DELIVERY_ZONE_MODES
            road_quality:   'good' | 'moderate' | 'poor'

        Returns:
            DeliveryZone dataclass
        """
        mode_cfg    = self.modes.get(delivery_mode, self.modes['quick_commerce'])
        vehicle_key = mode_cfg['vehicle']
        vehicle_cfg = self.vehicles.get(vehicle_key, self.vehicles['two_wheeler'])
        time_min    = mode_cfg['time_limit_min']

        # Effective speed after road quality factor
        roughness   = self.cfg.ROUGHNESS_FACTOR.get(road_quality, 1.25)
        eff_speed   = vehicle_cfg['speed_kmh'] / roughness

        # Radius = half time limit (round trip buffer)
        radius_km   = (eff_speed * time_min / 60) * 0.5

        # Area approximation
        area_sq_km  = np.pi * radius_km ** 2

        # Coverage = fraction reachable given road network completeness
        coverage_pct = self._estimate_coverage(road_quality, delivery_mode)

        # Isochrone boundary (approximate polygon — production: use OSRM/Valhalla)
        boundary    = self._compute_isochrone_boundary(depot_lat, depot_lon, radius_km)

        # Demand estimate
        demand_kg, n_orders = self._estimate_zone_demand(area_sq_km, delivery_mode)

        # Vehicle count required
        n_vehicles  = self._vehicles_required(n_orders, time_min, vehicle_cfg)

        # Daily cost
        daily_cost  = n_vehicles * (
            radius_km * 2 * vehicle_cfg['cost_per_km'] +
            self.cfg.COST_PARAMS['driver_per_day']
        )

        zone = DeliveryZone(
            zone_id              = f"ZONE_{depot_id}_{delivery_mode.upper()[:3]}",
            zone_name            = f"{depot_name} — {mode_cfg['name']}",
            depot_id             = depot_id,
            depot_name           = depot_name,
            depot_lat            = depot_lat,
            depot_lon            = depot_lon,
            delivery_mode        = delivery_mode,
            time_limit_min       = time_min,
            radius_km            = round(radius_km, 2),
            area_sq_km           = round(area_sq_km, 2),
            estimated_demand     = round(demand_kg, 1),
            estimated_orders     = n_orders,
            vehicle_type         = vehicle_key,
            n_vehicles_required  = n_vehicles,
            operational_cost_inr = round(daily_cost, 0),
            coverage_pct         = round(coverage_pct, 1),
            boundary_points      = boundary,
            color                = self._zone_colors[0],
        )

        logger.info(
            f"Isochrone computed: {depot_name} | Mode: {delivery_mode} | "
            f"Radius: {radius_km:.1f} km | Area: {area_sq_km:.1f} km²"
        )
        return zone

    def optimize_territories(
        self,
        depots:         List[Dict],
        demand_heatmap: Optional[Dict] = None,
        delivery_mode:  str  = "food_delivery",
        region_name:    str  = "Region",
        balance_demand: bool = True,
    ) -> ZoneOptimizationResult:
        """
        Optimize delivery territories for a multi-depot network.
        Assigns demand cells to nearest depot, balancing workload.

        Args:
            depots:         List of depot dicts with [id, name, lat, lon]
            demand_heatmap: Output from LastMileOptimizer.build_demand_heatmap()
            delivery_mode:  Delivery mode key
            region_name:    Region name
            balance_demand: If True, equalize demand per zone; else use nearest-depot

        Returns:
            ZoneOptimizationResult dataclass
        """
        logger.info(f"\nTERRITORY OPTIMIZATION — {region_name} | Depots: {len(depots)} | Mode: {delivery_mode}")

        mode_cfg = self.modes.get(delivery_mode, self.modes['food_delivery'])
        zones    = []
        gaps     = []
        overlaps = []

        # — Compute base isochrones per depot
        for i, depot in enumerate(depots):
            zone = self.compute_isochrone(
                depot_lat     = depot.get('lat', depot.get('latitude', 0)),
                depot_lon     = depot.get('lon', depot.get('longitude', 0)),
                depot_name    = depot.get('name', f"Depot {i+1}"),
                depot_id      = depot.get('id', f"D{i+1}"),
                delivery_mode = delivery_mode,
            )
            zone_dict = self._zone_to_dict(zone, color=self._zone_colors[i % len(self._zone_colors)])
            zones.append(zone_dict)

        # — Assign demand heatmap cells to zones
        if demand_heatmap and demand_heatmap.get('cells'):
            zones, gaps = self._assign_demand_to_zones(zones, demand_heatmap, depots, mode_cfg)

        # — Balance demand across zones if requested
        if balance_demand and len(zones) > 1:
            zones = self._balance_zones(zones)

        # — Detect overlaps
        overlaps = self._detect_overlaps(zones)

        # — Aggregate metrics
        total_area = sum(z.get('area_sq_km', 0) for z in zones)
        total_demand = demand_heatmap.get('total_demand_kg', 0) if demand_heatmap else 0
        covered_demand = sum(z.get('assigned_demand_kg', z.get('estimated_demand', 0)) for z in zones)
        coverage = min(covered_demand / total_demand * 100, 100) if total_demand > 0 else 80.0

        avg_time  = mode_cfg['time_limit_min'] * 0.7   # avg delivery is 70% of max
        total_cost = sum(z.get('operational_cost_inr', 0) for z in zones)

        result = ZoneOptimizationResult(
            optimization_id   = f"OPT_{date.today().strftime('%Y%m%d')}_{delivery_mode[:3].upper()}",
            region_name       = region_name,
            optimization_date = date.today().isoformat(),
            delivery_mode     = delivery_mode,
            n_depots          = len(depots),
            n_zones           = len(zones),
            total_area_sq_km  = round(total_area, 1),
            coverage_pct      = round(coverage, 1),
            zones             = zones,
            gaps              = gaps,
            overlaps          = overlaps,
            total_daily_cost  = round(total_cost, 0),
            avg_delivery_time = round(avg_time, 0),
            recommendations   = self._generate_recommendations(zones, gaps, overlaps, coverage, delivery_mode),
        )

        self._log_optimization_summary(result)
        return result

    def rebalance_zones(
        self,
        current_zones: List[Dict],
        new_orders:    Optional[pd.DataFrame] = None,
        max_imbalance: float = 0.30,
    ) -> List[Dict]:
        """
        Re-balance existing zones when demand shifts or new orders arrive.
        Zones with demand > (1 + max_imbalance) × average are split.

        Args:
            current_zones: List of existing zone dicts
            new_orders:    New order DataFrame to update demand
            max_imbalance: Allowed demand imbalance ratio (0.30 = ±30%)

        Returns:
            Rebalanced zone list
        """
        if not current_zones:
            return current_zones

        avg_demand = np.mean([z.get('estimated_demand', 0) for z in current_zones])
        rebalanced = []

        for zone in current_zones:
            zone_demand = zone.get('estimated_demand', 0)
            if zone_demand > avg_demand * (1 + max_imbalance):
                # Zone overloaded — mark for split recommendation
                zone['needs_split'] = True
                zone['split_reason'] = f"Demand {zone_demand:.0f} exceeds avg {avg_demand:.0f} by >{max_imbalance*100:.0f}%"
            elif zone_demand < avg_demand * (1 - max_imbalance):
                zone['needs_merge'] = True
                zone['merge_reason'] = f"Demand {zone_demand:.0f} below avg {avg_demand:.0f} by >{max_imbalance*100:.0f}%"
            rebalanced.append(zone)

        return rebalanced

    def export_zones_geojson(self, zones: List[Dict], output_path: str) -> str:
        """Export delivery zones as GeoJSON FeatureCollection."""
        features = []
        for zone in zones:
            boundary = zone.get('boundary_points', [])
            if boundary:
                coords = [[p['lon'], p['lat']] for p in boundary]
                coords.append(coords[0])   # Close polygon
                geometry = {'type': 'Polygon', 'coordinates': [coords]}
            else:
                geometry = None

            features.append({
                'type': 'Feature',
                'geometry': geometry,
                'properties': {
                    'zone_id':          zone.get('zone_id'),
                    'zone_name':        zone.get('zone_name'),
                    'depot_name':       zone.get('depot_name'),
                    'delivery_mode':    zone.get('delivery_mode'),
                    'radius_km':        zone.get('radius_km'),
                    'time_limit_min':   zone.get('time_limit_min'),
                    'coverage_pct':     zone.get('coverage_pct'),
                    'fill_color':       zone.get('color', '#999999'),
                    'estimated_orders': zone.get('estimated_orders'),
                },
            })

        geojson = {
            'type': 'FeatureCollection',
            'properties': {'generated_by': 'ZUTO Geotech Solutions'},
            'features': features,
        }

        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Zones GeoJSON exported → {output_path}")
        return output_path

    def export_api_response(self, result: ZoneOptimizationResult) -> Dict:
        """Export optimization result as REST API JSON response."""
        return {
            'optimization_id': result.optimization_id,
            'region':          result.region_name,
            'date':            result.optimization_date,
            'mode':            result.delivery_mode,
            'summary': {
                'n_depots':    result.n_depots,
                'n_zones':     result.n_zones,
                'coverage_pct': result.coverage_pct,
                'total_area_km2': result.total_area_sq_km,
                'daily_cost_inr': result.total_daily_cost,
                'avg_delivery_min': result.avg_delivery_time,
            },
            'zones': [
                {
                    'zone_id':    z['zone_id'],
                    'depot':      z.get('depot_name'),
                    'radius_km':  z.get('radius_km'),
                    'area_km2':   z.get('area_sq_km'),
                    'orders':     z.get('estimated_orders'),
                    'coverage_pct': z.get('coverage_pct'),
                    'color':      z.get('color'),
                    'boundary':   z.get('boundary_points', [])[:10],  # truncate for API
                }
                for z in result.zones
            ],
            'gaps':     result.gaps[:5],
            'generated_by': 'ZUTO Logistics Intelligence Platform',
        }

    # =========================================================================
    # INTERNAL — ISOCHRONE & GEOMETRY
    # =========================================================================

    def _compute_isochrone_boundary(
        self,
        lat: float,
        lon: float,
        radius_km: float,
        n_points: int = 36,
    ) -> List[Dict]:
        """
        Compute approximate circular isochrone boundary.
        In production: query OSRM or Valhalla isochrone API for real road-network shape.
        Here: approximated as ellipse (to account for road network vs crow-fly ratio = 1.3).
        """
        # Road network factor: actual road distance ≈ 1.3× crow-fly
        net_factor = 1.3
        true_radius = radius_km / net_factor

        boundary = []
        for i in range(n_points):
            angle = np.radians(i * 360 / n_points)
            # Slight ellipse to mimic road network shape
            r_lat = true_radius
            r_lon = true_radius * 1.1   # roads often faster east-west

            dlat  = (r_lat * np.cos(angle)) / 111.0
            dlon  = (r_lon * np.sin(angle)) / (111.0 * np.cos(np.radians(lat)))
            boundary.append({
                'lat': round(lat + dlat, 5),
                'lon': round(lon + dlon, 5),
            })

        return boundary

    def _estimate_coverage(self, road_quality: str, mode: str) -> float:
        """Estimate % of isochrone area actually reachable within time limit."""
        base = {'good': 90, 'moderate': 75, 'poor': 55, 'very_poor': 35}.get(road_quality, 70)
        mode_adj = {
            'quick_commerce': -5,    # Dense urban — some lanes inaccessible
            'food_delivery':  -5,
            'agri_last_mile': -15,   # Rural roads highly variable
            'b2b_distribution': 0,
        }.get(mode, 0)
        return float(np.clip(base + mode_adj, 20, 98))

    # =========================================================================
    # INTERNAL — DEMAND ASSIGNMENT
    # =========================================================================

    def _estimate_zone_demand(self, area_sq_km: float, mode: str) -> Tuple[float, int]:
        """Estimate daily demand within a zone from area and mode density."""
        # Density assumptions (orders/km²/day) by mode
        density = {
            'quick_commerce': 15,
            'food_delivery':  20,
            'agri_last_mile': 3,
            'b2b_distribution': 5,
            'pharma': 8,
        }.get(mode, 10)
        orders   = int(area_sq_km * density)
        avg_kg   = {'quick_commerce': 2, 'food_delivery': 1.5, 'agri_last_mile': 50, 'b2b_distribution': 200}.get(mode, 10)
        demand_kg = orders * avg_kg
        return float(demand_kg), orders

    def _assign_demand_to_zones(
        self,
        zones:    List[Dict],
        heatmap:  Dict,
        depots:   List[Dict],
        mode_cfg: Dict,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Assign heatmap demand cells to nearest depot (within time radius)."""
        cells = heatmap.get('cells', [])
        gaps  = []

        for zone in zones:
            zone['assigned_demand_kg'] = 0.0
            zone['assigned_cells']     = 0

        for cell in cells:
            clat, clon = cell['lat_center'], cell['lon_center']
            assigned = False

            for i, depot in enumerate(depots):
                dlat = depot.get('lat', depot.get('latitude', 0))
                dlon = depot.get('lon', depot.get('longitude', 0))
                dist = self._haversine_km(dlat, dlon, clat, clon)

                if dist <= zones[i]['radius_km']:
                    zones[i]['assigned_demand_kg'] = zones[i].get('assigned_demand_kg', 0) + cell['total_demand_kg']
                    zones[i]['assigned_cells']     = zones[i].get('assigned_cells', 0) + 1
                    assigned = True
                    break

            if not assigned:
                gaps.append({
                    'lat':     clat, 'lon': clon,
                    'demand_kg': cell['total_demand_kg'],
                    'note':    'Outside all depot service zones',
                })

        # Update zone demand from assignment
        for zone in zones:
            if zone.get('assigned_demand_kg', 0) > 0:
                zone['estimated_demand'] = zone['assigned_demand_kg']

        return zones, gaps

    def _balance_zones(self, zones: List[Dict]) -> List[Dict]:
        """Soft balance demand: adjust zone radii proportionally."""
        if len(zones) < 2:
            return zones

        demands = [z.get('estimated_demand', 1) for z in zones]
        avg_dem = np.mean(demands)

        for zone in zones:
            demand = zone.get('estimated_demand', avg_dem)
            ratio  = avg_dem / demand if demand > 0 else 1.0
            # Slightly scale radius to compensate (±15% max)
            adj = np.clip(ratio, 0.85, 1.15)
            zone['radius_km']   = round(zone.get('radius_km', 3.0) * adj, 2)
            zone['area_sq_km']  = round(np.pi * zone['radius_km'] ** 2, 2)
            zone['balanced']    = True

        return zones

    def _detect_overlaps(self, zones: List[Dict]) -> List[Dict]:
        """Detect zone pairs with overlapping radii."""
        overlaps = []
        for i in range(len(zones)):
            for j in range(i+1, len(zones)):
                z1, z2 = zones[i], zones[j]
                dist = self._haversine_km(
                    z1['depot_lat'], z1['depot_lon'],
                    z2['depot_lat'], z2['depot_lon'],
                )
                r1 = z1.get('radius_km', 0)
                r2 = z2.get('radius_km', 0)
                if dist < r1 + r2:
                    overlap_km2 = max(0, (r1 + r2 - dist) ** 2 * np.pi * 0.5)
                    overlaps.append({
                        'zone_1':       z1['zone_id'],
                        'zone_2':       z2['zone_id'],
                        'distance_km':  round(dist, 2),
                        'overlap_km2':  round(overlap_km2, 2),
                        'pct_z1':       round(overlap_km2 / (np.pi * r1**2) * 100, 1) if r1 > 0 else 0,
                        'recommendation': f"Adjust boundary between {z1['depot_name']} and {z2['depot_name']} depots to eliminate {overlap_km2:.1f} km² overlap.",
                    })
        return overlaps

    def _vehicles_required(
        self, n_orders: int, time_min: int, vehicle_cfg: Dict
    ) -> int:
        """Estimate vehicles needed to serve zone orders within time limit."""
        orders_per_vehicle = max(int(time_min / 15), 1)   # ~15 min per stop
        return max(int(np.ceil(n_orders / orders_per_vehicle)), 1)

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _zone_to_dict(self, zone: DeliveryZone, color: str = '#999999') -> Dict:
        """Convert DeliveryZone dataclass to dict."""
        return {
            'zone_id':             zone.zone_id,
            'zone_name':           zone.zone_name,
            'depot_id':            zone.depot_id,
            'depot_name':          zone.depot_name,
            'depot_lat':           zone.depot_lat,
            'depot_lon':           zone.depot_lon,
            'delivery_mode':       zone.delivery_mode,
            'time_limit_min':      zone.time_limit_min,
            'radius_km':           zone.radius_km,
            'area_sq_km':          zone.area_sq_km,
            'estimated_demand':    zone.estimated_demand,
            'estimated_orders':    zone.estimated_orders,
            'vehicle_type':        zone.vehicle_type,
            'n_vehicles_required': zone.n_vehicles_required,
            'operational_cost_inr': zone.operational_cost_inr,
            'coverage_pct':        zone.coverage_pct,
            'boundary_points':     zone.boundary_points,
            'color':               color,
        }

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp, dl = np.radians(lat2-lat1), np.radians(lon2-lon1)
        a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return float(2 * R * np.arcsin(np.sqrt(a)))

    def _generate_recommendations(
        self,
        zones:     List[Dict],
        gaps:      List[Dict],
        overlaps:  List[Dict],
        coverage:  float,
        mode:      str,
    ) -> List[str]:
        recs = []
        if coverage < 80:
            uncovered_demand = sum(g.get('demand_kg', 0) for g in gaps)
            recs.append(
                f"Coverage at {coverage:.1f}% — {len(gaps)} demand hotspots outside all zones "
                f"({uncovered_demand:.0f} kg/day uncovered). "
                f"Consider adding {'1-2' if len(gaps) < 5 else '3+'} new depots."
            )
        if overlaps:
            total_overlap = sum(o['overlap_km2'] for o in overlaps)
            recs.append(
                f"{len(overlaps)} zone overlap(s) totaling {total_overlap:.1f} km². "
                f"Adjust boundaries to eliminate duplication and improve resource utilization."
            )
        if mode == 'quick_commerce':
            recs.append("Quick commerce zones: target ≥90% coverage of demand within 1.5 km radius. Review monthly as order patterns shift.")
        elif mode == 'agri_last_mile':
            recs.append("Agri zones: coordinate with FPO calendar for peak harvest periods. Increase vehicle allocation Oct–Nov (Kharif) and Mar–Apr (Rabi).")
        if not recs:
            recs.append("Zone configuration looks optimal. Re-run optimization monthly to account for demand drift.")
        return recs

    def _log_optimization_summary(self, r: ZoneOptimizationResult):
        logger.info(
            f"\n{'='*60}\n"
            f"  ZONE OPTIMIZATION — {r.region_name}\n"
            f"  Mode        : {r.delivery_mode}\n"
            f"  Depots      : {r.n_depots}\n"
            f"  Zones       : {r.n_zones}\n"
            f"  Coverage    : {r.coverage_pct:.1f}%\n"
            f"  Total Area  : {r.total_area_sq_km:.1f} km²\n"
            f"  Daily Cost  : ₹{r.total_daily_cost:,.0f}\n"
            f"  Avg Delivery: {r.avg_delivery_time:.0f} min\n"
            f"  Gaps        : {len(r.gaps)} hotspots\n"
            f"  Overlaps    : {len(r.overlaps)} pairs\n"
            f"{'='*60}"
        )
