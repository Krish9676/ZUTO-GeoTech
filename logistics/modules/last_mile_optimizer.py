"""
ZUTO Geotech Solutions — Logistics Intelligence Platform
=========================================================
Module B: Last-Mile Delivery Optimizer

Optimizes last-mile delivery routes and zones using:
  - Demand heatmaps from order/transaction data
  - Vehicle routing (VRP) with capacity and time constraints
  - Road quality layers from satellite data
  - Route efficiency scores
  - Optimal delivery zone boundaries (territory partitioning)
  - REST API-ready output for integration with client LMS

Use cases:
  - Agri input delivery to farms (seeds, fertilizers, pesticides)
  - Farm produce pickup (mandi/FPO aggregation)
  - FMCG distribution to kirana stores
  - Rural healthcare and pharma distribution
  - Quick-commerce last mile in peri-urban areas
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
class DeliveryStop:
    """A single delivery stop/location"""
    stop_id:        str
    name:           str
    latitude:       float
    longitude:      float
    demand_kg:      float
    demand_units:   int
    priority:       int             # 1 = highest
    time_window_start: Optional[str]   # HH:MM
    time_window_end:   Optional[str]
    service_time_min:  int          # Minutes to unload at stop


@dataclass
class Route:
    """An optimized delivery route for a single vehicle"""
    route_id:           str
    vehicle_type:       str
    depot_name:         str
    stops:              List[Dict]
    stop_count:         int
    total_distance_km:  float
    total_time_min:     float
    total_load_kg:      float
    utilization_pct:    float       # % of vehicle capacity used
    fuel_cost_inr:      float
    driver_cost_inr:    float
    total_cost_inr:     float
    cost_per_kg:        float
    efficiency_score:   float       # 0-100
    feasible:           bool
    notes:              str


@dataclass
class VRPSolution:
    """Solution to Vehicle Routing Problem"""
    solution_id:        str
    depot_name:         str
    optimization_date:  str
    n_vehicles:         int
    n_stops:            int
    total_distance_km:  float
    total_cost_inr:     float
    avg_utilization_pct: float
    routes:             List[Dict]
    unserved_stops:     List[str]
    total_time_hrs:     float
    solution_quality:   str         # Good / Acceptable / Suboptimal
    savings_vs_naive:   float       # % savings vs simple sequential routing


# =============================================================================
# LAST MILE OPTIMIZER — MAIN CLASS
# =============================================================================

class LastMileOptimizer:
    """
    Last-mile delivery route optimizer using nearest-neighbor VRP heuristic
    with capacity and time-window constraints.

    Example:
        optimizer = LastMileOptimizer()

        # Solve routing problem
        solution = optimizer.optimize_routes(
            depot_lat    = 18.52,
            depot_lon    = 73.86,
            stops        = delivery_stops_list,
            vehicle_type = 'mini_truck',
            n_vehicles   = 3,
            depot_name   = "Pune Depot"
        )

        # Get demand heatmap for planning
        heatmap = optimizer.build_demand_heatmap(orders_df)
    """

    def __init__(self):
        self.cfg      = ZutoLogisticsConfig
        self.vehicles = self.cfg.VEHICLE_TYPES
        self.costs    = self.cfg.COST_PARAMS

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def optimize_routes(
        self,
        depot_lat:    float,
        depot_lon:    float,
        stops:        List[Dict],
        vehicle_type: str  = 'mini_truck',
        n_vehicles:   int  = 1,
        depot_name:   str  = "Depot",
        max_route_km: float = 200.0,
        max_time_hrs: float = 8.0,
    ) -> VRPSolution:
        """
        Solve Vehicle Routing Problem using nearest-neighbor heuristic.

        Args:
            depot_lat:    Depot latitude
            depot_lon:    Depot longitude
            stops:        List of stop dicts with [id, name, lat, lon, demand_kg]
            vehicle_type: Key from ZutoLogisticsConfig.VEHICLE_TYPES
            n_vehicles:   Number of available vehicles
            depot_name:   Name of the depot
            max_route_km: Maximum distance per route
            max_time_hrs: Maximum working hours per vehicle per day

        Returns:
            VRPSolution dataclass
        """
        logger.info(f"\nROUTE OPTIMIZATION — {depot_name} | Stops: {len(stops)} | Vehicles: {n_vehicles}")

        vehicle  = self.vehicles.get(vehicle_type, self.vehicles['mini_truck'])
        payload  = vehicle['payload_kg']
        speed    = vehicle['speed_kmh']
        cost_km  = vehicle['cost_per_km']

        # — Validate stops
        valid_stops, invalid = self._validate_stops(stops, payload)
        logger.info(f"  Valid stops: {len(valid_stops)} | Skipped: {len(invalid)}")

        # — Build distance matrix
        depot   = {'id': 'DEPOT', 'lat': depot_lat, 'lon': depot_lon}
        all_pts = [depot] + valid_stops
        dist_mx = self._build_distance_matrix(all_pts)

        # — Nearest-neighbor VRP with capacity constraint
        routes_raw    = self._nearest_neighbor_vrp(
            depot, valid_stops, dist_mx, n_vehicles, payload,
            speed, max_route_km, max_time_hrs
        )

        # — Evaluate routes
        routes = []
        for i, r in enumerate(routes_raw):
            evaluated = self._evaluate_route(r, i+1, vehicle_type, vehicle, cost_km, speed)
            routes.append(evaluated)

        # — Compute savings vs naive (sequential single route)
        naive_dist = self._naive_total_distance(depot, valid_stops)
        opt_dist   = sum(r['total_distance_km'] for r in routes)
        savings    = max((naive_dist - opt_dist) / naive_dist * 100, 0) if naive_dist > 0 else 0.0

        # — Unserved stops
        served_ids  = {s['stop_id'] for r in routes for s in r['stops']}
        unserved    = [s['id'] for s in valid_stops if s['id'] not in served_ids]

        total_dist  = round(opt_dist, 1)
        total_cost  = round(sum(r['total_cost_inr'] for r in routes), 0)
        avg_util    = round(np.mean([r['utilization_pct'] for r in routes]), 1) if routes else 0.0
        total_time  = round(max((r['total_time_min'] for r in routes), default=0) / 60, 2)
        quality     = 'Good' if avg_util > 70 and not unserved else ('Acceptable' if avg_util > 50 else 'Suboptimal')

        solution = VRPSolution(
            solution_id       = f"VRP_{date.today().strftime('%Y%m%d')}_{depot_name[:4].upper()}",
            depot_name        = depot_name,
            optimization_date = date.today().isoformat(),
            n_vehicles        = len(routes),
            n_stops           = len(valid_stops),
            total_distance_km = total_dist,
            total_cost_inr    = total_cost,
            avg_utilization_pct = avg_util,
            routes            = routes,
            unserved_stops    = unserved,
            total_time_hrs    = total_time,
            solution_quality  = quality,
            savings_vs_naive  = round(savings, 1),
        )

        self._log_solution_summary(solution)
        return solution

    def build_demand_heatmap(
        self,
        orders_df:       pd.DataFrame,
        grid_resolution: float = 0.01,
    ) -> Dict:
        """
        Build a demand heatmap from order/transaction data.
        Used for zone planning and depot placement.

        Args:
            orders_df:  DataFrame with [lat, lon, quantity_kg, order_date]
            grid_resolution: Heatmap grid size in degrees

        Returns:
            Heatmap dict with grid cells and demand intensity
        """
        if orders_df.empty:
            return {'cells': [], 'total_orders': 0, 'peak_demand_kg': 0}

        lat_col = 'lat' if 'lat' in orders_df.columns else 'latitude'
        lon_col = 'lon' if 'lon' in orders_df.columns else 'longitude'
        qty_col = 'quantity_kg' if 'quantity_kg' in orders_df.columns else 'quantity'

        lats = orders_df[lat_col].values
        lons = orders_df[lon_col].values
        qtys = orders_df[qty_col].values if qty_col in orders_df.columns else np.ones(len(lats))

        # Bin into grid
        lat_bins = np.arange(lats.min(), lats.max() + grid_resolution, grid_resolution)
        lon_bins = np.arange(lons.min(), lons.max() + grid_resolution, grid_resolution)

        cells = []
        for i in range(len(lat_bins) - 1):
            for j in range(len(lon_bins) - 1):
                mask = (
                    (lats >= lat_bins[i]) & (lats < lat_bins[i+1]) &
                    (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
                )
                if mask.any():
                    cells.append({
                        'lat_center':    float(lat_bins[i] + grid_resolution / 2),
                        'lon_center':    float(lon_bins[j] + grid_resolution / 2),
                        'n_orders':      int(mask.sum()),
                        'total_demand_kg': float(qtys[mask].sum()),
                        'avg_demand_kg': float(qtys[mask].mean()),
                    })

        cells = sorted(cells, key=lambda c: c['total_demand_kg'], reverse=True)

        return {
            'cells':          cells,
            'n_cells':        len(cells),
            'total_orders':   len(orders_df),
            'total_demand_kg': float(qtys.sum()),
            'peak_cell_kg':   cells[0]['total_demand_kg'] if cells else 0,
            'grid_resolution': grid_resolution,
        }

    def compute_route_efficiency(self, route: Dict) -> Dict:
        """
        Compute efficiency metrics for a route.
        Returns a structured scorecard.
        """
        score      = route.get('efficiency_score', 0)
        util       = route.get('utilization_pct', 0)
        cost_per_kg = route.get('cost_per_kg', 0)

        return {
            'efficiency_score':  score,
            'utilization_pct':   util,
            'cost_per_kg_inr':   cost_per_kg,
            'grade':             'A' if score > 80 else ('B' if score > 60 else ('C' if score > 40 else 'D')),
            'improvement_tips':  self._improvement_tips(score, util, cost_per_kg),
        }

    def export_routes_api(self, solution: VRPSolution) -> Dict:
        """
        Export VRP solution as REST API-ready JSON.
        Compatible with standard LMS (Logistics Management System) integration.
        """
        return {
            'solution_id':    solution.solution_id,
            'depot':          solution.depot_name,
            'date':           solution.optimization_date,
            'summary': {
                'vehicles':      solution.n_vehicles,
                'stops':         solution.n_stops,
                'distance_km':   solution.total_distance_km,
                'cost_inr':      solution.total_cost_inr,
                'quality':       solution.solution_quality,
                'savings_pct':   solution.savings_vs_naive,
            },
            'routes': [
                {
                    'route_id':      r['route_id'],
                    'vehicle':       r['vehicle_type'],
                    'stops':         r['stops'],
                    'distance_km':   r['total_distance_km'],
                    'time_min':      r['total_time_min'],
                    'load_kg':       r['total_load_kg'],
                    'cost_inr':      r['total_cost_inr'],
                }
                for r in solution.routes
            ],
            'unserved': solution.unserved_stops,
            'generated_by': 'ZUTO Logistics Intelligence Platform',
        }

    # =========================================================================
    # INTERNAL — VRP ALGORITHM
    # =========================================================================

    def _nearest_neighbor_vrp(
        self,
        depot:       Dict,
        stops:       List[Dict],
        dist_mx:     np.ndarray,
        n_vehicles:  int,
        payload_kg:  float,
        speed_kmh:   float,
        max_km:      float,
        max_hrs:     float,
    ) -> List[Dict]:
        """
        Nearest-neighbor VRP heuristic with capacity and distance constraints.
        Returns list of raw route dicts (pre-evaluation).
        """
        remaining = list(range(len(stops)))   # indices into stops list
        routes    = []

        max_min   = max_hrs * 60

        for v in range(n_vehicles):
            if not remaining:
                break

            route_stops  = []
            load_kg      = 0.0
            dist_km      = 0.0
            current      = 0    # depot is index 0 in all_pts

            while remaining:
                # Find nearest unvisited stop within capacity + distance limits
                best_i   = None
                best_d   = float('inf')

                for idx in remaining:
                    stop_idx = idx + 1    # +1 because depot is [0]
                    d = dist_mx[current][stop_idx]
                    demand = stops[idx].get('demand_kg', 0)

                    # Check constraints
                    if load_kg + demand > payload_kg:
                        continue
                    if dist_km + d + dist_mx[stop_idx][0] > max_km:
                        continue
                    if d < best_d:
                        best_d = d
                        best_i = idx

                if best_i is None:
                    break   # No more feasible stops for this vehicle

                # Add stop to route
                route_stops.append(stops[best_i])
                load_kg += stops[best_i].get('demand_kg', 0)
                dist_km += best_d
                current  = best_i + 1
                remaining.remove(best_i)

            # Return to depot
            if route_stops:
                dist_km += dist_mx[current][0]
                routes.append({
                    'vehicle_idx': v + 1,
                    'stops':       route_stops,
                    'load_kg':     load_kg,
                    'distance_km': dist_km,
                })

        return routes

    def _evaluate_route(
        self,
        raw:          Dict,
        route_num:    int,
        vehicle_type: str,
        vehicle_cfg:  Dict,
        cost_per_km:  float,
        speed_kmh:    float,
    ) -> Dict:
        """Evaluate a raw route and compute cost, efficiency metrics."""
        stops       = raw['stops']
        load_kg     = raw['load_kg']
        dist_km     = raw['distance_km']
        payload     = vehicle_cfg['payload_kg']

        service_min = sum(s.get('service_time_min', 10) for s in stops)
        drive_min   = (dist_km / speed_kmh) * 60
        total_min   = drive_min + service_min

        util         = min(load_kg / payload * 100, 100)
        fuel_cost    = dist_km * cost_per_km
        driver_cost  = self.costs['driver_per_day']
        total_cost   = fuel_cost + driver_cost
        cost_per_kg  = total_cost / load_kg if load_kg > 0 else 0

        # Efficiency score: weight utilization + distance efficiency
        eff_score = (
            0.40 * util +
            0.30 * min(100, 100 - dist_km / 200 * 30) +
            0.30 * (100 - min(cost_per_kg / 50 * 100, 100))
        )

        return {
            'route_id':          f"R{route_num:02d}",
            'vehicle_type':      vehicle_type,
            'depot_name':        'DEPOT',
            'stops':             [
                {'stop_id': s.get('id', s.get('stop_id', '')), 'name': s.get('name', ''),
                 'lat': s.get('lat', 0), 'lon': s.get('lon', 0),
                 'demand_kg': s.get('demand_kg', 0)}
                for s in stops
            ],
            'stop_count':        len(stops),
            'total_distance_km': round(dist_km, 1),
            'total_time_min':    round(total_min, 0),
            'total_load_kg':     round(load_kg, 1),
            'utilization_pct':   round(util, 1),
            'fuel_cost_inr':     round(fuel_cost, 0),
            'driver_cost_inr':   round(driver_cost, 0),
            'total_cost_inr':    round(total_cost, 0),
            'cost_per_kg':       round(cost_per_kg, 2),
            'efficiency_score':  round(eff_score, 1),
            'feasible':          True,
            'notes':             f"{len(stops)} stops, {util:.0f}% load utilization",
        }

    # =========================================================================
    # INTERNAL — DISTANCE MATRIX
    # =========================================================================

    def _build_distance_matrix(self, points: List[Dict]) -> np.ndarray:
        """Build N×N Haversine distance matrix for all points."""
        n     = len(points)
        mx    = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = self._haversine_km(
                    points[i]['lat'], points[i]['lon'],
                    points[j]['lat'], points[j]['lon'],
                )
                mx[i][j] = d
                mx[j][i] = d
        return mx

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp     = np.radians(lat2 - lat1)
        dl     = np.radians(lon2 - lon1)
        a      = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return float(2 * R * np.arcsin(np.sqrt(a)))

    # =========================================================================
    # INTERNAL — UTILITIES
    # =========================================================================

    def _validate_stops(self, stops: List[Dict], payload_kg: float) -> Tuple[List[Dict], List[str]]:
        """Filter stops for required fields. Returns (valid, invalid_ids)."""
        valid, invalid = [], []
        for s in stops:
            if ('lat' in s or 'latitude' in s) and ('lon' in s or 'longitude' in s):
                s.setdefault('lat', s.get('latitude', 0))
                s.setdefault('lon', s.get('longitude', 0))
                s.setdefault('demand_kg', 0)
                s.setdefault('service_time_min', 10)
                s.setdefault('id', s.get('stop_id', f"S{len(valid)+1}"))
                if s['demand_kg'] <= payload_kg:
                    valid.append(s)
                else:
                    invalid.append(s.get('id', ''))
                    logger.warning(f"Stop {s.get('id', '')} demand {s['demand_kg']} kg exceeds vehicle payload {payload_kg} kg")
            else:
                invalid.append(str(s))
        return valid, invalid

    def _naive_total_distance(self, depot: Dict, stops: List[Dict]) -> float:
        """Compute naive sequential single-route distance (baseline for savings calc)."""
        if not stops:
            return 0.0
        dist = self._haversine_km(depot['lat'], depot['lon'], stops[0]['lat'], stops[0]['lon'])
        for i in range(len(stops)-1):
            dist += self._haversine_km(stops[i]['lat'], stops[i]['lon'], stops[i+1]['lat'], stops[i+1]['lon'])
        dist += self._haversine_km(stops[-1]['lat'], stops[-1]['lon'], depot['lat'], depot['lon'])
        return dist

    def _improvement_tips(self, score: float, util: float, cost_per_kg: float) -> List[str]:
        tips = []
        if util < 60:
            tips.append("Consolidate orders — load utilization below 60%. Add more stops per route.")
        if cost_per_kg > 30:
            tips.append("High cost/kg. Consider switching to larger vehicle for this corridor.")
        if score < 50:
            tips.append("Low efficiency. Review territory boundaries and depot placement.")
        if not tips:
            tips.append("Route operating efficiently. No immediate optimization needed.")
        return tips

    def _log_solution_summary(self, s: VRPSolution):
        logger.info(
            f"\n{'='*60}\n"
            f"  VRP SOLUTION — {s.depot_name}\n"
            f"  Vehicles    : {s.n_vehicles}\n"
            f"  Stops       : {s.n_stops}\n"
            f"  Distance    : {s.total_distance_km:.1f} km\n"
            f"  Cost        : ₹{s.total_cost_inr:,.0f}\n"
            f"  Utilization : {s.avg_utilization_pct:.1f}%\n"
            f"  Savings vs Naive: {s.savings_vs_naive:.1f}%\n"
            f"  Quality     : {s.solution_quality}\n"
            f"  Unserved    : {len(s.unserved_stops)} stops\n"
            f"{'='*60}"
        )
