"""
╔══════════════════════════════════════════════════════════════════╗
║   ZUTO GEOTECH SOLUTIONS — AgriTech Platform                    ║
║   Component 1 — Module E: Market Analytics                      ║
╚══════════════════════════════════════════════════════════════════╝

WHAT IT DOES:
  Overlays crop production maps with market infrastructure layers
  to produce farmer-actionable market intelligence:

    1. MarketProximityAnalyzer  — Nearest APMC/mandi, cold storage,
                                   processing unit per field
    2. PriceIntelligence        — Real-time + historical MSP/mandi prices
    3. CropProductionHeatmap    — Aggregates yield forecasts → production map
    4. MarketLinkageScorer      — Scores farmer's market access (0–100)
    5. SupplyChainOptimizer     — Recommends best market route for produce

INPUTS:
  - field_location      : (lat, lon) of farm
  - crop_name           : detected/known crop
  - yield_result        : from YieldForecaster
  - market_data         : dict of {facility_type: [{name, lat, lon, ...}]}
                          (loaded from CSV, GeoJSON, or government API)
  - bbox                : for regional heatmap generation

KEY UNIQUENESS FOR ZUTO:
  This is the cross-component bridge between AgriTech + Logistics.
  The same market database feeds the Logistics corridor mapping module.

HOW TO USE IN COLAB:
  market = MarketAnalytics(market_data=market_db)

  # Field-level
  result = market.analyze_field_market_access(
      latitude=18.52, longitude=73.85,
      crop_name='Onion', yield_result=yield_result
  )

  # Regional heatmap
  heatmap = market.generate_production_heatmap(
      field_results=[...], bbox=[73.0, 18.0, 74.0, 19.0]
  )
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import math
import warnings
warnings.filterwarnings('ignore')

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

# Facility types ZUTO tracks in its market database
FACILITY_TYPES = [
    'APMC_Mandi',
    'Procurement_Centre',
    'Cold_Storage',
    'Processing_Unit',
    'Retail_Aggregator',
    'Export_Hub',
    'FPO_Aggregation',
]

# Maximum practical travel distance (km) by facility type
MAX_SEARCH_RADIUS_KM = {
    'APMC_Mandi':         30,
    'Procurement_Centre': 20,
    'Cold_Storage':       50,
    'Processing_Unit':    60,
    'Retail_Aggregator':  25,
    'Export_Hub':         150,
    'FPO_Aggregation':    15,
}

# Crops that MUST have cold storage access (perishables)
COLD_CHAIN_CROPS = [
    'Potato', 'Onion', 'Tomato', 'Cabbage', 'Chilli', 'Grapes',
    'Banana', 'Papaya', 'Pomegranate',
]

# Government MSP (₹/quintal) — FY 2023-24 CACP norms
MSP_PRICES = {
    'Bajra':     2350,  'Rice':     2183,  'Wheat':    2275,
    'Jowar':     2970,  'Maize':    1850,  'Gram':     5440,
    'Tur':       7000,  'Mustard':  5650,  'Groundnut': 6377,
    'Soyabean':  4600,  'Sunflower': 6760, 'Cotton':   6620,
    'Sugarcane':  340,
}

# Market access scoring weights
ACCESS_SCORE_WEIGHTS = {
    'nearest_mandi_dist':    0.30,  # distance to APMC mandi
    'cold_storage_access':   0.20,  # cold storage within range (for perishables)
    'processing_access':     0.15,  # processing unit access
    'price_premium_access':  0.15,  # higher-value market (export/retail)
    'road_connectivity':     0.10,  # road quality proxy (from DEM/LULC)
    'market_competition':    0.10,  # number of markets competing = more options
}


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 1 OF 5: MarketProximityAnalyzer
# ═════════════════════════════════════════════════════════════════════════════

class MarketProximityAnalyzer:
    """
    Finds nearest market facilities for a field using Haversine distance.

    Input: market_data dict structure:
      {
        'APMC_Mandi': [
          {'name': 'Pune APMC', 'lat': 18.52, 'lon': 73.85,
           'crops': ['Onion', 'Tomato'], 'capacity_t': 5000},
          ...
        ],
        'Cold_Storage': [...],
        ...
      }
    """

    def __init__(self, market_data: Dict[str, List[Dict]]):
        self.market_data = market_data
        self._validate_market_data()

    def find_nearest(
        self,
        latitude:  float,
        longitude: float,
        crop_name: Optional[str] = None,
        top_n:     int = 3,
    ) -> Dict[str, List[Dict]]:
        """
        Find nearest facilities of each type for a farm location.

        Args:
            latitude, longitude: Farm location.
            crop_name: Filter mandis by crop compatibility.
            top_n: Return top N nearest per facility type.

        Returns:
            {
              'APMC_Mandi':   [{name, distance_km, travel_time_min, ...}],
              'Cold_Storage': [...],
              ...
              'summary': {nearest_mandi_km, has_cold_storage, ...}
            }
        """
        results  = {}
        summary  = {}

        for ftype, facilities in self.market_data.items():
            if not facilities:
                results[ftype] = []
                continue

            scored = []
            for fac in facilities:
                dist_km = self._haversine(
                    latitude, longitude,
                    fac.get('lat', 0), fac.get('lon', 0)
                )
                max_r = MAX_SEARCH_RADIUS_KM.get(ftype, 100)
                if dist_km > max_r:
                    continue

                # Crop compatibility filter for mandis
                if crop_name and ftype == 'APMC_Mandi':
                    supported = fac.get('crops', [])
                    if supported and crop_name not in supported:
                        continue  # skip mandis that don't trade this crop

                travel_min = self._estimate_travel_time(dist_km, road_type='rural')

                scored.append({
                    **fac,
                    'distance_km':       round(dist_km, 2),
                    'travel_time_min':   round(travel_min, 0),
                    'within_max_radius': True,
                })

            scored.sort(key=lambda x: x['distance_km'])
            results[ftype] = scored[:top_n]

        # Build summary
        mandis      = results.get('APMC_Mandi', [])
        cold_stores = results.get('Cold_Storage', [])
        procs       = results.get('Processing_Unit', [])

        summary['nearest_mandi_km']       = mandis[0]['distance_km']    if mandis      else None
        summary['nearest_mandi_name']     = mandis[0].get('name', '?')  if mandis      else None
        summary['nearest_cold_store_km']  = cold_stores[0]['distance_km'] if cold_stores else None
        summary['has_cold_storage_30km']  = any(c['distance_km'] <= 30 for c in cold_stores)
        summary['has_processing_50km']    = any(p['distance_km'] <= 50 for p in procs)
        summary['n_mandis_available']     = len(mandis)
        summary['n_cold_stores_available'] = len(cold_stores)

        results['summary'] = summary
        return results

    def find_best_market(
        self,
        latitude:  float,
        longitude: float,
        crop_name: str,
        quantity_t: float,
        price_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Recommend the single BEST market considering price + distance.

        Score = price_score × 0.6 + proximity_score × 0.4
        """
        nearby = self.find_nearest(latitude, longitude, crop_name, top_n=5)
        mandis = nearby.get('APMC_Mandi', [])

        if not mandis:
            return {'recommendation': 'No mandi found within range',
                    'fallback': 'Contact FPO or direct buyer'}

        scored = []
        for mandi in mandis:
            dist    = mandi['distance_km']
            prox_sc = max(0, 100 - dist * 2)   # 100 at 0km, 0 at 50km

            # Price score
            mandi_price = None
            if price_data:
                mandi_price = price_data.get(mandi.get('name', ''), {}).get(crop_name)
            if not mandi_price:
                mandi_price = MSP_PRICES.get(crop_name, 2000)

            # Normalize price: assume range of MSP ± 50%
            msp = MSP_PRICES.get(crop_name, 2000)
            price_sc = min(100, (mandi_price / max(msp, 1)) * 70)

            total_sc = price_sc * 0.6 + prox_sc * 0.4

            # Transport cost estimate (₹/tonne/km × distance × quantity)
            transport_cost = dist * 2.5 * quantity_t  # ₹2.5/t/km approx

            revenue = mandi_price * quantity_t * 10  # ₹ (price/quintal × tonnes × 10)
            net_revenue = revenue - transport_cost

            scored.append({
                **mandi,
                'price_per_quintal':  mandi_price,
                'total_score':        round(total_sc, 1),
                'transport_cost_inr': round(transport_cost, 0),
                'gross_revenue_inr':  round(revenue, 0),
                'net_revenue_inr':    round(net_revenue, 0),
            })

        scored.sort(key=lambda x: x['total_score'], reverse=True)
        best = scored[0]

        return {
            'recommended_market': best,
            'alternatives':       scored[1:3],
            'crop':               crop_name,
            'quantity_t':         quantity_t,
            'decision_factors':   {
                'proximity_weight': '40%',
                'price_weight':     '60%',
                'transport_cost_considered': True,
            },
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in km."""
        R = 6371.0
        φ1, φ2 = math.radians(lat1), math.radians(lat2)
        dφ = math.radians(lat2 - lat1)
        dλ = math.radians(lon2 - lon1)
        a  = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    @staticmethod
    def _estimate_travel_time(dist_km: float, road_type: str = 'rural') -> float:
        """Estimate travel time in minutes based on road type."""
        speed = {'highway': 70, 'paved': 50, 'rural': 30, 'kutcha': 15}
        return (dist_km / speed.get(road_type, 30)) * 60

    def _validate_market_data(self):
        """Basic validation of market_data structure."""
        for ftype, facilities in self.market_data.items():
            if not isinstance(facilities, list):
                raise ValueError(f"market_data['{ftype}'] must be a list")


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 2 OF 5: PriceIntelligence
# ═════════════════════════════════════════════════════════════════════════════

class PriceIntelligence:
    """
    Fetches and analyzes mandi price data for crops.

    Data sources:
      - Agmarknet API (government — free, real-time India mandi prices)
      - Cached CSV of historical prices
      - MSP fallback if API unavailable

    Use:
      prices = PriceIntelligence()
      current = prices.get_current_price('Onion', 'Lasalgaon APMC')
      history = prices.get_price_history('Wheat', days=90)
      signal  = prices.get_market_signal('Wheat', production_forecast_t=50000)
    """

    AGMARKNET_API = "https://agmarknet.gov.in/api"   # real API endpoint

    def __init__(
        self,
        price_csv:  Optional[str] = None,
        use_api:    bool = False,
        verbose:    bool = True,
    ):
        """
        Args:
            price_csv: Path to historical price CSV.
                       Columns: date, market, state, crop, price_per_quintal.
            use_api:   If True, attempt live Agmarknet API fetch.
            verbose:   Print status.
        """
        self.verbose  = verbose
        self.use_api  = use_api
        self.price_df = pd.DataFrame()

        if price_csv:
            self._load_price_csv(price_csv)

    def get_current_price(
        self, crop_name: str, market_name: Optional[str] = None
    ) -> Dict:
        """
        Get most recent price for a crop.

        Returns:
            {price_per_quintal, date, market, vs_msp_pct, trend_7d}
        """
        if self.use_api:
            api_result = self._fetch_agmarknet(crop_name, market_name)
            if api_result:
                return api_result

        if not self.price_df.empty:
            filtered = self.price_df[self.price_df['crop'] == crop_name]
            if market_name:
                filtered = filtered[filtered['market'].str.contains(market_name, na=False)]
            if not filtered.empty:
                latest = filtered.sort_values('date').iloc[-1]
                msp    = MSP_PRICES.get(crop_name, 0)
                vs_msp = ((latest['price_per_quintal'] - msp) / max(msp, 1) * 100
                          if msp else None)
                return {
                    'price_per_quintal': float(latest['price_per_quintal']),
                    'date':              str(latest.get('date', '?')),
                    'market':            str(latest.get('market', '?')),
                    'vs_msp_pct':        round(vs_msp, 1) if vs_msp else None,
                    'source':            'csv',
                }

        # MSP fallback
        msp = MSP_PRICES.get(crop_name)
        if msp:
            return {
                'price_per_quintal': msp,
                'date':              datetime.now().strftime('%Y-%m-%d'),
                'market':            'MSP (fallback)',
                'vs_msp_pct':        0.0,
                'source':            'msp_fallback',
            }
        return {'price_per_quintal': None, 'source': 'unavailable'}

    def get_price_history(
        self, crop_name: str, market_name: Optional[str] = None, days: int = 90
    ) -> pd.DataFrame:
        """
        Get historical price time-series for a crop.

        Returns:
            DataFrame with columns: date, market, price_per_quintal, msp
        """
        if self.price_df.empty:
            self._log("⚠ No price data loaded. Pass price_csv= to constructor.")
            return pd.DataFrame()

        filtered = self.price_df[self.price_df['crop'] == crop_name].copy()
        if market_name:
            filtered = filtered[filtered['market'].str.contains(market_name, na=False)]

        filtered['date'] = pd.to_datetime(filtered['date'])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        filtered = filtered[filtered['date'] >= cutoff]
        filtered = filtered.sort_values('date')
        filtered['msp'] = MSP_PRICES.get(crop_name, None)
        return filtered

    def get_market_signal(
        self,
        crop_name:             str,
        production_forecast_t: Optional[float] = None,
    ) -> Dict:
        """
        Generate buy/sell/hold market signal based on price trend
        and supply forecast.

        Returns:
            {signal: 'sell_now'/'wait'/'negotiate', rationale, price_trend}
        """
        msp      = MSP_PRICES.get(crop_name, 0)
        current  = self.get_current_price(crop_name)
        price    = current.get('price_per_quintal', msp)

        signals  = []
        rationale = []

        # Price vs MSP
        vs_msp = (price - msp) / max(msp, 1) * 100 if msp else 0
        if vs_msp > 20:
            signals.append('sell_now')
            rationale.append(f"Price {vs_msp:+.0f}% above MSP — favorable market")
        elif vs_msp < -10:
            signals.append('wait')
            rationale.append(f"Price {vs_msp:+.0f}% below MSP — consider PM-AASHA")
        else:
            signals.append('hold')
            rationale.append("Price near MSP — market is fair")

        # Supply forecast signal
        if production_forecast_t:
            norm = production_forecast_t / max(msp, 1)  # rough normalization
            if norm > 1.3:
                signals.append('sell_early')
                rationale.append("High supply forecast — sell before harvest glut")

        final_signal = signals[0] if signals else 'monitor'

        return {
            'crop':             crop_name,
            'signal':           final_signal,
            'rationale':        rationale,
            'current_price':    price,
            'msp':              msp,
            'vs_msp_pct':       round(vs_msp, 1),
            'recommendation':   self._signal_to_action(final_signal, crop_name),
        }

    def compute_price_stats(self, crop_name: str, days: int = 180) -> Dict:
        """Compute price statistics: avg, volatility, trend, min, max."""
        hist = self.get_price_history(crop_name, days=days)
        if hist.empty:
            msp = MSP_PRICES.get(crop_name, 0)
            return {'avg': msp, 'min': msp, 'max': msp, 'volatility': 0, 'trend': 'unknown'}

        prices = hist['price_per_quintal'].values
        return {
            'avg':        round(float(np.mean(prices)), 0),
            'min':        round(float(np.min(prices)), 0),
            'max':        round(float(np.max(prices)), 0),
            'std':        round(float(np.std(prices)), 0),
            'volatility': round(float(np.std(prices) / max(np.mean(prices), 1)), 3),
            'trend':      ('rising'   if len(prices) > 5 and prices[-1] > prices[-5] else
                           'falling'  if len(prices) > 5 and prices[-1] < prices[-5] else 'flat'),
        }

    def _load_price_csv(self, path: str):
        try:
            self.price_df = pd.read_csv(path, parse_dates=['date'])
            self._log(f"✔ Loaded {len(self.price_df)} price records from {path}")
        except Exception as e:
            self._log(f"⚠ Could not load price CSV: {e}")

    def _fetch_agmarknet(self, crop_name: str, market_name: Optional[str]) -> Optional[Dict]:
        """Attempt live fetch from Agmarknet API."""
        try:
            import requests
            params = {'commodity': crop_name, 'market': market_name or ''}
            resp   = requests.get(f"{self.AGMARKNET_API}/prices", params=params, timeout=10)
            if resp.ok:
                data  = resp.json()
                price = data.get('modal_price') or data.get('price')
                if price:
                    return {
                        'price_per_quintal': float(price),
                        'date':              datetime.now().strftime('%Y-%m-%d'),
                        'market':            market_name or data.get('market', 'API'),
                        'source':            'agmarknet_api',
                    }
        except Exception:
            pass
        return None

    @staticmethod
    def _signal_to_action(signal: str, crop_name: str) -> str:
        actions = {
            'sell_now':   f'Sell {crop_name} immediately at current price',
            'sell_early': f'Sell {crop_name} early before supply-driven price fall',
            'wait':       f'Hold {crop_name} stock; explore PM-AASHA / FPO channels',
            'hold':       f'Market is fair; sell at nearest APMC or negotiate with traders',
            'monitor':    f'Monitor {crop_name} prices weekly before deciding',
        }
        return actions.get(signal, 'Consult local mandi committee')

    def _log(self, msg: str):
        print(msg)


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 3 OF 5: CropProductionHeatmap
# ═════════════════════════════════════════════════════════════════════════════

class CropProductionHeatmap:
    """
    Aggregates field-level yield forecasts into a regional production map.

    Inputs: List of field records:
      [
        {'lat': 18.5, 'lon': 73.8, 'crop': 'Onion',
         'yield_t_ha': 22.0, 'area_ha': 3.5},
        ...
      ]

    Outputs:
      - grid DataFrame: {cell_id, lat, lon, crop, avg_yield, total_production}
      - district summary: {total_production, dominant_crop, yield_map}
    """

    def __init__(self, grid_resolution_deg: float = 0.1):
        """
        Args:
            grid_resolution_deg: Grid cell size in degrees.
                                  0.1° ≈ 11km. 0.05° ≈ 5.5km.
        """
        self.resolution = grid_resolution_deg

    def generate(
        self,
        field_records: List[Dict],
        bbox:          Optional[List[float]] = None,
    ) -> Dict:
        """
        Generate production heatmap from field records.

        Args:
            field_records: List of {lat, lon, crop, yield_t_ha, area_ha}.
            bbox:          [min_lon, min_lat, max_lon, max_lat] — grid extent.
                           If None, derived from field_records bounds.

        Returns:
            {
              'grid':           pd.DataFrame (one row per cell),
              'district_summary': {total_t, dominant_crop, ...},
              'crop_breakdown': {crop: {area_ha, production_t, avg_yield}},
              'n_fields':       int,
            }
        """
        if not field_records:
            return {'grid': pd.DataFrame(), 'n_fields': 0}

        df = pd.DataFrame(field_records)
        if bbox is None:
            pad = self.resolution
            bbox = [
                df['lon'].min() - pad, df['lat'].min() - pad,
                df['lon'].max() + pad, df['lat'].max() + pad,
            ]

        # Assign each field to a grid cell
        df['cell_lon'] = (df['lon'] // self.resolution) * self.resolution
        df['cell_lat'] = (df['lat'] // self.resolution) * self.resolution
        df['cell_id']  = df['cell_lat'].astype(str) + '_' + df['cell_lon'].astype(str)

        # Aggregate per cell
        grid_rows = []
        for cell_id, cell_df in df.groupby('cell_id'):
            dominant_crop = (cell_df.groupby('crop')['area_ha'].sum()
                             .idxmax() if 'crop' in cell_df.columns else 'Unknown')
            total_prod    = (cell_df['yield_t_ha'] * cell_df['area_ha']).sum()
            total_area    = cell_df['area_ha'].sum()
            avg_yield     = total_prod / max(total_area, 0.01)

            grid_rows.append({
                'cell_id':        cell_id,
                'cell_lat':       cell_df['cell_lat'].iloc[0],
                'cell_lon':       cell_df['cell_lon'].iloc[0],
                'dominant_crop':  dominant_crop,
                'n_fields':       len(cell_df),
                'total_area_ha':  round(total_area, 2),
                'total_prod_t':   round(total_prod, 2),
                'avg_yield_t_ha': round(avg_yield, 2),
            })

        grid_df = pd.DataFrame(grid_rows)

        # District-level crop breakdown
        crop_breakdown = {}
        for crop, cdf in df.groupby('crop'):
            prod = (cdf['yield_t_ha'] * cdf['area_ha']).sum()
            area = cdf['area_ha'].sum()
            crop_breakdown[crop] = {
                'area_ha':      round(area, 2),
                'production_t': round(prod, 2),
                'avg_yield':    round(prod / max(area, 0.01), 2),
                'n_fields':     len(cdf),
            }

        total_prod    = df['yield_t_ha'].mul(df['area_ha']).sum()
        dominant_crop = max(crop_breakdown, key=lambda c: crop_breakdown[c]['production_t'],
                            default='Unknown')

        return {
            'grid':           grid_df,
            'district_summary': {
                'total_production_t': round(total_prod, 1),
                'total_area_ha':      round(df['area_ha'].sum(), 1),
                'dominant_crop':      dominant_crop,
                'n_crops':            len(crop_breakdown),
            },
            'crop_breakdown':  crop_breakdown,
            'n_fields':        len(field_records),
            'bbox':            bbox,
        }

    def to_geojson(self, heatmap_result: Dict) -> Dict:
        """Convert heatmap grid to GeoJSON FeatureCollection."""
        grid = heatmap_result.get('grid', pd.DataFrame())
        if grid.empty:
            return {'type': 'FeatureCollection', 'features': []}

        features = []
        r = self.resolution
        for _, row in grid.iterrows():
            lat, lon = row['cell_lat'], row['cell_lon']
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [lon, lat], [lon+r, lat], [lon+r, lat+r],
                        [lon, lat+r], [lon, lat],
                    ]],
                },
                'properties': {
                    'dominant_crop':  row['dominant_crop'],
                    'total_prod_t':   row['total_prod_t'],
                    'avg_yield_t_ha': row['avg_yield_t_ha'],
                    'n_fields':       row['n_fields'],
                },
            })
        return {'type': 'FeatureCollection', 'features': features}


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 4 OF 5: MarketLinkageScorer
# ═════════════════════════════════════════════════════════════════════════════

class MarketLinkageScorer:
    """
    Scores a farmer's market linkage quality (0–100).

    Higher score = better access to markets, cold storage, processing,
    and price information. Used for:
      - FPO prioritization for intervention
      - Loan risk assessment (better market access = lower risk)
      - Government scheme targeting
    """

    def score(
        self,
        proximity_result: Dict,
        crop_name:        str,
        price_stats:      Optional[Dict] = None,
    ) -> Dict:
        """
        Compute market linkage score.

        Args:
            proximity_result: Output from MarketProximityAnalyzer.find_nearest().
            crop_name:        Crop grown by farmer.
            price_stats:      Output from PriceIntelligence.compute_price_stats().

        Returns:
            {score, category, component_scores, gaps, recommendations}
        """
        components = {}
        summary    = proximity_result.get('summary', {})
        mandis     = proximity_result.get('APMC_Mandi', [])
        cold_st    = proximity_result.get('Cold_Storage', [])
        procs      = proximity_result.get('Processing_Unit', [])

        # ── 1. Nearest mandi distance (30%) ───────────────────────────────
        mandi_km = summary.get('nearest_mandi_km', 999)
        if mandi_km <= 5:
            mandi_sc = 100
        elif mandi_km <= 15:
            mandi_sc = 80
        elif mandi_km <= 30:
            mandi_sc = 55
        elif mandi_km <= 50:
            mandi_sc = 30
        else:
            mandi_sc = 10
        components['mandi_proximity'] = round(mandi_sc, 1)

        # ── 2. Cold storage (20%) — extra weight for perishables ──────────
        needs_cold   = crop_name in COLD_CHAIN_CROPS
        has_cold_30  = summary.get('has_cold_storage_30km', False)
        cold_km      = summary.get('nearest_cold_store_km', 999)

        if not needs_cold:
            cold_sc = 80  # not needed, partial score
        elif has_cold_30:
            cold_sc = 90 if cold_km <= 15 else 70
        else:
            cold_sc = 20
        components['cold_storage'] = round(cold_sc, 1)

        # ── 3. Processing access (15%) ────────────────────────────────────
        has_proc = summary.get('has_processing_50km', False)
        proc_km  = procs[0]['distance_km'] if procs else 999
        if proc_km <= 20:
            proc_sc = 100
        elif proc_km <= 40:
            proc_sc = 70
        elif has_proc:
            proc_sc = 40
        else:
            proc_sc = 15
        components['processing'] = round(proc_sc, 1)

        # ── 4. Market competition (10%) — more options = better ──────────
        n_mandis = summary.get('n_mandis_available', 0)
        comp_sc  = min(100, n_mandis * 25)
        components['competition'] = round(comp_sc, 1)

        # ── 5. Price information access (15%) ─────────────────────────────
        if price_stats:
            vol = price_stats.get('volatility', 0.5)
            price_info_sc = max(30, 100 - vol * 150)
        else:
            price_info_sc = 40  # no price data = limited access
        components['price_info'] = round(price_info_sc, 1)

        # ── 6. Export/premium market (10%) ───────────────────────────────
        exports = proximity_result.get('Export_Hub', [])
        exp_km  = exports[0]['distance_km'] if exports else 999
        exp_sc  = 90 if exp_km <= 50 else 60 if exp_km <= 100 else 25
        components['premium_market'] = round(exp_sc, 1)

        # ── Weighted total ────────────────────────────────────────────────
        weights = {
            'mandi_proximity':  0.30,
            'cold_storage':     0.20,
            'processing':       0.15,
            'competition':      0.10,
            'price_info':       0.15,
            'premium_market':   0.10,
        }
        total_score = sum(components[k] * weights[k] for k in components)

        # Category
        category = ('Excellent' if total_score >= 80 else
                    'Good'      if total_score >= 65 else
                    'Moderate'  if total_score >= 45 else
                    'Poor'      if total_score >= 25 else 'Critical')

        # Gaps and recommendations
        gaps = []
        recs = []
        if mandi_km > 30:
            gaps.append('No nearby APMC mandi')
            recs.append('Explore FPO aggregation or direct procurement arrangements')
        if needs_cold and not has_cold_30:
            gaps.append(f'{crop_name} needs cold storage but none within 30km')
            recs.append('Prioritize cold storage development; use FPO pre-cooling')
        if not has_proc:
            gaps.append('No processing unit within 50km')
            recs.append('Explore value addition at farm level')

        return {
            'score':             round(total_score, 1),
            'category':          category,
            'component_scores':  components,
            'gaps':              gaps,
            'recommendations':   recs,
            'crop':              crop_name,
        }


# ═════════════════════════════════════════════════════════════════════════════
# CLASS 5 OF 5: MarketAnalytics (Orchestrator)
# ═════════════════════════════════════════════════════════════════════════════

class MarketAnalytics:
    """
    Orchestrates all 4 market modules into a unified field-level and
    regional market intelligence report.

    Ties Component 1 (AgriTech) to Component 3 (Logistics):
      - Field yield + market proximity → optimal route
      - Regional production map → corridor mapping input
    """

    def __init__(
        self,
        market_data:  Optional[Dict[str, List[Dict]]] = None,
        price_csv:    Optional[str] = None,
        use_api:      bool = False,
        verbose:      bool = True,
    ):
        self.verbose = verbose
        market_data  = market_data or {}

        self.proximity  = MarketProximityAnalyzer(market_data)
        self.prices     = PriceIntelligence(price_csv=price_csv, use_api=use_api)
        self.heatmap    = CropProductionHeatmap()
        self.scorer     = MarketLinkageScorer()

    def analyze_field_market_access(
        self,
        latitude:      float,
        longitude:     float,
        crop_name:     str,
        yield_result:  Optional[Dict] = None,
    ) -> Dict:
        """
        Full market intelligence for a single field.

        Returns:
            {
              'proximity':       {APMC_Mandi: [...], Cold_Storage: [...], ...},
              'best_market':     {recommended_market, alternatives, net_revenue},
              'price_signal':    {signal, current_price, vs_msp_pct, recommendation},
              'linkage_score':   {score, category, gaps, recommendations},
              'farmer_summary':  str   — human-readable guidance
            }
        """
        self._log(f"\n{'━'*60}")
        self._log(f"MARKET ANALYTICS — ({latitude:.4f}, {longitude:.4f}) | Crop: {crop_name}")
        self._log(f"{'━'*60}")

        quantity_t = 0.0
        if yield_result:
            curr = yield_result.get('current_season', {})
            quantity_t = curr.get('production_t', 0)

        # Step 1: Proximity
        proximity = self.proximity.find_nearest(latitude, longitude, crop_name)

        # Step 2: Best market recommendation
        price_data = {}  # could load from PriceIntelligence
        best_market = self.proximity.find_best_market(
            latitude, longitude, crop_name, max(quantity_t, 1), price_data
        )

        # Step 3: Price signal
        price_stats  = self.prices.compute_price_stats(crop_name)
        price_signal = self.prices.get_market_signal(crop_name, quantity_t or None)

        # Step 4: Linkage score
        linkage = self.scorer.score(proximity, crop_name, price_stats)

        self._log(f"  Nearest mandi:   {proximity['summary'].get('nearest_mandi_name','?')} "
                  f"({proximity['summary'].get('nearest_mandi_km','?')} km)")
        self._log(f"  Market signal:   {price_signal['signal']}")
        self._log(f"  Linkage score:   {linkage['score']}/100 ({linkage['category']})")

        return {
            'proximity':      proximity,
            'best_market':    best_market,
            'price_signal':   price_signal,
            'price_stats':    price_stats,
            'linkage_score':  linkage,
            'farmer_summary': self._generate_farmer_summary(
                crop_name, proximity, price_signal, linkage, quantity_t
            ),
        }

    def generate_production_heatmap(
        self,
        field_results: List[Dict],
        bbox:          Optional[List[float]] = None,
    ) -> Dict:
        """
        Generate district-level production heatmap from field results.
        Feeds into the Logistics Component 3 corridor mapping module.
        """
        field_records = []
        for fr in field_results:
            loc   = fr.get('location', {})
            curr  = fr.get('yield_forecast', {}).get('current_season', {})
            if curr and loc:
                field_records.append({
                    'lat':        loc.get('latitude'),
                    'lon':        loc.get('longitude'),
                    'crop':       curr.get('season', 'Unknown'),
                    'yield_t_ha': curr.get('yield_t_ha', 0),
                    'area_ha':    fr.get('field_area_ha', 1),
                })
        return self.heatmap.generate(field_records, bbox)

    @staticmethod
    def _generate_farmer_summary(
        crop_name:    str,
        proximity:    Dict,
        price_signal: Dict,
        linkage:      Dict,
        quantity_t:   float,
    ) -> str:
        summary_parts = []

        mandi = proximity.get('summary', {}).get('nearest_mandi_name', '?')
        dist  = proximity.get('summary', {}).get('nearest_mandi_km', '?')
        summary_parts.append(f"Nearest mandi: {mandi} ({dist} km).")

        sig   = price_signal.get('signal', 'monitor')
        action = price_signal.get('recommendation', '')
        summary_parts.append(f"Market signal: {sig.upper()}. {action}.")

        if quantity_t:
            summary_parts.append(f"Estimated produce: {quantity_t:.1f} tonnes.")

        if linkage.get('gaps'):
            summary_parts.append(f"Key gap: {linkage['gaps'][0]}.")

        if linkage.get('recommendations'):
            summary_parts.append(f"Advice: {linkage['recommendations'][0]}.")

        return ' '.join(summary_parts)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
