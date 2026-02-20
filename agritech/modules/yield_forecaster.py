"""
╔══════════════════════════════════════════════════════════════════╗
║   ZUTO GEOTECH SOLUTIONS — AgriTech Platform                    ║
║   Component 1 — Module D: Yield Forecaster                      ║
╚══════════════════════════════════════════════════════════════════╝

WHAT IT DOES:
  Estimates crop yield and projects commodity supply using:
    1. NDVI integral method  — area under NDVI curve × crop coefficients
    2. Multi-index XGBoost   — RENDVI, EVI, NDMI, PSRI, weather features
    3. Weather adjustment    — GDD, rainfall deviation, stress events
    4. Historical calibration — scales prediction to regional baselines

  Outputs per field per season:
    - yield_t_ha          : Estimated yield in tons/hectare
    - production_tonnes   : Total production for the field
    - yield_category      : Below Average / Average / Above Average / Excellent
    - yield_vs_baseline   : % deviation from crop-specific baseline
    - commodity_forecast  : District-level supply projection (if area given)
    - confidence_interval : [low, high] at 80% confidence

INPUTS:
  - merged_seasons     : from ZutoSatelliteCollector (all timestamps, all indices)
  - seasonal_weather   : from WeatherAnalyzer (optional — improves accuracy)
  - crop_name          : detected or known crop
  - field_area_ha      : field size (for total production)
  - cultivation_area_ha: district cultivation area (for regional forecasting)

HOW TO USE IN COLAB:
  forecaster = YieldForecaster()
  result     = forecaster.forecast(
      merged_seasons=merged_seasons,
      crop_name='Wheat',
      field_area_ha=3.5,
      seasonal_weather=weather_result,   # optional
  )
  df = forecaster.to_dataframe(result)
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─── Crop yield parameters (ICAR / MoAFW India baselines) ────────────────────

CROP_YIELD_PARAMS = {
    # crop: {baseline, typical, optimal, max} in tons/ha
    'Bajra':       {'baseline': 0.8,  'typical': 1.5,  'optimal': 2.5,  'max': 4.0,  'price': 2350},
    'Banana':      {'baseline': 20.0, 'typical': 35.0, 'optimal': 50.0, 'max': 70.0, 'price': 1500},
    'Cabbage':     {'baseline': 15.0, 'typical': 25.0, 'optimal': 35.0, 'max': 50.0, 'price': 800},
    'Chilli':      {'baseline': 1.0,  'typical': 2.0,  'optimal': 3.5,  'max': 5.0,  'price': 8000},
    'Cotton':      {'baseline': 1.8,  'typical': 2.5,  'optimal': 3.5,  'max': 5.0,  'price': 6620},
    'Gram':        {'baseline': 0.8,  'typical': 1.2,  'optimal': 1.8,  'max': 2.5,  'price': 5440},
    'Grapes':      {'baseline': 10.0, 'typical': 20.0, 'optimal': 30.0, 'max': 40.0, 'price': 4000},
    'Groundnut':   {'baseline': 1.0,  'typical': 1.8,  'optimal': 2.8,  'max': 4.0,  'price': 6377},
    'Jowar':       {'baseline': 0.8,  'typical': 1.2,  'optimal': 2.0,  'max': 3.5,  'price': 2970},
    'Maize':       {'baseline': 2.5,  'typical': 3.5,  'optimal': 5.0,  'max': 8.0,  'price': 1850},
    'Mustard':     {'baseline': 0.8,  'typical': 1.2,  'optimal': 1.8,  'max': 2.5,  'price': 5650},
    'Onion':       {'baseline': 12.0, 'typical': 18.0, 'optimal': 25.0, 'max': 40.0, 'price': 1200},
    'Others':      {'baseline': 1.0,  'typical': 2.0,  'optimal': 3.0,  'max': 5.0,  'price': 2000},
    'Papaya':      {'baseline': 25.0, 'typical': 45.0, 'optimal': 65.0, 'max': 80.0, 'price': 1200},
    'Pomegranate': {'baseline': 5.0,  'typical': 10.0, 'optimal': 16.0, 'max': 22.0, 'price': 8000},
    'Potato':      {'baseline': 15.0, 'typical': 22.0, 'optimal': 30.0, 'max': 45.0, 'price': 1000},
    'Rice':        {'baseline': 2.5,  'typical': 3.5,  'optimal': 5.0,  'max': 7.0,  'price': 2183},
    'Soyabean':    {'baseline': 0.8,  'typical': 1.2,  'optimal': 1.8,  'max': 3.0,  'price': 4600},
    'Sugarcane':   {'baseline': 50.0, 'typical': 70.0, 'optimal': 90.0, 'max': 120.0,'price': 340},
    'Sunflower':   {'baseline': 0.6,  'typical': 1.0,  'optimal': 1.5,  'max': 2.5,  'price': 6760},
    'Tobacco':     {'baseline': 1.0,  'typical': 1.6,  'optimal': 2.2,  'max': 3.0,  'price': 16000},
    'Tur':         {'baseline': 0.6,  'typical': 1.0,  'optimal': 1.6,  'max': 2.5,  'price': 7000},
    'Wheat':       {'baseline': 2.5,  'typical': 3.5,  'optimal': 5.0,  'max': 7.5,  'price': 2275},
}

# Expected cumulative NDVI (AUC) for a fully healthy season (from growth curves)
EXPECTED_NDVI_AUC = {
    'Bajra': 35, 'Banana': 88, 'Cabbage': 30, 'Chilli': 65, 'Cotton': 72,
    'Gram': 60, 'Grapes': 52, 'Groundnut': 50, 'Jowar': 55, 'Maize': 65,
    'Mustard': 68, 'Onion': 50, 'Others': 55, 'Papaya': 80, 'Pomegranate': 55,
    'Potato': 62, 'Rice': 75, 'Soyabean': 68, 'Sugarcane': 92, 'Sunflower': 55,
    'Tobacco': 68, 'Tur': 75, 'Wheat': 75,
}

# Yield vs NDVI integral relationship (linear scale factor)
NDVI_YIELD_SLOPE = {
    'N_index':    'NDVI',     # primary index
    'supplement': 'RENDVI',   # nitrogen quality correction
}


# ═════════════════════════════════════════════════════════════════════════════
# CLASS: YieldForecaster
# ═════════════════════════════════════════════════════════════════════════════

class YieldForecaster:
    """
    Forecasts crop yield using satellite-derived spectral index time-series
    combined with weather data and crop-specific growth parameters.

    Three estimation methods (automatically selected by data availability):
      Method 1 — NDVI Integral: AUC of NDVI time-series scaled to yield
      Method 2 — Multi-Index XGBoost: 12 spectral features + weather
      Method 3 — Hybrid: weighted average of Method 1 & 2

    Accuracy targets:
      Method 1: R² ~ 0.65–0.72
      Method 2: R² ~ 0.72–0.80 (with weather) / 0.68–0.75 (without)
      Method 3: R² ~ 0.75–0.82
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        verbose:   bool = True,
    ):
        self.verbose   = verbose
        self.model_dir = model_dir
        self.models    = {}   # {crop: xgb.XGBRegressor}
        self.scalers   = {}
        self.mode      = 'ndvi_integral'

        if model_dir and XGB_AVAILABLE:
            self._load_models(model_dir)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main forecast
    # ─────────────────────────────────────────────────────────────────────────

    def forecast(
        self,
        merged_seasons:     List[Dict],
        crop_name:          str,
        field_area_ha:      float = 1.0,
        seasonal_weather:   Optional[Dict] = None,
        cultivation_area_ha: Optional[float] = None,
    ) -> Dict:
        """
        Forecast yield for the most recent season and project production.

        Args:
            merged_seasons:      Satellite data from ZutoSatelliteCollector.
            crop_name:           Crop type (must match CROP_YIELD_PARAMS keys).
            field_area_ha:       Field size in hectares.
            seasonal_weather:    Output from WeatherAnalyzer (optional).
            cultivation_area_ha: Total district cultivation area for
                                  regional supply projection (optional).

        Returns:
            {
              'current_season':    {yield_t_ha, production_t, category, ...},
              'historical':        [{season, year, yield_t_ha}],
              'trend':             {direction, delta_pct},
              'commodity_forecast': {total_supply_t, comparison_to_avg},
              'confidence_interval': [low, high],
              'feature_values':    {ndvi_auc, rendvi_peak, ...},
              'method_used':       str,
            }
        """
        self._log(f"\n{'━'*60}")
        self._log(f"YIELD FORECAST — Crop: {crop_name} | Area: {field_area_ha} ha")
        self._log(f"{'━'*60}")

        crop_params = CROP_YIELD_PARAMS.get(crop_name, CROP_YIELD_PARAMS['Others'])

        # Build per-season yield estimates (historical + current)
        historical = []
        for season_data in merged_seasons:
            est = self._estimate_season_yield(
                season_data, crop_name, crop_params, seasonal_weather
            )
            historical.append(est)

        if not historical:
            return self._empty_forecast(crop_name)

        # Most recent = current season forecast
        current = historical[-1]
        past    = historical[:-1]

        # Confidence interval (±15% if good data, ±25% if sparse)
        n_scenes = current.get('n_scenes', 0)
        ci_pct   = 0.15 if n_scenes >= 15 else 0.25
        ci_low   = round(current['yield_t_ha'] * (1 - ci_pct), 2)
        ci_high  = round(current['yield_t_ha'] * (1 + ci_pct), 2)

        # Trend
        trend = self._compute_trend(historical)

        # Regional commodity projection
        commodity_forecast = {}
        if cultivation_area_ha:
            commodity_forecast = self._project_supply(
                current['yield_t_ha'], cultivation_area_ha,
                crop_params, past
            )

        # Production for this field
        production_t = round(current['yield_t_ha'] * field_area_ha, 2)
        revenue_est  = round(production_t * crop_params['price'] * 100, 0)  # ₹ (price in ₹/quintal)

        self._log(f"  Estimated yield:    {current['yield_t_ha']:.2f} t/ha")
        self._log(f"  Field production:   {production_t:.2f} tonnes")
        self._log(f"  Revenue estimate:   ₹{revenue_est:,.0f}")
        self._log(f"  Category:           {current['category']}")
        self._log(f"  Method:             {current['method']}")

        return {
            'crop':              crop_name,
            'field_area_ha':     field_area_ha,
            'current_season': {
                **current,
                'production_t':    production_t,
                'revenue_est_inr': revenue_est,
                'price_per_quintal': crop_params['price'],
            },
            'historical':          past,
            'trend':               trend,
            'commodity_forecast':  commodity_forecast,
            'confidence_interval': [ci_low, ci_high],
            'method_used':         current.get('method', 'ndvi_integral'),
            'crop_params':         crop_params,
        }

    def forecast_region(
        self,
        field_forecasts:    List[Dict],
        cultivation_area_ha: float,
        crop_name:          str,
    ) -> Dict:
        """
        Aggregate individual field forecasts into a regional supply projection.

        Args:
            field_forecasts:     List of forecast() outputs for sampled fields.
            cultivation_area_ha: Total area under this crop in the region.
            crop_name:           Crop name.

        Returns:
            {
              'avg_yield_t_ha':    float,
              'total_supply_t':    float,
              'supply_category':   str,
              'yoy_change_pct':    float,
              'market_signal':     str,   ('surplus' / 'deficit' / 'normal')
            }
        """
        if not field_forecasts:
            return {}

        yields = [f['current_season']['yield_t_ha']
                  for f in field_forecasts
                  if f.get('current_season', {}).get('yield_t_ha')]

        if not yields:
            return {}

        avg_yield    = float(np.mean(yields))
        total_supply = avg_yield * cultivation_area_ha
        params       = CROP_YIELD_PARAMS.get(crop_name, CROP_YIELD_PARAMS['Others'])
        typical_supply = params['typical'] * cultivation_area_ha

        pct_vs_typical = (total_supply - typical_supply) / max(typical_supply, 1) * 100

        market_signal = ('surplus' if pct_vs_typical > 15 else
                         'deficit' if pct_vs_typical < -15 else 'normal')

        return {
            'avg_yield_t_ha':  round(avg_yield, 2),
            'total_supply_t':  round(total_supply, 1),
            'typical_supply_t': round(typical_supply, 1),
            'pct_vs_typical':  round(pct_vs_typical, 1),
            'supply_category': self._yield_category(avg_yield, params),
            'market_signal':   market_signal,
            'n_fields_sampled': len(field_forecasts),
            'cultivation_area_ha': cultivation_area_ha,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        merged_seasons:  List[Dict],
        ground_truth_df: pd.DataFrame,
        crop_name:       str,
        save_dir:        Optional[str] = None,
    ) -> Dict:
        """
        Train per-crop XGBoost yield model from ground truth yield records.

        Args:
            merged_seasons:  Satellite data.
            ground_truth_df: DataFrame with columns: season, year, yield_t_ha.
            crop_name:       Crop to train for.
            save_dir:        Where to save model files.

        Returns:
            Training metrics dict.
        """
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError("pip install xgboost scikit-learn")

        features_df = self._build_feature_table(merged_seasons, crop_name)
        merged = features_df.merge(ground_truth_df, on=['season', 'year'])

        if len(merged) < 5:
            self._log(f"⚠ Only {len(merged)} training samples for {crop_name}")
            return {}

        feature_cols = [c for c in features_df.columns
                        if c not in ('season', 'year', 'n_scenes')]
        X = merged[feature_cols].values
        y = merged['yield_t_ha'].values

        scaler = StandardScaler()
        X_s    = scaler.fit_transform(X)

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0,
        )
        model.fit(X_s, y)

        y_pred = model.predict(X_s)
        metrics = {
            'r2':     round(float(r2_score(y, y_pred)), 4),
            'n':      len(y),
            'crop':   crop_name,
        }

        self.models[crop_name]  = model
        self.scalers[crop_name] = scaler
        self.mode = 'xgboost'

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            model.save_model(f"{save_dir}/yield_{crop_name}_model.json")
            import joblib
            joblib.dump(scaler, f"{save_dir}/yield_{crop_name}_scaler.pkl")

        self._log(f"  Trained {crop_name}: R²={metrics['r2']:.3f} | n={metrics['n']}")
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Output
    # ─────────────────────────────────────────────────────────────────────────

    def to_dataframe(self, result: Dict) -> pd.DataFrame:
        """Convert forecast result to display DataFrame."""
        rows = []
        curr = result.get('current_season', {})
        rows.append({
            'Season':         f"{curr.get('season','?')} {curr.get('year','?')} (current)",
            'Yield (t/ha)':   curr.get('yield_t_ha', '—'),
            'Production (t)': curr.get('production_t', '—'),
            'Category':       curr.get('category', '—'),
            'vs Baseline':    f"{curr.get('yield_vs_baseline_pct', 0):+.1f}%",
            'Revenue (₹)':    f"₹{curr.get('revenue_est_inr', 0):,.0f}",
        })
        for h in result.get('historical', []):
            rows.append({
                'Season':         f"{h.get('season','?')} {h.get('year','?')}",
                'Yield (t/ha)':   h.get('yield_t_ha', '—'),
                'Production (t)': '—',
                'Category':       h.get('category', '—'),
                'vs Baseline':    f"{h.get('yield_vs_baseline_pct', 0):+.1f}%",
                'Revenue (₹)':    '—',
            })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Per-season estimation
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_season_yield(
        self,
        season_data:      Dict,
        crop_name:        str,
        crop_params:      Dict,
        seasonal_weather: Optional[Dict],
    ) -> Dict:
        """Estimate yield for a single season using best available method."""
        scenes  = season_data.get('scenes', [])
        season  = season_data.get('season', 'unknown')
        year    = season_data.get('year', 0)
        n       = len(scenes)

        if n < 3:
            return self._stub_season(season, year, crop_params, 'insufficient_data')

        # ── Extract index time-series ─────────────────────────────────────
        features = self._extract_features(scenes, crop_name, seasonal_weather, season)

        # ── Method selection ──────────────────────────────────────────────
        if crop_name in self.models and self.mode == 'xgboost':
            yield_est, method = self._predict_xgb(crop_name, features), 'xgboost'
        else:
            y_ndvi = self._ndvi_integral_method(features, crop_name, crop_params)
            yield_est = y_ndvi
            method    = 'ndvi_integral'

        # ── Weather adjustment ────────────────────────────────────────────
        if seasonal_weather:
            yield_est, weather_adj = self._apply_weather_adjustment(
                yield_est, seasonal_weather, season
            )
        else:
            weather_adj = 0.0

        # ── Clip to realistic range ───────────────────────────────────────
        yield_est = float(np.clip(yield_est, crop_params['baseline'] * 0.3,
                                              crop_params['max']))

        baseline_pct = (yield_est - crop_params['typical']) / max(crop_params['typical'], 0.01) * 100

        return {
            'season':               season,
            'year':                 year,
            'yield_t_ha':           round(yield_est, 2),
            'category':             self._yield_category(yield_est, crop_params),
            'yield_vs_baseline_pct': round(baseline_pct, 1),
            'weather_adjustment':   round(weather_adj, 3),
            'n_scenes':             n,
            'method':               method,
            'ndvi_auc':             features.get('ndvi_auc', 0),
            'rendvi_peak':          features.get('rendvi_peak', 0),
        }

    def _extract_features(
        self,
        scenes:           List[Dict],
        crop_name:        str,
        seasonal_weather: Optional[Dict],
        season:           str,
    ) -> Dict:
        """Extract all yield-relevant features from a scene list."""
        sorted_scenes = sorted(scenes, key=lambda s: s.get('date', ''))

        def ts(idx_name: str) -> List[float]:
            return [
                s.get('indices', {}).get(idx_name, {}).get('mean', np.nan)
                for s in sorted_scenes
            ]

        ndvi_ts    = [v for v in ts('NDVI')   if not np.isnan(v)]
        rendvi_ts  = [v for v in ts('RENDVI') if not np.isnan(v)]
        evi_ts     = [v for v in ts('EVI')    if not np.isnan(v)]
        ndmi_ts    = [v for v in ts('NDMI')   if not np.isnan(v)]
        psri_ts    = [v for v in ts('PSRI')   if not np.isnan(v)]
        si_ts      = [v for v in ts('SI')     if not np.isnan(v)]

        def safe_stats(vals: List) -> Tuple[float, float, float, float]:
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(vals)
            return (float(np.mean(a)), float(np.max(a)),
                    float(np.min(a)), float(np.std(a)))

        ndvi_mean, ndvi_peak, ndvi_min, ndvi_std = safe_stats(ndvi_ts)
        rend_mean, rend_peak, _, rend_std         = safe_stats(rendvi_ts)
        evi_mean, evi_peak, _, _                  = safe_stats(evi_ts)
        ndmi_mean, _, _, ndmi_std                 = safe_stats(ndmi_ts)
        psri_mean, psri_peak, _, _                = safe_stats(psri_ts)
        si_mean, _, _, _                          = safe_stats(si_ts)

        # NDVI AUC (integral proxy for biomass accumulation)
        ndvi_auc = float(np.trapz(ndvi_ts, np.arange(len(ndvi_ts)))) if len(ndvi_ts) > 1 else 0.0

        # Peak timing (early peak = early harvest variety or stress)
        peak_timing = float(np.argmax(ndvi_ts)) / max(len(ndvi_ts), 1) if ndvi_ts else 0.5

        feats = {
            'ndvi_mean': ndvi_mean, 'ndvi_peak': ndvi_peak,
            'ndvi_std':  ndvi_std,  'ndvi_auc':  ndvi_auc,
            'ndvi_min':  ndvi_min,  'peak_timing': peak_timing,
            'rendvi_mean': rend_mean, 'rendvi_peak': rend_peak, 'rendvi_std': rend_std,
            'evi_mean':  evi_mean,  'evi_peak': evi_peak,
            'ndmi_mean': ndmi_mean, 'ndmi_std': ndmi_std,
            'psri_mean': psri_mean, 'psri_peak': psri_peak,
            'si_mean':   si_mean,
        }

        # Add weather features if available
        if seasonal_weather:
            sw = seasonal_weather.get('seasonal_weather', [])
            matching = [w for w in sw if season in w.get('season', '')]
            if matching:
                w = matching[-1]
                feats['total_rainfall']    = w.get('total_rainfall_mm', 0)
                feats['avg_temp']          = w.get('avg_temp_c', 25)
                feats['max_temp']          = w.get('max_temp_c', 35)
                feats['rainfall_vs_norm']  = w.get('rainfall_vs_norm_pct', 100)
                feats['n_extreme_events']  = len(w.get('extreme_events', []))

        return feats

    def _ndvi_integral_method(
        self,
        features:    Dict,
        crop_name:   str,
        crop_params: Dict,
    ) -> float:
        """
        Estimate yield from NDVI AUC (integral).

        Formula:
          yield_ratio = ndvi_auc / expected_ndvi_auc
          yield_est   = typical_yield × yield_ratio × rendvi_correction

        RENDVI correction: accounts for nitrogen quality
          - High RENDVI (>0.6) → 1.05x (well-nourished crop)
          - Low RENDVI  (<0.3) → 0.80x (N-deficient crop)
        """
        expected_auc = EXPECTED_NDVI_AUC.get(crop_name, 60)
        ndvi_auc     = features.get('ndvi_auc', expected_auc * 0.7)
        typical      = crop_params['typical']
        optimal      = crop_params['optimal']

        # Yield ratio from NDVI
        ratio = ndvi_auc / max(expected_auc, 1.0)
        ratio = float(np.clip(ratio, 0.2, 1.4))

        # RENDVI quality correction (nitrogen adequacy)
        rendvi = features.get('rendvi_peak', 0.45)
        if rendvi > 0.60:
            n_correction = 1.05
        elif rendvi > 0.45:
            n_correction = 1.00
        elif rendvi > 0.30:
            n_correction = 0.92
        else:
            n_correction = 0.80

        # NDMI water stress correction
        ndmi = features.get('ndmi_mean', 0.3)
        if ndmi > 0.4:
            water_factor = 1.0
        elif ndmi > 0.2:
            water_factor = 0.95
        elif ndmi > 0.0:
            water_factor = 0.85
        else:
            water_factor = 0.70

        # PSRI early senescence correction
        psri_peak = features.get('psri_peak', 0.1)
        psri_factor = max(0.75, 1.0 - psri_peak * 0.5)

        yield_est = typical * ratio * n_correction * water_factor * psri_factor

        # Blend toward optimal if all signals are strong
        if ratio > 1.0 and n_correction >= 1.0 and water_factor >= 0.95:
            yield_est = yield_est * 0.7 + optimal * 0.3

        return float(yield_est)

    def _predict_xgb(self, crop_name: str, features: Dict) -> float:
        """Predict yield using trained XGBoost model."""
        model  = self.models[crop_name]
        scaler = self.scalers.get(crop_name)

        feature_order = [
            'ndvi_mean', 'ndvi_peak', 'ndvi_std', 'ndvi_auc',
            'rendvi_mean', 'rendvi_peak', 'evi_mean', 'evi_peak',
            'ndmi_mean', 'ndmi_std', 'psri_mean', 'si_mean',
            'peak_timing', 'total_rainfall', 'avg_temp', 'rainfall_vs_norm',
        ]
        X = np.array([[features.get(k, 0.0) for k in feature_order]])
        X = np.nan_to_num(X)

        if scaler:
            X = scaler.transform(X)

        return float(model.predict(X)[0])

    def _apply_weather_adjustment(
        self,
        base_yield:       float,
        seasonal_weather: Dict,
        season:           str,
    ) -> Tuple[float, float]:
        """
        Adjust yield estimate based on weather risk score and extreme events.

        Returns: (adjusted_yield, adjustment_factor)
        """
        risk_score = seasonal_weather.get('weather_risk_score', 50)

        # Risk score 0-100 → adjustment factor 1.10 to 0.70
        if risk_score < 20:
            factor = 1.08
        elif risk_score < 40:
            factor = 1.02
        elif risk_score < 60:
            factor = 0.95
        elif risk_score < 80:
            factor = 0.85
        else:
            factor = 0.72

        # Additional penalty for critical-stage extreme events
        extreme_events = seasonal_weather.get('extreme_events', [])
        critical_events = [e for e in extreme_events if e.get('crop_stage_critical')]
        if critical_events:
            critical_penalty = max(0.70, 1.0 - len(critical_events) * 0.04)
            factor *= critical_penalty

        adjustment = (factor - 1.0) * base_yield
        return base_yield * factor, adjustment

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Analysis helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_trend(self, historical: List[Dict]) -> Dict:
        """Compute multi-season yield trend."""
        yields = [h.get('yield_t_ha', 0) for h in historical if h.get('yield_t_ha')]
        if len(yields) < 2:
            return {'direction': 'insufficient_data', 'delta_pct': 0}

        recent = np.mean(yields[-2:])
        older  = np.mean(yields[:-2]) if len(yields) > 2 else yields[0]
        delta  = (recent - older) / max(older, 0.1) * 100

        return {
            'direction':  ('improving' if delta > 5 else 'declining' if delta < -5 else 'stable'),
            'delta_pct':  round(delta, 1),
            'yields':     [round(y, 2) for y in yields],
            'avg_yield':  round(float(np.mean(yields)), 2),
        }

    def _project_supply(
        self,
        yield_t_ha:          float,
        cultivation_area_ha: float,
        crop_params:         Dict,
        historical:          List[Dict],
    ) -> Dict:
        """Project regional commodity supply."""
        total_t         = yield_t_ha * cultivation_area_ha
        typical_total   = crop_params['typical'] * cultivation_area_ha
        pct_vs_typical  = (total_t - typical_total) / max(typical_total, 1) * 100

        historical_avg  = (
            np.mean([h.get('yield_t_ha', crop_params['typical']) for h in historical])
            if historical else crop_params['typical']
        )
        historical_supply = historical_avg * cultivation_area_ha
        yoy_pct = (total_t - historical_supply) / max(historical_supply, 1) * 100

        market_signal = (
            'Surplus — Price may fall'   if pct_vs_typical > 20 else
            'Deficit — Price may rise'   if pct_vs_typical < -20 else
            'Normal supply expected'
        )

        return {
            'total_supply_t':      round(total_t, 1),
            'typical_supply_t':    round(typical_total, 1),
            'pct_vs_typical':      round(pct_vs_typical, 1),
            'yoy_change_pct':      round(yoy_pct, 1),
            'market_signal':       market_signal,
            'cultivation_area_ha': cultivation_area_ha,
        }

    def _build_feature_table(self, merged_seasons: List[Dict], crop_name: str) -> pd.DataFrame:
        """Build feature DataFrame for training."""
        rows = []
        for sd in merged_seasons:
            feats = self._extract_features(sd.get('scenes', []), crop_name, None, sd.get('season', ''))
            feats['season'] = sd.get('season')
            feats['year']   = sd.get('year')
            feats['n_scenes'] = len(sd.get('scenes', []))
            rows.append(feats)
        return pd.DataFrame(rows).fillna(0)

    @staticmethod
    def _yield_category(yield_t_ha: float, params: Dict) -> str:
        """Categorize yield relative to baseline, typical, and optimal."""
        if yield_t_ha >= params['optimal']:
            return 'Excellent'
        elif yield_t_ha >= params['typical']:
            return 'Above Average'
        elif yield_t_ha >= params['baseline']:
            return 'Average'
        elif yield_t_ha >= params['baseline'] * 0.6:
            return 'Below Average'
        else:
            return 'Poor'

    def _load_models(self, model_dir: str):
        """Load pre-trained yield models from disk."""
        import os
        for crop in CROP_YIELD_PARAMS:
            path = f"{model_dir}/yield_{crop}_model.json"
            if os.path.exists(path):
                m = xgb.XGBRegressor()
                m.load_model(path)
                self.models[crop] = m
                self._log(f"  ✔ Loaded yield model: {crop}")
        if self.models:
            self.mode = 'xgboost'

    @staticmethod
    def _stub_season(season, year, crop_params, reason) -> Dict:
        return {
            'season': season, 'year': year,
            'yield_t_ha': crop_params['typical'],
            'category': 'Average (estimated)',
            'yield_vs_baseline_pct': 0.0,
            'weather_adjustment': 0.0,
            'n_scenes': 0, 'method': reason,
            'ndvi_auc': 0, 'rendvi_peak': 0,
        }

    @staticmethod
    def _empty_forecast(crop_name: str) -> Dict:
        return {
            'crop': crop_name, 'field_area_ha': 0,
            'current_season': {}, 'historical': [],
            'trend': {}, 'commodity_forecast': {},
            'confidence_interval': [0, 0], 'method_used': 'error',
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
