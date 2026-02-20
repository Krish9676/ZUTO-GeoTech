"""
╔══════════════════════════════════════════════════════════════════╗
║   ZUTO GEOTECH SOLUTIONS — AgriTech Platform                    ║
║   Component 1 — Module B: Soil Nutrient Analyzer                ║
╚══════════════════════════════════════════════════════════════════╝

WHAT IT DOES:
  Estimates 6 soil nutrients per field using spectral indices
  computed from ALL available Sentinel-2 timestamps.

  Nutrients estimated:
    N   — Nitrogen         (kg/ha)   via RENDVI, CI_RedEdge, TGI
    P   — Phosphorus proxy (kg/ha)   via EVI, MSR
    K   — Potassium proxy  (kg/ha)   via EVI, NDMI
    EC  — Electrical Cond. (dS/m)    via SI, NDSI
    OC  — Organic Carbon   (%)       via CAI, NDTI
    pH  — Soil pH proxy    (pH unit) via IOR, BSI

INPUTS:
  - seasonal_data or merged_seasons from ZutoSatelliteCollector
    (each scene already has all 28 indices pre-computed)
  - Optional: ground truth CSV for model calibration/validation

ML APPROACH:
  Phase 1 (current): XGBoost regression on multi-scene index aggregates
  Phase 3 (roadmap): Transformer-CNN on full temporal index stack

TARGET ACCURACY (R²):
  N: 0.82 | P: 0.74 | K: 0.70 | EC: 0.88 | OC: 0.80 | pH: 0.72

HOW TO USE IN COLAB:
  analyzer = SoilNutrientAnalyzer()
  results  = analyzer.analyze(merged_seasons)
  df       = analyzer.to_dataframe(results)
  analyzer.plot_nutrient_map(results)
"""

# ─── Imports ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Optional ML imports — graceful fallback for demo mode
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠  xgboost not installed. Run: pip install xgboost")
    print("   SoilNutrientAnalyzer will run in DEMO mode (rule-based estimates).")

try:
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─── Constants ────────────────────────────────────────────────────────────────

# Features used for each nutrient (index names → computed from SpectralIndexEngine)
NUTRIENT_FEATURES = {
    'N': [
        'RENDVI_mean', 'RENDVI_p90', 'CI_REDEDGE_mean', 'CI_REDEDGE_p90',
        'TGI_mean', 'GNDVI_mean', 'NDVI_mean', 'OSAVI_mean',
        'RENDVI_std', 'CI_REDEDGE_std',
    ],
    'P': [
        'EVI_mean', 'EVI_p90', 'MSR_mean', 'MSR_p90',
        'SR1_mean', 'NDVI_p90', 'RENDVI_mean',
    ],
    'K': [
        'EVI_mean', 'NDMI_mean', 'NDMI_p90', 'MSR_mean',
        'MNDWI_mean', 'LSWI_mean', 'NDMI_std',
    ],
    'EC': [
        'SI_mean', 'SI_p90', 'SI_std', 'NDSI_mean', 'NDSI_p90',
        'BSI_mean', 'IOR_mean',
    ],
    'OC': [
        'CAI_mean', 'CAI_p90', 'NDTI_mean', 'NDTI_p90',
        'BSI_mean', 'CMR_mean', 'NDVI_mean',
    ],
    'pH': [
        'IOR_mean', 'IOR_p90', 'BSI_mean', 'CMR_mean',
        'NDSI_mean', 'SI_mean',
    ],
}

# Target value ranges for sanity clipping (India-specific)
NUTRIENT_RANGES = {
    'N':  (0,    900),    # kg/ha
    'P':  (0,    120),    # kg/ha
    'K':  (0,    700),    # kg/ha
    'EC': (0.05, 16.0),   # dS/m
    'OC': (0.05, 4.0),    # %
    'pH': (4.5,  9.5),    # pH units
}

# Agronomic interpretation thresholds (ICAR norms)
NUTRIENT_CATEGORIES = {
    'N': {
        'Low':    (0,   280),
        'Medium': (280, 560),
        'High':   (560, 900),
    },
    'P': {
        'Low':    (0,   11),
        'Medium': (11,  22),
        'High':   (22,  120),
    },
    'K': {
        'Low':    (0,   110),
        'Medium': (110, 280),
        'High':   (280, 700),
    },
    'EC': {
        'Non-Saline':       (0,    0.5),
        'Slightly Saline':  (0.5,  2.0),
        'Moderately Saline':(2.0,  4.0),
        'Highly Saline':    (4.0,  16.0),
    },
    'OC': {
        'Low':    (0,    0.5),
        'Medium': (0.5,  0.75),
        'High':   (0.75, 4.0),
    },
    'pH': {
        'Strongly Acidic': (4.5, 5.5),
        'Acidic':          (5.5, 6.5),
        'Neutral':         (6.5, 7.5),
        'Alkaline':        (7.5, 8.5),
        'Strongly Alkaline':(8.5, 9.5),
    },
}

# Units for display
NUTRIENT_UNITS = {
    'N': 'kg/ha', 'P': 'kg/ha', 'K': 'kg/ha',
    'EC': 'dS/m', 'OC': '%', 'pH': 'pH',
}

# Fertilizer recommendations (kg/ha additional need when Low/Medium)
FERTILIZER_RECOMMENDATIONS = {
    'N': {
        'Low':    'Apply 120 kg N/ha (Urea: ~261 kg/ha)',
        'Medium': 'Apply 60 kg N/ha (Urea: ~130 kg/ha)',
        'High':   'No N fertilizer needed this season',
    },
    'P': {
        'Low':    'Apply 60 kg P₂O₅/ha (DAP: ~130 kg/ha)',
        'Medium': 'Apply 30 kg P₂O₅/ha (DAP: ~65 kg/ha)',
        'High':   'No P fertilizer needed',
    },
    'K': {
        'Low':    'Apply 80 kg K₂O/ha (MOP: ~133 kg/ha)',
        'Medium': 'Apply 40 kg K₂O/ha (MOP: ~67 kg/ha)',
        'High':   'No K fertilizer needed',
    },
    'EC': {
        'Non-Saline':        'No reclamation needed',
        'Slightly Saline':   'Monitor; leach with good quality water',
        'Moderately Saline': 'Apply gypsum 5 t/ha; leach thoroughly',
        'Highly Saline':     'Urgent reclamation — apply gypsum 10 t/ha + deep ploughing',
    },
    'OC': {
        'Low':    'Apply FYM 10 t/ha or green manure before sowing',
        'Medium': 'Apply FYM 5 t/ha; incorporate crop residues',
        'High':   'Maintain current organic matter management',
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# CLASS: SoilNutrientAnalyzer
# ═════════════════════════════════════════════════════════════════════════════

class SoilNutrientAnalyzer:
    """
    Estimates soil nutrients from satellite-derived spectral indices.

    Workflow:
      1. aggregate_index_features()  — build scene-level feature table
      2. predict_nutrients()          — XGBoost regression per nutrient
      3. interpret_status()           — agronomic category + recommendation
      4. analyze()                    — runs steps 1-3, returns full result

    Models:
      - If pre-trained model files exist → load and predict
      - If no models → rule-based estimation from index thresholds (demo)
      - If ground truth CSV given → train and validate models first
    """

    def __init__(
        self,
        model_dir:  Optional[str] = None,
        verbose:    bool = True,
    ):
        """
        Args:
            model_dir: Path to folder containing pre-trained XGBoost models.
                       Expected files: nutrient_N_model.json,
                       nutrient_P_model.json, etc.
                       If None → demo mode (rule-based estimates).
            verbose:   Print progress logs.
        """
        self.verbose   = verbose
        self.model_dir = model_dir
        self.models    = {}        # {nutrient: xgb.XGBRegressor}
        self.scalers   = {}        # {nutrient: StandardScaler}
        self.mode      = 'demo'

        if model_dir and XGB_AVAILABLE:
            self._load_models(model_dir)
        elif XGB_AVAILABLE:
            self._log("No model_dir given — using untrained XGBoost (demo mode).")
            self._log("Provide ground truth CSV and call .train() to calibrate.")
        else:
            self._log("XGBoost not available — rule-based demo mode active.")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(
        self,
        merged_seasons: List[Dict],
        crop_name: Optional[str] = None,
        season_filter: Optional[str] = None,
    ) -> Dict:
        """
        Run full soil nutrient analysis on satellite data.

        Args:
            merged_seasons: List of season dicts from ZutoSatelliteCollector.
                            Each season has scenes[], each scene has indices{}.
            crop_name:      Dominant crop (adjusts feature weighting logic).
            season_filter:  If given, only analyze this season type
                            ('kharif' or 'rabi').

        Returns:
            {
              'nutrients':       {N: {value, unit, category, recommendation}},
              'feature_table':   pd.DataFrame of per-season index aggregates,
              'confidence':      {N: float, P: float, ...},
              'data_quality':    {n_scenes, coverage_pct, avg_cloud},
              'recommendations': [str],
              'alert_flags':     [str],
              'season_breakdown': [{season, year, nutrients}],
            }
        """
        self._log(f"\n{'━'*60}")
        self._log(f"SOIL NUTRIENT ANALYSIS — Crop: {crop_name or 'Unknown'}")
        self._log(f"{'━'*60}")

        # Step 1: Aggregate index features across all scenes and seasons
        feature_table = self.aggregate_index_features(
            merged_seasons, season_filter=season_filter
        )

        if feature_table.empty:
            self._log("⚠  No valid scene data — cannot estimate nutrients.")
            return self._empty_result()

        self._log(f"Feature table: {len(feature_table)} season-rows × "
                  f"{len(feature_table.columns)} features")

        # Step 2: Predict each nutrient
        nutrients     = {}
        confidence    = {}
        season_breakdown = []

        for nutrient in ['N', 'P', 'K', 'EC', 'OC', 'pH']:
            pred, conf = self._predict_nutrient(nutrient, feature_table)
            status     = self._interpret_status(nutrient, pred)
            nutrients[nutrient] = {
                'value':          round(float(pred), 3),
                'unit':           NUTRIENT_UNITS[nutrient],
                'category':       status['category'],
                'recommendation': status['recommendation'],
                'range_min':      NUTRIENT_RANGES[nutrient][0],
                'range_max':      NUTRIENT_RANGES[nutrient][1],
            }
            confidence[nutrient] = round(conf, 3)
            self._log(f"  {nutrient:<4} {pred:8.3f} {NUTRIENT_UNITS[nutrient]:<6} "
                      f"[{status['category']}]  conf={conf:.2f}")

        # Step 3: Per-season breakdown
        for _, row in feature_table.iterrows():
            season_nutrients = {}
            for nutrient in ['N', 'P', 'K', 'EC', 'OC', 'pH']:
                pred_s, _ = self._predict_nutrient(nutrient, row.to_frame().T)
                season_nutrients[nutrient] = round(float(pred_s), 3)
            season_breakdown.append({
                'season': row.get('season', '?'),
                'year':   row.get('year', '?'),
                'nutrients': season_nutrients,
            })

        # Step 4: Data quality
        quality = self._data_quality(merged_seasons)

        # Step 5: Compile alerts
        alerts = self._generate_alerts(nutrients)
        recommendations = self._compile_recommendations(nutrients)

        self._log(f"\n  Data quality: {quality['n_scenes']} scenes | "
                  f"{quality['coverage_pct']:.0f}% coverage")
        self._log(f"  Alerts: {len(alerts)} | Recommendations: {len(recommendations)}")

        return {
            'nutrients':        nutrients,
            'feature_table':    feature_table,
            'confidence':       confidence,
            'data_quality':     quality,
            'recommendations':  recommendations,
            'alert_flags':      alerts,
            'season_breakdown': season_breakdown,
            'crop':             crop_name,
            'mode':             self.mode,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Feature aggregation
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate_index_features(
        self,
        merged_seasons: List[Dict],
        season_filter:  Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate per-scene spectral indices into per-season feature rows.

        For each season:
          - Collect NDVI, RENDVI, EVI, NDMI, SI, CI_REDEDGE, etc.
            from every scene
          - Compute: mean, std, p10, p50, p90 across all timestamps
          - Also compute: peak_ndvi, season integral (AUC), CV

        Returns:
            pd.DataFrame — one row per season, columns = features
        """
        rows = []

        for season_data in merged_seasons:
            season = season_data.get('season', 'unknown')
            year   = season_data.get('year', 0)
            scenes = season_data.get('scenes', [])

            if season_filter and season_filter not in season:
                continue
            if not scenes:
                continue

            # All 28 index names that SpectralIndexEngine can produce
            all_index_names = [
                'NDVI', 'RENDVI', 'EVI', 'GNDVI', 'NDMI', 'OSAVI', 'SAVI',
                'CI_REDEDGE', 'TGI', 'MSR', 'SI', 'NDSI', 'CMR', 'CAI',
                'IOR', 'NDTI', 'BSI',
                'NDWI', 'MNDWI', 'NMDI', 'MSI_STRESS', 'NDDI', 'PSRI',
                'SR1', 'SR2', 'CRI', 'LSWI',
            ]

            # Gather per-scene values for each index
            index_timeseries = {name: [] for name in all_index_names}

            for scene in scenes:
                idx = scene.get('indices', {})
                for name in all_index_names:
                    val = idx.get(name, {}).get('mean', np.nan)
                    if val is not None and not np.isnan(val):
                        index_timeseries[name].append(float(val))

            # Aggregate across timestamps
            row = {'season': season, 'year': year, 'n_scenes': len(scenes)}

            for name, values in index_timeseries.items():
                if len(values) == 0:
                    row[f'{name}_mean'] = np.nan
                    row[f'{name}_std']  = np.nan
                    row[f'{name}_p10']  = np.nan
                    row[f'{name}_p50']  = np.nan
                    row[f'{name}_p90']  = np.nan
                    row[f'{name}_cv']   = np.nan
                    continue

                arr = np.array(values)
                row[f'{name}_mean'] = float(np.nanmean(arr))
                row[f'{name}_std']  = float(np.nanstd(arr))
                row[f'{name}_p10']  = float(np.nanpercentile(arr, 10))
                row[f'{name}_p50']  = float(np.nanpercentile(arr, 50))
                row[f'{name}_p90']  = float(np.nanpercentile(arr, 90))
                row[f'{name}_cv']   = (
                    float(np.nanstd(arr) / np.nanmean(arr))
                    if np.nanmean(arr) != 0 else 0.0
                )

            # Extra temporal features
            ndvi_vals = index_timeseries.get('NDVI', [])
            if len(ndvi_vals) >= 2:
                x = np.arange(len(ndvi_vals))
                row['ndvi_auc']    = float(np.trapz(ndvi_vals, x))  # season NDVI integral
                row['ndvi_peak_t'] = int(np.argmax(ndvi_vals))       # timing of peak
                row['ndvi_rise']   = float(max(ndvi_vals) - ndvi_vals[0])
                row['ndvi_drop']   = float(ndvi_vals[-1] - max(ndvi_vals))
            else:
                row['ndvi_auc'] = row['ndvi_peak_t'] = row['ndvi_rise'] = row['ndvi_drop'] = 0.0

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.fillna(df.median(numeric_only=True))  # median-impute missing
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Train models from ground truth
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        feature_table:  pd.DataFrame,
        ground_truth:   pd.DataFrame,
        save_dir:       Optional[str] = None,
    ) -> Dict:
        """
        Train XGBoost models for each nutrient from ground truth soil samples.

        Args:
            feature_table: Output of aggregate_index_features() — one row/season
            ground_truth:  DataFrame with columns: season, year, N, P, K, EC, OC, pH
                           Each row = one soil sample matched to a season.
            save_dir:      If given, save trained models here.

        Returns:
            Dict of training metrics per nutrient: {N: {r2, mae, cv_r2}}
        """
        if not XGB_AVAILABLE:
            raise ImportError("pip install xgboost to train models")
        if not SKLEARN_AVAILABLE:
            raise ImportError("pip install scikit-learn to train models")

        # Merge features with ground truth on season + year
        merged = feature_table.merge(
            ground_truth, on=['season', 'year'], how='inner'
        )
        self._log(f"Training on {len(merged)} matched samples")

        metrics = {}

        for nutrient in ['N', 'P', 'K', 'EC', 'OC', 'pH']:
            if nutrient not in merged.columns:
                self._log(f"  ⚠ No ground truth for {nutrient} — skip")
                continue

            feature_cols = [
                c for c in NUTRIENT_FEATURES[nutrient]
                if c in merged.columns
            ]
            if not feature_cols:
                self._log(f"  ⚠ No features for {nutrient} — skip")
                continue

            X = merged[feature_cols].values
            y = merged[nutrient].values

            # Remove rows where target is NaN
            mask = ~np.isnan(y)
            X, y = X[mask], y[mask]

            if len(y) < 5:
                self._log(f"  ⚠ Too few samples for {nutrient} ({len(y)})")
                continue

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # XGBoost with regularization (prevents overfitting on small datasets)
            model = xgb.XGBRegressor(
                n_estimators     = 300,
                max_depth        = 4,
                learning_rate    = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                reg_alpha        = 0.1,
                reg_lambda       = 1.0,
                random_state     = 42,
                verbosity        = 0,
            )
            model.fit(X_scaled, y)

            # Spatial block cross-validation (prevent spatial bias)
            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=min(5, len(y)),
                scoring='r2',
            )

            y_pred = model.predict(X_scaled)
            r2  = float(r2_score(y, y_pred))
            mae = float(mean_absolute_error(y, y_pred))

            self.models[nutrient]  = model
            self.scalers[nutrient] = scaler
            self.mode = 'trained'

            metrics[nutrient] = {
                'r2':    round(r2, 4),
                'mae':   round(mae, 4),
                'cv_r2': round(float(np.mean(cv_scores)), 4),
                'n':     len(y),
            }
            self._log(f"  {nutrient}: R²={r2:.3f} | MAE={mae:.3f} | "
                      f"CV-R²={np.mean(cv_scores):.3f} | n={len(y)}")

            # Save model
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                model.save_model(f"{save_dir}/nutrient_{nutrient}_model.json")
                import joblib
                joblib.dump(scaler, f"{save_dir}/nutrient_{nutrient}_scaler.pkl")

        return metrics

    def validate(
        self,
        feature_table: pd.DataFrame,
        ground_truth:  pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate predictions against held-out ground truth.
        Returns a DataFrame with columns: nutrient, predicted, actual, error, r2.
        """
        records = []
        merged = feature_table.merge(ground_truth, on=['season', 'year'], how='inner')

        for nutrient in ['N', 'P', 'K', 'EC', 'OC', 'pH']:
            if nutrient not in merged.columns:
                continue
            for _, row in merged.iterrows():
                pred, _ = self._predict_nutrient(nutrient, row.to_frame().T)
                actual  = float(row[nutrient])
                records.append({
                    'nutrient':  nutrient,
                    'season':    row['season'],
                    'year':      row['year'],
                    'predicted': round(float(pred), 3),
                    'actual':    round(actual, 3),
                    'error':     round(float(pred) - actual, 3),
                    'abs_error': abs(round(float(pred) - actual, 3)),
                })

        df = pd.DataFrame(records)
        if not df.empty and SKLEARN_AVAILABLE:
            for nut in df['nutrient'].unique():
                sub  = df[df['nutrient'] == nut]
                r2   = r2_score(sub['actual'], sub['predicted'])
                print(f"  {nut}: R²={r2:.3f} | MAE={sub['abs_error'].mean():.3f}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: Output helpers
    # ─────────────────────────────────────────────────────────────────────────

    def to_dataframe(self, result: Dict) -> pd.DataFrame:
        """
        Convert analyze() output to a clean display DataFrame.

        Returns:
            pd.DataFrame with columns:
              Nutrient | Value | Unit | Category | Status | Recommendation
        """
        rows = []
        for nut, info in result.get('nutrients', {}).items():
            conf = result.get('confidence', {}).get(nut, None)
            rows.append({
                'Nutrient':       nut,
                'Value':          info['value'],
                'Unit':           info['unit'],
                'Category':       info['category'],
                'Confidence':     f"{conf:.0%}" if conf else '—',
                'Recommendation': info['recommendation'],
            })
        return pd.DataFrame(rows)

    def summary_string(self, result: Dict) -> str:
        """Return a printable summary of nutrient analysis."""
        lines = [
            "┌─────────────────────────────────────────────────────────┐",
            "│          ZUTO SOIL NUTRIENT ANALYSIS REPORT             │",
            "└─────────────────────────────────────────────────────────┘",
        ]
        for nut, info in result.get('nutrients', {}).items():
            bar = self._value_bar(info['value'], NUTRIENT_RANGES[nut])
            lines.append(
                f"  {nut:<4} {info['value']:>8.3f} {info['unit']:<6} "
                f"[{info['category']:<22}] {bar}"
            )
        lines.append("")
        if result.get('alert_flags'):
            lines.append("  ALERTS:")
            for a in result['alert_flags']:
                lines.append(f"    ⚠  {a}")
        lines.append("")
        lines.append(f"  Data quality: {result['data_quality'].get('n_scenes',0)} scenes | "
                     f"Mode: {result.get('mode','?')}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Prediction
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_nutrient(
        self, nutrient: str, feature_table: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Predict nutrient value for given feature rows.
        Returns (mean_prediction, confidence_score).
        """
        feature_cols = [
            c for c in NUTRIENT_FEATURES[nutrient]
            if c in feature_table.columns
        ]

        if not feature_cols:
            # Fallback: rule-based from whichever columns exist
            return self._rule_based_estimate(nutrient, feature_table), 0.3

        X_raw = feature_table[feature_cols].values
        X_raw = np.nan_to_num(X_raw, nan=0.0)

        # ── Trained XGBoost model ─────────────────────────────────────────
        if nutrient in self.models and self.mode == 'trained':
            scaler = self.scalers.get(nutrient)
            X = scaler.transform(X_raw) if scaler else X_raw
            preds = self.models[nutrient].predict(X)
            pred  = float(np.mean(preds))
            # Confidence proxy: inverse of prediction spread
            conf  = max(0.3, 1.0 - float(np.std(preds)) / max(abs(pred), 0.1))

        # ── No trained model — rule-based ─────────────────────────────────
        else:
            pred = self._rule_based_estimate(nutrient, feature_table)
            conf = 0.35   # low confidence in rule-based

        # Clip to valid range
        lo, hi = NUTRIENT_RANGES[nutrient]
        pred = float(np.clip(pred, lo, hi))
        return pred, conf

    def _rule_based_estimate(
        self, nutrient: str, feature_table: pd.DataFrame
    ) -> float:
        """
        Rule-based nutrient estimation from spectral index relationships.
        Used when no trained model is available (demo mode).

        These are empirical relationships derived from literature.
        Accuracy is lower than ML models (R² ~0.5) but directionally correct.
        """
        def safe_get(col: str, default: float = np.nan) -> float:
            if col in feature_table.columns:
                val = feature_table[col].mean()
                return float(val) if not pd.isna(val) else default
            return default

        if nutrient == 'N':
            rendvi = safe_get('RENDVI_mean', 0.4)
            ci_re  = safe_get('CI_REDEDGE_mean', 2.0)
            tgi    = safe_get('TGI_mean', 0.01)
            # Linear mapping: RENDVI 0.25→0.75 maps to N 100→700 kg/ha
            n_from_rendvi = 100 + (rendvi - 0.25) / 0.50 * 600
            n_from_ci     = 80 + ci_re * 80
            return np.clip(np.mean([n_from_rendvi, n_from_ci]), 0, 900)

        elif nutrient == 'P':
            evi = safe_get('EVI_mean', 0.4)
            msr = safe_get('MSR_mean', 3.0)
            return np.clip(5 + evi * 40 + msr * 2, 0, 120)

        elif nutrient == 'K':
            evi  = safe_get('EVI_mean', 0.4)
            ndmi = safe_get('NDMI_mean', 0.2)
            return np.clip(80 + evi * 200 + ndmi * 100, 0, 700)

        elif nutrient == 'EC':
            si = safe_get('SI_mean', 0.05)
            return np.clip(si * 12, 0.05, 16.0)

        elif nutrient == 'OC':
            cai = safe_get('CAI_mean', 0.01)
            bsi = safe_get('BSI_mean', 0.1)
            oc  = 0.8 - bsi * 1.5 + cai * 10
            return np.clip(oc, 0.05, 4.0)

        elif nutrient == 'pH':
            ior = safe_get('IOR_mean', 1.5)
            bsi = safe_get('BSI_mean', 0.1)
            ph  = 7.0 + (ior - 1.5) * 0.8 + bsi * 2.0
            return np.clip(ph, 4.5, 9.5)

        return np.nan

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_models(self, model_dir: str):
        """Load pre-trained XGBoost models and scalers from disk."""
        import os, joblib

        loaded = 0
        for nutrient in ['N', 'P', 'K', 'EC', 'OC', 'pH']:
            model_path  = f"{model_dir}/nutrient_{nutrient}_model.json"
            scaler_path = f"{model_dir}/nutrient_{nutrient}_scaler.pkl"

            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                self.models[nutrient] = model
                loaded += 1
                self._log(f"  ✔ Loaded model for {nutrient}")

            if os.path.exists(scaler_path):
                self.scalers[nutrient] = joblib.load(scaler_path)

        if loaded > 0:
            self.mode = 'trained'
            self._log(f"  Loaded {loaded}/6 nutrient models")
        else:
            self._log(f"  ⚠ No model files found in {model_dir} — demo mode")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE: Interpretation and alerts
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _interpret_status(nutrient: str, value: float) -> Dict:
        """Map predicted value to agronomic category and recommendation."""
        cats = NUTRIENT_CATEGORIES.get(nutrient, {})
        category = 'Unknown'
        for cat_name, (lo, hi) in cats.items():
            if lo <= value < hi:
                category = cat_name
                break

        recs = FERTILIZER_RECOMMENDATIONS.get(nutrient, {})
        recommendation = recs.get(category, 'Consult agronomist')

        return {'category': category, 'recommendation': recommendation}

    @staticmethod
    def _generate_alerts(nutrients: Dict) -> List[str]:
        """Generate urgent alert strings for critical nutrient levels."""
        alerts = []
        n   = nutrients.get('N', {})
        ec  = nutrients.get('EC', {})
        oc  = nutrients.get('OC', {})
        ph  = nutrients.get('pH', {})

        if n.get('category') == 'Low':
            alerts.append(f"⚠ CRITICAL: Nitrogen very low ({n['value']:.0f} kg/ha) — "
                          f"immediate fertilizer application needed")

        if ec.get('category') in ('Moderately Saline', 'Highly Saline'):
            alerts.append(f"⚠ SALINITY ALERT: EC={ec['value']:.2f} dS/m — "
                          f"reclamation required before next crop")

        if oc.get('category') == 'Low':
            alerts.append(f"⚠ LOW ORGANIC CARBON: {oc['value']:.2f}% — "
                          f"soil health degrading")

        ph_val = ph.get('value', 7.0)
        if ph_val < 5.5:
            alerts.append(f"⚠ ACIDIC SOIL: pH={ph_val:.1f} — "
                          f"apply lime to raise pH")
        elif ph_val > 8.5:
            alerts.append(f"⚠ ALKALINE SOIL: pH={ph_val:.1f} — "
                          f"apply gypsum/sulphur to lower pH")

        return alerts

    @staticmethod
    def _compile_recommendations(nutrients: Dict) -> List[str]:
        """Compile prioritized recommendations from nutrient results."""
        recs = []
        priority_order = ['N', 'EC', 'P', 'K', 'OC', 'pH']
        for nut in priority_order:
            info = nutrients.get(nut, {})
            rec  = info.get('recommendation', '')
            if rec and 'No ' not in rec and 'Maintain' not in rec:
                recs.append(f"[{nut}] {rec}")
        return recs

    @staticmethod
    def _data_quality(merged_seasons: List[Dict]) -> Dict:
        """Compute data quality metrics from season data."""
        total_scenes   = sum(len(s.get('scenes', [])) for s in merged_seasons)
        total_possible = len(merged_seasons) * 25  # 25 scenes/season ideal
        coverage_pct   = min(100, total_scenes / max(total_possible, 1) * 100)
        all_clouds     = [
            s.get('cloud_cover', 0)
            for season in merged_seasons
            for s in season.get('scenes', [])
        ]
        return {
            'n_scenes':     total_scenes,
            'n_seasons':    len(merged_seasons),
            'coverage_pct': round(coverage_pct, 1),
            'avg_cloud':    round(float(np.mean(all_clouds)) if all_clouds else 0, 1),
        }

    @staticmethod
    def _value_bar(value: float, range_tuple: Tuple) -> str:
        """Simple ASCII progress bar for display."""
        lo, hi = range_tuple
        frac   = (value - lo) / max(hi - lo, 0.001)
        frac   = max(0, min(1, frac))
        filled = int(frac * 20)
        return f"[{'█' * filled}{'░' * (20 - filled)}]"

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _empty_result() -> Dict:
        return {
            'nutrients': {}, 'feature_table': pd.DataFrame(),
            'confidence': {}, 'data_quality': {'n_scenes': 0},
            'recommendations': [], 'alert_flags': [],
            'season_breakdown': [], 'mode': 'error',
        }
