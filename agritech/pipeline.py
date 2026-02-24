"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Module: AgriTech Pipeline Orchestrator

The main entry point for ZUTO Component 1 — Agriculture Intelligence.
Orchestrates all modules in sequence:

  1. ZutoSatelliteCollector  → Collect ALL timestamps, compute 27 indices
  2. CropHealthMonitor       → Module A: Per-scene health scoring + alerts
  3. SoilNutrientAnalyzer    → Module B: N, P, K, EC, OC, pH estimation
  4. RegionalCropClassifier  → Module C: Crop type mapping (regional scale)
  5. YieldForecaster         → Module D: Supply forecasting
  6. MarketAnalytics         → Module E: Market proximity + price overlays

Usage:
    from agritech.pipeline import ZutoAgriPipeline

    pipeline = ZutoAgriPipeline()

    # Field-level analysis
    result = pipeline.analyze_field(
        latitude=18.52, longitude=73.85, field_area_ha=2.5,
        crop='Wheat', sowing_date='2024-11-20'
    )

    # Regional analysis
    result = pipeline.analyze_region(
        bbox=[73.5, 18.2, 74.0, 18.8],
        season='rabi', year=2024
    )
"""

import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

from config.agri_config import ZutoAgriConfig
from agritech.modules.satellite_collector import ZutoSatelliteCollector
from agritech.modules.spectral_index_engine import SpectralIndexEngine
from agritech.modules.crop_health_monitor import CropHealthMonitor
from agritech.modules.soil_nutrient_analyzer import SoilNutrientAnalyzer
from agritech.modules.yield_forecaster import YieldForecaster
from agritech.modules.market_analytics import MarketAnalytics
from utils.data_processing import DataProcessor
from utils.geometry_utils import GeometryUtils

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=ZutoAgriConfig.LOG_FORMAT,
)


class ZutoAgriPipeline:
    """
    ZUTO AgriTech Intelligence Pipeline.

    Provides field-level and regional-level agricultural analysis
    using dense Sentinel-2 time-series and the full 27-index spectral suite.

    All 6 modules are wired and run in sequence on every analysis:
      1. ZutoSatelliteCollector  — collect all timestamps, compute 27 indices
      2. CropHealthMonitor       — per-scene health scoring + alerts
      3. SoilNutrientAnalyzer    — N, P, K, EC, OC, pH estimation
      4. YieldForecaster         — seasonal yield + production forecast
      5. MarketAnalytics         — market proximity, price signal, linkage score
    """

    def __init__(self, verbose: bool = True):
        self.cfg     = ZutoAgriConfig
        self.verbose = verbose

        logger.info(f"{'='*65}")
        logger.info(f"  {self.cfg.PLATFORM_NAME}")
        logger.info(f"  Version {self.cfg.PLATFORM_VERSION}")
        logger.info(f"  {self.cfg.COMPANY}")
        logger.info(f"{'='*65}")

        # Initialize module instances
        self.collector        = ZutoSatelliteCollector(verbose=verbose)
        self.health_monitor   = CropHealthMonitor()
        self.index_engine     = SpectralIndexEngine()
        self.soil_analyzer    = SoilNutrientAnalyzer(verbose=verbose)
        self.yield_forecaster = YieldForecaster(verbose=verbose)
        self.market_analytics = MarketAnalytics(verbose=verbose)

        logger.info(f"  Modules loaded: SatelliteCollector | HealthMonitor | "
                    f"SoilAnalyzer | YieldForecaster | MarketAnalytics")

    # =========================================================================
    # FIELD-LEVEL ANALYSIS
    # =========================================================================

    def analyze_field(
        self,
        latitude:      Optional[float] = None,
        longitude:     Optional[float] = None,
        field_area_ha: float = 1.0,
        geometry:      Optional[object] = None,
        crop:          Optional[str] = None,
        sowing_date:   Optional[str] = None,
        num_seasons:   int = 4,
        seasons:       Optional[List[str]] = None,
        market_data:   Optional[Dict] = None,
        output_path:   Optional[str] = None,
    ) -> Dict:
        """
        Full field-level agricultural intelligence analysis.

        Runs all 5 modules in sequence:
          1. Satellite collection  — dense time-series, 27 indices per scene
          2. Crop health           — per-scene scoring, alerts, trajectory
          3. Soil nutrients        — N, P, K, EC, OC, pH from spectral indices
          4. Yield forecast        — seasonal yield estimate + revenue projection
          5. Market intelligence   — proximity, price signal, linkage score

        Args:
            latitude, longitude:  Field center (WGS84)
            field_area_ha:        Field area in hectares
            geometry:             Shapely Polygon / GeoJSON (alternative to lat/lon)
            crop:                 Crop name (if known) — improves all module accuracy
            sowing_date:          Crop sowing date 'YYYY-MM-DD' (optional)
            num_seasons:          Number of past seasons to analyze
            seasons:              Which seasons. Default: ['kharif', 'rabi']
            market_data:          Optional market facility dict for MarketAnalytics.
                                  Format: {'APMC_Mandi': [{name, lat, lon, ...}], ...}
                                  If None, MarketAnalytics runs in proximity-only mode.
            output_path:          If given, save JSON result to this path

        Returns:
            Dict with keys:
              meta               — platform info, location, timing, data summary
              current_season     — latest season health score, alerts, trajectory
              historical         — prior seasons health assessments
              trend              — multi-season health direction + delta
              index_timeseries   — chronological index values (for charts)
              soil_nutrients     — N/P/K/EC/OC/pH estimates + recommendations
              yield_forecast     — yield t/ha, production t, revenue estimate
              market_intelligence — proximity, price signal, linkage score
              recommendations    — unified prioritised action list
        """
        start_time = datetime.now()
        logger.info(f"\n{'#'*65}")
        logger.info(f"ZUTO FIELD ANALYSIS")
        logger.info(f"Location: ({latitude}, {longitude}) | Crop: {crop or 'Auto-detect'}")
        logger.info(f"{'#'*65}")

        # ── Step 1: Satellite Data Collection ────────────────────────────
        logger.info("\n[STEP 1/5] Satellite Data Collection")
        sat_data = self.collector.collect_field_data(
            latitude=latitude, longitude=longitude,
            field_area_ha=field_area_ha, geometry=geometry,
            num_seasons=num_seasons, seasons=seasons,
        )
        merged_seasons = sat_data['merged_seasons']
        field_lat      = sat_data['location']['latitude']
        field_lon      = sat_data['location']['longitude']

        # ── Step 2: Crop Health Assessment ───────────────────────────────
        logger.info("\n[STEP 2/5] Crop Health Assessment")
        health_assessments = self.health_monitor.assess_multi_season(
            merged_seasons, crop_name=crop
        )
        current_assessment     = health_assessments[-1] if health_assessments else {}
        historical_assessments = health_assessments[:-1] if len(health_assessments) > 1 else []
        trend                  = self._compute_trend(health_assessments)
        index_timeseries       = self._build_index_timeseries(merged_seasons)

        # ── Step 3: Soil Nutrient Analysis ───────────────────────────────
        logger.info("\n[STEP 3/5] Soil Nutrient Analysis")
        soil_result = {}
        try:
            soil_result = self.soil_analyzer.analyze(
                merged_seasons=merged_seasons,
                crop_name=crop,
            )
            # feature_table is a DataFrame — not JSON-serialisable; drop it
            soil_result.pop('feature_table', None)
        except Exception as e:
            logger.warning(f"  SoilNutrientAnalyzer failed: {e}")

        # ── Step 4: Yield Forecast ────────────────────────────────────────
        logger.info("\n[STEP 4/5] Yield Forecast")
        yield_result = {}
        try:
            yield_result = self.yield_forecaster.forecast(
                merged_seasons=merged_seasons,
                crop_name=crop or 'Others',
                field_area_ha=field_area_ha,
            )
        except Exception as e:
            logger.warning(f"  YieldForecaster failed: {e}")

        # ── Step 5: Market Intelligence ───────────────────────────────────
        logger.info("\n[STEP 5/5] Market Intelligence")
        market_result = {}
        try:
            if market_data:
                self.market_analytics.proximity = \
                    self.market_analytics.proximity.__class__(market_data)
            market_result = self.market_analytics.analyze_field_market_access(
                latitude=field_lat,
                longitude=field_lon,
                crop_name=crop or 'Others',
                yield_result=yield_result or None,
            )
        except Exception as e:
            logger.warning(f"  MarketAnalytics failed: {e}")

        # ── Compile unified result ────────────────────────────────────────
        elapsed = (datetime.now() - start_time).total_seconds()

        result = {
            'meta': {
                'platform':        self.cfg.PLATFORM_NAME,
                'version':         self.cfg.PLATFORM_VERSION,
                'analysis_date':   datetime.now().isoformat(),
                'elapsed_seconds': round(elapsed, 1),
                'modules_run':     [
                    'SatelliteCollector', 'CropHealthMonitor',
                    'SoilNutrientAnalyzer', 'YieldForecaster', 'MarketAnalytics',
                ],
                'field': {
                    'latitude':   field_lat,
                    'longitude':  field_lon,
                    'area_ha':    sat_data['field_area_ha'],
                    'bbox':       sat_data['bbox'],
                    'crop':       crop,
                },
                'data_summary': sat_data['summary'],
            },
            'current_season':      current_assessment,
            'historical':          historical_assessments,
            'trend':               trend,
            'index_timeseries':    index_timeseries,
            'soil_nutrients':      soil_result,
            'yield_forecast':      yield_result,
            'market_intelligence': market_result,
            'recommendations':     self._generate_recommendations(
                current_assessment, trend, soil_result, yield_result, market_result
            ),
        }

        logger.info(f"\n{'='*65}")
        logger.info(f"FIELD ANALYSIS COMPLETE in {elapsed:.1f}s")
        if current_assessment:
            logger.info(f"  Health:  {current_assessment.get('health_score', 'N/A')} "
                        f"({current_assessment.get('health_category', 'N/A')})")
        if yield_result.get('current_season'):
            cs = yield_result['current_season']
            logger.info(f"  Yield:   {cs.get('yield_t_ha', 'N/A')} t/ha  "
                        f"({cs.get('category', '')})")
        if soil_result.get('nutrients'):
            n = soil_result['nutrients'].get('N', {})
            logger.info(f"  Soil N:  {n.get('value', 'N/A')} {n.get('unit','')}  "
                        f"[{n.get('category', '')}]")
        if market_result.get('linkage_score'):
            ls = market_result['linkage_score']
            logger.info(f"  Market:  {ls.get('score', 'N/A')}/100 ({ls.get('category', '')})")
        logger.info(f"  Scenes:  {sat_data['summary']['total_scenes']}")
        logger.info(f"{'='*65}")

        if output_path:
            self._save_result(result, output_path)

        return result

    # =========================================================================
    # REGIONAL ANALYSIS
    # =========================================================================

    def analyze_region(
        self,
        bbox:        List[float],
        season:      str = 'kharif',
        year:        int = None,
        crop:        Optional[str] = None,
        num_seasons: int = 3,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Regional crop mapping and analytics (district / block level).

        Runs satellite collection, health monitoring, soil nutrient analysis,
        and yield forecasting at regional scale.

        Args:
            bbox:        [min_lon, min_lat, max_lon, max_lat]
            season:      'kharif', 'rabi', or 'zaid'
            year:        Target year (default: latest complete season)
            crop:        Dominant crop in the region (optional)
            num_seasons: Historical seasons for trend analysis

        Returns:
            Dict with keys:
              meta               — platform info, region bbox, timing
              regional_health    — latest season health assessment
              regional_stats     — aggregated index stats across all scenes
              seasonal_trend     — multi-season health direction
              index_timeseries   — chronological index values
              regional_nutrients — N/P/K/EC/OC/pH estimates for the region
              regional_yield     — regional yield and supply projection
        """
        logger.info(f"\n{'#'*65}")
        logger.info(f"ZUTO REGIONAL ANALYSIS")
        logger.info(f"BBox: {bbox} | Season: {season.upper()} | Crop: {crop or 'All'}")
        logger.info(f"{'#'*65}")

        # ── Step 1: Satellite Data Collection ────────────────────────────
        logger.info("\n[STEP 1/4] Satellite Data Collection")
        sat_data = self.collector.collect_regional_data(
            bbox=bbox, num_seasons=num_seasons, seasons=[season],
        )
        merged_seasons = sat_data['merged_seasons']

        # ── Step 2: Regional Health + Index Stats ────────────────────────
        logger.info("\n[STEP 2/4] Regional Health Assessment")
        regional_stats     = self._regional_index_stats(sat_data['seasonal_data'])
        health_assessments = self.health_monitor.assess_multi_season(merged_seasons)

        # ── Step 3: Regional Soil Nutrients ──────────────────────────────
        logger.info("\n[STEP 3/4] Regional Soil Nutrient Analysis")
        regional_nutrients = {}
        try:
            soil_result = self.soil_analyzer.analyze(
                merged_seasons=merged_seasons,
                crop_name=crop,
            )
            soil_result.pop('feature_table', None)
            regional_nutrients = soil_result
        except Exception as e:
            logger.warning(f"  SoilNutrientAnalyzer failed: {e}")

        # ── Step 4: Regional Yield Forecast ──────────────────────────────
        logger.info("\n[STEP 4/4] Regional Yield Forecast")
        regional_yield = {}
        try:
            regional_yield = self.yield_forecaster.forecast(
                merged_seasons=merged_seasons,
                crop_name=crop or 'Others',
                field_area_ha=1.0,   # per-hectare basis for regional
            )
        except Exception as e:
            logger.warning(f"  YieldForecaster failed: {e}")

        result = {
            'meta': {
                'platform':      self.cfg.PLATFORM_NAME,
                'analysis_date': datetime.now().isoformat(),
                'region':        {'bbox': bbox, 'season': season, 'crop': crop},
                'data_summary':  sat_data['summary'],
            },
            'regional_health':     health_assessments[-1] if health_assessments else {},
            'regional_stats':      regional_stats,
            'seasonal_trend':      self._compute_trend(health_assessments),
            'index_timeseries':    self._build_index_timeseries(merged_seasons),
            'regional_nutrients':  regional_nutrients,
            'regional_yield':      regional_yield,
        }

        if output_path:
            self._save_result(result, output_path)

        return result

    # =========================================================================
    # ANALYTICS HELPERS
    # =========================================================================

    def _compute_trend(self, assessments: List[Dict]) -> Dict:
        """Compute multi-season trend in crop health."""
        if len(assessments) < 2:
            return {'direction': 'insufficient_data', 'seasons': len(assessments)}

        scores = [a.get('health_score', 0) for a in assessments if a.get('health_score')]
        if len(scores) < 2:
            return {'direction': 'unknown', 'seasons': len(assessments)}

        recent_avg = sum(scores[-2:]) / 2
        older_avg  = sum(scores[:-2]) / max(len(scores) - 2, 1)
        delta      = recent_avg - older_avg

        direction = ('improving'  if delta > 5  else
                     'declining'  if delta < -5 else
                     'stable')

        return {
            'direction':    direction,
            'delta':        round(delta, 1),
            'recent_avg':   round(recent_avg, 1),
            'historical_avg': round(older_avg, 1),
            'seasons':      len(assessments),
            'scores':       [round(s, 1) for s in scores],
        }

    def _build_index_timeseries(self, merged_seasons: List[Dict]) -> List[Dict]:
        """
        Build a flat chronological time-series of key indices across all seasons.
        Used for dashboard visualization (line charts).
        """
        timeseries = []
        key_indices = ['NDVI', 'RENDVI', 'EVI', 'NDMI', 'SI', 'CI_REDEDGE',
                       'PSRI', 'BSI', 'CMR', 'CAI']

        for season_data in merged_seasons:
            for scene in season_data.get('scenes', []):
                entry = {
                    'date':   scene.get('date'),
                    'season': season_data.get('season'),
                    'year':   season_data.get('year'),
                }
                for idx in key_indices:
                    entry[idx] = scene.get('indices', {}).get(idx, {}).get('mean')
                timeseries.append(entry)

        timeseries.sort(key=lambda x: x.get('date', ''))
        return timeseries

    def _regional_index_stats(self, seasonal_data: List[Dict]) -> Dict:
        """Aggregate index statistics at regional level across all seasons."""
        all_scenes = []
        for s in seasonal_data:
            all_scenes.extend(s.get('scenes', []))

        if not all_scenes:
            return {}

        stats = {}
        key_indices = ['NDVI', 'RENDVI', 'EVI', 'NDMI', 'SI', 'BSI']
        for idx in key_indices:
            vals = [
                s.get('indices', {}).get(idx, {}).get('mean')
                for s in all_scenes
                if s.get('indices', {}).get(idx, {}).get('mean') is not None
            ]
            if vals:
                stats[idx] = {
                    'mean': round(float(np.mean(vals)), 4),
                    'max':  round(float(np.max(vals)), 4),
                    'min':  round(float(np.min(vals)), 4),
                    'n':    len(vals),
                }
        return stats

    def _generate_recommendations(
        self,
        assessment:     Dict,
        trend:          Dict,
        soil_result:    Optional[Dict] = None,
        yield_result:   Optional[Dict] = None,
        market_result:  Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generate a unified, priority-ordered action list from all 5 modules.

        Sources:
          - Health alerts     → HIGH / CRITICAL actions
          - Soil deficiencies → nutrient application recommendations
          - Yield category    → harvest / storage timing advice
          - Market signal     → sell / hold / wait guidance
          - Trend direction   → seasonal management review
        """
        recommendations = []
        alerts = assessment.get('alerts', [])
        score  = assessment.get('health_score', 0)

        # ── 1. Health alerts (critical + high only) ───────────────────────
        for alert in alerts[:5]:
            if alert['severity'] in ('critical', 'high'):
                recommendations.append({
                    'priority': 'HIGH',
                    'source':   'crop_health',
                    'category': alert['type'].replace('_', ' ').title(),
                    'action':   alert.get('action', 'Monitor'),
                    'urgency':  'Immediate (1–3 days)' if alert['severity'] == 'critical'
                                else 'Within 7 days',
                })

        # ── 2. Soil nutrient deficiencies ─────────────────────────────────
        if soil_result:
            for nutrient, info in soil_result.get('nutrients', {}).items():
                cat = info.get('category', '')
                if cat in ('Low', 'Very Low', 'Deficient'):
                    recommendations.append({
                        'priority': 'HIGH' if cat == 'Very Low' else 'MEDIUM',
                        'source':   'soil_nutrients',
                        'category': f'{nutrient} Deficiency',
                        'action':   info.get('recommendation', f'Apply {nutrient} fertilizer'),
                        'urgency':  'Before next irrigation cycle',
                    })
            for flag in soil_result.get('alert_flags', [])[:2]:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'source':   'soil_nutrients',
                    'category': 'Soil Alert',
                    'action':   flag,
                    'urgency':  'This season',
                })

        # ── 3. Yield category advice ──────────────────────────────────────
        if yield_result:
            cs  = yield_result.get('current_season', {})
            cat = cs.get('category', '')
            if cat in ('Low', 'Very Low', 'Below Average'):
                recommendations.append({
                    'priority': 'MEDIUM',
                    'source':   'yield_forecast',
                    'category': 'Yield Risk',
                    'action':   (f"Yield forecast is {cat} "
                                 f"({cs.get('yield_t_ha', '?')} t/ha). "
                                 f"Review crop nutrition and irrigation."),
                    'urgency':  'This season',
                })
            yld_trend = yield_result.get('trend', {})
            if yld_trend.get('direction') == 'declining':
                recommendations.append({
                    'priority': 'MEDIUM',
                    'source':   'yield_forecast',
                    'category': 'Yield Trend',
                    'action':   (f"Yield declining ({yld_trend.get('delta_pct', 0):+.1f}% vs prior). "
                                 f"Consider soil testing and precision input management."),
                    'urgency':  'Next season planning',
                })

        # ── 4. Market signal ──────────────────────────────────────────────
        if market_result:
            signal = market_result.get('price_signal', {})
            sig    = signal.get('signal', '')
            if sig in ('sell_now', 'sell_early'):
                recommendations.append({
                    'priority': 'HIGH',
                    'source':   'market_intelligence',
                    'category': 'Market Signal',
                    'action':   signal.get('recommendation', f'Market signal: {sig.upper()}'),
                    'urgency':  'Within 48 hours',
                })
            elif sig in ('wait', 'hold'):
                recommendations.append({
                    'priority': 'LOW',
                    'source':   'market_intelligence',
                    'category': 'Market Signal',
                    'action':   signal.get('recommendation', f'Market signal: {sig.upper()}'),
                    'urgency':  'Monitor weekly',
                })
            gaps = market_result.get('linkage_score', {}).get('gaps', [])
            for gap in gaps[:1]:
                recommendations.append({
                    'priority': 'LOW',
                    'source':   'market_intelligence',
                    'category': 'Market Access Gap',
                    'action':   gap,
                    'urgency':  'Medium term',
                })

        # ── 5. Multi-season health trend ──────────────────────────────────
        if trend.get('direction') == 'declining':
            recommendations.append({
                'priority': 'MEDIUM',
                'source':   'crop_health',
                'category': 'Seasonal Trend',
                'action':   (f"Health declining ({trend.get('delta', 0):+.1f} pts over "
                             f"{trend.get('seasons', '?')} seasons). "
                             f"Review irrigation, nutrition, and pest management."),
                'urgency':  'This season',
            })

        # ── 6. Overall score floor ────────────────────────────────────────
        if score < 40:
            recommendations.append({
                'priority': 'HIGH',
                'source':   'crop_health',
                'category': 'Overall Health',
                'action':   'Field inspection required — multiple stress indicators active.',
                'urgency':  'Immediate',
            })
        elif score < 60:
            recommendations.append({
                'priority': 'MEDIUM',
                'source':   'crop_health',
                'category': 'Overall Health',
                'action':   'Schedule field visit. Address top 2 alerts first.',
                'urgency':  'Within 2 weeks',
            })

        # Sort: HIGH first, then MEDIUM, then LOW
        priority_rank = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda r: priority_rank.get(r['priority'], 9))

        return recommendations

    # =========================================================================
    # OUTPUT
    # =========================================================================

    @staticmethod
    def _save_result(result: Dict, output_path: str):
        """Save analysis result to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"  ✔ Result saved to: {path}")

    def get_supported_crops(self) -> List[str]:
        """Return list of all supported crop names."""
        return self.cfg.SUPPORTED_CROPS

    def get_index_catalog(self) -> Dict:
        """Return full catalog of computed spectral indices."""
        return {
            'total':             len(self.cfg.ALL_INDICES),
            'tier1_vegetation':  self.cfg.TIER1_INDICES,
            'tier2_nutrients':   self.cfg.TIER2_INDICES,
            'tier3_moisture':    self.cfg.TIER3_INDICES,
            'tier4_canopy':      self.cfg.TIER4_INDICES,
        }

    def get_pipeline_status(self) -> Dict:
        """
        Return status of all instantiated modules.
        Useful for health-checks and API readiness probes.
        """
        return {
            'platform':  self.cfg.PLATFORM_NAME,
            'version':   self.cfg.PLATFORM_VERSION,
            'modules': {
                'satellite_collector':  'ready',
                'crop_health_monitor':  'ready',
                'soil_nutrient_analyzer': (
                    f"ready ({self.soil_analyzer.mode} mode)"
                ),
                'yield_forecaster': (
                    f"ready ({self.yield_forecaster.mode} mode)"
                ),
                'market_analytics': 'ready',
            },
            'index_count':   len(self.cfg.ALL_INDICES),
            'crop_count':    len(self.cfg.SUPPORTED_CROPS),
            'nutrient_count': len(self.cfg.NUTRIENT_TARGETS),
        }

    def train_soil_models(
        self,
        merged_seasons: List[Dict],
        ground_truth_df,            # pd.DataFrame: season, year, N, P, K, EC, OC, pH
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Train soil nutrient models from ground-truth soil sample data.
        Delegates directly to SoilNutrientAnalyzer.train().

        Args:
            merged_seasons:  Satellite data (same format as analyze_field output).
            ground_truth_df: DataFrame with columns: season, year, N, P, K, EC, OC, pH.
            save_dir:        Directory to save trained model files.

        Returns:
            Training metrics per nutrient: {N: {r2, mae, cv_r2, n}, ...}
        """
        feature_table = self.soil_analyzer.aggregate_index_features(merged_seasons)
        return self.soil_analyzer.train(feature_table, ground_truth_df, save_dir)

    def train_yield_models(
        self,
        merged_seasons:   List[Dict],
        ground_truth_df,             # pd.DataFrame: season, year, yield_t_ha
        crop_name:        str,
        save_dir:         Optional[str] = None,
    ) -> Dict:
        """
        Train a yield forecasting model from ground-truth yield records.
        Delegates directly to YieldForecaster.train().

        Args:
            merged_seasons:  Satellite data.
            ground_truth_df: DataFrame with columns: season, year, yield_t_ha.
            crop_name:       Crop to train for.
            save_dir:        Directory to save trained model files.

        Returns:
            Training metrics: {r2, n, crop}
        """
        return self.yield_forecaster.train(
            merged_seasons, ground_truth_df, crop_name, save_dir
        )