"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Module: AgriTech Pipeline Orchestrator

The main entry point for ZUTO Component 1 — Agriculture Intelligence.
Orchestrates all modules in sequence:

  1. ZutoSatelliteCollector  → Collect ALL timestamps, compute 28 indices
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
from datetime import datetime
from typing import Dict, List, Optional, Union
from pathlib import Path

from config.agri_config import ZutoAgriConfig
from agritech.modules.satellite_collector import ZutoSatelliteCollector
from agritech.modules.spectral_index_engine import SpectralIndexEngine
from agritech.modules.crop_health_monitor import CropHealthMonitor
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
    using dense Sentinel-2 time-series and the full 28-index spectral suite.
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
        self.collector      = ZutoSatelliteCollector(verbose=verbose)
        self.health_monitor = CropHealthMonitor()
        self.index_engine   = SpectralIndexEngine()

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
        output_path:   Optional[str] = None,
    ) -> Dict:
        """
        Full field-level agricultural intelligence analysis.

        Runs all modules: satellite collection → health monitoring →
        nutrient interpretation → alert generation → summary report.

        Args:
            latitude, longitude: Field center (WGS84)
            field_area_ha:       Field area in hectares
            geometry:            Shapely Polygon / GeoJSON (alternative)
            crop:                Crop name (if known) — improves accuracy
            sowing_date:         Crop sowing date 'YYYY-MM-DD' (optional)
            num_seasons:         Number of past seasons to analyze
            seasons:             Which seasons. Default: ['kharif', 'rabi']
            output_path:         If given, save JSON result to this path

        Returns:
            Complete field analysis dict
        """
        start_time = datetime.now()
        logger.info(f"\n{'#'*65}")
        logger.info(f"ZUTO FIELD ANALYSIS")
        logger.info(f"Location: ({latitude}, {longitude}) | Crop: {crop or 'Auto-detect'}")
        logger.info(f"{'#'*65}")

        # ── Step 1: Satellite Data Collection ────────────────────────────
        logger.info("\n[STEP 1] Satellite Data Collection")
        sat_data = self.collector.collect_field_data(
            latitude=latitude, longitude=longitude,
            field_area_ha=field_area_ha, geometry=geometry,
            num_seasons=num_seasons, seasons=seasons,
        )

        # ── Step 2: Crop Health Assessment ───────────────────────────────
        logger.info("\n[STEP 2] Crop Health Assessment")
        health_assessments = self.health_monitor.assess_multi_season(
            sat_data['merged_seasons'], crop_name=crop
        )

        # ── Step 3: Current Season Summary ───────────────────────────────
        current_assessment = health_assessments[-1] if health_assessments else {}
        historical_assessments = health_assessments[:-1] if len(health_assessments) > 1 else []

        # ── Step 4: Trend Analysis ───────────────────────────────────────
        logger.info("\n[STEP 3] Trend Analysis")
        trend = self._compute_trend(health_assessments)

        # ── Step 5: Index Time-Series for Dashboard ───────────────────────
        index_timeseries = self._build_index_timeseries(sat_data['merged_seasons'])

        # ── Step 6: Compile Result ────────────────────────────────────────
        elapsed = (datetime.now() - start_time).total_seconds()

        result = {
            'meta': {
                'platform':       self.cfg.PLATFORM_NAME,
                'version':        self.cfg.PLATFORM_VERSION,
                'analysis_date':  datetime.now().isoformat(),
                'elapsed_seconds': round(elapsed, 1),
                'field': {
                    'latitude':      sat_data['location']['latitude'],
                    'longitude':     sat_data['location']['longitude'],
                    'area_ha':       sat_data['field_area_ha'],
                    'bbox':          sat_data['bbox'],
                },
                'data_summary':   sat_data['summary'],
            },
            'current_season': current_assessment,
            'historical':     historical_assessments,
            'trend':          trend,
            'index_timeseries': index_timeseries,
            'recommendations':  self._generate_recommendations(
                current_assessment, trend
            ),
        }

        logger.info(f"\n{'='*65}")
        logger.info(f"ANALYSIS COMPLETE in {elapsed:.1f}s")
        if current_assessment:
            logger.info(f"  Current Health: {current_assessment.get('health_score', 'N/A')} "
                        f"({current_assessment.get('health_category', 'N/A')})")
        logger.info(f"  Total scenes analyzed: {sat_data['summary']['total_scenes']}")
        logger.info(f"{'='*65}")

        # ── Save if output path given ─────────────────────────────────────
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
        num_seasons: int = 3,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Regional crop mapping and analytics (district / block level).

        Used for:
          - Module C: Regional Crop Classification
          - Module D: Commodity Supply Forecasting
          - Module E: Market Analytics overlay

        Args:
            bbox:   [min_lon, min_lat, max_lon, max_lat]
            season: 'kharif', 'rabi', or 'zaid'
            year:   Target year (default: latest complete season)
            num_seasons: Historical seasons for trend analysis

        Returns:
            Regional analytics dict
        """
        logger.info(f"\n{'#'*65}")
        logger.info(f"ZUTO REGIONAL ANALYSIS")
        logger.info(f"BBox: {bbox} | Season: {season.upper()}")
        logger.info(f"{'#'*65}")

        # Collect regional satellite data
        sat_data = self.collector.collect_regional_data(
            bbox=bbox, num_seasons=num_seasons, seasons=[season],
        )

        # Regional index statistics across all scenes
        regional_stats = self._regional_index_stats(sat_data['seasonal_data'])

        # Crop health at regional scale
        health_assessments = self.health_monitor.assess_multi_season(
            sat_data['merged_seasons']
        )

        result = {
            'meta': {
                'platform':      self.cfg.PLATFORM_NAME,
                'analysis_date': datetime.now().isoformat(),
                'region': {'bbox': bbox, 'season': season},
                'data_summary':  sat_data['summary'],
            },
            'regional_health':    health_assessments[-1] if health_assessments else {},
            'regional_stats':     regional_stats,
            'seasonal_trend':     self._compute_trend(health_assessments),
            'index_timeseries':   self._build_index_timeseries(sat_data['merged_seasons']),
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
                import numpy as np
                stats[idx] = {
                    'mean': round(float(sum(vals) / len(vals)), 4),
                    'max':  round(float(max(vals)), 4),
                    'min':  round(float(min(vals)), 4),
                    'n':    len(vals),
                }
        return stats

    def _generate_recommendations(
        self, assessment: Dict, trend: Dict
    ) -> List[Dict]:
        """Generate actionable recommendations from health assessment."""
        recommendations = []
        alerts = assessment.get('alerts', [])
        score  = assessment.get('health_score', 0)

        # Priority-ordered recommendations from alerts
        for alert in alerts[:5]:  # top 5 most severe
            if alert['severity'] in ('critical', 'high'):
                recommendations.append({
                    'priority':   'HIGH',
                    'category':   alert['type'].replace('_', ' ').title(),
                    'action':     alert.get('action', 'Monitor'),
                    'urgency':    'Immediate (1–3 days)' if alert['severity'] == 'critical'
                                  else 'Within 7 days',
                })

        # Trend-based recommendation
        if trend.get('direction') == 'declining':
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Seasonal Trend',
                'action':   f"Health declining ({trend.get('delta', 0):+.1f} pts). "
                            f"Review irrigation, nutrition, and pest management.",
                'urgency':  'This season',
            })

        # Score-based general advice
        if score < 40:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Overall Health',
                'action':   'Field inspection required — multiple stress indicators active.',
                'urgency':  'Immediate',
            })
        elif score < 60:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Overall Health',
                'action':   'Schedule field visit. Address top 2 alerts first.',
                'urgency':  'Within 2 weeks',
            })

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
            'total': len(self.cfg.ALL_INDICES),
            'tier1_vegetation':  self.cfg.TIER1_INDICES,
            'tier2_nutrients':   self.cfg.TIER2_INDICES,
            'tier3_moisture':    self.cfg.TIER3_INDICES,
            'tier4_canopy':      self.cfg.TIER4_INDICES,
        }
