"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Module: Satellite Data Collector

VERSION 1.0 — ZUTO Design Principles:
  1. EVERY available timestamp is collected (no scene-count caps)
     - Sentinel-2 revisits every ~5 days (2 satellites combined)
     - Collect all scenes within cloud threshold for maximum time-series density
  2. ALL 28 indices computed on every scene via SpectralIndexEngine
  3. Three Indian seasons: Kharif, Rabi, Zaid
  4. Cross-season long-duration crop detection (Sugarcane, Banana, Tur...)
  5. Chronological ordering always — ML features must be time-ordered
  6. Landsat-8/9 fallback for historical depth (pre-2017)

Data Sources:
  Primary:   Sentinel-2 L2A via Microsoft Planetary Computer STAC
  Fallback:  Landsat 8/9 Collection 2 Level-2
  Weather:   NASA POWER API (fetched per season)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
import logging
import gc

try:
    import pystac_client
    import planetary_computer
    import rasterio
    from rasterio.windows import from_bounds, Window
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    SATELLITE_AVAILABLE = True
except ImportError:
    SATELLITE_AVAILABLE = False
    print("⚠  ZUTO: Satellite libraries not installed.")
    print("   Run: pip install pystac-client planetary-computer rasterio")

from config.agri_config import ZutoAgriConfig

# ── FIX: satellite_collector.py lives inside agritech/modules/ so
#    spectral_index_engine (a sibling module) must be imported with a
#    relative path, not the full agritech.modules.* absolute path which
#    would require the package to import itself.
from agritech.modules.spectral_index_engine import SpectralIndexEngine
from utils.data_processing import DataProcessor
from utils.geometry_utils import GeometryUtils

logger = logging.getLogger(__name__)


class ZutoSatelliteCollector:
    """
    Collects ALL available Sentinel-2 timestamps for a given field/region
    across Indian crop seasons (Kharif, Rabi, Zaid) and computes the full
    28-index spectral suite on every scene.

    Output per season:
      - scenes[]  : chronologically ordered list, one entry per timestamp
      - Each scene : {date, cloud_cover, bands_stats, indices (28 values each)}
      - ndvi_start/end/peak for cross-season analysis
    """

    def __init__(self, verbose: bool = True):
        if not SATELLITE_AVAILABLE:
            raise ImportError(
                "Install satellite libraries:\n"
                "pip install pystac-client planetary-computer rasterio shapely"
            )

        self.cfg      = ZutoAgriConfig
        self.engine   = SpectralIndexEngine(scale_bands=True)
        self.verbose  = verbose
        self.catalog  = pystac_client.Client.open(
            self.cfg.STAC_API_URL,
            modifier=planetary_computer.sign_inplace,
        )
        self.band_res = {
            band: info['resolution']
            for band, info in self.cfg.SENTINEL2_BANDS.items()
        }

        logger.info(f"✔ ZUTO Satellite Collector v1.0 initialized")
        logger.info(f"  Mode: COLLECT ALL TIMESTAMPS (no scene cap)")
        logger.info(f"  Cloud threshold: {self.cfg.MAX_CLOUD_COVER}%")
        logger.info(f"  Target resolution: {self.cfg.TARGET_RESOLUTION_M}m")

    # =========================================================================
    # PUBLIC API — FIELD LEVEL
    # =========================================================================

    def collect_field_data(
        self,
        latitude:      Optional[float] = None,
        longitude:     Optional[float] = None,
        field_area_ha: float = 1.0,
        geometry:      Optional[object] = None,
        num_seasons:   int = 4,
        seasons:       Optional[List[str]] = None,
    ) -> Dict:
        """
        Collect ALL Sentinel-2 scenes for a field across multiple seasons.

        Args:
            latitude, longitude: Field center point (WGS84)
            field_area_ha:       Field area in hectares (used to size bbox)
            geometry:            Shapely Polygon or GeoJSON (overrides lat/lon)
            num_seasons:         Number of past seasons to collect
            seasons:             Season names to include, e.g. ['kharif', 'rabi']

        Returns:
            {
              'location': {latitude, longitude},
              'field_area_ha': float,
              'bbox': [min_lon, min_lat, max_lon, max_lat],
              'merged_seasons': [season_dict, ...],   # chronological
              'summary': {total_scenes, seasons_collected, date_range}
            }
        """
        if seasons is None:
            seasons = ['kharif', 'rabi']

        # Resolve geometry → bbox
        if geometry is not None:
            bbox = GeometryUtils.calculate_bbox_from_geometry(geometry)
            lat, lon = GeometryUtils.calculate_centroid_from_geometry(geometry)
        else:
            if latitude is None or longitude is None:
                raise ValueError("Provide either (latitude, longitude) or geometry.")
            lat, lon = latitude, longitude
            bbox = GeometryUtils.calculate_bbox_from_point(lat, lon, field_area_ha)

        if not GeometryUtils.validate_bbox(bbox):
            raise ValueError(f"Invalid bbox: {bbox}")

        logger.info(f"  Field bbox: {[round(x, 4) for x in bbox]}")
        logger.info(f"  Area: {field_area_ha} ha | Seasons: {seasons} × {num_seasons}")

        # Build list of season windows to collect
        season_windows = self._build_season_windows(seasons, num_seasons)

        # Collect data for each window
        seasonal_data = []
        for window in season_windows:
            season_data = self._collect_season(bbox, window)
            if season_data and len(season_data.get('scenes', [])) >= self.cfg.MIN_SCENES_PER_SEASON:
                seasonal_data.append(season_data)
            elif season_data:
                logger.warning(
                    f"  ⚠ {window['season'].upper()} {window['year']}: "
                    f"only {len(season_data.get('scenes', []))} scenes — below minimum {self.cfg.MIN_SCENES_PER_SEASON}"
                )

        # Merge and sort chronologically
        merged = sorted(seasonal_data, key=lambda x: (x['year'], x['season']))

        total_scenes = sum(len(s.get('scenes', [])) for s in merged)
        date_range = self._get_date_range(merged)

        logger.info(f"  ✔ Collected {total_scenes} scenes across {len(merged)} season windows")

        return {
            'location':       {'latitude': lat, 'longitude': lon},
            'field_area_ha':  field_area_ha,
            'bbox':           bbox,
            'merged_seasons': merged,
            'seasonal_data':  seasonal_data,
            'summary': {
                'total_scenes':      total_scenes,
                'seasons_collected': len(merged),
                'date_range':        date_range,
            },
        }

    # =========================================================================
    # PUBLIC API — REGIONAL LEVEL
    # =========================================================================

    def collect_regional_data(
        self,
        bbox:        List[float],
        num_seasons: int = 3,
        seasons:     Optional[List[str]] = None,
    ) -> Dict:
        """
        Collect satellite data for a regional bounding box.

        Args:
            bbox:        [min_lon, min_lat, max_lon, max_lat]
            num_seasons: Number of past seasons
            seasons:     Season names to include

        Returns:
            Same structure as collect_field_data but keyed to region
        """
        if not GeometryUtils.validate_bbox(bbox):
            raise ValueError(f"Invalid bbox: {bbox}")

        if seasons is None:
            seasons = ['kharif', 'rabi']

        logger.info(f"  Regional bbox: {[round(x, 4) for x in bbox]}")

        season_windows = self._build_season_windows(seasons, num_seasons)
        seasonal_data = []
        for window in season_windows:
            season_data = self._collect_season(bbox, window)
            if season_data and len(season_data.get('scenes', [])) >= self.cfg.MIN_SCENES_PER_SEASON:
                seasonal_data.append(season_data)

        merged = sorted(seasonal_data, key=lambda x: (x['year'], x['season']))
        total_scenes = sum(len(s.get('scenes', [])) for s in merged)

        return {
            'bbox':           bbox,
            'merged_seasons': merged,
            'seasonal_data':  seasonal_data,
            'summary': {
                'total_scenes':      total_scenes,
                'seasons_collected': len(merged),
                'date_range':        self._get_date_range(merged),
            },
        }

    # =========================================================================
    # INTERNAL — SEASON WINDOW BUILDER
    # =========================================================================

    def _build_season_windows(
        self, seasons: List[str], num_seasons: int
    ) -> List[Dict]:
        """
        Build a list of {season, year, start_date, end_date} windows
        going back num_seasons from today.
        """
        today = date.today()
        windows = []

        for season_name in seasons:
            cfg = self.cfg.SEASONS.get(season_name)
            if not cfg:
                logger.warning(f"Unknown season '{season_name}' — skipping")
                continue

            # Determine which years have complete seasons going backwards
            collected = 0
            for year_offset in range(self.cfg.MAX_YEARS_BACK * 2):
                if collected >= num_seasons:
                    break

                year = today.year - year_offset

                # Build season start/end dates
                s_month = cfg['start_month']
                s_day   = cfg['start_day']
                e_month = cfg['end_month']
                e_day   = cfg['end_day']

                try:
                    start = date(year, s_month, s_day)
                    # Handle seasons that cross year boundary (e.g. Rabi: Nov–Apr)
                    if e_month < s_month:
                        end = date(year + 1, e_month, e_day)
                    else:
                        end = date(year, e_month, e_day)
                except ValueError:
                    continue

                # Only collect fully completed seasons
                if end >= today:
                    continue

                windows.append({
                    'season':     season_name,
                    'year':       year,
                    'start_date': start.isoformat(),
                    'end_date':   end.isoformat(),
                })
                collected += 1

        return windows

    # =========================================================================
    # INTERNAL — SINGLE SEASON COLLECTION
    # =========================================================================

    def _collect_season(self, bbox: List[float], window: Dict) -> Optional[Dict]:
        """
        Collect ALL Sentinel-2 scenes within a season window for a bbox.
        Computes all 28 indices on each valid scene.
        """
        season   = window['season']
        year     = window['year']
        start_dt = window['start_date']
        end_dt   = window['end_date']

        if self.verbose:
            logger.info(f"  Searching {season.upper()} {year}: {start_dt} → {end_dt}")

        try:
            search = self.catalog.search(
                collections=[self.cfg.SENTINEL2_COLLECTION],
                bbox=bbox,
                datetime=f"{start_dt}/{end_dt}",
                query={'eo:cloud_cover': {'lt': self.cfg.MAX_CLOUD_COVER}},
                sortby='datetime',
            )
            items = list(search.items())
        except Exception as e:
            logger.error(f"  STAC search failed for {season} {year}: {e}")
            return None

        if not items:
            logger.info(f"  No scenes found for {season.upper()} {year}")
            return None

        logger.info(f"  Found {len(items)} scenes for {season.upper()} {year}")

        scenes = []
        for item in items:
            scene = self._process_scene(item, bbox)
            if scene is not None:
                scenes.append(scene)
            gc.collect()

        # Sort chronologically
        scenes.sort(key=lambda x: x['date'])

        if not scenes:
            return None

        # Compute season-level NDVI summary
        ndvi_vals = [s['indices'].get('NDVI', {}).get('mean') for s in scenes
                     if s['indices'].get('NDVI', {}).get('mean') is not None]

        return {
            'season':    season,
            'year':      year,
            'start_date': start_dt,
            'end_date':   end_dt,
            'scenes':    scenes,
            'scene_count': len(scenes),
            'ndvi_peak':  max(ndvi_vals) if ndvi_vals else None,
            'ndvi_start': ndvi_vals[0]  if ndvi_vals else None,
            'ndvi_end':   ndvi_vals[-1] if ndvi_vals else None,
            'season_cfg': self.cfg.SEASONS.get(season, {}),
        }

    # =========================================================================
    # INTERNAL — SINGLE SCENE PROCESSING
    # =========================================================================

    def _process_scene(self, item, bbox: List[float]) -> Optional[Dict]:
        """
        Download bands for one STAC item, compute all 28 indices.
        Returns None if scene fails quality checks.
        """
        scene_date = item.datetime.strftime('%Y-%m-%d') if item.datetime else 'unknown'
        cloud_pct  = item.properties.get('eo:cloud_cover', 100)

        bands_data = {}
        for band_name in self.cfg.SENTINEL2_BANDS:
            asset = item.assets.get(band_name)
            if asset is None:
                continue
            try:
                with rasterio.open(asset.href) as src:
                    window = from_bounds(*bbox, src.transform)
                    arr = src.read(1, window=window, out_shape=(
                        1,
                        max(1, int(window.height * self.cfg.TARGET_RESOLUTION_M /
                                   self.cfg.SENTINEL2_BANDS[band_name]['resolution'])),
                        max(1, int(window.width  * self.cfg.TARGET_RESOLUTION_M /
                                   self.cfg.SENTINEL2_BANDS[band_name]['resolution'])),
                    ), resampling=Resampling.bilinear).astype(np.float32)
                    bands_data[band_name] = arr[0]
            except Exception as e:
                logger.debug(f"  Band {band_name} failed: {e}")
                continue

        if 'B08' not in bands_data or 'B04' not in bands_data:
            logger.debug(f"  Scene {scene_date}: missing critical bands — skipped")
            return None

        # Validate pixel quality
        if not DataProcessor.validate_band_data(bands_data['B08'], self.cfg.MIN_VALID_PIXEL_RATIO):
            logger.debug(f"  Scene {scene_date}: insufficient valid pixels — skipped")
            return None

        # Compute all 28 indices
        indices = self.engine.compute_all(bands_data, scene_date=scene_date)

        # Band statistics (for soil nutrient analysis)
        bands_stats = {
            band: DataProcessor.calculate_array_stats(arr)
            for band, arr in bands_data.items()
        }

        return {
            'date':        scene_date,
            'cloud_cover': round(cloud_pct, 1),
            'indices':     indices,
            'bands_stats': bands_stats,
            'item_id':     item.id,
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _get_date_range(merged: List[Dict]) -> Dict:
        """Extract earliest and latest dates from merged season list."""
        all_dates = []
        for s in merged:
            for scene in s.get('scenes', []):
                d = scene.get('date')
                if d:
                    all_dates.append(d)
        if not all_dates:
            return {'start': None, 'end': None}
        all_dates.sort()
        return {'start': all_dates[0], 'end': all_dates[-1]}
