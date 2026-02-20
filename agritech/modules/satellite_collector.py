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
        self.catalog  = pystac_client.Client.open(self.cfg.STAC_API_URL)
        self.band_res = {
            band: info['resolution']
            for band, info in self.cfg.SENTINEL2_BANDS.items()
        }

        logger.info(f"✔ ZUTO Satellite Collector v1.0 initialized")
        logger.info(f"  Mode: COLLECT ALL TIMESTAMPS (no scene cap)")
        logger.info(f"  Cloud threshold: {self.cfg.MAX_CLOUD_COVER}%")
        logger.info(f"  Indices per scene: {len(self.cfg.ALL_INDICES)}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def collect_field_data(
        self,
        latitude:      Optional[float] = None,
        longitude:     Optional[float] = None,
        field_area_ha: Optional[float] = None,
        geometry:      Optional[object] = None,
        num_seasons:   int = None,
        seasons:       Optional[List[str]] = None,
    ) -> Dict:
        """
        Collect ALL Sentinel-2 timestamps for a field across crop seasons.

        Args:
            latitude, longitude: Field centre point (WGS84)
            field_area_ha:       Field area in hectares (for bbox buffer)
            geometry:            Shapely Polygon or GeoJSON (alternative to lat/lon)
            num_seasons:         Number of past seasons to collect
            seasons:             Which seasons to include. Default: all 3.
                                 Options: ['kharif', 'rabi', 'zaid']

        Returns:
            {
              'location': {lat, lon},
              'bbox': [w, s, e, n],
              'field_area_ha': float,
              'seasonal_data': [{season, year, start_date, end_date, scenes[]}],
              'merged_seasons': [{...cross-season resolved...}],
              'summary': {total_scenes, scenes_per_season, ...},
              'collection_date': ISO string,
            }
        """
        if num_seasons is None:
            num_seasons = self.cfg.NUM_SEASONS_HISTORY
        if seasons is None:
            seasons = list(self.cfg.SEASONS.keys())

        logger.info(f"\n{'='*70}")
        logger.info(f"ZUTO SATELLITE COLLECTION — {num_seasons} SEASONS")
        logger.info(f"Seasons: {seasons}")
        logger.info(f"{'='*70}")

        # ── Resolve bounding box ──────────────────────────────────────────
        bbox, centroid_lat, centroid_lon, field_area = self._resolve_geometry(
            latitude, longitude, field_area_ha, geometry
        )
        logger.info(f"Location: ({centroid_lat:.5f}, {centroid_lon:.5f}) | "
                    f"Area: {field_area:.2f} ha")
        logger.info(f"BBox: {[round(v, 5) for v in bbox]}")

        # ── Generate season windows ───────────────────────────────────────
        season_windows = self._generate_season_windows(num_seasons, seasons)
        logger.info(f"\nSeason windows: {len(season_windows)}")
        for w in season_windows:
            logger.info(f"  {w['season'].upper():10s} {w['year']}  "
                        f"{w['start_date']} → {w['end_date']}")

        # ── Collect data per season ───────────────────────────────────────
        seasonal_data = []
        for i, win in enumerate(season_windows):
            logger.info(f"\n[{i+1}/{len(season_windows)}] "
                        f"{win['season'].upper()} {win['year']}  "
                        f"({win['start_date']} → {win['end_date']})")
            try:
                result = self._collect_season(
                    bbox,
                    win['start_date'], win['end_date'],
                    win['season'], win['year'],
                )
                seasonal_data.append(result)
                logger.info(f"  ✔ Collected {result['n_scenes']} scenes | "
                            f"Peak NDVI: {result['ndvi_peak']:.3f}")
            except Exception as e:
                logger.error(f"  ✗ Season failed: {e}")
                seasonal_data.append(
                    self._empty_season(win['season'], win['year'],
                                       win['start_date'], win['end_date'], str(e))
                )
            gc.collect()

        # ── Cross-season continuity analysis ─────────────────────────────
        merged_seasons = self._cross_season_analysis(seasonal_data)

        # ── Summary ──────────────────────────────────────────────────────
        total_scenes    = sum(s.get('n_scenes', 0) for s in seasonal_data)
        seasons_with_data = sum(1 for s in seasonal_data
                                if s.get('n_scenes', 0) >= self.cfg.MIN_SCENES_PER_SEASON)
        cross_season    = sum(1 for m in merged_seasons if m.get('is_cross_season'))

        logger.info(f"\n{'='*70}")
        logger.info(f"COLLECTION COMPLETE — ZUTO AGRITECH")
        logger.info(f"  Total scenes collected:    {total_scenes}")
        logger.info(f"  Seasons with valid data:   {seasons_with_data}/{len(seasonal_data)}")
        logger.info(f"  Avg scenes/season:         "
                    f"{total_scenes/max(seasons_with_data,1):.1f}")
        logger.info(f"  Cross-season crops:        {cross_season}")
        logger.info(f"{'='*70}")

        return {
            'location':        {'latitude': centroid_lat, 'longitude': centroid_lon},
            'bbox':            bbox,
            'field_area_ha':   field_area,
            'num_seasons':     num_seasons,
            'seasons_requested': seasons,
            'seasonal_data':   seasonal_data,
            'merged_seasons':  merged_seasons,
            'collection_date': datetime.now().isoformat(),
            'summary': {
                'total_scenes':          total_scenes,
                'seasons_with_data':     seasons_with_data,
                'total_seasons':         len(seasonal_data),
                'cross_season_crops':    cross_season,
                'avg_scenes_per_season': round(total_scenes / max(seasons_with_data, 1), 1),
                'indices_per_scene':     len(self.cfg.ALL_INDICES),
            },
        }

    def collect_regional_data(
        self,
        bbox:        List[float],
        num_seasons: int = 4,
        seasons:     Optional[List[str]] = None,
    ) -> Dict:
        """
        Collect data for a regional bounding box (district/block level).
        Used for crop classification and regional analytics modules.

        Args:
            bbox:    [min_lon, min_lat, max_lon, max_lat]
            num_seasons: historical seasons
            seasons: which seasons to include

        Returns:
            Same structure as collect_field_data but without field-level detail
        """
        if seasons is None:
            seasons = ['kharif', 'rabi']  # Zaid less relevant at regional scale

        lat_c = (bbox[1] + bbox[3]) / 2
        lon_c = (bbox[0] + bbox[2]) / 2
        area_approx = (
            GeometryUtils.calculate_distance_km(bbox[1], bbox[0], bbox[1], bbox[2]) *
            GeometryUtils.calculate_distance_km(bbox[1], bbox[0], bbox[3], bbox[0])
        ) * 100  # approx ha

        logger.info(f"ZUTO Regional Collection: {bbox} | ~{area_approx:.0f} ha")

        season_windows = self._generate_season_windows(num_seasons, seasons)
        seasonal_data  = []

        for win in season_windows:
            try:
                result = self._collect_season(
                    bbox, win['start_date'], win['end_date'],
                    win['season'], win['year'],
                )
                seasonal_data.append(result)
            except Exception as e:
                logger.error(f"Regional season failed: {e}")
                seasonal_data.append(
                    self._empty_season(win['season'], win['year'],
                                       win['start_date'], win['end_date'], str(e))
                )
            gc.collect()

        merged = self._cross_season_analysis(seasonal_data)

        return {
            'location':       {'latitude': lat_c, 'longitude': lon_c},
            'bbox':           bbox,
            'field_area_ha':  area_approx,
            'is_regional':    True,
            'seasonal_data':  seasonal_data,
            'merged_seasons': merged,
            'collection_date': datetime.now().isoformat(),
            'summary': {
                'total_scenes': sum(s.get('n_scenes', 0) for s in seasonal_data),
                'seasons':      len(seasonal_data),
            },
        }

    # =========================================================================
    # SEASON WINDOW GENERATION
    # =========================================================================

    def _generate_season_windows(
        self, num_seasons: int, seasons: List[str]
    ) -> List[Dict]:
        """
        Generate the last `num_seasons` windows for the requested season types.
        Returns list sorted oldest → newest.

        Kharif: Jun 1  → Nov 30 (contained in one calendar year)
        Rabi:   Nov 1  → Apr 30 (spans two calendar years)
        Zaid:   Mar 1  → Jun 30 (contained in one calendar year)
        """
        today     = datetime.now().date()
        cfg_seasons = self.cfg.SEASONS
        windows   = []
        seen      = set()

        for year_offset in range(self.cfg.MAX_YEARS_BACK + 1):
            yr = today.year - year_offset

            for season_name in ['kharif', 'rabi', 'zaid']:
                if season_name not in seasons:
                    continue
                cfg = cfg_seasons[season_name]

                if season_name == 'rabi':
                    # Nov yr → Apr yr+1
                    start = date(yr,   cfg['start_month'], cfg['start_day'])
                    end   = date(yr+1, cfg['end_month'],   cfg['end_day'])
                    label_year = yr
                else:
                    start = date(yr, cfg['start_month'], cfg['start_day'])
                    end   = date(yr, cfg['end_month'],   cfg['end_day'])
                    label_year = yr

                key = (season_name, label_year)
                if key in seen:
                    continue
                if end >= today:
                    continue  # skip incomplete seasons

                seen.add(key)
                windows.append({
                    'season':       season_name,
                    'year':         label_year,
                    'start_date':   start.strftime('%Y-%m-%d'),
                    'end_date':     end.strftime('%Y-%m-%d'),
                    '_end_obj':     end,
                })

        # Sort newest → oldest, take num_seasons, reverse to oldest → newest
        windows.sort(key=lambda x: x['_end_obj'], reverse=True)
        selected = windows[:num_seasons]
        selected.sort(key=lambda x: x['start_date'])

        for w in selected:
            del w['_end_obj']

        return selected

    # =========================================================================
    # PER-SEASON COLLECTION — ALL TIMESTAMPS
    # =========================================================================

    def _collect_season(
        self,
        bbox:       List[float],
        start_date: str,
        end_date:   str,
        season:     str,
        year:       int,
    ) -> Dict:
        """
        Collect ALL valid Sentinel-2 scenes for a season window.

        Unlike reference pipeline (which bins into N slots and picks 1 per bin),
        ZUTO collects EVERY scene that meets cloud threshold.
        This gives maximum time-series density for:
          - Better index time-series interpolation
          - More accurate NDVI phenology curve fitting
          - More ML features for crop classification
        """
        candidates = self._search_scenes(bbox, start_date, end_date)
        logger.info(f"  Found {len(candidates)} candidate scenes")

        if not candidates:
            return self._empty_season(season, year, start_date, end_date,
                                      'No scenes found')

        # Sort ALL candidates chronologically — no binning, no dropping
        candidates.sort(
            key=lambda c: c.datetime if c.datetime else datetime.min
        )

        processed_scenes = []
        for i, item in enumerate(candidates):
            try:
                scene_date  = item.datetime.strftime('%Y-%m-%d') if item.datetime else '?'
                cloud_pct   = float(item.properties.get('eo:cloud_cover', 0))

                logger.info(f"    [{i+1:03d}/{len(candidates)}] {scene_date}  "
                            f"cloud={cloud_pct:.1f}%")

                band_data = self._download_bands(item, bbox)

                if len(band_data) < 4:
                    logger.warning(f"       ⚠ Only {len(band_data)} bands — skip")
                    continue

                # Compute FULL 28-index suite on this scene
                index_stats = self.engine.compute_all(band_data, scene_date=scene_date)

                # Band summary stats
                band_stats = {
                    k: DataProcessor.calculate_array_stats(v)
                    for k, v in band_data.items()
                }

                # Key metrics for quick access
                ndvi  = index_stats.get('NDVI',   {}).get('mean', np.nan)
                rendvi= index_stats.get('RENDVI', {}).get('mean', np.nan)
                evi   = index_stats.get('EVI',    {}).get('mean', np.nan)
                ndmi  = index_stats.get('NDMI',   {}).get('mean', np.nan)
                si    = index_stats.get('SI',     {}).get('mean', np.nan)

                logger.info(
                    f"       ✔ NDVI={ndvi:.3f}  RENDVI={rendvi:.3f}  "
                    f"EVI={evi:.3f}  NDMI={ndmi:.3f}  SI={si:.3f}"
                )

                processed_scenes.append({
                    'date':        scene_date,
                    'cloud_cover': cloud_pct,
                    'bands':       band_stats,
                    'indices':     {
                        name: {
                            'mean': stats.get('mean', np.nan),
                            'std':  stats.get('std',  np.nan),
                            'p10':  stats.get('p10',  np.nan),
                            'p50':  stats.get('p50',  np.nan),
                            'p90':  stats.get('p90',  np.nan),
                        }
                        for name, stats in index_stats.items()
                    },
                    'n_bands': len(band_data),
                })

                del band_data   # free memory
                gc.collect()

            except Exception as e:
                logger.warning(f"       ✗ Scene failed: {str(e)[:80]}")
                continue

        # Temporal gap diagnostics
        self._log_temporal_gaps(processed_scenes)

        # Compute NDVI window stats for cross-season analysis
        return {
            'season':      season,
            'year':        year,
            'start_date':  start_date,
            'end_date':    end_date,
            'scenes':      processed_scenes,
            'n_scenes':    len(processed_scenes),
            'ndvi_start':  self._window_ndvi(processed_scenes, 'start'),
            'ndvi_end':    self._window_ndvi(processed_scenes, 'end'),
            'ndvi_peak':   max(
                (s['indices'].get('NDVI', {}).get('mean', 0)
                 for s in processed_scenes), default=0.0
            ),
            'rendvi_peak': max(
                (s['indices'].get('RENDVI', {}).get('mean', 0)
                 for s in processed_scenes), default=0.0
            ),
        }

    # =========================================================================
    # SCENE SEARCH
    # =========================================================================

    def _search_scenes(
        self, bbox: List[float], start_date: str, end_date: str
    ) -> List:
        """Search STAC for all Sentinel-2 scenes within cloud threshold."""
        try:
            search = self.catalog.search(
                collections=[self.cfg.SENTINEL2_COLLECTION],
                bbox=bbox,
                datetime=f"{start_date}/{end_date}",
                query={"eo:cloud_cover": {"lt": self.cfg.MAX_CLOUD_COVER}},
                limit=500,  # high limit — we want everything
            )
            items = list(search.get_items())
            items.sort(key=lambda i: i.datetime if i.datetime else datetime.min)
            return items
        except Exception as e:
            logger.error(f"  Scene search error: {e}")
            return []

    # =========================================================================
    # BAND DOWNLOAD
    # =========================================================================

    def _download_bands(
        self,
        item,
        bbox_wgs84: List[float],
        target_resolution: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Download all available Sentinel-2 bands for an item.
        Resamples all bands to target_resolution (default 10m).
        """
        band_data = {}

        for band_name, native_res in self.band_res.items():
            if band_name not in item.assets:
                continue
            try:
                asset = planetary_computer.sign(item.assets[band_name])

                with rasterio.open(asset.href) as src:
                    # Reproject bbox to source CRS
                    bbox_t = transform_bounds('EPSG:4326', src.crs, *bbox_wgs84)
                    window = from_bounds(*bbox_t, transform=src.transform)
                    src_win = Window(0, 0, src.width, src.height)
                    window  = window.intersection(src_win)

                    if window.width <= 0 or window.height <= 0:
                        continue

                    # Integer window
                    window = Window(
                        int(np.floor(window.col_off)),
                        int(np.floor(window.row_off)),
                        max(1, int(np.ceil(window.width))),
                        max(1, int(np.ceil(window.height))),
                    )
                    if window.width < 3 or window.height < 3:
                        continue

                    # Output shape after resampling to target_resolution
                    out_h, out_w = DataProcessor.resample_to_resolution(
                        np.zeros((int(window.height), int(window.width))),
                        native_res, target_resolution,
                    )

                    data = src.read(
                        1, window=window,
                        out_shape=(out_h, out_w),
                        resampling=Resampling.bilinear,
                    ).astype(np.float32)

                    # Clean: remove nodata, clip to valid range
                    data = DataProcessor.clean_satellite_data(
                        data, nodata_value=src.nodata,
                        min_value=0.0, max_value=10000.0,
                    )

                    if DataProcessor.validate_band_data(
                        data,
                        min_valid_ratio=self.cfg.MIN_VALID_PIXEL_RATIO
                    ):
                        band_data[band_name] = data

            except Exception as e:
                logger.debug(f"  Band {band_name} download failed: {str(e)[:60]}")
                continue

        return band_data

    # =========================================================================
    # CROSS-SEASON CONTINUITY ANALYSIS
    # =========================================================================

    def _cross_season_analysis(self, seasonal_data: List[Dict]) -> List[Dict]:
        """
        Analyse consecutive season pairs to detect long-duration crops
        (Sugarcane, Banana, Tur, Cotton) that span across season boundaries.

        Logic:
          - If NDVI at END of season-A is HIGH and NDVI at START of season-B
            is also HIGH → same crop continuing (no harvest gap)
          - Merge these two seasons into one crop event record
          - Prevents double-counting the same crop as two plantings
        """
        if not seasonal_data:
            return []

        CONTINUITY = self.cfg.CROSS_SEASON_NDVI_CONTINUITY         # 0.38
        DROP       = self.cfg.CROSS_SEASON_NDVI_DROP_FOR_NEW_CROP   # 0.15

        merged    = []
        skip_next = False

        for i, season in enumerate(seasonal_data):
            if skip_next:
                skip_next = False
                continue

            if i + 1 >= len(seasonal_data):
                merged.append(self._single_season_record(season))
                continue

            nxt = seasonal_data[i + 1]

            if not self._are_consecutive(season, nxt):
                merged.append(self._single_season_record(season))
                continue

            end_ndvi   = season.get('ndvi_end',   0.0)
            start_ndvi = nxt.get('ndvi_start', 0.0)
            ndvi_drop  = end_ndvi - start_ndvi

            if end_ndvi >= CONTINUITY and start_ndvi >= CONTINUITY:
                # Long-duration crop continues across boundary
                logger.info(
                    f"  ⟳ CROSS-SEASON LONG-DURATION: "
                    f"{season['season']} {season['year']} → "
                    f"{nxt['season']} {nxt['year']} "
                    f"(end={end_ndvi:.3f}, start={start_ndvi:.3f})"
                )
                merged.append(self._merged_record(season, nxt, 'long_duration'))
                skip_next = True

            elif (end_ndvi >= CONTINUITY
                  and start_ndvi >= CONTINUITY * 0.7
                  and ndvi_drop < DROP):
                # Late-sown crop straddling the season boundary
                logger.info(
                    f"  ⟳ CROSS-SEASON LATE-SOWN: "
                    f"{season['season']} {season['year']} → "
                    f"{nxt['season']} {nxt['year']} "
                    f"(end={end_ndvi:.3f}, start={start_ndvi:.3f})"
                )
                merged.append(self._merged_record(season, nxt, 'late_sown'))
                skip_next = True

            else:
                merged.append(self._single_season_record(season))

        return merged

    # =========================================================================
    # GEOMETRY HELPERS
    # =========================================================================

    def _resolve_geometry(
        self,
        latitude:      Optional[float],
        longitude:     Optional[float],
        field_area_ha: Optional[float],
        geometry:      Optional[object],
    ) -> Tuple[List[float], float, float, float]:
        """Resolve input to (bbox, centroid_lat, centroid_lon, area_ha)."""
        if geometry is not None:
            bbox = GeometryUtils.calculate_bbox_from_geometry(geometry)
            lat, lon = GeometryUtils.calculate_centroid_from_geometry(geometry)
            area = GeometryUtils.calculate_area_from_geometry(geometry)
        elif latitude is not None and longitude is not None:
            area = field_area_ha or 1.0
            bbox = GeometryUtils.calculate_bbox_from_point(latitude, longitude, area)
            lat, lon = latitude, longitude
        else:
            raise ValueError("Provide (latitude, longitude) or geometry")

        if not GeometryUtils.validate_bbox(bbox):
            raise ValueError(f"Invalid bbox: {bbox}")

        return bbox, lat, lon, area

    # =========================================================================
    # STATIC HELPERS
    # =========================================================================

    @staticmethod
    def _window_ndvi(scenes: List[Dict], position: str) -> float:
        """Return mean NDVI of first or last 3 scenes."""
        if not scenes:
            return 0.0
        n = min(3, len(scenes))
        subset = scenes[:n] if position == 'start' else scenes[-n:]
        vals = [s['indices'].get('NDVI', {}).get('mean', 0.0) for s in subset]
        return float(np.nanmean(vals)) if vals else 0.0

    @staticmethod
    def _are_consecutive(a: Dict, b: Dict) -> bool:
        """Check if season b starts immediately after season a ends."""
        try:
            end_a   = datetime.strptime(a['end_date'],   '%Y-%m-%d')
            start_b = datetime.strptime(b['start_date'], '%Y-%m-%d')
            return 0 <= (start_b - end_a).days <= 7
        except Exception:
            return False

    @staticmethod
    def _single_season_record(s: Dict) -> Dict:
        return {
            'season_keys':       [(s['season'], s['year'])],
            'is_cross_season':   False,
            'cross_season_type': 'normal',
            'season':            s['season'],
            'year':              s['year'],
            'start_date':        s['start_date'],
            'end_date':          s['end_date'],
            'scenes':            s.get('scenes', []),
            'n_scenes':          s.get('n_scenes', 0),
            'ndvi_peak':         s.get('ndvi_peak', 0.0),
            'rendvi_peak':       s.get('rendvi_peak', 0.0),
        }

    @staticmethod
    def _merged_record(a: Dict, b: Dict, cross_type: str) -> Dict:
        combined = sorted(
            a.get('scenes', []) + b.get('scenes', []),
            key=lambda s: s.get('date', ''),
        )
        return {
            'season_keys':       [(a['season'], a['year']), (b['season'], b['year'])],
            'is_cross_season':   True,
            'cross_season_type': cross_type,
            'season':            f"{a['season']}+{b['season']}",
            'year':              a['year'],
            'start_date':        a['start_date'],
            'end_date':          b['end_date'],
            'scenes':            combined,
            'n_scenes':          len(combined),
            'ndvi_peak':         max(a.get('ndvi_peak', 0), b.get('ndvi_peak', 0)),
            'rendvi_peak':       max(a.get('rendvi_peak', 0), b.get('rendvi_peak', 0)),
        }

    @staticmethod
    def _empty_season(season, year, start_date, end_date, reason='') -> Dict:
        return {
            'season': season, 'year': year,
            'start_date': start_date, 'end_date': end_date,
            'scenes': [], 'n_scenes': 0,
            'ndvi_start': 0.0, 'ndvi_end': 0.0, 'ndvi_peak': 0.0,
            'rendvi_peak': 0.0, 'warning': reason,
        }

    def _log_temporal_gaps(self, scenes: List[Dict]):
        """Log temporal gap diagnostics for scene collection."""
        if len(scenes) < 2:
            return
        try:
            dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in scenes]
            gaps  = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg   = sum(gaps) / len(gaps)
            mx    = max(gaps)
            flag  = f"  ⚠ MAX GAP={mx}d > {self.cfg.MAX_GAP_DAYS}d!" \
                    if mx > self.cfg.MAX_GAP_DAYS else "  ✔ Gaps OK"
            logger.info(f"  Temporal gaps: avg={avg:.1f}d | max={mx}d{flag}")
        except Exception:
            pass
