"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Module A: Crop Health Monitor

Assesses field-level crop health on every satellite timestamp using:
  - Full 28-index spectral suite (from SpectralIndexEngine)
  - Crop-specific NDVI growth curves (from CropGrowthCurves)
  - Alert generation for nitrogen, moisture, salinity, senescence
  - Time-series health trajectory for dashboard visualization
  - Stage-aware scoring (stress during critical stages penalized more)

Output per field per season:
  - health_score (0–100)
  - health_category: Excellent / Good / Moderate / Poor / Critical
  - per-timestamp health trajectory
  - alerts list with severity and recommended actions
  - nutrient interpretation (N, P/K proxy, EC, OC)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from config.agri_config import ZutoAgriConfig

logger = logging.getLogger(__name__)


class CropHealthMonitor:
    """
    Assesses crop health from dense satellite time-series.
    Uses ALL collected timestamps for high-resolution phenology tracking.
    """

    def __init__(self):
        self.cfg = ZutoAgriConfig
        self.health_cats = self.cfg.HEALTH_CATEGORIES
        self.alert_thresholds = self.cfg.ALERT_THRESHOLDS

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def assess_season_health(
        self,
        season_data: Dict,
        crop_name:   Optional[str] = None,
        sowing_date: Optional[str] = None,
    ) -> Dict:
        """
        Assess crop health for a full season using all collected timestamps.

        Args:
            season_data: One season entry from ZutoSatelliteCollector output
            crop_name:   Known or predicted crop (optional but improves accuracy)
            sowing_date: Crop sowing date for growth-stage alignment

        Returns:
            {
              'season': str, 'year': int,
              'crop': str,
              'health_score': float (0–100),
              'health_category': str,
              'trajectory': [{date, ndvi, rendvi, health_score, stage, alerts}],
              'alerts': [{type, severity, date, message, action}],
              'nutrient_summary': {N, EC, OC, moisture},
              'peak_health': {date, score, ndvi},
              'current_health': {date, score, ndvi},   # most recent scene
              'summary': str,
            }
        """
        scenes    = season_data.get('scenes', [])
        season    = season_data.get('season', 'unknown')
        year      = season_data.get('year', 0)
        crop      = crop_name or 'Unknown'

        if not scenes:
            return self._empty_assessment(season, year, crop, 'No scenes available')

        logger.info(f"\nCROP HEALTH ASSESSMENT — {season.upper()} {year} | "
                    f"Crop: {crop} | Scenes: {len(scenes)}")

        # ── Build time-series DataFrame ───────────────────────────────────
        ts = self._build_timeseries(scenes)

        # ── Compute per-scene health scores ───────────────────────────────
        trajectory  = self._compute_trajectory(ts, crop, sowing_date, season)

        # ── Generate alerts ───────────────────────────────────────────────
        alerts = self._generate_alerts(trajectory, crop)

        # ── Season-level health score (weighted recent + peak) ────────────
        health_score, health_cat = self._season_health_score(trajectory)

        # ── Nutrient summary (last 3 scenes average) ─────────────────────
        nutrient_summary = self._nutrient_summary(scenes[-3:] if len(scenes) >= 3 else scenes)

        # ── Peak and current health ───────────────────────────────────────
        peak    = max(trajectory, key=lambda x: x['health_score'], default={})
        current = trajectory[-1] if trajectory else {}

        result = {
            'season':           season,
            'year':             year,
            'crop':             crop,
            'n_scenes':         len(scenes),
            'health_score':     round(health_score, 1),
            'health_category':  health_cat,
            'trajectory':       trajectory,
            'alerts':           alerts,
            'nutrient_summary': nutrient_summary,
            'peak_health': {
                'date':         peak.get('date'),
                'score':        round(peak.get('health_score', 0), 1),
                'ndvi':         round(peak.get('ndvi', 0), 4),
                'rendvi':       round(peak.get('rendvi', 0), 4),
            },
            'current_health': {
                'date':         current.get('date'),
                'score':        round(current.get('health_score', 0), 1),
                'ndvi':         round(current.get('ndvi', 0), 4),
                'stage':        current.get('stage', 'Unknown'),
                'active_alerts': [a['type'] for a in alerts if not a.get('resolved')],
            },
            'summary': self._generate_summary(health_score, health_cat, crop, alerts),
        }

        logger.info(f"  Health Score: {health_score:.1f} ({health_cat}) | "
                    f"Alerts: {len(alerts)}")
        return result

    def assess_multi_season(
        self,
        merged_seasons: List[Dict],
        crop_name:      Optional[str] = None,
    ) -> List[Dict]:
        """
        Run health assessment across all seasons (including cross-season crops).

        Args:
            merged_seasons: List from satellite_collector.collect_field_data()
                            (cross-season resolved)
            crop_name:      Crop name if known

        Returns:
            List of per-season assessment dicts
        """
        assessments = []
        for season in merged_seasons:
            try:
                assessment = self.assess_season_health(season, crop_name)
                assessments.append(assessment)
            except Exception as e:
                logger.error(f"Assessment failed for {season.get('season')} "
                             f"{season.get('year')}: {e}")
        return assessments

    # =========================================================================
    # TIME-SERIES BUILDING
    # =========================================================================

    def _build_timeseries(self, scenes: List[Dict]) -> pd.DataFrame:
        """Build a tidy DataFrame from scene list for analysis."""
        records = []
        for s in scenes:
            idx = s.get('indices', {})
            rec = {
                'date':        s.get('date'),
                'cloud_cover': s.get('cloud_cover', 0),
                # Core health indices
                'ndvi':        self._safe_get(idx, 'NDVI',       'mean'),
                'rendvi':      self._safe_get(idx, 'RENDVI',     'mean'),
                'evi':         self._safe_get(idx, 'EVI',        'mean'),
                'gndvi':       self._safe_get(idx, 'GNDVI',      'mean'),
                'ndmi':        self._safe_get(idx, 'NDMI',       'mean'),
                'osavi':       self._safe_get(idx, 'OSAVI',      'mean'),
                # Nutrient indices
                'ci_rededge':  self._safe_get(idx, 'CI_REDEDGE', 'mean'),
                'tgi':         self._safe_get(idx, 'TGI',        'mean'),
                'msr':         self._safe_get(idx, 'MSR',        'mean'),
                'si':          self._safe_get(idx, 'SI',         'mean'),
                'ndsi':        self._safe_get(idx, 'NDSI',       'mean'),
                'cmr':         self._safe_get(idx, 'CMR',        'mean'),
                'cai':         self._safe_get(idx, 'CAI',        'mean'),
                'ior':         self._safe_get(idx, 'IOR',        'mean'),
                'bsi':         self._safe_get(idx, 'BSI',        'mean'),
                # Moisture & stress
                'ndwi':        self._safe_get(idx, 'NDWI',       'mean'),
                'nmdi':        self._safe_get(idx, 'NMDI',       'mean'),
                'msi_stress':  self._safe_get(idx, 'MSI_STRESS', 'mean'),
                'psri':        self._safe_get(idx, 'PSRI',       'mean'),
                'nddi':        self._safe_get(idx, 'NDDI',       'mean'),
            }
            records.append(rec)

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    # =========================================================================
    # TRAJECTORY COMPUTATION
    # =========================================================================

    def _compute_trajectory(
        self,
        ts:          pd.DataFrame,
        crop:        str,
        sowing_date: Optional[str],
        season:      str,
    ) -> List[Dict]:
        """Compute per-scene health score and stage for full trajectory."""
        trajectory = []

        # Load crop growth curve if available
        crop_curve    = self._get_crop_curve(crop)
        sowing_dt     = self._parse_date(sowing_date)
        season_start  = ts['date'].min()

        for _, row in ts.iterrows():
            ndvi   = row.get('ndvi', np.nan)
            rendvi = row.get('rendvi', np.nan)
            ndmi   = row.get('ndmi', np.nan)
            si     = row.get('si', np.nan)
            psri   = row.get('psri', np.nan)

            if pd.isna(ndvi):
                continue

            # ── Days since sowing (for stage identification) ──────────────
            ref_date = sowing_dt if sowing_dt else season_start
            days_since_sowing = max(0, (row['date'] - ref_date).days)

            # ── Expected NDVI from growth curve ──────────────────────────
            expected_ndvi, stage = self._interpolate_curve(
                crop_curve, days_since_sowing
            )

            # ── Per-scene health score (multi-index weighted) ─────────────
            score = self._compute_scene_health(
                ndvi, rendvi, ndmi, si, psri, expected_ndvi, crop
            )

            # ── Per-scene alerts ──────────────────────────────────────────
            scene_alerts = self._scene_alerts(row, stage, crop)

            trajectory.append({
                'date':              row['date'].strftime('%Y-%m-%d'),
                'ndvi':              float(ndvi) if not pd.isna(ndvi) else None,
                'rendvi':            float(rendvi) if not pd.isna(rendvi) else None,
                'evi':               float(row.get('evi', np.nan)) if not pd.isna(row.get('evi', np.nan)) else None,
                'ndmi':              float(ndmi) if not pd.isna(ndmi) else None,
                'si':                float(si) if not pd.isna(si) else None,
                'psri':              float(psri) if not pd.isna(psri) else None,
                'expected_ndvi':     round(expected_ndvi, 3) if expected_ndvi else None,
                'stage':             stage,
                'days_since_sowing': days_since_sowing,
                'health_score':      round(score, 1),
                'health_category':   self._categorize(score),
                'cloud_cover':       float(row.get('cloud_cover', 0)),
                'scene_alerts':      scene_alerts,
            })

        return trajectory

    def _compute_scene_health(
        self,
        ndvi:          float,
        rendvi:        float,
        ndmi:          float,
        si:            float,
        psri:          float,
        expected_ndvi: float,
        crop:          str,
    ) -> float:
        """
        Compute health score (0–100) for a single scene.

        Weighted components:
          - NDVI performance vs. expected (30%)
          - Nitrogen status via RENDVI (25%)
          - Moisture status via NDMI (20%)
          - Salinity stress via SI (15%)
          - Senescence via PSRI (10%)
        """
        components = []

        # ── 1. NDVI vs expected (30%) ─────────────────────────────────────
        if not pd.isna(ndvi) and expected_ndvi and expected_ndvi > 0:
            ratio = min(ndvi / expected_ndvi, 1.2)   # cap at 120% of expected
            ndvi_score = min(100.0, ratio * 83.33)   # 1.0 ratio → 83.33 → ~85 pts
            components.append(('ndvi_perf', ndvi_score, 0.30))
        elif not pd.isna(ndvi):
            # No curve — score from raw NDVI
            ndvi_score = min(100.0, (ndvi / 0.85) * 100)
            components.append(('ndvi_raw', ndvi_score, 0.30))

        # ── 2. RENDVI nitrogen score (25%) ────────────────────────────────
        if not pd.isna(rendvi):
            if rendvi > 0.60:
                n_score = 100.0
            elif rendvi > 0.40:
                n_score = 70.0 + (rendvi - 0.40) / 0.20 * 30
            elif rendvi > 0.25:
                n_score = 40.0 + (rendvi - 0.25) / 0.15 * 30
            else:
                n_score = max(0.0, rendvi / 0.25 * 40)
            components.append(('nitrogen', n_score, 0.25))

        # ── 3. Moisture score from NDMI (20%) ─────────────────────────────
        if not pd.isna(ndmi):
            if ndmi > 0.40:
                m_score = 100.0
            elif ndmi > 0.20:
                m_score = 70.0 + (ndmi - 0.20) / 0.20 * 30
            elif ndmi > 0.00:
                m_score = 40.0 + ndmi / 0.20 * 30
            else:
                m_score = max(0.0, 40.0 + ndmi * 40)
            components.append(('moisture', m_score, 0.20))

        # ── 4. Salinity score from SI (15%) ───────────────────────────────
        if not pd.isna(si):
            if si < 0.03:
                s_score = 100.0
            elif si < 0.08:
                s_score = 80.0 - (si - 0.03) / 0.05 * 40
            elif si < 0.15:
                s_score = 40.0 - (si - 0.08) / 0.07 * 30
            else:
                s_score = max(0.0, 10.0 - (si - 0.15) * 50)
            components.append(('salinity', s_score, 0.15))

        # ── 5. Senescence score from PSRI (10%) ───────────────────────────
        if not pd.isna(psri):
            if psri < 0.10:
                p_score = 100.0
            elif psri < 0.25:
                p_score = 80.0 - (psri - 0.10) / 0.15 * 40
            elif psri < 0.40:
                p_score = 40.0 - (psri - 0.25) / 0.15 * 30
            else:
                p_score = max(0.0, 10.0)
            components.append(('senescence', p_score, 0.10))

        if not components:
            return 50.0  # default if no indices available

        # Weighted average — normalize weights to sum to 1
        total_weight = sum(w for _, _, w in components)
        score = sum(s * w for _, s, w in components) / total_weight
        return round(max(0.0, min(100.0, score)), 1)

    # =========================================================================
    # ALERT GENERATION
    # =========================================================================

    def _generate_alerts(self, trajectory: List[Dict], crop: str) -> List[Dict]:
        """
        Generate consolidated alerts from full season trajectory.
        Groups individual scene alerts into season-level alerts with duration.
        """
        all_scene_alerts = []
        for point in trajectory:
            for alert in point.get('scene_alerts', []):
                alert['date'] = point['date']
                all_scene_alerts.append(alert)

        if not all_scene_alerts:
            return []

        # Group by alert type, find periods
        by_type = {}
        for a in all_scene_alerts:
            by_type.setdefault(a['type'], []).append(a)

        season_alerts = []
        for alert_type, incidents in by_type.items():
            if not incidents:
                continue
            dates    = [i['date'] for i in incidents]
            severities = [i['severity'] for i in incidents]
            worst_sev  = max(severities,
                             key=lambda s: {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}.get(s, 0))

            season_alerts.append({
                'type':        alert_type,
                'severity':    worst_sev,
                'occurrences': len(incidents),
                'first_date':  min(dates),
                'last_date':   max(dates),
                'resolved':    incidents[-1]['severity'] in ('low',),
                'message':     incidents[0].get('message', ''),
                'action':      incidents[0].get('action', 'Monitor'),
                'crop':        crop,
            })

        # Sort by severity descending
        sev_rank = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        season_alerts.sort(key=lambda a: sev_rank.get(a['severity'], 0), reverse=True)
        return season_alerts

    def _scene_alerts(
        self, row: pd.Series, stage: str, crop: str
    ) -> List[Dict]:
        """Generate alerts for a single scene observation."""
        alerts = []
        thresholds = self.alert_thresholds

        # Nitrogen deficiency (RENDVI based)
        rendvi = row.get('rendvi', np.nan)
        if not pd.isna(rendvi):
            if rendvi < 0.20:
                alerts.append({
                    'type': 'nitrogen_deficiency',
                    'severity': 'critical',
                    'message': f'Severe N deficiency (RENDVI={rendvi:.3f})',
                    'action': 'Apply nitrogen fertilizer immediately',
                })
            elif rendvi < thresholds['nitrogen_deficiency']['threshold']:
                alerts.append({
                    'type': 'nitrogen_deficiency',
                    'severity': 'high' if rendvi < 0.25 else 'medium',
                    'message': f'Nitrogen deficiency detected (RENDVI={rendvi:.3f})',
                    'action': 'Schedule N fertilizer application within 7 days',
                })

        # Water stress (NDMI based)
        ndmi = row.get('ndmi', np.nan)
        if not pd.isna(ndmi):
            if ndmi < -0.10:
                alerts.append({
                    'type': 'severe_water_stress',
                    'severity': 'critical',
                    'message': f'Severe drought/water stress (NDMI={ndmi:.3f})',
                    'action': 'Irrigate immediately — wilting risk',
                })
            elif ndmi < thresholds['water_stress']['threshold']:
                alerts.append({
                    'type': 'water_stress',
                    'severity': 'high' if ndmi < 0.05 else 'medium',
                    'message': f'Water stress detected (NDMI={ndmi:.3f})',
                    'action': 'Schedule irrigation within 3 days',
                })

        # Salinity stress (SI based)
        si = row.get('si', np.nan)
        if not pd.isna(si) and si > 0.12:
            alerts.append({
                'type': 'salinity_stress',
                'severity': 'critical' if si > 0.20 else 'high',
                'message': f'High soil salinity (SI={si:.4f})',
                'action': 'Apply leaching irrigation; consider gypsum amendment',
            })

        # Early senescence (PSRI based)
        psri = row.get('psri', np.nan)
        if not pd.isna(psri) and psri > thresholds['senescence']['threshold']:
            if stage not in ('Harvest Ready', 'Late Maturity', 'Dough Stage'):
                alerts.append({
                    'type': 'early_senescence',
                    'severity': 'high' if psri > 0.35 else 'medium',
                    'message': f'Premature senescence at {stage} (PSRI={psri:.3f})',
                    'action': 'Investigate cause — heat, pest, or disease stress',
                })

        # Bare soil in growing season (BSI based)
        bsi = row.get('bsi', np.nan)
        ndvi = row.get('ndvi', np.nan)
        if not pd.isna(bsi) and bsi > 0.20 and not pd.isna(ndvi) and ndvi < 0.25:
            alerts.append({
                'type': 'poor_establishment',
                'severity': 'medium',
                'message': f'Poor crop establishment (BSI={bsi:.3f}, NDVI={ndvi:.3f})',
                'action': 'Check germination; consider gap-filling or replanting',
            })

        return alerts

    # =========================================================================
    # NUTRIENT SUMMARY
    # =========================================================================

    def _nutrient_summary(self, recent_scenes: List[Dict]) -> Dict:
        """Generate nutrient status summary from most recent scenes."""
        if not recent_scenes:
            return {}

        def avg_index(name):
            vals = [
                s.get('indices', {}).get(name, {}).get('mean', np.nan)
                for s in recent_scenes
            ]
            valid = [v for v in vals if not np.isnan(v)]
            return float(np.mean(valid)) if valid else None

        rendvi_avg = avg_index('RENDVI')
        ci_re_avg  = avg_index('CI_REDEDGE')
        si_avg     = avg_index('SI')
        cai_avg    = avg_index('CAI')
        ndmi_avg   = avg_index('NDMI')
        cmr_avg    = avg_index('CMR')
        ior_avg    = avg_index('IOR')

        def n_status(v):
            if v is None: return 'Unknown'
            if v > 0.60: return 'Adequate'
            if v > 0.40: return 'Mild Deficiency'
            if v > 0.25: return 'Moderate Deficiency'
            return 'Severe Deficiency'

        def ec_status(v):
            if v is None: return 'Unknown'
            if v < 0.05: return 'Non-Saline'
            if v < 0.10: return 'Slightly Saline'
            if v < 0.20: return 'Moderately Saline'
            return 'Highly Saline'

        def moisture_status(v):
            if v is None: return 'Unknown'
            if v > 0.40: return 'Adequate'
            if v > 0.20: return 'Mild Stress'
            if v > 0.00: return 'Moderate Stress'
            return 'Severe Drought'

        return {
            'Nitrogen': {
                'status':   n_status(rendvi_avg),
                'RENDVI':   round(rendvi_avg, 4) if rendvi_avg else None,
                'CI_RedEdge': round(ci_re_avg, 4) if ci_re_avg else None,
            },
            'Phosphorus_Potassium': {
                'status':   'Monitor' if rendvi_avg and rendvi_avg > 0.4 else 'Check',
                'note':     'P/K assessed indirectly via EVI/MSR — ground truth recommended',
            },
            'Salinity_EC': {
                'status':   ec_status(si_avg),
                'SI':       round(si_avg, 4) if si_avg else None,
            },
            'Organic_Carbon': {
                'status':   'Adequate' if (cai_avg and cai_avg > 0.01) else 'Low',
                'CAI':      round(cai_avg, 4) if cai_avg else None,
            },
            'Moisture': {
                'status':   moisture_status(ndmi_avg),
                'NDMI':     round(ndmi_avg, 4) if ndmi_avg else None,
            },
            'Clay_CEC': {
                'CMR':      round(cmr_avg, 4) if cmr_avg else None,
                'note':     'Higher CMR = more clay = better nutrient retention',
            },
            'Soil_pH_proxy': {
                'IOR':      round(ior_avg, 4) if ior_avg else None,
                'note':     'IOR > 2.0 may indicate iron-rich acidic soil',
            },
        }

    # =========================================================================
    # SEASON HEALTH SCORE
    # =========================================================================

    def _season_health_score(
        self, trajectory: List[Dict]
    ) -> Tuple[float, str]:
        """
        Compute season-level health score.
        Weights: Peak health (40%) + Recent 3 scenes (40%) + Season average (20%)
        """
        if not trajectory:
            return 50.0, 'Unknown'

        scores = [t['health_score'] for t in trajectory if t.get('health_score') is not None]
        if not scores:
            return 50.0, 'Unknown'

        peak_score   = max(scores)
        recent_score = float(np.mean(scores[-3:])) if len(scores) >= 3 else float(np.mean(scores))
        avg_score    = float(np.mean(scores))

        final = 0.40 * peak_score + 0.40 * recent_score + 0.20 * avg_score
        return round(final, 1), self._categorize(final)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_crop_curve(self, crop: str) -> Optional[List]:
        """Load NDVI growth curve for crop from CropGrowthCurves."""
        try:
            from config.crop_parameters import CropGrowthCurves
            name = CropGrowthCurves._normalize_crop_name(crop)
            return CropGrowthCurves.EXPECTED_NDVI_CURVES.get(name)
        except ImportError:
            return None

    @staticmethod
    def _interpolate_curve(
        curve: Optional[List], days: int
    ) -> Tuple[Optional[float], str]:
        """Interpolate expected NDVI and stage from growth curve."""
        if not curve:
            return None, 'Unknown'

        if days <= curve[0][0]:
            return curve[0][1], curve[0][2]
        if days >= curve[-1][0]:
            return curve[-1][1], curve[-1][2]

        for i in range(len(curve) - 1):
            d1, n1, s1 = curve[i]
            d2, n2, s2 = curve[i + 1]
            if d1 <= days <= d2:
                t = (days - d1) / max(d2 - d1, 1)
                return n1 + (n2 - n1) * t, (s1 if t < 0.5 else s2)

        return None, 'Unknown'

    def _categorize(self, score: float) -> str:
        """Map numeric score to health category string."""
        for cat, (lo, hi) in self.health_cats.items():
            if lo <= score < hi:
                return cat
        return 'Unknown'

    @staticmethod
    def _safe_get(idx: Dict, name: str, key: str) -> float:
        return idx.get(name, {}).get(key, np.nan)

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
        if not date_str:
            return None
        try:
            return pd.to_datetime(date_str)
        except Exception:
            return None

    def _generate_summary(
        self, score: float, category: str, crop: str, alerts: List[Dict]
    ) -> str:
        critical = [a for a in alerts if a['severity'] in ('critical', 'high')]
        alert_str = (f" Critical issues: {', '.join(a['type'] for a in critical[:2])}."
                     if critical else " No critical alerts.")
        return (f"{crop} health is {category} (score: {score:.0f}/100).{alert_str} "
                f"{'Immediate intervention recommended.' if score < 40 else 'Continue monitoring.'}")

    @staticmethod
    def _empty_assessment(season, year, crop, reason) -> Dict:
        return {
            'season': season, 'year': year, 'crop': crop,
            'health_score': 0.0, 'health_category': 'Unknown',
            'trajectory': [], 'alerts': [], 'nutrient_summary': {},
            'peak_health': {}, 'current_health': {},
            'summary': reason, 'n_scenes': 0,
        }
