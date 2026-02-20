"""
ZUTO Geotech Solutions — Environmental Analysis Platform
=========================================================
Module C: Forest Cover & Deforestation Tracker

Monitors forest cover dynamics using bi-annual Sentinel-2 + Landsat time series.
Generates actionable alerts for NGOs, forest departments, and carbon platforms.

Workflow:
  1. Compute NDVI + NBR time-series for each scene
  2. Detect bi-temporal NDVI change (dNDVI) for gradual deforestation
  3. Compute dNBR for fire-caused forest loss
  4. Classify change: deforestation / fire / degradation / regeneration
  5. Estimate biomass loss using Indian tropical allometric equations
  6. Convert to carbon stock loss and potential carbon credits
  7. Generate alerts when forest loss exceeds configurable threshold

Target clients:
  - Forest Survey of India (FSI) / state forest departments
  - NGOs (WWF, WCS, ATREE) for protected area monitoring
  - Carbon market platforms (VCS, Gold Standard, REDD+)
  - Green bond issuers and ESG reporting entities
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.env_config import ZutoEnvConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ForestChangeEvent:
    """Detected forest change event between two time points"""
    event_id:           str
    change_type:        str         # deforestation / fire / degradation / regeneration
    severity:           str         # Critical / High / Medium / Info
    location_name:      str
    date_t1:            str
    date_t2:            str
    area_ha:            float
    ndvi_t1:            float
    ndvi_t2:            float
    dndvi:              float
    nbr_t1:             float
    nbr_t2:             float
    dnbr:               float
    agb_loss_t_dm_ha:   float       # Above-Ground Biomass loss (t dry matter/ha)
    total_biomass_loss_t: float     # Total AGB loss (tonnes DM)
    carbon_loss_t:      float       # Carbon loss (tonnes C)
    co2e_loss:          float       # CO2 equivalent loss
    carbon_credits_est: float       # Estimated carbon credits (tCO2e)
    confidence:         float       # Detection confidence 0-1
    alert_triggered:    bool
    notes:              str


@dataclass
class ForestCoverReport:
    """Full forest cover assessment for a monitoring area"""
    region_id:              str
    region_name:            str
    report_date:            str
    monitoring_period:      str
    total_area_ha:          float
    forest_cover_ha_t1:     float
    forest_cover_ha_t2:     float
    forest_cover_pct_t1:    float
    forest_cover_pct_t2:    float
    net_change_ha:          float       # negative = loss
    deforestation_ha:       float
    fire_damage_ha:         float
    degradation_ha:         float
    regeneration_ha:        float
    total_carbon_loss_t:    float
    total_co2e_loss:        float
    total_carbon_gain_t:    float
    net_co2e_balance:       float
    carbon_credits_available: float
    change_events:          List[Dict]
    alerts:                 List[Dict]
    fire_events:            List[Dict]
    redd_eligibility:       bool        # REDD+ eligible?
    summary:                str
    recommendations:        List[str]


# =============================================================================
# FOREST CHANGE TRACKER — MAIN CLASS
# =============================================================================

class ForestChangeTracker:
    """
    Tracks forest cover dynamics, deforestation, and fire events
    using NDVI, NBR, and dNBR from Sentinel-2 + Landsat archives.

    Example:
        tracker = ForestChangeTracker()

        # Detect changes between two periods
        report = tracker.generate_forest_report(
            scenes_t1   = baseline_scenes,
            scenes_t2   = current_scenes,
            region_name = "Sahyadri Forest Reserve",
            region_id   = "MH_FOR_001",
            total_area_ha = 5000.0,
            forest_type = "tropical_moist",
        )

        # Monitor for alerts continuously
        alert = tracker.check_deforestation_alert(
            latest_scenes  = recent_scenes,
            baseline_ndvi  = 0.65,
            threshold_ha   = 50.0,
        )
    """

    def __init__(self, forest_type: str = 'tropical_moist'):
        self.cfg          = ZutoEnvConfig
        self.thresh       = self.cfg.FOREST_THRESHOLDS
        self.alert_types  = self.cfg.FOREST_ALERT_TYPES
        self.biomass_eq   = self.cfg.BIOMASS_EQUATIONS.get(
            forest_type, self.cfg.BIOMASS_EQUATIONS['default']
        )
        self.CF           = self.cfg.CF
        self.CO2_FACTOR   = self.cfg.CO2_FACTOR
        self.forest_type  = forest_type

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate_forest_report(
        self,
        scenes_t1:     List[Dict],
        scenes_t2:     List[Dict],
        region_name:   str  = "Forest Area",
        region_id:     str  = "FOR_001",
        total_area_ha: float = 1000.0,
        forest_type:   str  = 'tropical_moist',
        threshold_ha:  float = 10.0,        # alert if loss > this
    ) -> ForestCoverReport:
        """
        Full bi-temporal forest change assessment.

        Args:
            scenes_t1:    Baseline satellite scenes (T1)
            scenes_t2:    Current satellite scenes (T2)
            region_name:  Name of monitored area
            region_id:    Unique region ID
            total_area_ha: Total region area in hectares
            forest_type:  'tropical_moist' | 'tropical_dry' | 'subtropical'
            threshold_ha: Alert threshold for forest loss (ha)

        Returns:
            ForestCoverReport dataclass
        """
        logger.info(f"\nFOREST CHANGE TRACKING — {region_name}")

        # — Composite indices for T1 and T2
        idx_t1 = self._composite_indices(scenes_t1)
        idx_t2 = self._composite_indices(scenes_t2)

        dates_t1 = self._scene_dates(scenes_t1)
        dates_t2 = self._scene_dates(scenes_t2)

        # — Forest cover estimation
        forest_ha_t1 = self._estimate_forest_cover(idx_t1, total_area_ha)
        forest_ha_t2 = self._estimate_forest_cover(idx_t2, total_area_ha)

        # — Change detection
        dndvi = idx_t2['ndvi'] - idx_t1['ndvi']
        dnbr  = idx_t1['nbr']  - idx_t2['nbr']    # pre - post (USGS convention)

        # — Classify change type
        events = self._classify_all_changes(idx_t1, idx_t2, dndvi, dnbr, total_area_ha, dates_t1, dates_t2, region_name)

        # — Aggregate loss/gain
        deforest_ha = sum(e['area_ha'] for e in events if e['change_type'] == 'deforestation')
        fire_ha     = sum(e['area_ha'] for e in events if e['change_type'] == 'fire_damage')
        degrade_ha  = sum(e['area_ha'] for e in events if e['change_type'] == 'degradation')
        regen_ha    = sum(e['area_ha'] for e in events if e['change_type'] == 'regeneration')

        # — Carbon accounting
        total_co2e_loss = sum(e['co2e_loss'] for e in events if e['co2e_loss'] > 0)
        total_c_loss    = total_co2e_loss / self.CO2_FACTOR
        total_c_gain    = sum(e.get('co2e_loss', 0) for e in events if e.get('co2e_loss', 0) < 0)
        net_co2e        = total_c_gain - total_co2e_loss
        credits_avail   = sum(e['carbon_credits_est'] for e in events)

        # — REDD+ eligibility
        redd_eligible = deforest_ha > 10 and (deforest_ha / total_area_ha) > 0.01

        # — Alerts
        alerts = [
            {'alert': e['notes'], 'severity': e['severity'],
             'change_type': e['change_type'], 'area_ha': e['area_ha']}
            for e in events if e['alert_triggered']
        ]

        fire_events = [e for e in events if e['change_type'] == 'fire_damage']

        net_ha  = forest_ha_t2 - forest_ha_t1
        summary = (
            f"Forest cover {('lost' if net_ha < 0 else 'gained')} "
            f"{abs(net_ha):.1f} ha over the monitoring period. "
            f"Deforestation: {deforest_ha:.1f} ha, Fire: {fire_ha:.1f} ha, "
            f"Regeneration: {regen_ha:.1f} ha. "
            f"Net CO2e loss: {total_co2e_loss:.1f} tCO2e."
        )

        report = ForestCoverReport(
            region_id             = region_id,
            region_name           = region_name,
            report_date           = date.today().isoformat(),
            monitoring_period     = f"{dates_t1[0]} → {dates_t2[-1]}",
            total_area_ha         = total_area_ha,
            forest_cover_ha_t1    = round(forest_ha_t1, 1),
            forest_cover_ha_t2    = round(forest_ha_t2, 1),
            forest_cover_pct_t1   = round(forest_ha_t1 / total_area_ha * 100, 1),
            forest_cover_pct_t2   = round(forest_ha_t2 / total_area_ha * 100, 1),
            net_change_ha         = round(net_ha, 1),
            deforestation_ha      = round(deforest_ha, 1),
            fire_damage_ha        = round(fire_ha, 1),
            degradation_ha        = round(degrade_ha, 1),
            regeneration_ha       = round(regen_ha, 1),
            total_carbon_loss_t   = round(total_c_loss, 2),
            total_co2e_loss       = round(total_co2e_loss, 2),
            total_carbon_gain_t   = round(abs(total_c_gain) / self.CO2_FACTOR, 2),
            net_co2e_balance      = round(net_co2e, 2),
            carbon_credits_available = round(credits_avail, 2),
            change_events         = events,
            alerts                = alerts,
            fire_events           = fire_events,
            redd_eligibility      = redd_eligible,
            summary               = summary,
            recommendations       = self._generate_recommendations(
                deforest_ha, fire_ha, regen_ha, net_co2e, redd_eligible
            ),
        )

        self._log_report_summary(report)
        return report

    def check_deforestation_alert(
        self,
        latest_scenes:  List[Dict],
        baseline_ndvi:  float,
        region_name:    str   = "Forest Area",
        threshold_ha:   float = 50.0,
        total_area_ha:  float = 1000.0,
    ) -> Dict:
        """
        Quick deforestation alert check against a baseline NDVI.
        Designed for near-real-time monitoring integration (e.g., 5-day cycle).

        Returns alert dict with triggered flag.
        """
        current = self._composite_indices(latest_scenes)
        dndvi   = current['ndvi'] - baseline_ndvi
        t       = self.thresh

        # Estimate deforested area from NDVI drop
        if dndvi < t['dndvi_loss_alert']:
            fraction_lost = abs(dndvi) / 0.5    # normalize: -0.5 NDVI = total loss
            est_loss_ha   = fraction_lost * total_area_ha
        else:
            est_loss_ha = 0.0

        triggered = est_loss_ha >= threshold_ha
        severity  = 'CRITICAL' if est_loss_ha >= threshold_ha * 2 else ('HIGH' if triggered else 'INFO')

        return {
            'region':          region_name,
            'check_date':      date.today().isoformat(),
            'current_ndvi':    round(current['ndvi'], 4),
            'baseline_ndvi':   round(baseline_ndvi, 4),
            'dndvi':           round(dndvi, 4),
            'estimated_loss_ha': round(est_loss_ha, 1),
            'alert_triggered': triggered,
            'severity':        severity,
            'threshold_ha':    threshold_ha,
            'message':         (
                f"ALERT: ~{est_loss_ha:.1f} ha potential forest loss detected in {region_name}."
                if triggered else
                f"No significant forest loss detected in {region_name}."
            ),
        }

    def compute_fire_severity(self, pre_scene: Dict, post_scene: Dict) -> Dict:
        """
        Compute dNBR-based fire severity classification.
        Used for targeted post-fire damage assessment.
        """
        nbr_pre  = self._get_nbr(pre_scene)
        nbr_post = self._get_nbr(post_scene)
        dnbr     = nbr_pre - nbr_post   # positive = burned area
        severity = self._classify_fire_severity(dnbr)

        return {
            'nbr_pre':  round(nbr_pre, 4),
            'nbr_post': round(nbr_post, 4),
            'dnbr':     round(dnbr, 4),
            'severity': severity,
            'interpretation': self._fire_severity_note(severity),
        }

    def estimate_carbon_credits(
        self,
        deforestation_ha:  float,
        forest_type:       str = 'tropical_moist',
        project_years:     int = 30,
    ) -> Dict:
        """
        Estimate REDD+ / VCS carbon credits from avoided deforestation.
        Returns total and annual credit estimates.
        """
        eq     = self.cfg.BIOMASS_EQUATIONS.get(forest_type, self.cfg.BIOMASS_EQUATIONS['default'])
        agb    = eq['a'] + eq['b'] * 10    # simplified: AGB = a + b × dbh proxy
        total_biomass = agb * deforestation_ha
        carbon_t      = total_biomass * self.CF * eq['BEF'] * (1 + eq['R'])
        co2e          = carbon_t * self.CO2_FACTOR
        annual_credits = co2e / project_years

        return {
            'deforestation_ha':      deforestation_ha,
            'forest_type':           forest_type,
            'agb_t_dm_ha':           round(agb, 2),
            'total_biomass_t_dm':    round(total_biomass, 2),
            'carbon_stock_tC':       round(carbon_t, 2),
            'total_co2e':            round(co2e, 2),
            'project_years':         project_years,
            'annual_credits_tco2e':  round(annual_credits, 2),
            'methodology':           'VCS/REDD+ Tier-2 allometric, IPCC 2006 GL',
        }

    # =========================================================================
    # INTERNAL — INDEX COMPUTATION
    # =========================================================================

    def _composite_indices(self, scenes: List[Dict]) -> Dict:
        """Build median composite of key indices from scene list."""
        keys = ['ndvi', 'nbr', 'ndwi', 'evi', 'b08', 'b11', 'b12']
        agg  = {k: [] for k in keys}

        for scene in scenes:
            idx = scene.get('indices', {})
            for k in keys:
                val = idx.get(k) or idx.get(k.upper())
                if val is not None:
                    agg[k].append(float(val))

        composite = {}
        for k, vals in agg.items():
            composite[k] = float(np.median(vals)) if vals else 0.0

        # Compute NBR if not directly available
        if composite['nbr'] == 0.0 and composite['b08'] != 0.0 and composite['b12'] != 0.0:
            b08, b12 = composite['b08'], composite['b12']
            denom    = b08 + b12
            composite['nbr'] = (b08 - b12) / denom if abs(denom) > 0.001 else 0.0

        return composite

    def _get_nbr(self, scene: Dict) -> float:
        """Extract or compute NBR from a single scene."""
        idx = scene.get('indices', {})
        if 'nbr' in idx:
            return float(idx['nbr'])
        b08 = float(idx.get('b08', 0.0))
        b12 = float(idx.get('b12', 0.0))
        denom = b08 + b12
        return (b08 - b12) / denom if abs(denom) > 0.001 else 0.0

    def _scene_dates(self, scenes: List[Dict]) -> List[str]:
        """Extract sorted dates from scenes."""
        dates = [s.get('timestamp', s.get('date', 'unknown')) for s in scenes]
        return sorted(set(d for d in dates if d != 'unknown')) or ['unknown']

    # =========================================================================
    # INTERNAL — FOREST COVER ESTIMATION
    # =========================================================================

    def _estimate_forest_cover(self, indices: Dict, total_ha: float) -> float:
        """Estimate forested area from NDVI threshold."""
        ndvi    = indices['ndvi']
        frac    = np.clip(
            (ndvi - self.thresh['nbr_low_severity']) /
            (1.0 - self.thresh['nbr_low_severity']),
            0.0, 1.0
        )
        return round(frac * total_ha, 1)

    # =========================================================================
    # INTERNAL — CHANGE CLASSIFICATION
    # =========================================================================

    def _classify_all_changes(
        self,
        idx_t1:      Dict,
        idx_t2:      Dict,
        dndvi:       float,
        dnbr:        float,
        total_ha:    float,
        dates_t1:    List[str],
        dates_t2:    List[str],
        location:    str,
    ) -> List[Dict]:
        """Detect and classify all change types from index composites."""
        events = []
        t      = self.thresh

        # — Regeneration (NDVI gain)
        if dndvi > t['dndvi_gain_regen']:
            frac    = np.clip(dndvi / 0.3, 0, 1)
            area_ha = frac * total_ha
            events.append(self._make_event(
                'regeneration', 'Info', location,
                dates_t1[0], dates_t2[-1],
                area_ha, idx_t1, idx_t2, dndvi, dnbr
            ))
            return events   # net gain — no loss events

        # — Deforestation (gradual NDVI loss without fire signature)
        if dndvi < t['dndvi_loss_alert'] and dnbr < t['dnbr_mod_sev']:
            frac    = np.clip(abs(dndvi) / 0.5, 0, 1)
            area_ha = frac * total_ha * 0.5   # deforestation in half the changed zone
            events.append(self._make_event(
                'deforestation', 'CRITICAL', location,
                dates_t1[0], dates_t2[-1],
                area_ha, idx_t1, idx_t2, dndvi, dnbr
            ))

        # — Fire damage (strong dNBR signal)
        if dnbr >= t['dnbr_mod_sev']:
            frac_fire = np.clip(dnbr / 0.66, 0, 1)
            area_fire = frac_fire * total_ha * 0.4
            events.append(self._make_event(
                'fire_damage', 'HIGH', location,
                dates_t1[0], dates_t2[-1],
                area_fire, idx_t1, idx_t2, dndvi, dnbr
            ))

        # — Degradation (mild NDVI loss, mild dNBR)
        if t['dndvi_loss_alert'] < dndvi < -0.05:
            frac_deg = abs(dndvi) / 0.15
            area_deg = frac_deg * total_ha * 0.3
            events.append(self._make_event(
                'degradation', 'MEDIUM', location,
                dates_t1[0], dates_t2[-1],
                area_deg, idx_t1, idx_t2, dndvi, dnbr
            ))

        return events if events else [self._stable_event(location, dates_t1[0], dates_t2[-1])]

    def _make_event(
        self,
        change_type: str,
        severity:    str,
        location:    str,
        date_t1:     str,
        date_t2:     str,
        area_ha:     float,
        idx_t1:      Dict,
        idx_t2:      Dict,
        dndvi:       float,
        dnbr:        float,
    ) -> Dict:
        """Build a change event dict with biomass and carbon estimates."""
        eq        = self.biomass_eq
        # AGB loss per ha using simple regression: AGB = a + b×(NDVI_drop×10)
        ndvi_drop = max(-dndvi, 0)
        agb_ha    = eq['a'] + eq['b'] * ndvi_drop * 10
        agb_ha    = max(agb_ha, 0.0)

        total_biomass = agb_ha * area_ha
        carbon_loss   = total_biomass * self.CF * eq['BEF'] * (1 + eq['R'])
        co2e_loss     = carbon_loss * self.CO2_FACTOR if change_type != 'regeneration' else -carbon_loss * self.CO2_FACTOR

        # Adjust for fire (higher combustion factor)
        if change_type == 'fire_damage':
            co2e_loss *= 1.15    # fire releases CH4 + N2O on top of CO2

        credits = max(co2e_loss * 0.85, 0)   # 85% after leakage buffer
        triggered = area_ha >= 10.0 and change_type in ['deforestation', 'fire_damage']

        alert_info = self.alert_types.get(change_type, {'severity': 'INFO', 'color': '#aaaaaa'})

        return {
            'event_id':          f"{change_type.upper()}_{date_t2}",
            'change_type':       change_type,
            'severity':          severity,
            'location_name':     location,
            'date_t1':           date_t1,
            'date_t2':           date_t2,
            'area_ha':           round(area_ha, 2),
            'ndvi_t1':           round(idx_t1['ndvi'], 4),
            'ndvi_t2':           round(idx_t2['ndvi'], 4),
            'dndvi':             round(dndvi, 4),
            'nbr_t1':            round(idx_t1['nbr'], 4),
            'nbr_t2':            round(idx_t2['nbr'], 4),
            'dnbr':              round(dnbr, 4),
            'agb_loss_t_dm_ha':  round(agb_ha, 2),
            'total_biomass_loss_t': round(total_biomass, 2),
            'carbon_loss_t':     round(carbon_loss, 2),
            'co2e_loss':         round(co2e_loss, 2),
            'carbon_credits_est': round(credits, 2),
            'confidence':        round(min(abs(dndvi) / 0.3, 1.0), 2),
            'alert_triggered':   triggered,
            'color':             alert_info['color'],
            'notes':             self._change_note(change_type, area_ha, co2e_loss),
        }

    def _stable_event(self, location: str, d1: str, d2: str) -> Dict:
        return {
            'event_id': f"STABLE_{d2}", 'change_type': 'stable',
            'severity': 'INFO', 'location_name': location,
            'date_t1': d1, 'date_t2': d2, 'area_ha': 0.0,
            'ndvi_t1': 0.0, 'ndvi_t2': 0.0, 'dndvi': 0.0,
            'nbr_t1': 0.0, 'nbr_t2': 0.0, 'dnbr': 0.0,
            'agb_loss_t_dm_ha': 0.0, 'total_biomass_loss_t': 0.0,
            'carbon_loss_t': 0.0, 'co2e_loss': 0.0,
            'carbon_credits_est': 0.0, 'confidence': 1.0,
            'alert_triggered': False, 'color': '#00cc44',
            'notes': 'No significant forest change detected.',
        }

    # =========================================================================
    # INTERNAL — FIRE CLASSIFICATION
    # =========================================================================

    def _classify_fire_severity(self, dnbr: float) -> str:
        t = self.thresh
        if dnbr < t['dnbr_unburned']:
            return 'Unburned'
        elif dnbr < t['dnbr_low_sev']:
            return 'Low Severity'
        elif dnbr < t['dnbr_mod_sev']:
            return 'Moderate Severity'
        elif dnbr < t['dnbr_high_sev']:
            return 'High Severity'
        else:
            return 'Very High Severity'

    def _fire_severity_note(self, severity: str) -> str:
        notes = {
            'Unburned':           'No fire impact detected.',
            'Low Severity':       'Surface fire, minimal canopy damage. Recovery 1–2 years.',
            'Moderate Severity':  'Mixed canopy damage. Recovery 3–5 years.',
            'High Severity':      'Stand-replacing fire. Recovery 10–20 years.',
            'Very High Severity': 'Catastrophic fire. Soil sterilization likely. Recovery 20+ years.',
        }
        return notes.get(severity, 'Unknown severity.')

    # =========================================================================
    # INTERNAL — RECOMMENDATIONS & REPORTING
    # =========================================================================

    def _change_note(self, change_type: str, area_ha: float, co2e: float) -> str:
        if change_type == 'deforestation':
            return (f"Deforestation of ~{area_ha:.1f} ha detected. "
                    f"Carbon loss: ~{co2e:.1f} tCO2e. Verify with ground survey.")
        elif change_type == 'fire_damage':
            return (f"Fire damage: ~{area_ha:.1f} ha affected. "
                    f"CO2e loss including non-CO2 GHGs: ~{co2e:.1f} tCO2e.")
        elif change_type == 'degradation':
            return f"Forest degradation: ~{area_ha:.1f} ha at risk. Early intervention recommended."
        elif change_type == 'regeneration':
            return f"Forest regeneration: ~{area_ha:.1f} ha gaining cover. Positive carbon sink signal."
        return "No significant change."

    def _generate_recommendations(
        self,
        deforest_ha:  float,
        fire_ha:      float,
        regen_ha:     float,
        net_co2e:     float,
        redd_eligible: bool,
    ) -> List[str]:
        recs = []

        if deforest_ha > 50:
            recs.append(f"CRITICAL: {deforest_ha:.0f} ha deforestation requires immediate DFO/forest department intervention. File FIR under Forest Conservation Act 1980.")
        elif deforest_ha > 10:
            recs.append(f"Moderate deforestation ({deforest_ha:.0f} ha). Conduct beat-level verification. Initiate compensatory plantation at 2× affected area.")

        if fire_ha > 20:
            recs.append(f"Fire damage {fire_ha:.0f} ha detected. Deploy forest fire suppression teams. Post-fire soil erosion risk is high — install temporary check dams.")

        if regen_ha > 0:
            recs.append(f"Active regeneration detected ({regen_ha:.0f} ha). Protect from grazing/encroachment to maximize carbon sequestration benefit.")

        if redd_eligible:
            recs.append("Area is potentially eligible for REDD+ carbon finance. Initiate baseline study and Project Design Document (PDD) preparation.")

        if net_co2e < -500:
            recs.append(f"Net CO2e loss of {abs(net_co2e):.0f} tCO2e. Recommend voluntary carbon market credit purchase to offset. Engage VCS or Gold Standard registry.")

        if not recs:
            recs.append("Forest cover stable. Continue bi-annual satellite monitoring. Increase frequency to monthly in fire-prone months (Feb–May).")

        return recs

    def _log_report_summary(self, r: ForestCoverReport):
        logger.info(
            f"\n{'='*60}\n"
            f"  FOREST COVER REPORT — {r.region_name}\n"
            f"  Period   : {r.monitoring_period}\n"
            f"  Cover T1 : {r.forest_cover_ha_t1:.1f} ha ({r.forest_cover_pct_t1:.1f}%)\n"
            f"  Cover T2 : {r.forest_cover_ha_t2:.1f} ha ({r.forest_cover_pct_t2:.1f}%)\n"
            f"  Net Chng : {r.net_change_ha:+.1f} ha\n"
            f"  Deforest : {r.deforestation_ha:.1f} ha\n"
            f"  Fire     : {r.fire_damage_ha:.1f} ha\n"
            f"  Regen    : {r.regeneration_ha:.1f} ha\n"
            f"  CO2e Loss: {r.total_co2e_loss:.1f} tCO2e\n"
            f"  Credits  : {r.carbon_credits_available:.1f} tCO2e (REDD+: {'Yes' if r.redd_eligibility else 'No'})\n"
            f"  Alerts   : {len(r.alerts)}\n"
            f"{'='*60}"
        )
