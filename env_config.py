"""
ZUTO Geotech Solutions — Environmental Analysis Platform
=========================================================
Component 2: Environmental Intelligence Configuration

VERSION 1.0
Author: ZUTO Geotech Solutions

Covers:
  - Carbon Footprint Mapping
  - Flood & Drought Risk Assessment
  - Forest Cover & Deforestation Tracking
  - Urban Heat Island Analysis
  - Air & Water Quality Spatial Dashboards

Data Sources:
  - Sentinel-2 (optical, 10m/20m)
  - Landsat 8/9 (thermal + long archive)
  - Sentinel-1 SAR (flood, soil moisture)
  - DEM/SRTM (terrain)
  - MODIS (regional/temporal)
  - OpenAQ / CPCB (air quality)
  - CWC/WRIS (water bodies)
"""


class ZutoEnvConfig:
    """Central configuration for ZUTO Environmental Analysis Platform"""

    # =========================================================================
    # PLATFORM IDENTITY
    # =========================================================================

    PLATFORM_NAME    = "ZUTO Environmental Intelligence Platform"
    PLATFORM_VERSION = "1.0.0"
    COMPANY          = "ZUTO Geotech Solutions"

    # =========================================================================
    # LAND USE LAND COVER (LULC) CLASSES
    # =========================================================================

    LULC_CLASSES = {
        0: {'name': 'Forest',     'color': '#1a7a1a', 'carbon_factor': 150.0},  # tC/ha
        1: {'name': 'Cropland',   'color': '#c8b400', 'carbon_factor': 5.0},
        2: {'name': 'Grassland',  'color': '#90ee90', 'carbon_factor': 3.5},
        3: {'name': 'Wetland',    'color': '#4169e1', 'carbon_factor': 200.0},  # high due to peat
        4: {'name': 'Urban',      'color': '#808080', 'carbon_factor': 0.5},
        5: {'name': 'Water',      'color': '#00bfff', 'carbon_factor': 0.0},
        6: {'name': 'Barren',     'color': '#d2b48c', 'carbon_factor': 0.8},
        7: {'name': 'Shrubland',  'color': '#8fbc8f', 'carbon_factor': 20.0},
    }

    # IPCC Tier-1 emission factors (tCO2e/ha/yr) by land-use transition
    IPCC_EMISSION_FACTORS = {
        'forest_to_cropland':  95.0,
        'forest_to_urban':     120.0,
        'forest_to_grassland': 45.0,
        'cropland_to_urban':   5.5,
        'wetland_to_cropland': 110.0,   # peat oxidation
        'grassland_to_cropland': 12.0,
    }

    # Carbon sequestration rates (tCO2e/ha/yr) — positive = sink
    CARBON_SEQUESTRATION = {
        'forest_afforestation':    6.0,
        'forest_natural_regen':    4.5,
        'agroforestry':            3.0,
        'grassland_restoration':   1.5,
        'wetland_restoration':     8.0,
    }

    # =========================================================================
    # SPECTRAL INDEX THRESHOLDS — ENVIRONMENTAL
    # =========================================================================

    LULC_THRESHOLDS = {
        # NDVI — vegetation vs non-vegetation
        'ndvi_forest_min':    0.50,
        'ndvi_cropland_min':  0.20,
        'ndvi_grassland_min': 0.15,

        # NDBI (Normalized Difference Built-up Index) — urban detection
        'ndbi_urban_min':     0.05,

        # MNDWI (Modified NDWI) — water bodies
        'mndwi_water_min':    0.10,

        # BSI (Bare Soil Index) — barren land
        'bsi_barren_min':     0.05,
    }

    # =========================================================================
    # FLOOD & DROUGHT THRESHOLDS
    # =========================================================================

    FLOOD_THRESHOLDS = {
        # SAR backscatter change (dB) — negative = water inundation
        'sar_flood_change_db':       -3.0,
        # MNDWI threshold for open water
        'mndwi_flood_water':          0.10,
        # Flood probability classes
        'prob_low':                   0.25,
        'prob_medium':                0.50,
        'prob_high':                  0.75,
    }

    DROUGHT_THRESHOLDS = {
        # NDDI = (NDVI - NDWI) / (NDVI + NDWI)
        'nddi_mild_drought':       0.40,
        'nddi_moderate_drought':   0.55,
        'nddi_severe_drought':     0.70,
        # NDMI moisture stress
        'ndmi_stress_mild':       -0.10,
        'ndmi_stress_moderate':   -0.20,
        'ndmi_stress_severe':     -0.35,
        # SPI (Standard Precipitation Index)
        'spi_mild_drought':       -1.0,
        'spi_moderate_drought':   -1.5,
        'spi_severe_drought':     -2.0,
    }

    DROUGHT_SEVERITY = {
        'None':     {'min': 0.00, 'max': 0.40, 'color': '#00cc44', 'risk_score': 0},
        'Mild':     {'min': 0.40, 'max': 0.55, 'color': '#ffff00', 'risk_score': 25},
        'Moderate': {'min': 0.55, 'max': 0.70, 'color': '#ff8800', 'risk_score': 55},
        'Severe':   {'min': 0.70, 'max': 1.00, 'color': '#cc0000', 'risk_score': 85},
    }

    FLOOD_RISK_ZONES = {
        'Safe':     {'prob_max': 0.25, 'color': '#00cc44', 'insurance_multiplier': 1.0},
        'Low':      {'prob_max': 0.50, 'color': '#ffff00', 'insurance_multiplier': 1.3},
        'Medium':   {'prob_max': 0.75, 'color': '#ff8800', 'insurance_multiplier': 1.8},
        'High':     {'prob_max': 1.00, 'color': '#cc0000', 'insurance_multiplier': 2.5},
    }

    # =========================================================================
    # FOREST CHANGE THRESHOLDS
    # =========================================================================

    FOREST_THRESHOLDS = {
        # NBR (Normalized Burn Ratio) — (B08 - B12) / (B08 + B12)
        'nbr_healthy_min':     0.30,
        'nbr_low_severity':    0.10,
        'nbr_mod_severity':   -0.10,
        'nbr_high_severity':  -0.25,

        # dNBR (pre - post) fire severity
        'dnbr_regrowth':      -0.25,
        'dnbr_unburned':       0.10,
        'dnbr_low_sev':        0.27,
        'dnbr_mod_sev':        0.44,
        'dnbr_high_sev':       0.66,

        # NDVI change for deforestation
        'dndvi_loss_alert':   -0.15,    # 15% NDVI drop = deforestation alert
        'dndvi_gain_regen':    0.10,    # 10% NDVI gain = regeneration
    }

    # Allometric equations for biomass estimation (tropical India)
    # AGB (tDM/ha) from canopy cover (0-1)
    BIOMASS_EQUATIONS = {
        'tropical_moist':    {'a': 21.3, 'b': 6.95, 'BEF': 1.74, 'R': 0.26, 'D': 0.6},
        'tropical_dry':      {'a': 15.5, 'b': 5.20, 'BEF': 1.65, 'R': 0.24, 'D': 0.6},
        'subtropical':       {'a': 18.0, 'b': 6.10, 'BEF': 1.70, 'R': 0.25, 'D': 0.6},
        'default':           {'a': 18.0, 'b': 6.10, 'BEF': 1.70, 'R': 0.25, 'D': 0.6},
    }
    # Carbon fraction of dry matter
    CF = 0.47
    # CO2 equivalent factor (44/12)
    CO2_FACTOR = 3.667

    FOREST_ALERT_TYPES = {
        'deforestation':  {'severity': 'CRITICAL', 'color': '#cc0000'},
        'fire_damage':    {'severity': 'HIGH',     'color': '#ff4400'},
        'degradation':    {'severity': 'MEDIUM',   'color': '#ff8800'},
        'regeneration':   {'severity': 'INFO',     'color': '#00cc44'},
    }

    # =========================================================================
    # URBAN HEAT ISLAND (UHI) THRESHOLDS
    # =========================================================================

    UHI_THRESHOLDS = {
        # LST anomaly above urban-rural background (°C)
        'uhi_weak':        2.0,
        'uhi_moderate':    4.0,
        'uhi_strong':      6.0,
        'uhi_extreme':     8.0,

        # NDBI for built-up detection
        'ndbi_high_density':  0.20,
        'ndbi_medium_density': 0.05,

        # NDVI for green cooling effect
        'ndvi_cooling_high':  0.40,
        'ndvi_cooling_med':   0.20,
    }

    UHI_CATEGORIES = {
        'Cool':     {'lst_anom_max': 2.0,  'color': '#4169e1', 'vulnerability': 'Low'},
        'Neutral':  {'lst_anom_max': 4.0,  'color': '#90ee90', 'vulnerability': 'Low'},
        'Warm':     {'lst_anom_max': 6.0,  'color': '#ffff00', 'vulnerability': 'Moderate'},
        'Hot':      {'lst_anom_max': 8.0,  'color': '#ff8800', 'vulnerability': 'High'},
        'Extreme':  {'lst_anom_max': 99.0, 'color': '#cc0000', 'vulnerability': 'Critical'},
    }

    # Landsat thermal band constants (Band 10)
    LANDSAT_THERMAL = {
        'K1': 774.8853,   # Calibration constant K1
        'K2': 1321.0789,  # Calibration constant K2
        'ML': 0.0003342,  # Radiance multiplier
        'AL': 0.1,        # Radiance additive
    }

    # Emissivity values for LST correction
    EMISSIVITY = {
        'vegetation': 0.986,
        'bare_soil':  0.914,
        'urban':      0.923,
        'water':      0.991,
    }

    # =========================================================================
    # AIR QUALITY (AQI) STANDARDS — CPCB India
    # =========================================================================

    AQI_BREAKPOINTS = {
        'PM2_5': [  # µg/m³ (24hr avg)
            (0,   30,   0,   50,  'Good'),
            (31,  60,   51,  100, 'Satisfactory'),
            (61,  90,   101, 200, 'Moderate'),
            (91,  120,  201, 300, 'Poor'),
            (121, 250,  301, 400, 'Very Poor'),
            (251, 500,  401, 500, 'Severe'),
        ],
        'PM10': [   # µg/m³ (24hr avg)
            (0,   50,   0,   50,  'Good'),
            (51,  100,  51,  100, 'Satisfactory'),
            (101, 250,  101, 200, 'Moderate'),
            (251, 350,  201, 300, 'Poor'),
            (351, 430,  301, 400, 'Very Poor'),
            (431, 600,  401, 500, 'Severe'),
        ],
        'NO2': [    # µg/m³
            (0,   40,   0,   50,  'Good'),
            (41,  80,   51,  100, 'Satisfactory'),
            (81,  180,  101, 200, 'Moderate'),
            (181, 280,  201, 300, 'Poor'),
            (281, 400,  301, 400, 'Very Poor'),
            (401, 999,  401, 500, 'Severe'),
        ],
        'SO2': [    # µg/m³
            (0,   40,   0,   50,  'Good'),
            (41,  80,   51,  100, 'Satisfactory'),
            (81,  380,  101, 200, 'Moderate'),
            (381, 800,  201, 300, 'Poor'),
            (801, 1600, 301, 400, 'Very Poor'),
            (1601,9999, 401, 500, 'Severe'),
        ],
    }

    AQI_CATEGORIES = {
        'Good':          {'min': 0,   'max': 50,  'color': '#00cc44', 'health_impact': 'Minimal'},
        'Satisfactory':  {'min': 51,  'max': 100, 'color': '#99dd00', 'health_impact': 'Minor breathing discomfort'},
        'Moderate':      {'min': 101, 'max': 200, 'color': '#ffcc00', 'health_impact': 'Breathing discomfort for sensitive groups'},
        'Poor':          {'min': 201, 'max': 300, 'color': '#ff8800', 'health_impact': 'Breathing discomfort for most'},
        'Very Poor':     {'min': 301, 'max': 400, 'color': '#cc0000', 'health_impact': 'Respiratory illness on prolonged exposure'},
        'Severe':        {'min': 401, 'max': 500, 'color': '#800000', 'health_impact': 'Affects healthy; serious risk to sensitive'},
    }

    # =========================================================================
    # WATER QUALITY THRESHOLDS — CPCB India / IS 10500
    # =========================================================================

    WATER_QUALITY = {
        # Turbidity (NTU) — proxy from Sentinel-2 B02/B03
        'turbidity': {
            'clear':     (0,   5),
            'moderate':  (5,   25),
            'turbid':    (25,  100),
            'very_turbid': (100, 9999),
        },
        # Chlorophyll-a (µg/L) — proxy from B02, B03, B04
        'chlorophyll_a': {
            'oligotrophic':   (0,   2),
            'mesotrophic':    (2,   10),
            'eutrophic':      (10,  50),
            'hypereutrophic': (50,  9999),
        },
        # NDWI variants for water extent
        'ndwi_water_min': 0.10,
        'mndwi_water_min': 0.15,
    }

    # =========================================================================
    # SENTINEL-2 BAND REFERENCES (for documentation clarity)
    # =========================================================================

    S2_BANDS = {
        'B02': {'name': 'Blue',          'wavelength': 490,  'resolution': 10},
        'B03': {'name': 'Green',         'wavelength': 560,  'resolution': 10},
        'B04': {'name': 'Red',           'wavelength': 665,  'resolution': 10},
        'B05': {'name': 'RedEdge1',      'wavelength': 705,  'resolution': 20},
        'B06': {'name': 'RedEdge2',      'wavelength': 740,  'resolution': 20},
        'B07': {'name': 'RedEdge3',      'wavelength': 783,  'resolution': 20},
        'B08': {'name': 'NIR',           'wavelength': 842,  'resolution': 10},
        'B8A': {'name': 'NarrowNIR',     'wavelength': 865,  'resolution': 20},
        'B11': {'name': 'SWIR1',         'wavelength': 1610, 'resolution': 20},
        'B12': {'name': 'SWIR2',         'wavelength': 2190, 'resolution': 20},
    }

    # =========================================================================
    # OUTPUT / REPORTING
    # =========================================================================

    ESG_REPORT_SECTIONS = [
        'executive_summary',
        'lulc_analysis',
        'carbon_stock_estimate',
        'emission_sources',
        'sequestration_potential',
        'flood_drought_risk',
        'recommendations',
        'methodology',
    ]

    REPORT_FORMATS = ['pdf', 'xlsx', 'geojson', 'json']

    # Spatial resolution for output rasters
    OUTPUT_RESOLUTION_M = 10    # metres (Sentinel-2 native)

    # =========================================================================
    # INTERPOLATION SETTINGS — AIR QUALITY SURFACES
    # =========================================================================

    INTERPOLATION = {
        'method':          'idw',    # 'idw' | 'kriging' | 'rbf'
        'idw_power':       2,
        'grid_resolution': 0.01,     # degrees (~1km at India lat)
        'kriging_model':   'spherical',
        'search_radius_km': 50,
    }
