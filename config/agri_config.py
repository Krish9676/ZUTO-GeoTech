"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Component 1: Agriculture Intelligence Configuration

VERSION 1.0
Author: ZUTO Geotech Solutions

Key differences from reference pipeline:
- EVERY available timestamp is collected (no scene-count caps per season)
- Full 20-index spectral suite per scene (not just NDVI/EVI/NDMI)
- Three Indian crop seasons: Kharif, Rabi, Zaid
- Multi-purpose outputs: health monitoring, soil nutrients,
  crop classification, yield forecasting, market analytics
- Sentinel-2 priority; Landsat-8/9 as historical fallback
"""


class ZutoAgriConfig:
    """Central configuration for ZUTO AgriTech Platform"""

    # =========================================================================
    # PLATFORM IDENTITY
    # =========================================================================

    PLATFORM_NAME    = "ZUTO AgriTech Intelligence Platform"
    PLATFORM_VERSION = "1.0.0"
    COMPANY          = "ZUTO Geotech Solutions"

    # =========================================================================
    # INDIAN CROP SEASONS (3 seasons)
    # =========================================================================

    SEASONS = {
        'kharif': {
            'start_month': 6,
            'start_day':   1,
            'end_month':   11,
            'end_day':     30,
            'description': 'Monsoon/Kharif season — Jun 1 to Nov 30',
            'primary_crops': [
                'Rice', 'Cotton', 'Soyabean', 'Maize', 'Bajra',
                'Jowar', 'Groundnut', 'Tur', 'Sugarcane', 'Banana',
                'Papaya', 'Pomegranate', 'Chilli', 'Onion',
            ],
            'rainfall_norm_mm': 700,
        },
        'rabi': {
            'start_month': 11,
            'start_day':   1,
            'end_month':   4,
            'end_day':     30,
            'description': 'Winter/Rabi season — Nov 1 to Apr 30',
            'primary_crops': [
                'Wheat', 'Gram', 'Mustard', 'Potato', 'Onion',
                'Sunflower', 'Tobacco', 'Chilli', 'Cabbage', 'Grapes',
                'Coriander', 'Tomato',
            ],
            'rainfall_norm_mm': 150,
        },
        'zaid': {
            'start_month': 3,
            'start_day':   1,
            'end_month':   6,
            'end_day':     30,
            'description': 'Summer/Zaid season — Mar 1 to Jun 30',
            'primary_crops': [
                'Watermelon', 'Muskmelon', 'Cucumber', 'Moong',
                'Sunflower', 'Maize', 'Vegetables',
            ],
            'rainfall_norm_mm': 60,
        },
    }

    # =========================================================================
    # ANALYSIS WINDOWS
    # =========================================================================

    NUM_SEASONS_HISTORY = 6      # seasons of historical data to analyse
    MAX_YEARS_BACK      = 4      # never look beyond 4 years

    # =========================================================================
    # SATELLITE DATA — EVERY TIMESTAMP APPROACH
    # =========================================================================
    # Unlike reference pipelines that cap scenes, ZUTO collects ALL
    # available timestamps within cloud threshold for maximum density.
    # This gives denser time-series for better index computation.

    MAX_CLOUD_COVER        = 60.0   # % — scenes above this are skipped
    MIN_VALID_PIXEL_RATIO  = 0.20   # minimum valid pixel fraction per scene
    TARGET_RESOLUTION_M    = 10     # all bands resampled to 10 m

    # No scene caps — collect everything within cloud threshold
    COLLECT_ALL_TIMESTAMPS = True
    MAX_SCENES_PER_SEASON  = 999    # effectively unlimited
    MIN_SCENES_PER_SEASON  = 3      # minimum for any valid analysis

    IDEAL_GAP_DAYS = 5    # Sentinel-2 revisit ~5 days with both satellites
    MAX_GAP_DAYS   = 15   # warn if gap exceeds this

    # =========================================================================
    # SENTINEL-2 BAND CONFIGURATION (ALL 12 analysis bands)
    # =========================================================================

    SENTINEL2_BANDS = {
        "B02": {"resolution": 10,  "wavelength_nm": 490,  "name": "Blue",         "use": "soil_props, salinity"},
        "B03": {"resolution": 10,  "wavelength_nm": 560,  "name": "Green",        "use": "chlorophyll, TGI"},
        "B04": {"resolution": 10,  "wavelength_nm": 665,  "name": "Red",          "use": "chlorophyll absorption"},
        "B05": {"resolution": 20,  "wavelength_nm": 705,  "name": "Red Edge 1",   "use": "NITROGEN — direct detection"},
        "B06": {"resolution": 20,  "wavelength_nm": 740,  "name": "Red Edge 2",   "use": "stress progression"},
        "B07": {"resolution": 20,  "wavelength_nm": 783,  "name": "Red Edge 3",   "use": "leaf area index"},
        "B08": {"resolution": 10,  "wavelength_nm": 842,  "name": "NIR",          "use": "biomass, NDVI"},
        "B8A": {"resolution": 20,  "wavelength_nm": 865,  "name": "Narrow NIR",   "use": "precise biomass"},
        "B11": {"resolution": 20,  "wavelength_nm": 1610, "name": "SWIR 1",       "use": "soil minerals, moisture"},
        "B12": {"resolution": 20,  "wavelength_nm": 2190, "name": "SWIR 2",       "use": "organic matter, composition"},
    }

    # =========================================================================
    # SPECTRAL INDEX SUITE — 27 INDICES
    # =========================================================================
    # Organised by tier and purpose, computed on every scene.
    # All names are UPPER_SNAKE_CASE to match SpectralIndexEngine output keys.

    # Tier 1: Core vegetation health (computed always)
    TIER1_INDICES = [
        'NDVI',      # Baseline vegetation
        'RENDVI',    # Nitrogen — red edge (BEST for N)
        'EVI',       # Dense vegetation, P/K proxy
        'GNDVI',     # Chlorophyll via green band
        'NDMI',      # Moisture stress
        'OSAVI',     # Soil-adjusted for sparse veg
        'SAVI',      # Soil adjusted (early season)
    ]

    # Tier 2: Soil nutrient indices
    TIER2_INDICES = [
        'CI_REDEDGE', # Chlorophyll Index Red Edge — chlorophyll/N direct
        'TGI',        # Triangle Greenness — early N stress
        'MSR',        # Modified Simple Ratio — P/K proxy
        'SI',         # Salinity Index (EC correlation)
        'NDSI',       # Normalized Difference Salinity
        'CMR',        # Clay Minerals Ratio (nutrient retention)
        'CAI',        # Cellulose Absorption — organic matter
        'IOR',        # Iron Oxide Ratio — pH proxy
        'NDTI',       # Tillage Index — residue management
        'BSI',        # Bare Soil Index
    ]

    # Tier 3: Moisture & stress indices
    # FIX: 'NMDI' was listed twice — duplicate removed. TIER3 is now 6 entries.
    TIER3_INDICES = [
        'NDWI',       # Water body detection
        'MNDWI',      # Modified water (urban areas)
        'NMDI',       # Normalized Multi-band Drought Index
        'MSI_STRESS', # Moisture Stress Index
        'NDDI',       # Drought index (NDVI-NDWI combo)
        'PSRI',       # Plant Senescence Reflectance Index
    ]

    # Tier 4: Growth stage & canopy
    TIER4_INDICES = [
        'SR1',   # Simple Ratio NIR/Red — biomass
        'SR2',   # Simple Ratio Blue/Green — early stress
        'CRI',   # Carotenoid Reflectance — stress
        'LSWI',  # Land Surface Water Index
    ]

    # 7 + 10 + 6 + 4 = 27 unique indices (matches SpectralIndexEngine exactly)
    ALL_INDICES = TIER1_INDICES + TIER2_INDICES + TIER3_INDICES + TIER4_INDICES

    # =========================================================================
    # ML MODEL CONFIGURATION
    # =========================================================================

    ML_MODELS = {
        'crop_classification': {
            'primary':   'RandomForest',
            'secondary': 'XGBoost',
            'advanced':  'TransformerCNN',
            'features':  'multi_temporal_indices',
            'target_accuracy': 0.82,
        },
        'nutrient_prediction': {
            'primary':   'XGBoost',
            'secondary': 'RandomForest',
            'advanced':  'TransformerCNN',
            'targets':   ['N', 'P', 'K', 'EC', 'OC', 'pH'],
            'target_r2': 0.78,
        },
        'yield_forecasting': {
            'primary':   'XGBoost',
            'secondary': 'CNN_LSTM',
            'features':  'cumulative_ndvi_weather',
            'target_r2': 0.75,
        },
    }

    # Feature scenes for ML (chronological)
    ML_FEATURE_SCENES  = 20     # more than reference due to denser sampling
    ML_FEATURE_INDICES = [
        'NDVI_mean', 'RENDVI_mean', 'EVI_mean', 'NDMI_mean',
        'CI_REDEDGE_mean', 'SI_mean', 'CMR_mean', 'CAI_mean',
    ]

    # =========================================================================
    # CROP DETECTION THRESHOLDS
    # =========================================================================

    CROP_DETECTION_NDVI_THRESHOLD   = 0.25
    MIN_NDVI_RISE                   = 0.10
    MIN_FRACTION_ABOVE_THRESHOLD    = 0.20
    MIN_NDVI_CV_FOR_CROP            = 0.08
    MAX_NDVI_CV_FOR_CROP            = 0.80

    # Cross-season continuity (long-duration crops)
    CROSS_SEASON_NDVI_CONTINUITY        = 0.38
    CROSS_SEASON_NDVI_DROP_FOR_NEW_CROP = 0.15

    LONG_DURATION_CROPS = {
        'Sugarcane':   330,
        'Banana':      330,
        'Papaya':      300,
        'Pomegranate': 180,
        'Tur':         180,
        'Cotton':      180,
        'Tobacco':     160,
    }

    # =========================================================================
    # SOIL NUTRIENT ANALYSIS
    # =========================================================================

    NUTRIENT_TARGETS = {
        'N':  {'unit': 'kg/ha',  'low': 280,  'medium': 560,  'high': 840},
        'P':  {'unit': 'kg/ha',  'low': 11,   'medium': 22,   'high': 55},
        'K':  {'unit': 'kg/ha',  'low': 110,  'medium': 280,  'high': 560},
        'EC': {'unit': 'dS/m',   'low': 0.5,  'medium': 2.0,  'high': 4.0},
        'OC': {'unit': '%',      'low': 0.5,  'medium': 0.75, 'high': 1.5},
        'pH': {'unit': 'pH',     'low': 6.0,  'medium': 7.0,  'high': 8.0},
    }

    # Index-to-nutrient mapping (for feature selection)
    INDEX_NUTRIENT_MAP = {
        'N':  ['RENDVI', 'CI_REDEDGE', 'TGI', 'NDVI', 'GNDVI'],
        'P':  ['EVI', 'MSR', 'SR1'],
        'K':  ['EVI', 'MSR', 'NDMI'],
        'EC': ['SI', 'NDSI'],
        'OC': ['CAI', 'NDTI'],
        'pH': ['IOR', 'BSI'],
    }

    # Accuracy targets per nutrient (R² values from literature)
    NUTRIENT_ACCURACY_TARGETS = {
        'N':  0.82,   # Best — direct red-edge sensitivity
        'P':  0.74,   # Moderate — indirect via plant health
        'K':  0.70,   # Moderate — indirect via water stress
        'EC': 0.88,   # Excellent — SI/NDSI direct correlation
        'OC': 0.80,   # Good — CAI well validated
        'pH': 0.72,   # Moderate — IOR proxy
    }

    # =========================================================================
    # HEALTH SCORING CATEGORIES
    # =========================================================================

    HEALTH_CATEGORIES = {
        'Excellent': (85, 101),
        'Good':      (70,  85),
        'Moderate':  (55,  70),
        'Poor':      (35,  55),
        'Critical':  ( 0,  35),
    }

    ALERT_THRESHOLDS = {
        'nitrogen_deficiency': {'index': 'RENDVI', 'threshold': 0.30},
        'water_stress':        {'index': 'NDMI',   'threshold': 0.10},
        'salinity_stress':     {'index': 'SI',     'threshold': 80.0},
        'senescence':          {'index': 'PSRI',   'threshold': 0.25},
        'bare_soil':           {'index': 'BSI',    'threshold': 0.20},
    }

    # =========================================================================
    # YIELD ESTIMATION
    # =========================================================================

    YIELD_METHOD = 'cumulative_ndvi'     # primary method
    YIELD_METHOD_ADVANCED = 'xgboost_multi_index'

    # Seasonal NDVI integral thresholds (area under NDVI time series)
    YIELD_NDVI_INTEGRAL = {
        'high_yield':   7.5,
        'medium_yield': 5.5,
        'low_yield':    3.5,
    }

    # =========================================================================
    # REGIONAL MARKET ANALYTICS
    # =========================================================================

    MARKET_FACILITY_TYPES = [
        'APMC_Mandi',
        'Procurement_Centre',
        'Cold_Storage',
        'Processing_Unit',
        'Retail_Aggregator',
        'Export_Hub',
    ]

    # Max distance (km) to search for market facilities
    MARKET_SEARCH_RADIUS_KM = {
        'APMC_Mandi':        25,
        'Cold_Storage':      40,
        'Processing_Unit':   50,
        'Export_Hub':        100,
    }

    # =========================================================================
    # WEATHER INTEGRATION
    # =========================================================================

    WEATHER_PARAMETERS = [
        "T2M",          # Temperature at 2m
        "T2M_MAX",      # Max temperature
        "T2M_MIN",      # Min temperature
        "PRECTOTCORR",  # Precipitation
        "RH2M",         # Relative humidity
        "WS2M",         # Wind speed
        "ALLSKY_SFC_SW_DWN",  # Solar radiation (for GDD)
    ]

    # Growing Degree Days (GDD) base temperatures per crop group
    GDD_BASE_TEMPS = {
        'cereals':    10.0,
        'pulses':     10.0,
        'oilseeds':   7.0,
        'vegetables': 5.0,
        'fruits':     10.0,
        'fiber':      15.0,
        'sugarcane':  15.0,
    }

    WEATHER_API_TIMEOUT     = 120
    WEATHER_API_MAX_RETRIES = 3

    HEATWAVE_THRESHOLD_C      = 40
    HEATWAVE_MIN_DAYS         =  3
    COLD_WAVE_THRESHOLD_C     = 10
    COLD_WAVE_MIN_DAYS        =  3
    HEAVY_RAIN_SINGLE_DAY_MM  = 100
    HEAVY_RAIN_3DAY_MM        = 200
    DROUGHT_MIN_DAYS          =  21   # tightened from 30 — earlier warning
    DROUGHT_DAILY_RAINFALL_MM =   2

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    MIN_FIELD_BUFFER_KM = 0.5
    EARTH_RADIUS_KM     = 6371.0
    KM_PER_DEGREE_LAT   = 111.0

    # =========================================================================
    # SUPPORTED CROPS (23 + Zaid additions)
    # =========================================================================

    SUPPORTED_CROPS = [
        'Bajra', 'Banana', 'Cabbage', 'Chilli', 'Cotton',
        'Gram', 'Grapes', 'Groundnut', 'Jowar', 'Maize',
        'Mustard', 'Onion', 'Others', 'Papaya', 'Pomegranate',
        'Potato', 'Rice', 'Soyabean', 'Sugarcane', 'Sunflower',
        'Tobacco', 'Tur', 'Wheat',
        # Zaid season additions
        'Watermelon', 'Muskmelon', 'Cucumber', 'Moong', 'Tomato',
    ]

    HIGH_VALUE_CROPS = [
        'Chilli', 'Cotton', 'Grapes', 'Groundnut', 'Mustard',
        'Onion', 'Papaya', 'Pomegranate', 'Potato',
        'Sunflower', 'Tobacco', 'Tur', 'Tomato',
    ]

    # =========================================================================
    # API ENDPOINTS
    # =========================================================================

    STAC_API_URL            = "https://planetarycomputer.microsoft.com/api/stac/v1"
    SENTINEL2_COLLECTION    = "sentinel-2-l2a"
    LANDSAT_COLLECTION      = "landsat-c2-l2"
    NASA_POWER_BASE_URL     = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_POWER_COMMUNITY    = "AG"
    AGMARKNET_API_BASE   = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
    AGMARKNET_API_FORMAT = "json"
    # Register for API key at: https://data.gov.in/user/register

    # =========================================================================
    # LOGGING / OUTPUT
    # =========================================================================

    LOG_LEVEL  = "INFO"
    LOG_FORMAT = '%(asctime)s [ZUTO-AGRI] %(name)s — %(levelname)s — %(message)s'
    OUTPUT_FORMATS = ['json', 'geojson', 'csv', 'pdf_report']

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @classmethod
    def validate(cls):
        errors = []
        if cls.MAX_CLOUD_COVER <= 0 or cls.MAX_CLOUD_COVER > 100:
            errors.append("MAX_CLOUD_COVER must be between 0–100")
        if cls.MIN_VALID_PIXEL_RATIO <= 0 or cls.MIN_VALID_PIXEL_RATIO > 1:
            errors.append("MIN_VALID_PIXEL_RATIO must be 0–1")

        # Duplicate index detection — catches naming bugs at import time
        seen, duplicates = set(), []
        for idx in cls.ALL_INDICES:
            if idx in seen:
                duplicates.append(idx)
            seen.add(idx)
        if duplicates:
            errors.append(
                f"Duplicate index names in ALL_INDICES: {duplicates}. "
                f"Each index must appear exactly once across all TIER lists."
            )

        if errors:
            raise ValueError(f"ZutoAgriConfig errors: {errors}")
        return True


ZutoAgriConfig.validate()