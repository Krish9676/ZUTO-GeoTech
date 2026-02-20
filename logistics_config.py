"""
ZUTO Geotech Solutions — Logistics Intelligence Platform
=========================================================
Component 3: Logistics Configuration

VERSION 1.0
Author: ZUTO Geotech Solutions

Covers:
  - Agri-Logistics Corridor Mapping
  - Last-Mile Delivery Optimization
  - Warehouse & Hub Placement Intelligence
  - Address Intelligence & Geocoding
  - Delivery Zone Management

Data Sources:
  - OSM road network (pgRouting / NetworkX)
  - DEM/SRTM (terrain, road quality)
  - Sentinel-2 LULC (from AgriTech/Env components)
  - Crop production maps (from AgriTech Component 1)
  - Google Maps/HERE API (travel time)
  - Admin boundary shapefiles (state/district/tehsil/village)
  - India Post address database, Bhuvan API
"""


class ZutoLogisticsConfig:
    """Central configuration for ZUTO Logistics Intelligence Platform"""

    # =========================================================================
    # PLATFORM IDENTITY
    # =========================================================================

    PLATFORM_NAME    = "ZUTO Logistics Intelligence Platform"
    PLATFORM_VERSION = "1.0.0"
    COMPANY          = "ZUTO Geotech Solutions"

    # =========================================================================
    # INDIA ROAD NETWORK CLASSIFICATION
    # =========================================================================

    ROAD_CLASSES = {
        'NH':   {'name': 'National Highway',   'speed_kmh': 80, 'quality': 'good',    'truck_capable': True},
        'SH':   {'name': 'State Highway',      'speed_kmh': 60, 'quality': 'good',    'truck_capable': True},
        'MDR':  {'name': 'Major District Road', 'speed_kmh': 45, 'quality': 'moderate','truck_capable': True},
        'ODR':  {'name': 'Other District Road', 'speed_kmh': 30, 'quality': 'poor',   'truck_capable': False},
        'VR':   {'name': 'Village Road',        'speed_kmh': 20, 'quality': 'poor',   'truck_capable': False},
        'PMGSY':{'name': 'PMGSY Rural Road',   'speed_kmh': 35, 'quality': 'moderate','truck_capable': False},
    }

    # Road surface roughness factor (multiplier on travel time)
    ROUGHNESS_FACTOR = {
        'good':     1.0,
        'moderate': 1.25,
        'poor':     1.60,
        'very_poor': 2.20,
    }

    # Seasonal road accessibility (monsoon impact)
    SEASONAL_FACTOR = {
        'kharif':   {'ODR': 0.6, 'VR': 0.4},   # 40-60% roads accessible in monsoon
        'rabi':     {'ODR': 1.0, 'VR': 0.9},
        'zaid':     {'ODR': 1.0, 'VR': 1.0},
    }

    # =========================================================================
    # VEHICLE TYPES
    # =========================================================================

    VEHICLE_TYPES = {
        'mini_truck': {
            'name': 'Mini Truck (Tata Ace)',
            'payload_kg': 750,
            'volume_m3':  3.5,
            'speed_kmh':  50,
            'cost_per_km': 12.0,
            'road_access': ['NH', 'SH', 'MDR', 'ODR', 'VR', 'PMGSY'],
        },
        'medium_truck': {
            'name': 'Medium Truck (8T)',
            'payload_kg': 8000,
            'volume_m3':  30.0,
            'speed_kmh':  60,
            'cost_per_km': 25.0,
            'road_access': ['NH', 'SH', 'MDR'],
        },
        'large_truck': {
            'name': 'Large Truck (20T)',
            'payload_kg': 20000,
            'volume_m3':  70.0,
            'speed_kmh':  70,
            'cost_per_km': 40.0,
            'road_access': ['NH', 'SH'],
        },
        'refrigerated': {
            'name': 'Reefer Van',
            'payload_kg': 5000,
            'volume_m3':  20.0,
            'speed_kmh':  55,
            'cost_per_km': 35.0,
            'road_access': ['NH', 'SH', 'MDR'],
            'temp_range_c': (-25, 8),
        },
        'two_wheeler': {
            'name': 'Two-wheeler (Delivery)',
            'payload_kg': 25,
            'volume_m3':  0.1,
            'speed_kmh':  30,
            'cost_per_km': 3.5,
            'road_access': ['NH', 'SH', 'MDR', 'ODR', 'VR', 'PMGSY'],
        },
    }

    # =========================================================================
    # AGRI-LOGISTICS CORRIDOR PARAMETERS
    # =========================================================================

    CORRIDOR_THRESHOLDS = {
        # Minimum crop production to justify corridor priority (MT/year)
        'min_production_mt':      500,
        # Maximum distance farm → mandi to be considered connected (km)
        'max_farm_mandi_km':      50,
        # Bottleneck: road quality score < this = infrastructure gap
        'road_quality_gap':       40,   # 0-100 scale
        # Critical volume on a road segment = potential bottleneck (MT/day)
        'volume_bottleneck_mt_day': 200,
    }

    # Commodity-specific logistics requirements
    COMMODITY_LOGISTICS = {
        'perishables': {
            'max_transport_hours': 12,
            'requires_cold_chain': True,
            'priority': 'HIGH',
        },
        'grains': {
            'max_transport_hours': 72,
            'requires_cold_chain': False,
            'priority': 'MEDIUM',
        },
        'vegetables': {
            'max_transport_hours': 18,
            'requires_cold_chain': True,
            'priority': 'HIGH',
        },
        'oilseeds': {
            'max_transport_hours': 48,
            'requires_cold_chain': False,
            'priority': 'MEDIUM',
        },
    }

    # =========================================================================
    # WAREHOUSE & HUB PLACEMENT
    # =========================================================================

    WAREHOUSE_TYPES = {
        'primary_hub': {
            'name': 'Primary Hub (50T+ capacity)',
            'min_capacity_t': 500,
            'min_area_sqm':   5000,
            'road_access_req': 'NH',
            'catchment_km':    100,
            'cold_chain_req':  False,
        },
        'cold_storage': {
            'name': 'Cold Storage (10°C)',
            'min_capacity_t': 200,
            'min_area_sqm':   2000,
            'road_access_req': 'SH',
            'catchment_km':    50,
            'cold_chain_req':  True,
            'temp_c':          10,
        },
        'aggregation_center': {
            'name': 'Farm Aggregation Center',
            'min_capacity_t': 50,
            'min_area_sqm':   500,
            'road_access_req': 'MDR',
            'catchment_km':    20,
            'cold_chain_req':  False,
        },
        'dark_store': {
            'name': 'Quick Commerce Dark Store',
            'min_capacity_t': 10,
            'min_area_sqm':   200,
            'road_access_req': 'ODR',
            'catchment_km':    5,
            'cold_chain_req':  False,
        },
    }

    PLACEMENT_WEIGHTS = {
        'demand_score':          0.35,
        'road_accessibility':    0.25,
        'land_availability':     0.20,
        'competition_proximity': 0.10,
        'cost_surface':          0.10,
    }

    # =========================================================================
    # ADDRESS INTELLIGENCE
    # =========================================================================

    ADDRESS_COMPONENTS = [
        'flat_door',        # Flat/Door/House number
        'building_name',    # Building or society name
        'street',           # Street/road name
        'locality',         # Mohalla/Nagar/Colony
        'subdistrict',      # Tehsil/Taluka
        'district',         # District
        'state',            # State
        'pincode',          # PIN code (6-digit India)
        'landmark',         # Near landmark
    ]

    # Indian state codes (ISO 3166-2:IN)
    INDIA_STATES = {
        'MH': 'Maharashtra', 'KA': 'Karnataka', 'TN': 'Tamil Nadu',
        'AP': 'Andhra Pradesh', 'TS': 'Telangana', 'GJ': 'Gujarat',
        'RJ': 'Rajasthan', 'MP': 'Madhya Pradesh', 'UP': 'Uttar Pradesh',
        'DL': 'Delhi', 'HR': 'Haryana', 'PB': 'Punjab', 'WB': 'West Bengal',
        'OR': 'Odisha', 'BR': 'Bihar', 'AS': 'Assam', 'KL': 'Kerala',
        'HP': 'Himachal Pradesh', 'UK': 'Uttarakhand', 'GA': 'Goa',
        'CG': 'Chhattisgarh', 'JH': 'Jharkhand', 'JK': 'Jammu & Kashmir',
        'MN': 'Manipur', 'ML': 'Meghalaya', 'NL': 'Nagaland', 'TR': 'Tripura',
        'SK': 'Sikkim', 'AR': 'Arunachal Pradesh', 'MZ': 'Mizoram',
    }

    # Address parsing patterns (regex keys)
    ADDRESS_PATTERNS = {
        'pincode':   r'\b[1-9]\d{5}\b',
        'flat_no':   r'(?:flat|f|apt|apartment|plot|door)\s*[#:\-]?\s*\d+[a-zA-Z]?',
        'survey_no': r'(?:survey|s\.no|s/n|gat)\s*[#:\-]?\s*\d+',
        'village':   r'(?:at|at/po|a/p|village|vill?)\s*[:.\-]?\s*([a-zA-Z\s]+)',
        'po':        r'(?:po|p\.o\.|post)\s*[:.\-]?\s*([a-zA-Z\s]+)',
    }

    # Geocoding confidence thresholds
    GEOCODING_CONFIDENCE = {
        'HIGH':   0.85,     # Exact match to building/door level
        'MEDIUM': 0.60,     # Street/locality level
        'LOW':    0.40,     # Sub-district level
        'FAILED': 0.0,
    }

    # =========================================================================
    # DELIVERY ZONE (ISOCHRONE) PARAMETERS
    # =========================================================================

    DELIVERY_ZONE_MODES = {
        'quick_commerce': {
            'name': 'Quick Commerce (30 min)',
            'time_limit_min': 30,
            'vehicle': 'two_wheeler',
            'service_type': 'hyperlocal',
        },
        'food_delivery': {
            'name': 'Food Delivery (45 min)',
            'time_limit_min': 45,
            'vehicle': 'two_wheeler',
            'service_type': 'hyperlocal',
        },
        'agri_last_mile': {
            'name': 'Agri Last-Mile (Same day)',
            'time_limit_min': 240,
            'vehicle': 'mini_truck',
            'service_type': 'agri',
        },
        'b2b_distribution': {
            'name': 'B2B Distribution (Next day)',
            'time_limit_min': 480,
            'vehicle': 'medium_truck',
            'service_type': 'commercial',
        },
        'pharma': {
            'name': 'Pharmaceutical Distribution',
            'time_limit_min': 360,
            'vehicle': 'refrigerated',
            'service_type': 'pharma',
        },
    }

    # Max zones per depot (to avoid over-splitting)
    MAX_ZONES_PER_DEPOT = 20

    # =========================================================================
    # COST PARAMETERS (INR)
    # =========================================================================

    COST_PARAMS = {
        'land_cost_per_sqm': {
            'urban':     5000,    # INR/sqm
            'peri_urban': 2000,
            'rural':      500,
        },
        'construction_per_sqm': {
            'warehouse': 3000,
            'cold_storage': 7000,
            'dark_store': 4000,
        },
        'fuel_per_litre':    105,   # INR (diesel)
        'driver_per_day':    800,   # INR
        'loading_per_mt':    150,   # INR/MT
    }

    # =========================================================================
    # API INTEGRATION KEYS (environment variable names — not actual keys)
    # =========================================================================

    API_KEYS = {
        'google_maps':  'GOOGLE_MAPS_API_KEY',
        'here_maps':    'HERE_API_KEY',
        'openrouteservice': 'ORS_API_KEY',
        'bhuvan':       'BHUVAN_API_KEY',
        'india_post':   'INDIA_POST_API_KEY',
    }
