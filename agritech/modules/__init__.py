"""
ZUTO GeoTech Solutions — AgriTech Modules
==========================================
Package initializer for all AgriTech analysis modules.

Public API:
    from agritech.modules.satellite_collector import ZutoSatelliteCollector
    from agritech.modules.spectral_index_engine import SpectralIndexEngine
    from agritech.modules.crop_health_monitor import CropHealthMonitor
    from agritech.modules.soil_nutrient_analyzer import SoilNutrientAnalyzer
    from agritech.modules.yield_forecaster import YieldForecaster
    from agritech.modules.market_analytics import MarketAnalytics
"""

from agritech.modules.spectral_index_engine import SpectralIndexEngine
from agritech.modules.crop_health_monitor import CropHealthMonitor
from agritech.modules.soil_nutrient_analyzer import SoilNutrientAnalyzer
from agritech.modules.yield_forecaster import YieldForecaster
from agritech.modules.market_analytics import MarketAnalytics

# ZutoSatelliteCollector has optional heavy dependencies (pystac, rasterio)
# Import separately to avoid breaking the package if satellite libs are absent
try:
    from agritech.modules.satellite_collector import ZutoSatelliteCollector
    __all__ = [
        'ZutoSatelliteCollector',
        'SpectralIndexEngine',
        'CropHealthMonitor',
        'SoilNutrientAnalyzer',
        'YieldForecaster',
        'MarketAnalytics',
    ]
except ImportError:
    __all__ = [
        'SpectralIndexEngine',
        'CropHealthMonitor',
        'SoilNutrientAnalyzer',
        'YieldForecaster',
        'MarketAnalytics',
    ]
