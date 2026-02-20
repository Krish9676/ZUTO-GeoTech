"""
ZUTO Geotech Solutions — AgriTech Platform
===========================================
Example Usage & Quick Start

Run this file to see the pipeline structure and what it will produce.
Actual satellite data collection requires pystac-client and rasterio.
"""

from agritech.pipeline import ZutoAgriPipeline
from agritech.modules.spectral_index_engine import SpectralIndexEngine
from config.agri_config import ZutoAgriConfig


def example_field_analysis():
    """Example: Analyze a wheat field in Punjab."""
    pipeline = ZutoAgriPipeline(verbose=True)

    result = pipeline.analyze_field(
        latitude=30.9010,       # Ludhiana, Punjab
        longitude=75.8573,
        field_area_ha=3.5,
        crop='Wheat',
        sowing_date='2024-11-20',
        num_seasons=4,
        seasons=['rabi', 'kharif'],
        output_path='outputs/wheat_field_ludhiana.json',
    )
    return result


def example_cotton_field():
    """Example: Monitor cotton crop in Vidarbha, Maharashtra."""
    pipeline = ZutoAgriPipeline()

    result = pipeline.analyze_field(
        latitude=20.7002,       # Nagpur district
        longitude=79.0882,
        field_area_ha=5.0,
        crop='Cotton',
        sowing_date='2024-06-15',
        seasons=['kharif'],
        num_seasons=3,
    )
    return result


def example_regional_analysis():
    """Example: Regional crop mapping for a district."""
    pipeline = ZutoAgriPipeline()

    result = pipeline.analyze_region(
        bbox=[75.5, 30.5, 76.5, 31.5],  # District bbox (Ludhiana area)
        season='rabi',
        num_seasons=3,
        output_path='outputs/ludhiana_regional_rabi.json',
    )
    return result


def example_index_demo():
    """
    Demo: Show all 28 indices the engine computes on a synthetic scene.
    No satellite data needed.
    """
    import numpy as np

    print("\n" + "="*60)
    print("ZUTO SpectralIndexEngine — 28-Index Suite Demo")
    print("="*60)

    engine = SpectralIndexEngine(scale_bands=False)  # already in 0-1 range

    # Synthetic band data simulating a healthy wheat field (NDVI ~0.75)
    shape  = (50, 50)
    np.random.seed(42)
    bands = {
        'B02': np.random.uniform(0.05, 0.10, shape),  # Blue
        'B03': np.random.uniform(0.08, 0.12, shape),  # Green
        'B04': np.random.uniform(0.06, 0.10, shape),  # Red (low = healthy)
        'B05': np.random.uniform(0.20, 0.30, shape),  # Red Edge 1
        'B06': np.random.uniform(0.30, 0.40, shape),  # Red Edge 2
        'B07': np.random.uniform(0.40, 0.50, shape),  # Red Edge 3
        'B08': np.random.uniform(0.55, 0.70, shape),  # NIR (high = healthy)
        'B8A': np.random.uniform(0.55, 0.68, shape),  # Narrow NIR
        'B11': np.random.uniform(0.15, 0.25, shape),  # SWIR 1
        'B12': np.random.uniform(0.08, 0.15, shape),  # SWIR 2
    }

    results = engine.compute_all(bands, scene_date='2025-01-15')

    print(f"\n{'Index':<15} {'Mean':>8} {'Std':>8} {'P10':>8} {'P90':>8}")
    print("-" * 55)

    tier_labels = {
        **{i: 'VEGETATION' for i in ZutoAgriConfig.TIER1_INDICES},
        **{i: 'NUTRIENT  ' for i in ZutoAgriConfig.TIER2_INDICES},
        **{i: 'MOISTURE  ' for i in ZutoAgriConfig.TIER3_INDICES},
        **{i: 'CANOPY    ' for i in ZutoAgriConfig.TIER4_INDICES},
    }

    for idx_name, stats in results.items():
        label = tier_labels.get(idx_name, '          ')
        print(f"[{label}] {idx_name:<12} "
              f"{stats['mean']:8.4f} {stats['std']:8.4f} "
              f"{stats['p10']:8.4f} {stats['p90']:8.4f}")

    print(f"\nTotal indices computed: {len(results)}/28")

    # Nutrient interpretation
    from agritech.modules.spectral_index_engine import SpectralIndexEngine
    interp = SpectralIndexEngine.interpret_nutrient_levels(results)
    print("\n--- Nutrient Interpretation ---")
    for nutrient, info in interp.items():
        print(f"  {nutrient}: {info['status']}")


def show_config_summary():
    """Show ZUTO AgriConfig summary."""
    cfg = ZutoAgriConfig
    print(f"\n{'='*60}")
    print(f"  {cfg.PLATFORM_NAME} v{cfg.PLATFORM_VERSION}")
    print(f"  {cfg.COMPANY}")
    print(f"{'='*60}")
    print(f"  Seasons:         {list(cfg.SEASONS.keys())}")
    print(f"  Analysis window: {cfg.NUM_SEASONS_HISTORY} seasons")
    print(f"  Cloud threshold: {cfg.MAX_CLOUD_COVER}%")
    print(f"  All timestamps:  {cfg.COLLECT_ALL_TIMESTAMPS}")
    print(f"  Indices/scene:   {len(cfg.ALL_INDICES)}")
    print(f"  Supported crops: {len(cfg.SUPPORTED_CROPS)}")
    print(f"  Nutrients:       {list(cfg.NUTRIENT_TARGETS.keys())}")
    print(f"\n  Index Breakdown:")
    print(f"    Tier 1 Vegetation:  {len(cfg.TIER1_INDICES)} → {cfg.TIER1_INDICES}")
    print(f"    Tier 2 Nutrients:   {len(cfg.TIER2_INDICES)} → {cfg.TIER2_INDICES}")
    print(f"    Tier 3 Moisture:    {len(cfg.TIER3_INDICES)} → {cfg.TIER3_INDICES}")
    print(f"    Tier 4 Canopy:      {len(cfg.TIER4_INDICES)} → {cfg.TIER4_INDICES}")


if __name__ == '__main__':
    show_config_summary()
    example_index_demo()

    # Uncomment to run actual satellite collection (requires libraries):
    # result = example_field_analysis()
    # result = example_cotton_field()
    # result = example_regional_analysis()
