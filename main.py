"""
ZUTO GeoTech Solutions — Unified Platform Entry Point
=====================================================
Run this file to verify the full project loads correctly.

Usage:
    python main.py
"""

import os
import sys
from pathlib import Path

# ── Load environment variables from .env before any other imports ────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠  python-dotenv not installed. Run: pip install python-dotenv")
    print("   Environment variables must be set manually.")

from agritech.pipeline import ZutoAgriPipeline
from environment.modules.carbon_mapper import CarbonMapper
from logistics.modules.last_mile_optimizer import LastMileOptimizer


def show_project_summary() -> None:
    """Print high-level project structure and readiness."""
    print('\n' + '='*60)
    print('  ZUTO GeoTech — Unified Platform')
    print('='*60)
    print('\n  Components:')
    print('  1) AgriTech      : field/regional agronomy analytics')
    print('  2) Environment   : climate-risk and ESG intelligence')
    print('  3) Logistics     : corridor and last-mile optimization')
    print('\n  Core classes loaded:')
    print(f'  ✔ {ZutoAgriPipeline.__name__}   (agritech/pipeline.py)')
    print(f'  ✔ {CarbonMapper.__name__}         (environment/modules/carbon_mapper.py)')
    print(f'  ✔ {LastMileOptimizer.__name__}  (logistics/modules/last_mile_optimizer.py)')
    print('\n  MongoDB URI:', 'configured' if os.getenv('MONGO_URI') else '⚠ NOT SET — add to .env')
    print('='*60 + '\n')


if __name__ == '__main__':
    show_project_summary()
