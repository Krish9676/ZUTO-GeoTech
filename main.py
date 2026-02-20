"""Unified entrypoint for ZUTO GeoTech project."""

from agritech.pipeline import ZutoAgriPipeline
from environment.modules.carbon_mapper import CarbonMapper
from logistics.modules.last_mile_optimizer import LastMileOptimizer


def show_project_summary() -> None:
    """Print high-level project structure and readiness."""
    print('\nZUTO GeoTech — Unified Platform')
    print('1) AgriTech      : field/regional agronomy analytics')
    print('2) Environment   : climate-risk and ESG intelligence')
    print('3) Logistics     : corridor and last-mile optimization')
    print('\nCore classes:')
    print(f'- {ZutoAgriPipeline.__name__}')
    print(f'- {CarbonMapper.__name__}')
    print(f'- {LastMileOptimizer.__name__}')


if __name__ == '__main__':
    show_project_summary()
