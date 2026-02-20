"""
Backward-compatible import wrapper for ZutoAgriPipeline.
Allows: from agritech_pipeline import ZutoAgriPipeline
at the project root level without going through the package.
"""

from agritech.pipeline import ZutoAgriPipeline

__all__ = ['ZutoAgriPipeline']
