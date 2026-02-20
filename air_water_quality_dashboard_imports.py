import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# scipy is optional — used for RBF/kriging spatial interpolation fallback
try:
    from scipy.interpolate import RBFInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    RBFInterpolator = None  # type: ignore[assignment,misc]

from config.env_config import ZutoEnvConfig
