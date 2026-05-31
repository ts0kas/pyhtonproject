"""Shared type aliases used across the framework.

Centralising aliases keeps signatures readable and gives a single place to
tighten types (e.g. to ``pandera`` schemas) later without touching call sites.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd

# A vector of asset weights keyed by ticker.
Weights = Mapping[str, float]

# A 1-D float array (returns, weights, etc.).
FloatArray = np.ndarray

# Price or return panels indexed by a DatetimeIndex with one column per asset.
PriceFrame = pd.DataFrame
ReturnFrame = pd.DataFrame

# A single price/return series.
PriceSeries = pd.Series
ReturnSeries = pd.Series

# Universe specification.
Universe = Sequence[str]

# Generic JSON-like config payload.
ConfigDict = Dict[str, Union[str, int, float, bool, None]]
