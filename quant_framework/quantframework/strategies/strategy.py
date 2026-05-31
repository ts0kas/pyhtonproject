"""Trading strategies.

A :class:`Strategy` maps a price panel to a panel of *target weights* (one row
per timestamp, one column per asset, each row summing to <= 1 in absolute
terms). Decoupling signal generation from position sizing and execution lets
the same strategy run under different cost models or rebalance schedules.

Three reference strategies are provided:

* :class:`EqualWeightStrategy` -- a naive 1/N benchmark.
* :class:`MomentumStrategy` -- cross-sectional time-series momentum.
* :class:`MeanReversionStrategy` -- z-score reversion around a moving average.
"""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd

from quantframework.utils.exceptions import StrategyError
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)


class Strategy(abc.ABC):
    """Abstract base class for signal-generating strategies."""

    name: str = "base"

    @abc.abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a panel of target weights aligned to ``prices``.

        Parameters
        ----------
        prices : pandas.DataFrame
            Adjusted close prices, DatetimeIndex rows, one column per asset.

        Returns
        -------
        pandas.DataFrame
            Target weights with the same index/columns as ``prices``.
        """
        raise NotImplementedError

    @staticmethod
    def _validate(prices: pd.DataFrame) -> None:
        if prices.empty or prices.shape[1] == 0:
            raise StrategyError("Price panel is empty.")
        if prices.isna().all().any():
            raise StrategyError("One or more assets are entirely NaN.")


class EqualWeightStrategy(Strategy):
    """Allocate equally across all assets every period (1/N)."""

    name = "equal_weight"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        self._validate(prices)
        n = prices.shape[1]
        return pd.DataFrame(
            1.0 / n, index=prices.index, columns=prices.columns
        )


class MomentumStrategy(Strategy):
    """Cross-sectional momentum.

    Each rebalance, rank assets by trailing total return over ``lookback``
    periods and go long the top ``top_n`` (equally weighted). A skip window
    avoids the well-documented short-term reversal contaminating the signal.

    Parameters
    ----------
    lookback : int, optional
        Formation window in periods, by default 126 (~6 months daily).
    skip : int, optional
        Most-recent periods to skip, by default 21 (~1 month).
    top_n : int, optional
        Number of assets to hold long. ``None`` holds all with positive
        momentum. By default ``None``.
    """

    name = "momentum"

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 21,
        top_n: int | None = None,
    ) -> None:
        if lookback <= skip:
            raise StrategyError("`lookback` must exceed `skip`.")
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        self._validate(prices)
        # Trailing return from t-lookback to t-skip (vectorised, no loops).
        momentum = prices.shift(self.skip) / prices.shift(self.lookback) - 1.0

        if self.top_n is not None:
            ranks = momentum.rank(axis=1, ascending=False)
            selected = ranks <= self.top_n
        else:
            selected = momentum > 0.0

        selected = selected.astype(float)
        counts = selected.sum(axis=1).replace(0.0, np.nan)
        weights = selected.div(counts, axis=0).fillna(0.0)
        return weights


class MeanReversionStrategy(Strategy):
    """Z-score mean reversion around a rolling moving average.

    For each asset, compute the z-score of price relative to its rolling mean
    and standard deviation. Take positions opposite to the deviation: long when
    cheap (negative z), short when rich (positive z). Positions are scaled so
    gross exposure sums to one.

    Parameters
    ----------
    lookback : int, optional
        Rolling window for the mean/std, by default 21.
    entry_z : float, optional
        Absolute z-score beyond which a position is taken, by default 1.0.
    allow_short : bool, optional
        If ``False``, negative legs are dropped (long-only), by default ``True``.
    """

    name = "mean_reversion"

    def __init__(
        self,
        lookback: int = 21,
        entry_z: float = 1.0,
        allow_short: bool = True,
    ) -> None:
        if lookback < 2:
            raise StrategyError("`lookback` must be >= 2.")
        self.lookback = lookback
        self.entry_z = entry_z
        self.allow_short = allow_short

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        self._validate(prices)
        roll_mean = prices.rolling(self.lookback).mean()
        roll_std = prices.rolling(self.lookback).std(ddof=0)
        zscore = (prices - roll_mean) / roll_std.replace(0.0, np.nan)

        # Signal is opposite the deviation; only act beyond the entry band.
        raw = -zscore.where(zscore.abs() >= self.entry_z, 0.0)
        if not self.allow_short:
            raw = raw.clip(lower=0.0)

        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        weights = raw.div(gross, axis=0).fillna(0.0)
        return weights
