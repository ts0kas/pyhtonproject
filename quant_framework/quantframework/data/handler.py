"""Data ingestion pipeline.

Defines the abstract :class:`DataHandler` contract and a concrete
:class:`YFinanceDataHandler` that downloads, caches, cleans, and resamples
historical OHLCV data. The abstraction means the rest of the framework is
vendor-agnostic: swapping in an Alpaca or Polygon handler requires only a new
subclass, not changes to the backtester or analytics.
"""

from __future__ import annotations

import abc
from datetime import timedelta
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from quantframework.data.cache import DataCache
from quantframework.utils.exceptions import (
    DataIngestionError,
    InsufficientDataError,
)
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)

# Pandas offset aliases accepted for resampling.
_VALID_RESAMPLE = {"D", "W", "M", "Q", "Y", "B", "W-FRI", "ME", "QE", "YE"}


class DataHandler(abc.ABC):
    """Abstract base class for all market-data providers.

    Concrete subclasses implement :meth:`_download`, returning a tidy panel of
    adjusted close prices. Shared concerns -- caching, missing-data handling,
    resampling, and return computation -- live here so every provider behaves
    identically downstream.
    """

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        self._cache = cache or DataCache(ttl=timedelta(days=1))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_prices(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
        adjusted: bool = True,
        max_missing_pct: float = 0.10,
    ) -> pd.DataFrame:
        """Return a clean panel of (adjusted) close prices.

        Parameters
        ----------
        tickers : sequence of str
            Asset symbols to download.
        start, end : str
            ISO-8601 date bounds (``"YYYY-MM-DD"``).
        interval : str, optional
            Vendor sampling interval, by default ``"1d"``.
        adjusted : bool, optional
            If ``True``, prices are adjusted for splits and dividends so that
            returns reflect total return. By default ``True``.
        max_missing_pct : float, optional
            Maximum fraction of missing observations tolerated per asset before
            that asset is dropped, by default ``0.10``.

        Returns
        -------
        pandas.DataFrame
            DatetimeIndex rows, one float column per surviving ticker.

        Raises
        ------
        InsufficientDataError
            If no asset survives the cleaning step.
        DataIngestionError
            If the underlying download fails.
        """
        if not tickers:
            raise DataIngestionError("`tickers` must be non-empty.")

        cached = self._cache.get(tickers, start, end, interval, adjusted)
        if cached is not None:
            return cached

        raw = self._download(tickers, start, end, interval, adjusted)
        clean = self._clean(raw, max_missing_pct=max_missing_pct)
        self._cache.put(clean, tickers, start, end, interval, adjusted)
        return clean

    def get_returns(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
        method: str = "log",
        **kwargs: object,
    ) -> pd.DataFrame:
        """Return periodic returns derived from clean prices.

        Parameters
        ----------
        method : {"log", "simple"}, optional
            ``"log"`` yields continuously compounded returns (time-additive);
            ``"simple"`` yields arithmetic returns. By default ``"log"``.

        Returns
        -------
        pandas.DataFrame
            Returns with the first (NaN) row dropped.
        """
        prices = self.get_prices(tickers, start, end, interval, **kwargs)  # type: ignore[arg-type]
        return self.compute_returns(prices, method=method)

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """Compute returns from a price panel using vectorised operations."""
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        elif method == "simple":
            returns = prices.pct_change()
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"Unknown return method: {method!r}")
        return returns.dropna(how="all")

    @staticmethod
    def resample(
        prices: pd.DataFrame, rule: str, how: str = "last"
    ) -> pd.DataFrame:
        """Resample a price panel to a coarser frequency.

        Parameters
        ----------
        rule : str
            Pandas offset alias (e.g. ``"W"``, ``"ME"``).
        how : {"last", "first", "mean", "ohlc"}, optional
            Aggregation applied within each bucket, by default ``"last"``
            (the convention for close-price panels).
        """
        if rule not in _VALID_RESAMPLE:
            logger.warning("Resample rule %r not in known set; passing through.", rule)
        resampler = prices.resample(rule)
        return getattr(resampler, how)().dropna(how="all")

    def _clean(
        self, raw: pd.DataFrame, max_missing_pct: float
    ) -> pd.DataFrame:
        """Handle gaps, duplicate timestamps, and degenerate columns.

        The cleaning policy is deliberately conservative:

        * drop assets missing more than ``max_missing_pct`` of observations;
        * forward-fill short internal gaps (e.g. half-day holidays) then
          back-fill any leading NaNs from late-listing assets;
        * sort the index and remove duplicate timestamps.
        """
        if raw.empty:
            raise InsufficientDataError("Download returned an empty frame.")

        raw = raw[~raw.index.duplicated(keep="first")].sort_index()

        missing_frac = raw.isna().mean()
        survivors = missing_frac[missing_frac <= max_missing_pct].index.tolist()
        dropped = set(raw.columns) - set(survivors)
        if dropped:
            logger.warning(
                "Dropping %d asset(s) over missing-data threshold: %s",
                len(dropped),
                sorted(dropped),
            )
        clean = raw[survivors]

        if clean.empty or clean.shape[1] == 0:
            raise InsufficientDataError(
                "No assets survived the missing-data filter; "
                "relax `max_missing_pct` or widen the date range."
            )

        # Forward-fill intraday/holiday gaps, then back-fill leading NaNs.
        clean = clean.ffill().bfill()
        return clean

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def _download(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str,
        adjusted: bool,
    ) -> pd.DataFrame:
        """Download a raw close-price panel from the vendor."""
        raise NotImplementedError


class YFinanceDataHandler(DataHandler):
    """Yahoo! Finance implementation of :class:`DataHandler`.

    Uses the ``auto_adjust`` flag so the returned close already incorporates
    splits and dividends, giving a clean total-return series.
    """

    def _download(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str,
        adjusted: bool,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf  # imported lazily to keep import side-effect-free
        except ImportError as exc:  # pragma: no cover
            raise DataIngestionError(
                "yfinance is not installed; `pip install yfinance`."
            ) from exc

        logger.info(
            "Downloading %d ticker(s) from yfinance [%s -> %s @ %s].",
            len(tickers),
            start,
            end,
            interval,
        )
        try:
            data = yf.download(
                tickers=list(tickers),
                start=start,
                end=end,
                interval=interval,
                auto_adjust=adjusted,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalise vendor errors
            raise DataIngestionError(f"yfinance download failed: {exc}") from exc

        return self._extract_close(data, tickers)

    @staticmethod
    def _extract_close(
        data: pd.DataFrame, tickers: Sequence[str]
    ) -> pd.DataFrame:
        """Normalise yfinance's variable column layout to a close-price panel."""
        if data.empty:
            raise InsufficientDataError("yfinance returned no rows.")

        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker frame: columns are (field, ticker).
            if "Close" not in data.columns.get_level_values(0):
                raise DataIngestionError("No 'Close' field in download.")
            close = data["Close"].copy()
        else:
            # Single-ticker frame: flat columns.
            close = data[["Close"]].copy()
            close.columns = [tickers[0]]

        close.index = pd.to_datetime(close.index)
        close.index.name = "date"
        return close


def synthetic_price_panel(
    tickers: Sequence[str],
    periods: int = 1000,
    seed: int = 42,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Generate a reproducible geometric-Brownian-motion price panel.

    This offline generator lets the framework, examples, and tests run without
    network access while still exercising the full analytics and backtest path.

    Parameters
    ----------
    tickers : sequence of str
        Column names for the synthetic assets.
    periods : int, optional
        Number of business days to simulate, by default 1000.
    seed : int, optional
        RNG seed for reproducibility, by default 42.
    start : str, optional
        First date in the index.

    Returns
    -------
    pandas.DataFrame
        Synthetic adjusted-close panel.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    dates = pd.bdate_range(start=start, periods=periods, name="date")

    # Give each asset a distinct drift/vol and induce cross-sectional
    # correlation via a shared market factor.
    drifts = rng.uniform(0.03, 0.18, size=n) / 252.0
    vols = rng.uniform(0.15, 0.45, size=n) / np.sqrt(252.0)
    market = rng.standard_normal(periods)
    betas = rng.uniform(0.5, 1.4, size=n)

    shocks = rng.standard_normal((periods, n))
    factor = np.outer(market, betas)
    daily_rets = drifts + vols * (0.6 * factor + 0.4 * shocks)

    prices = 100.0 * np.exp(np.cumsum(daily_rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=list(tickers))
