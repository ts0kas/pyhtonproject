"""On-disk caching layer for market data.

Caching avoids hammering data vendors during research iterations and makes
backtests reproducible. Data is persisted as Parquet (columnar, typed,
compressed) keyed by a deterministic hash of the request parameters.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataCache:
    """A simple, TTL-aware Parquet cache for price panels.

    Parameters
    ----------
    cache_dir : str or pathlib.Path, optional
        Directory in which cached files are stored. Created if missing.
    ttl : datetime.timedelta, optional
        Maximum age before a cache entry is considered stale. ``None`` disables
        expiry (entries live forever), which is appropriate for fully
        historical date ranges.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/market_data",
        ttl: Optional[timedelta] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    @staticmethod
    def _key(
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str,
        adjusted: bool,
    ) -> str:
        """Build a deterministic cache key from request parameters."""
        payload = json.dumps(
            {
                "tickers": sorted(tickers),
                "start": start,
                "end": end,
                "interval": interval,
                "adjusted": adjusted,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str,
        adjusted: bool,
    ) -> Optional[pd.DataFrame]:
        """Return a cached frame if present and fresh, otherwise ``None``."""
        path = self._path(self._key(tickers, start, end, interval, adjusted))
        if not path.exists():
            return None

        if self.ttl is not None:
            age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
            if age > self.ttl:
                logger.debug("Cache entry %s expired (age=%s).", path.name, age)
                return None

        logger.info("Cache hit: %s", path.name)
        return pd.read_parquet(path)

    def put(
        self,
        frame: pd.DataFrame,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str,
        adjusted: bool,
    ) -> None:
        """Persist a frame to the cache."""
        path = self._path(self._key(tickers, start, end, interval, adjusted))
        frame.to_parquet(path)
        logger.info("Cached %d rows -> %s", len(frame), path.name)
