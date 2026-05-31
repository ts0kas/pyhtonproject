"""Performance and risk-adjusted return metrics.

Computes the standard quant tearsheet: annualised return/volatility, Sharpe and
Sortino ratios, maximum drawdown and Calmar, win-rate, profit factor, and the
CAPM alpha/beta of the strategy against a benchmark. All metrics are computed
from a periodic net-return series and an annualisation factor so the same code
serves daily, weekly, or monthly data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from quantframework.utils.exceptions import InsufficientDataError
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)

_EPS = 1e-12


@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable tearsheet of performance statistics."""

    total_return: float
    cagr: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    alpha: Optional[float]
    beta: Optional[float]
    skew: float
    kurtosis: float

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return the metrics as a plain dictionary."""
        return asdict(self)

    def to_frame(self) -> pd.DataFrame:
        """Return a single-column DataFrame suitable for display/export."""
        return pd.DataFrame.from_dict(
            self.to_dict(), orient="index", columns=["value"]
        )


class PerformanceAnalyzer:
    """Compute tearsheet metrics from a return series.

    Parameters
    ----------
    returns : pandas.Series
        Periodic net portfolio returns.
    periods_per_year : int, optional
        Annualisation factor, by default 252.
    risk_free_rate : float, optional
        Annualised risk-free rate, by default 0.0.
    """

    def __init__(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        clean = returns.dropna()
        if clean.empty:
            raise InsufficientDataError("Return series is empty.")
        self.returns = clean
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self._rf_per_period = risk_free_rate / periods_per_year

    # ------------------------------------------------------------------ #
    # Equity / drawdown
    # ------------------------------------------------------------------ #
    def equity_curve(self) -> pd.Series:
        """Cumulative growth of one unit of capital."""
        return (1.0 + self.returns).cumprod()

    def drawdown_series(self) -> pd.Series:
        """Drawdown at each point relative to the running peak."""
        equity = self.equity_curve()
        running_max = equity.cummax()
        return equity / running_max - 1.0

    def max_drawdown(self) -> float:
        """Worst peak-to-trough decline (negative number)."""
        return float(self.drawdown_series().min())

    # ------------------------------------------------------------------ #
    # Return / risk
    # ------------------------------------------------------------------ #
    def cagr(self) -> float:
        """Compound annual growth rate."""
        equity = self.equity_curve()
        n_years = len(self.returns) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)

    def annual_volatility(self) -> float:
        """Annualised standard deviation of returns."""
        return float(self.returns.std(ddof=1) * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio."""
        excess = self.returns - self._rf_per_period
        denom = excess.std(ddof=1)
        if denom < _EPS:
            return 0.0
        return float(excess.mean() / denom * np.sqrt(self.periods_per_year))

    def sortino_ratio(self) -> float:
        """Annualised Sortino ratio (downside-deviation denominator)."""
        excess = self.returns - self._rf_per_period
        downside = excess[excess < 0.0]
        downside_dev = np.sqrt((downside**2).mean()) if not downside.empty else 0.0
        if downside_dev < _EPS:
            return 0.0
        return float(
            excess.mean() / downside_dev * np.sqrt(self.periods_per_year)
        )

    def calmar_ratio(self) -> float:
        """CAGR divided by the absolute maximum drawdown."""
        mdd = abs(self.max_drawdown())
        return float(self.cagr() / mdd) if mdd > _EPS else 0.0

    # ------------------------------------------------------------------ #
    # Trade-quality statistics
    # ------------------------------------------------------------------ #
    def win_rate(self) -> float:
        """Fraction of periods with a positive return."""
        return float((self.returns > 0).mean())

    def profit_factor(self) -> float:
        """Gross gains divided by gross losses."""
        gains = self.returns[self.returns > 0].sum()
        losses = -self.returns[self.returns < 0].sum()
        return float(gains / losses) if losses > _EPS else np.inf

    # ------------------------------------------------------------------ #
    # CAPM regression vs benchmark
    # ------------------------------------------------------------------ #
    def alpha_beta(
        self, benchmark_returns: Optional[pd.Series]
    ) -> tuple[Optional[float], Optional[float]]:
        """Annualised Jensen's alpha and beta versus a benchmark.

        Uses an OLS regression of strategy excess returns on benchmark excess
        returns. Returns ``(None, None)`` when no benchmark is supplied.
        """
        if benchmark_returns is None:
            return None, None

        aligned = pd.concat(
            [self.returns, benchmark_returns], axis=1, join="inner"
        ).dropna()
        if len(aligned) < 2:
            return None, None

        strat = aligned.iloc[:, 0] - self._rf_per_period
        bench = aligned.iloc[:, 1] - self._rf_per_period

        cov = np.cov(strat, bench, ddof=1)
        var_bench = cov[1, 1]
        if var_bench < _EPS:
            return None, None

        beta = float(cov[0, 1] / var_bench)
        alpha_per_period = float(strat.mean() - beta * bench.mean())
        alpha_annual = alpha_per_period * self.periods_per_year
        return alpha_annual, beta

    # ------------------------------------------------------------------ #
    # Aggregate
    # ------------------------------------------------------------------ #
    def compute(
        self, benchmark_returns: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """Compute the full tearsheet in a single call."""
        alpha, beta = self.alpha_beta(benchmark_returns)
        metrics = PerformanceMetrics(
            total_return=float(self.equity_curve().iloc[-1] - 1.0),
            cagr=self.cagr(),
            annual_volatility=self.annual_volatility(),
            sharpe_ratio=self.sharpe_ratio(),
            sortino_ratio=self.sortino_ratio(),
            max_drawdown=self.max_drawdown(),
            calmar_ratio=self.calmar_ratio(),
            win_rate=self.win_rate(),
            profit_factor=self.profit_factor(),
            alpha=alpha,
            beta=beta,
            skew=float(self.returns.skew()),
            kurtosis=float(self.returns.kurtosis()),
        )
        return metrics
