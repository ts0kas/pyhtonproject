"""Value-at-Risk and Expected Shortfall (CVaR) estimation.

Provides three estimators:

* **Historical** -- empirical quantile of realised returns; assumption-free but
  bounded by the sample.
* **Parametric (Gaussian)** -- closed-form using the normal inverse CDF; fast
  but mis-states tail risk for fat-tailed assets.
* **Monte Carlo** -- simulates correlated multivariate-normal portfolio paths
  via Cholesky factorisation, giving smooth tail estimates and the ability to
  stress horizons longer than one period.

By convention VaR/CVaR are reported as positive loss magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantframework.utils.exceptions import InsufficientDataError
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskEstimate:
    """Container for a paired VaR/CVaR estimate.

    Attributes
    ----------
    var : float
        Value at Risk as a positive loss fraction at ``confidence``.
    cvar : float
        Conditional VaR (Expected Shortfall): mean loss beyond the VaR.
    confidence : float
        Confidence level, e.g. 0.95.
    method : str
        Estimator name.
    horizon : int
        Holding-period horizon in base periods.
    """

    var: float
    cvar: float
    confidence: float
    method: str
    horizon: int


class RiskAnalyzer:
    """Compute tail-risk metrics for a portfolio return series.

    Parameters
    ----------
    portfolio_returns : pandas.Series
        Periodic portfolio returns.
    """

    def __init__(self, portfolio_returns: pd.Series) -> None:
        clean = portfolio_returns.dropna()
        if clean.empty:
            raise InsufficientDataError("Return series is empty.")
        self.returns = clean

    # ------------------------------------------------------------------ #
    # Historical
    # ------------------------------------------------------------------ #
    def historical(
        self, confidence: float = 0.95, horizon: int = 1
    ) -> RiskEstimate:
        """Empirical (historical-simulation) VaR and CVaR."""
        alpha = 1.0 - confidence
        var_quantile = float(np.quantile(self.returns, alpha))
        tail = self.returns[self.returns <= var_quantile]
        cvar = float(tail.mean()) if not tail.empty else var_quantile

        scale = np.sqrt(horizon)
        return RiskEstimate(
            var=abs(var_quantile) * scale,
            cvar=abs(cvar) * scale,
            confidence=confidence,
            method="historical",
            horizon=horizon,
        )

    # ------------------------------------------------------------------ #
    # Parametric Gaussian
    # ------------------------------------------------------------------ #
    def parametric(
        self, confidence: float = 0.95, horizon: int = 1
    ) -> RiskEstimate:
        """Variance-covariance (Gaussian) VaR and CVaR."""
        mu = float(self.returns.mean())
        sigma = float(self.returns.std(ddof=1))
        alpha = 1.0 - confidence
        z = norm.ppf(alpha)

        var = -(mu + sigma * z)
        # Closed-form Gaussian Expected Shortfall.
        cvar = -(mu - sigma * norm.pdf(z) / alpha)

        scale = np.sqrt(horizon)
        return RiskEstimate(
            var=abs(var) * scale,
            cvar=abs(cvar) * scale,
            confidence=confidence,
            method="parametric",
            horizon=horizon,
        )

    # ------------------------------------------------------------------ #
    # Monte Carlo
    # ------------------------------------------------------------------ #
    @staticmethod
    def monte_carlo(
        asset_returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1,
        n_sims: int = 50_000,
        seed: Optional[int] = 42,
    ) -> RiskEstimate:
        """Monte Carlo VaR/CVaR from correlated multivariate-normal draws.

        Parameters
        ----------
        asset_returns : pandas.DataFrame
            Periodic per-asset returns used to estimate the mean vector and
            covariance matrix.
        weights : numpy.ndarray
            Portfolio weights aligned to the columns of ``asset_returns``.
        n_sims : int, optional
            Number of simulated horizon paths, by default 50,000.
        seed : int, optional
            RNG seed for reproducibility.

        Notes
        -----
        Multi-period horizons are built by summing ``horizon`` i.i.d. one-period
        simulated vectors, i.e. assuming serial independence of returns.
        """
        rng = np.random.default_rng(seed)
        mu = asset_returns.mean().to_numpy()
        cov = asset_returns.cov().to_numpy()

        # Cholesky factor for correlated draws; jitter guards against a
        # numerically non-PSD sample covariance.
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = cov + np.eye(cov.shape[0]) * 1e-10
            chol = np.linalg.cholesky(cov)

        n_assets = len(weights)
        sim_port = np.zeros(n_sims)
        for _ in range(horizon):
            z = rng.standard_normal((n_sims, n_assets))
            draws = mu + z @ chol.T
            sim_port += draws @ weights

        alpha = 1.0 - confidence
        var_q = float(np.quantile(sim_port, alpha))
        tail = sim_port[sim_port <= var_q]
        cvar = float(tail.mean()) if tail.size else var_q

        logger.info(
            "Monte Carlo VaR(%.0f%%, h=%d): %.4f over %d sims.",
            confidence * 100,
            horizon,
            abs(var_q),
            n_sims,
        )
        return RiskEstimate(
            var=abs(var_q),
            cvar=abs(cvar),
            confidence=confidence,
            method="monte_carlo",
            horizon=horizon,
        )
