"""Modern Portfolio Theory optimisation.

Implements mean-variance optimisation to recover the efficient frontier, the
maximum-Sharpe (tangency) portfolio, and the global minimum-variance portfolio.
Uses :func:`scipy.optimize.minimize` (SLSQP) with linear constraints, with a
closed-form fallback for the unconstrained minimum-variance case.

Covariances are annualised; the caller supplies the periods-per-year factor so
the same code path works for daily, weekly, or monthly returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantframework.utils.exceptions import OptimizationError
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PortfolioResult:
    """Immutable container describing an optimised portfolio.

    Attributes
    ----------
    weights : dict of str to float
        Asset weights summing to one.
    expected_return : float
        Annualised expected return.
    volatility : float
        Annualised standard deviation.
    sharpe : float
        Annualised Sharpe ratio at the configured risk-free rate.
    """

    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float

    def weight_array(self, order: List[str]) -> np.ndarray:
        """Return weights as an array aligned to ``order``."""
        return np.array([self.weights[a] for a in order], dtype=float)


@dataclass
class MeanVarianceOptimizer:
    """Mean-variance optimiser over a panel of asset returns.

    Parameters
    ----------
    returns : pandas.DataFrame
        Periodic (not annualised) returns, one column per asset.
    risk_free_rate : float, optional
        Annualised risk-free rate used in Sharpe calculations, by default 0.0.
    periods_per_year : int, optional
        Annualisation factor (252 for daily, 52 weekly, 12 monthly), by
        default 252.
    allow_short : bool, optional
        If ``False`` (default) weights are bounded to ``[0, 1]``; if ``True``
        the bound is ``[-1, 1]``.
    """

    returns: pd.DataFrame
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    allow_short: bool = False

    assets: List[str] = field(init=False)
    mean_returns: np.ndarray = field(init=False, repr=False)
    cov_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.returns.empty or self.returns.shape[1] < 2:
            raise OptimizationError(
                "Need at least two assets with data to optimise."
            )
        self.assets = list(self.returns.columns)
        # Annualise moments once; reuse across every optimisation call.
        self.mean_returns = (
            self.returns.mean().to_numpy() * self.periods_per_year
        )
        self.cov_matrix = (
            self.returns.cov().to_numpy() * self.periods_per_year
        )

    # ------------------------------------------------------------------ #
    # Core portfolio statistics
    # ------------------------------------------------------------------ #
    def _portfolio_stats(
        self, weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Return (annual return, annual vol, Sharpe) for given weights."""
        exp_ret = float(weights @ self.mean_returns)
        variance = float(weights @ self.cov_matrix @ weights)
        vol = float(np.sqrt(max(variance, 0.0)))
        sharpe = (
            (exp_ret - self.risk_free_rate) / vol if vol > 1e-12 else 0.0
        )
        return exp_ret, vol, sharpe

    def _bounds(self) -> Tuple[Tuple[float, float], ...]:
        lo = -1.0 if self.allow_short else 0.0
        return tuple((lo, 1.0) for _ in self.assets)

    def _result_from_weights(self, weights: np.ndarray) -> PortfolioResult:
        exp_ret, vol, sharpe = self._portfolio_stats(weights)
        return PortfolioResult(
            weights=dict(zip(self.assets, weights.tolist())),
            expected_return=exp_ret,
            volatility=vol,
            sharpe=sharpe,
        )

    # ------------------------------------------------------------------ #
    # Optimisation routines
    # ------------------------------------------------------------------ #
    def max_sharpe(self) -> PortfolioResult:
        """Find the maximum-Sharpe (tangency) portfolio."""

        def neg_sharpe(w: np.ndarray) -> float:
            return -self._portfolio_stats(w)[2]

        result = self._solve(objective=neg_sharpe)
        logger.info("Max-Sharpe portfolio Sharpe=%.3f", -result.fun)
        return self._result_from_weights(result.x)

    def min_volatility(self) -> PortfolioResult:
        """Find the global minimum-variance portfolio."""

        def variance(w: np.ndarray) -> float:
            return float(w @ self.cov_matrix @ w)

        result = self._solve(objective=variance)
        return self._result_from_weights(result.x)

    def efficient_return(self, target_return: float) -> PortfolioResult:
        """Minimise variance subject to a target annual return."""

        def variance(w: np.ndarray) -> float:
            return float(w @ self.cov_matrix @ w)

        extra = {
            "type": "eq",
            "fun": lambda w: float(w @ self.mean_returns) - target_return,
        }
        result = self._solve(objective=variance, extra_constraints=[extra])
        return self._result_from_weights(result.x)

    def efficient_frontier(
        self, n_points: int = 50
    ) -> pd.DataFrame:
        """Trace the efficient frontier between min-vol and max-return assets.

        Parameters
        ----------
        n_points : int, optional
            Number of target-return levels to solve for, by default 50.

        Returns
        -------
        pandas.DataFrame
            Columns ``return``, ``volatility``, ``sharpe`` plus one column per
            asset weight, sorted by volatility.
        """
        min_vol = self.min_volatility()
        lo = min_vol.expected_return
        hi = float(self.mean_returns.max())
        targets = np.linspace(lo, hi, n_points)

        rows: List[Dict[str, float]] = []
        for target in targets:
            try:
                res = self.efficient_return(target)
            except OptimizationError:
                continue
            row: Dict[str, float] = {
                "return": res.expected_return,
                "volatility": res.volatility,
                "sharpe": res.sharpe,
            }
            row.update(res.weights)
            rows.append(row)

        if not rows:
            raise OptimizationError("Efficient frontier produced no points.")
        return pd.DataFrame(rows).sort_values("volatility").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Solver plumbing
    # ------------------------------------------------------------------ #
    def _solve(
        self,
        objective,
        extra_constraints: Optional[List[dict]] = None,
    ):
        """Run SLSQP with the budget constraint and optional extras."""
        n = len(self.assets)
        x0 = np.repeat(1.0 / n, n)
        constraints = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        ]
        if extra_constraints:
            constraints.extend(extra_constraints)

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=self._bounds(),
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if not result.success:
            raise OptimizationError(
                f"Optimisation failed to converge: {result.message}"
            )
        return result
