"""Vectorised backtesting engine.

The :class:`Backtester` orchestrates the full pipeline:

#. ask the :class:`~quantframework.strategies.strategy.Strategy` for target
   weights;
#. apply the rebalance schedule via the
   :class:`~quantframework.backtest.execution.ExecutionHandler`;
#. lag executed weights by one period to avoid look-ahead (we trade on the
   *next* bar after a signal);
#. compute gross strategy returns, subtract transaction costs, and compound
   into an equity curve.

The whole computation is vectorised over the price panel -- no per-row Python
loops -- so multi-year daily backtests run in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantframework.backtest.execution import ExecutionHandler
from quantframework.backtest.primitives import BacktestConfig
from quantframework.strategies.strategy import Strategy
from quantframework.utils.exceptions import BacktestError
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Outputs of a backtest run.

    Attributes
    ----------
    equity_curve : pandas.Series
        Portfolio equity over time, starting at ``initial_capital``.
    returns : pandas.Series
        Net periodic portfolio returns.
    gross_returns : pandas.Series
        Periodic returns before transaction costs.
    weights : pandas.DataFrame
        Executed (held) weights per period.
    costs : pandas.Series
        Per-period transaction costs as a fraction of equity.
    turnover : pandas.Series
        Per-period one-way turnover.
    config : BacktestConfig
        The configuration used for the run.
    """

    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    costs: pd.Series
    turnover: pd.Series
    config: BacktestConfig

    @property
    def total_return(self) -> float:
        """Cumulative net return over the full sample."""
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)


class Backtester:
    """Run a strategy over a price panel under realistic frictions.

    Parameters
    ----------
    config : BacktestConfig, optional
        Run configuration; a sensible default is used if omitted.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.execution = ExecutionHandler(self.config.cost_model)

    def run(self, strategy: Strategy, prices: pd.DataFrame) -> BacktestResult:
        """Execute ``strategy`` over ``prices`` and return performance.

        Parameters
        ----------
        strategy : Strategy
            The signal generator to evaluate.
        prices : pandas.DataFrame
            Adjusted close prices.

        Returns
        -------
        BacktestResult
            Equity curve, returns, weights, and cost diagnostics.

        Raises
        ------
        BacktestError
            If inputs are degenerate or produce no tradeable periods.
        """
        if prices.shape[0] < 2:
            raise BacktestError("Need at least two price observations.")

        logger.info(
            "Backtesting '%s' over %d periods x %d assets.",
            strategy.name,
            prices.shape[0],
            prices.shape[1],
        )

        target = strategy.generate_weights(prices)
        executed = self.execution.apply_rebalance_schedule(
            target, self.config.rebalance_freq
        )

        # Asset returns; align to executed-weight columns.
        asset_returns = prices.pct_change().fillna(0.0)
        executed = executed.reindex(columns=asset_returns.columns).fillna(0.0)

        # Lag weights by one period: a signal observed at t is traded at t+1.
        lagged = executed.shift(1).fillna(0.0)

        gross_returns = (lagged * asset_returns).sum(axis=1)
        costs = self.execution.transaction_costs(executed)
        turnover = self.execution.compute_turnover(executed)
        net_returns = gross_returns - costs

        equity = (1.0 + net_returns).cumprod() * self.config.initial_capital

        if equity.isna().all():
            raise BacktestError("Backtest produced an all-NaN equity curve.")

        logger.info(
            "Backtest complete: total return=%.2f%%, avg turnover=%.3f.",
            float(equity.iloc[-1] / equity.iloc[0] - 1.0) * 100,
            float(turnover.mean()),
        )

        return BacktestResult(
            equity_curve=equity,
            returns=net_returns,
            gross_returns=gross_returns,
            weights=executed,
            costs=costs,
            turnover=turnover,
            config=self.config,
        )
