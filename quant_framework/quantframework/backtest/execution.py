"""Execution layer.

The :class:`ExecutionHandler` converts a strategy's continuous target-weight
panel into the weights actually held after applying the rebalance schedule and
no-trade-between-rebalances logic, and computes the per-period turnover used to
charge transaction costs. Keeping this separate from the portfolio accounting
means cost/slippage assumptions can be swapped without touching PnL logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantframework.backtest.primitives import TransactionCostModel
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExecutionHandler:
    """Translate target weights into executed weights and trading costs.

    Parameters
    ----------
    cost_model : TransactionCostModel
        Commission and slippage assumptions.
    """

    def __init__(self, cost_model: TransactionCostModel) -> None:
        self.cost_model = cost_model

    @staticmethod
    def apply_rebalance_schedule(
        target_weights: pd.DataFrame, rebalance_freq: str
    ) -> pd.DataFrame:
        """Hold weights constant between scheduled rebalance dates.

        The strategy may emit a fresh target every period, but trading every
        period is unrealistic and expensive. We sample the targets on the
        rebalance grid and forward-fill, so positions only change on rebalance
        dates.

        Parameters
        ----------
        target_weights : pandas.DataFrame
            Continuous target weights from the strategy.
        rebalance_freq : str
            Pandas offset alias (e.g. ``"W-FRI"``, ``"ME"``).

        Returns
        -------
        pandas.DataFrame
            Step-function weight panel aligned to the original index.
        """
        # Identify the last available observation in each rebalance bucket.
        rebal_dates = (
            target_weights.resample(rebalance_freq).last().index
        )
        valid = [d for d in rebal_dates if d in target_weights.index]
        held = target_weights.copy()
        # Keep only rows on rebalance dates, then forward-fill so positions are
        # held constant between rebalances. Using .loc assignment avoids the
        # shape-mismatch pitfalls of DataFrame.where with a 1-D mask.
        non_rebal = ~held.index.isin(valid)
        held.loc[non_rebal, :] = np.nan
        held = held.ffill().fillna(0.0)
        logger.info(
            "Applied '%s' rebalance schedule: %d rebalance dates.",
            rebalance_freq,
            len(valid),
        )
        return held

    def compute_turnover(self, executed_weights: pd.DataFrame) -> pd.Series:
        """Compute per-period one-way turnover (sum of absolute weight changes)."""
        delta = executed_weights.diff().abs().sum(axis=1)
        if len(delta) > 0:
            # First period: cost of establishing the initial book.
            delta.iloc[0] = executed_weights.iloc[0].abs().sum()
        return delta

    def transaction_costs(self, executed_weights: pd.DataFrame) -> pd.Series:
        """Per-period transaction cost as a fraction of equity."""
        turnover = self.compute_turnover(executed_weights)
        return self.cost_model.cost(turnover)
