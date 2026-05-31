"""Core value objects shared across the backtest engine.

These light dataclasses give the engine typed, self-documenting interfaces
between the strategy, portfolio, and execution layers without the overhead of a
full event queue. The framework uses a *vectorised* design (signals computed
over the whole panel at once) for speed, while keeping the component boundaries
that an event-driven system would have so logic stays testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    """Linear transaction-cost and slippage model.

    Costs are charged on traded notional (the absolute change in position
    weight times portfolio equity).

    Parameters
    ----------
    commission_bps : float
        Commission in basis points of traded notional (e.g. 1.0 = 1 bp).
    slippage_bps : float
        Slippage in basis points of traded notional, modelling adverse fill
        relative to the reference price.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 2.0

    @property
    def total_bps(self) -> float:
        """Combined cost in basis points."""
        return self.commission_bps + self.slippage_bps

    def cost(self, traded_notional: pd.Series | float) -> pd.Series | float:
        """Return the cash cost for a given traded notional."""
        return traded_notional * (self.total_bps / 1e4)


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration controlling a backtest run.

    Parameters
    ----------
    initial_capital : float
        Starting equity in account currency.
    rebalance_freq : str
        Pandas offset alias controlling rebalance cadence (e.g. ``"W-FRI"``,
        ``"ME"``). Signals are held constant between rebalances.
    periods_per_year : int
        Annualisation factor for the data frequency.
    cost_model : TransactionCostModel
        Transaction-cost and slippage assumptions.
    """

    initial_capital: float = 1_000_000.0
    rebalance_freq: str = "W-FRI"
    periods_per_year: int = 252
    cost_model: TransactionCostModel = TransactionCostModel()
