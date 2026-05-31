"""Compare the bundled strategies on a common universe (offline).

Runs each strategy through the backtester under identical cost assumptions and
prints a side-by-side metrics table. Useful as a template for strategy research.
"""

from __future__ import annotations

import pandas as pd

from quantframework.backtest import Backtester, BacktestConfig
from quantframework.data import synthetic_price_panel
from quantframework.metrics import PerformanceAnalyzer
from quantframework.strategies import (
    EqualWeightStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
)
from quantframework.utils import configure_logging


def main() -> None:
    """Backtest every strategy and tabulate the headline metrics."""
    configure_logging()
    prices = synthetic_price_panel(
        ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"], periods=1260, seed=7
    )
    config = BacktestConfig(rebalance_freq="W-FRI")

    strategies = [
        EqualWeightStrategy(),
        MomentumStrategy(lookback=126, skip=21, top_n=2),
        MeanReversionStrategy(lookback=21, entry_z=1.0, allow_short=False),
    ]

    rows = {}
    for strat in strategies:
        result = Backtester(config).run(strat, prices)
        m = PerformanceAnalyzer(result.returns, risk_free_rate=0.02).compute()
        rows[strat.name] = {
            "Total Return": f"{m.total_return:.2%}",
            "CAGR": f"{m.cagr:.2%}",
            "Sharpe": f"{m.sharpe_ratio:.2f}",
            "Max DD": f"{m.max_drawdown:.2%}",
            "Win Rate": f"{m.win_rate:.2%}",
        }

    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
