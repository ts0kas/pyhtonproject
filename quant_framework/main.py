"""End-to-end demonstration of the quantframework pipeline.

Runs the full research workflow on a sample technology universe:

#. ingest adjusted daily prices (live via yfinance, or a reproducible synthetic
   panel when ``--offline`` is set or the network is unavailable);
#. solve for the maximum-Sharpe portfolio and trace the efficient frontier;
#. estimate Value-at-Risk and Expected Shortfall (historical, parametric, and
   Monte Carlo) for the optimal portfolio;
#. backtest a momentum strategy with realistic costs against a benchmark;
#. compute tearsheet metrics and render the charts to ``./artifacts``.

Run with::

    python main.py                 # attempt live data, fall back to synthetic
    python main.py --offline       # force the synthetic generator
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantframework.analytics import MeanVarianceOptimizer, RiskAnalyzer
from quantframework.backtest import (
    Backtester,
    BacktestConfig,
    TransactionCostModel,
)
from quantframework.data import YFinanceDataHandler, synthetic_price_panel
from quantframework.metrics import PerformanceAnalyzer
from quantframework.strategies import MomentumStrategy
from quantframework.utils import DataIngestionError, configure_logging, get_logger
from quantframework.viz import plot_efficient_frontier, plot_tearsheet

logger = get_logger("main")

UNIVERSE = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META"]
BENCHMARK = "SPY"
START, END = "2019-01-01", "2024-01-01"
RISK_FREE = 0.02
PERIODS_PER_YEAR = 252


def load_prices(offline: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Load the universe and benchmark prices, falling back to synthetic data."""
    if not offline:
        try:
            handler = YFinanceDataHandler()
            prices = handler.get_prices(UNIVERSE, START, END)
            bench = handler.get_prices([BENCHMARK], START, END)[BENCHMARK]
            logger.info("Loaded live data for %d assets.", prices.shape[1])
            return prices, bench
        except (DataIngestionError, Exception) as exc:  # noqa: BLE001
            logger.warning("Live data unavailable (%s); using synthetic.", exc)

    prices = synthetic_price_panel(UNIVERSE, periods=1260, seed=7)
    bench = synthetic_price_panel([BENCHMARK], periods=1260, seed=99)[BENCHMARK]
    return prices, bench


def main(offline: bool = False) -> None:
    """Run the complete demonstration pipeline."""
    configure_logging()
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    # 1) Data ----------------------------------------------------------------
    prices, bench_prices = load_prices(offline)
    returns = prices.pct_change().dropna()
    bench_returns = bench_prices.pct_change().dropna()

    # 2) Optimisation --------------------------------------------------------
    optimizer = MeanVarianceOptimizer(
        returns, risk_free_rate=RISK_FREE, periods_per_year=PERIODS_PER_YEAR
    )
    max_sharpe = optimizer.max_sharpe()
    min_vol = optimizer.min_volatility()
    frontier = optimizer.efficient_frontier(n_points=60)

    logger.info("Max-Sharpe weights:")
    for asset, w in sorted(
        max_sharpe.weights.items(), key=lambda kv: -kv[1]
    ):
        logger.info("  %-6s %6.2f%%", asset, w * 100)
    logger.info(
        "Max-Sharpe: ret=%.2f%% vol=%.2f%% sharpe=%.2f",
        max_sharpe.expected_return * 100,
        max_sharpe.volatility * 100,
        max_sharpe.sharpe,
    )

    fig_frontier = plot_efficient_frontier(frontier, max_sharpe, min_vol)
    fig_frontier.savefig(artifacts / "efficient_frontier.png", dpi=150)

    # 3) Tail risk on the optimal portfolio ----------------------------------
    weight_vec = max_sharpe.weight_array(list(returns.columns))
    port_returns = returns @ weight_vec
    risk = RiskAnalyzer(port_returns)
    hist = risk.historical(confidence=0.95)
    param = risk.parametric(confidence=0.95)
    mc = RiskAnalyzer.monte_carlo(returns, weight_vec, confidence=0.95)

    logger.info("VaR/CVaR (95%%, 1-day) for the max-Sharpe portfolio:")
    for est in (hist, param, mc):
        logger.info(
            "  %-12s VaR=%.2f%%  CVaR=%.2f%%",
            est.method,
            est.var * 100,
            est.cvar * 100,
        )

    # 4) Backtest a momentum strategy ----------------------------------------
    config = BacktestConfig(
        initial_capital=1_000_000.0,
        rebalance_freq="W-FRI",
        periods_per_year=PERIODS_PER_YEAR,
        cost_model=TransactionCostModel(commission_bps=1.0, slippage_bps=2.0),
    )
    strategy = MomentumStrategy(lookback=126, skip=21, top_n=3)
    result = Backtester(config).run(strategy, prices)

    # 5) Performance + tearsheet --------------------------------------------
    perf = PerformanceAnalyzer(
        result.returns,
        periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE,
    )
    metrics = perf.compute(benchmark_returns=bench_returns)

    logger.info("Backtest tearsheet:")
    for name, value in metrics.to_dict().items():
        if value is None:
            continue
        logger.info("  %-18s %s", name, f"{value:.4f}")

    fig_tearsheet = plot_tearsheet(
        result.returns,
        result.equity_curve,
        benchmark=(1.0 + bench_returns).cumprod(),
        periods_per_year=PERIODS_PER_YEAR,
        risk_free_rate=RISK_FREE,
        save_path=artifacts / "tearsheet.png",
    )

    logger.info("Artifacts written to %s/", artifacts.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the synthetic price generator instead of live downloads.",
    )
    args = parser.parse_args()
    main(offline=args.offline)
