"""Unit tests for the core framework components.

Tests run fully offline using the synthetic price generator, so the suite is
deterministic and network-independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantframework.analytics import MeanVarianceOptimizer, RiskAnalyzer
from quantframework.backtest import (
    Backtester,
    BacktestConfig,
    TransactionCostModel,
)
from quantframework.data import DataHandler, synthetic_price_panel
from quantframework.metrics import PerformanceAnalyzer
from quantframework.strategies import (
    EqualWeightStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
)
from quantframework.utils.exceptions import (
    BacktestError,
    InsufficientDataError,
    OptimizationError,
    StrategyError,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return synthetic_price_panel(TICKERS, periods=750, seed=11)


@pytest.fixture(scope="module")
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
def test_synthetic_panel_shape(prices: pd.DataFrame) -> None:
    assert prices.shape == (750, 4)
    assert (prices > 0).all().all()
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_log_returns_are_time_additive(prices: pd.DataFrame) -> None:
    log_ret = DataHandler.compute_returns(prices, method="log")
    # Sum of log returns equals total log growth.
    recovered = np.exp(log_ret.sum()) * prices.iloc[0]
    np.testing.assert_allclose(recovered.values, prices.iloc[-1].values, rtol=1e-6)


def test_resample_reduces_rows(prices: pd.DataFrame) -> None:
    weekly = DataHandler.resample(prices, "W")
    assert len(weekly) < len(prices)


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one(returns: pd.DataFrame) -> None:
    opt = MeanVarianceOptimizer(returns)
    res = opt.max_sharpe()
    assert pytest.approx(sum(res.weights.values()), abs=1e-6) == 1.0


def test_min_vol_not_above_max_sharpe_vol(returns: pd.DataFrame) -> None:
    opt = MeanVarianceOptimizer(returns)
    assert opt.min_volatility().volatility <= opt.max_sharpe().volatility + 1e-9


def test_frontier_monotonic_volatility(returns: pd.DataFrame) -> None:
    opt = MeanVarianceOptimizer(returns)
    frontier = opt.efficient_frontier(n_points=25)
    assert frontier["volatility"].is_monotonic_increasing


def test_optimizer_rejects_single_asset() -> None:
    one = synthetic_price_panel(["X"], periods=100).pct_change().dropna()
    with pytest.raises(OptimizationError):
        MeanVarianceOptimizer(one)


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
def test_cvar_at_least_var(returns: pd.DataFrame) -> None:
    w = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    port = returns @ w
    est = RiskAnalyzer(port).historical(0.95)
    assert est.cvar >= est.var - 1e-9


def test_monte_carlo_var_positive(returns: pd.DataFrame) -> None:
    w = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    est = RiskAnalyzer.monte_carlo(returns, w, n_sims=5000, seed=1)
    assert est.var > 0 and est.cvar > 0


def test_empty_returns_raises() -> None:
    with pytest.raises(InsufficientDataError):
        RiskAnalyzer(pd.Series(dtype=float))


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def test_equal_weight_rows_sum_to_one(prices: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices)
    np.testing.assert_allclose(w.sum(axis=1).values, 1.0)


def test_momentum_gross_leq_one(prices: pd.DataFrame) -> None:
    w = MomentumStrategy(lookback=120, skip=20, top_n=2).generate_weights(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_mean_reversion_runs(prices: pd.DataFrame) -> None:
    w = MeanReversionStrategy(lookback=15, entry_z=1.0).generate_weights(prices)
    assert w.shape == prices.shape


def test_bad_momentum_config_raises() -> None:
    with pytest.raises(StrategyError):
        MomentumStrategy(lookback=10, skip=20)


# --------------------------------------------------------------------------- #
# Backtest + metrics
# --------------------------------------------------------------------------- #
def test_backtest_produces_equity_curve(prices: pd.DataFrame) -> None:
    config = BacktestConfig(rebalance_freq="W-FRI")
    result = Backtester(config).run(EqualWeightStrategy(), prices)
    assert len(result.equity_curve) == len(prices)
    assert result.equity_curve.iloc[0] > 0


def test_costs_reduce_returns(prices: pd.DataFrame) -> None:
    no_cost = BacktestConfig(
        cost_model=TransactionCostModel(0.0, 0.0), rebalance_freq="W-FRI"
    )
    high_cost = BacktestConfig(
        cost_model=TransactionCostModel(50.0, 50.0), rebalance_freq="W-FRI"
    )
    strat = MomentumStrategy(lookback=100, skip=20, top_n=2)
    cheap = Backtester(no_cost).run(strat, prices).total_return
    pricey = Backtester(high_cost).run(strat, prices).total_return
    assert pricey <= cheap


def test_backtest_rejects_short_series() -> None:
    tiny = synthetic_price_panel(TICKERS, periods=1)
    with pytest.raises(BacktestError):
        Backtester().run(EqualWeightStrategy(), tiny)


def test_metrics_internally_consistent(prices: pd.DataFrame) -> None:
    result = Backtester().run(EqualWeightStrategy(), prices)
    m = PerformanceAnalyzer(result.returns).compute()
    assert -1.0 <= m.max_drawdown <= 0.0
    assert 0.0 <= m.win_rate <= 1.0
    assert m.annual_volatility >= 0.0


def test_alpha_beta_against_benchmark(prices: pd.DataFrame) -> None:
    result = Backtester().run(EqualWeightStrategy(), prices)
    bench = synthetic_price_panel(["BENCH"], periods=len(prices), seed=3)[
        "BENCH"
    ].pct_change().dropna()
    bench.index = result.returns.index[: len(bench)]
    alpha, beta = PerformanceAnalyzer(result.returns).alpha_beta(bench)
    assert alpha is not None and beta is not None
