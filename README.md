# quantframework

An advanced **portfolio optimisation & algorithmic backtesting framework**, built
to production standards: modular OOP architecture, strict separation of
concerns, full type hinting, NumPy-style docstrings, structured logging, custom
exceptions, and a vectorised backtest engine.

## Features

- **Data ingestion** — vendor-agnostic `DataHandler` interface with a `yfinance`
  implementation, Parquet disk caching, missing-data handling, split/dividend
  adjustment (total-return prices), and multi-timeframe resampling. A
  reproducible synthetic GBM generator lets everything run offline.
- **Quantitative analysis** — Modern Portfolio Theory (efficient frontier,
  max-Sharpe tangency, global min-variance) via SLSQP, plus Value-at-Risk and
  Expected Shortfall (CVaR) using historical, parametric-Gaussian, and
  Monte Carlo (Cholesky-correlated) methods.
- **Backtesting** — vectorised engine with pluggable strategies, rebalance
  scheduling, one-period signal lag (no look-ahead), and a linear
  commission + slippage cost model.
- **Strategies** — equal-weight benchmark, cross-sectional momentum, and
  z-score mean reversion, all behind a common `Strategy` ABC.
- **Performance metrics** — Sharpe, Sortino, Calmar, max drawdown, win rate,
  profit factor, skew/kurtosis, and CAPM alpha/beta vs a benchmark.
- **Visualisation** — efficient frontier, cumulative returns, underwater plots,
  and a composite tearsheet (Matplotlib, headless-safe).

## Architecture

```
DataHandler ──▶ MeanVarianceOptimizer ──▶ RiskAnalyzer
     │
     └──▶ Strategy ──▶ ExecutionHandler ──▶ Backtester ──▶ PerformanceAnalyzer ──▶ plots
```

Each layer depends only on the abstraction below it, so any component (data
vendor, strategy, cost model) can be swapped without touching the others.

## Installation

```bash
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + test/lint/type tooling
pip install -e .                       # editable install
```

## Quick start

```bash
python main.py             # live data via yfinance, falls back to synthetic
python main.py --offline   # force the deterministic synthetic generator
```

Outputs (`efficient_frontier.png`, `tearsheet.png`) are written to `./artifacts`.

```python
from quantframework.data import YFinanceDataHandler
from quantframework.analytics import MeanVarianceOptimizer
from quantframework.backtest import Backtester, BacktestConfig
from quantframework.strategies import MomentumStrategy

prices = YFinanceDataHandler().get_prices(
    ["AAPL", "MSFT", "GOOGL", "NVDA"], "2019-01-01", "2024-01-01"
)
opt = MeanVarianceOptimizer(prices.pct_change().dropna(), risk_free_rate=0.02)
print(opt.max_sharpe().weights)

result = Backtester(BacktestConfig()).run(MomentumStrategy(top_n=2), prices)
print(result.total_return)
```

## Testing & quality

```bash
pytest                 # 19 unit tests, fully offline
ruff check .           # lint (PEP 8, import order, docstrings)
mypy quantframework    # static type checking
```

## Project layout

```
quantframework/
├── data/        # ingestion, caching, cleaning
├── analytics/   # MPT optimiser, VaR/CVaR
├── strategies/  # Strategy ABC + concrete strategies
├── backtest/    # execution model + vectorised engine
├── metrics/     # tearsheet performance analytics
├── viz/         # matplotlib plots
└── utils/       # logging, exceptions, type aliases
```

## Notes

- VaR/CVaR are reported as positive loss magnitudes.
- Returns default to log (time-additive) for analytics; the backtester uses
  simple returns for PnL compounding.
- The synthetic generator is for offline development/CI only; use real data for
  any research conclusions.
