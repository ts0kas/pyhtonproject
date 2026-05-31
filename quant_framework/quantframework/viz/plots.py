"""Visualisation utilities.

Matplotlib-based plotting for the efficient frontier, cumulative returns, and
underwater (drawdown) charts, plus a composite tearsheet figure. Matplotlib is
chosen over Plotly for headless reproducibility (figures save to PNG without a
browser/kaleido dependency); the API returns the :class:`~matplotlib.figure.Figure`
so callers can display interactively or persist to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; callers may override before importing.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantframework.analytics.optimizer import PortfolioResult
from quantframework.metrics.performance import PerformanceAnalyzer
from quantframework.utils.logging_config import get_logger

logger = get_logger(__name__)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    max_sharpe: Optional[PortfolioResult] = None,
    min_vol: Optional[PortfolioResult] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot the efficient frontier coloured by Sharpe ratio.

    Parameters
    ----------
    frontier : pandas.DataFrame
        Output of
        :meth:`~quantframework.analytics.optimizer.MeanVarianceOptimizer.efficient_frontier`.
    max_sharpe, min_vol : PortfolioResult, optional
        Highlighted reference portfolios.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; a new figure is created if omitted.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    scatter = ax.scatter(
        frontier["volatility"],
        frontier["return"],
        c=frontier["sharpe"],
        cmap="viridis",
        s=18,
    )
    fig.colorbar(scatter, ax=ax, label="Sharpe ratio")

    if max_sharpe is not None:
        ax.scatter(
            max_sharpe.volatility,
            max_sharpe.expected_return,
            marker="*",
            color="crimson",
            s=320,
            edgecolor="black",
            label="Max Sharpe",
            zorder=5,
        )
    if min_vol is not None:
        ax.scatter(
            min_vol.volatility,
            min_vol.expected_return,
            marker="D",
            color="navy",
            s=110,
            edgecolor="black",
            label="Min Volatility",
            zorder=5,
        )

    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised expected return")
    ax.set_title("Efficient Frontier")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_cumulative_returns(
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a strategy equity curve, optionally against a benchmark."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    normed = equity_curve / equity_curve.iloc[0]
    ax.plot(normed.index, normed.values, label="Strategy", linewidth=1.6)

    if benchmark is not None:
        bench = benchmark.reindex(equity_curve.index).ffill()
        bench = bench / bench.iloc[0]
        ax.plot(
            bench.index,
            bench.values,
            label="Benchmark",
            linewidth=1.3,
            alpha=0.8,
            linestyle="--",
        )

    ax.set_ylabel("Growth of $1")
    ax.set_title("Cumulative Returns")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_underwater(
    returns: pd.Series, ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot the underwater (drawdown) curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    analyzer = PerformanceAnalyzer(returns)
    dd = analyzer.drawdown_series()
    ax.fill_between(dd.index, dd.values * 100, 0.0, color="crimson", alpha=0.4)
    ax.plot(dd.index, dd.values * 100, color="crimson", linewidth=1.0)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Underwater Plot")
    fig.tight_layout()
    return fig


def plot_tearsheet(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark: Optional[pd.Series] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Render a composite tearsheet: equity, drawdown, return distribution.

    Parameters
    ----------
    save_path : str or pathlib.Path, optional
        If provided, the figure is written to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
        The composite figure.
    """
    analyzer = PerformanceAnalyzer(
        returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )
    metrics = analyzer.compute(benchmark)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.4, 1.4])

    # Equity curve spanning the top row.
    ax_eq = fig.add_subplot(gs[0, :])
    plot_cumulative_returns(equity_curve, benchmark, ax=ax_eq)

    # Underwater plot.
    ax_dd = fig.add_subplot(gs[1, :])
    plot_underwater(returns, ax=ax_dd)

    # Return distribution.
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_hist.hist(returns.values * 100, bins=60, color="steelblue", alpha=0.8)
    ax_hist.axvline(0.0, color="black", linewidth=0.8)
    ax_hist.set_xlabel("Periodic return (%)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Return Distribution")

    # Metrics table.
    ax_tbl = fig.add_subplot(gs[2, 1])
    ax_tbl.axis("off")
    rows = [
        ("Total Return", f"{metrics.total_return:.2%}"),
        ("CAGR", f"{metrics.cagr:.2%}"),
        ("Volatility", f"{metrics.annual_volatility:.2%}"),
        ("Sharpe", f"{metrics.sharpe_ratio:.2f}"),
        ("Sortino", f"{metrics.sortino_ratio:.2f}"),
        ("Max Drawdown", f"{metrics.max_drawdown:.2%}"),
        ("Calmar", f"{metrics.calmar_ratio:.2f}"),
        ("Win Rate", f"{metrics.win_rate:.2%}"),
    ]
    if metrics.alpha is not None and metrics.beta is not None:
        rows.append(("Alpha (ann.)", f"{metrics.alpha:.2%}"))
        rows.append(("Beta", f"{metrics.beta:.2f}"))

    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax_tbl.set_title("Performance Summary", pad=20)

    fig.suptitle("Strategy Tearsheet", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Tearsheet saved to %s", save_path)

    return fig
