"""Custom exception hierarchy for the quant framework.

Centralising exceptions lets callers catch broad categories
(:class:`QuantFrameworkError`) or precise failure modes without coupling to
third-party error types that may change between library versions.
"""

from __future__ import annotations


class QuantFrameworkError(Exception):
    """Base class for every exception raised by this framework."""


class DataIngestionError(QuantFrameworkError):
    """Raised when market data cannot be downloaded, parsed, or cached."""


class InsufficientDataError(DataIngestionError):
    """Raised when a request returns fewer observations than required."""


class OptimizationError(QuantFrameworkError):
    """Raised when a portfolio optimisation routine fails to converge."""


class StrategyError(QuantFrameworkError):
    """Raised when a strategy is misconfigured or produces invalid signals."""


class BacktestError(QuantFrameworkError):
    """Raised for invalid backtest configuration or execution state."""


class ConfigurationError(QuantFrameworkError):
    """Raised when user-supplied configuration is invalid."""
