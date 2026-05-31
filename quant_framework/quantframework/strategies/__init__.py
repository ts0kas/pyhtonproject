"""Pluggable trading strategies."""

from quantframework.strategies.strategy import (
    EqualWeightStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
)

__all__ = [
    "EqualWeightStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "Strategy",
]
