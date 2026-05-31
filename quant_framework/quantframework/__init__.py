"""quantframework: an advanced portfolio optimisation & backtesting framework.

A modular, production-grade toolkit covering data ingestion, mean-variance
optimisation, tail-risk estimation, vectorised backtesting, and performance
analytics.
"""

__version__ = "0.1.0"

from quantframework.utils.logging_config import configure_logging, get_logger

__all__ = ["configure_logging", "get_logger", "__version__"]
