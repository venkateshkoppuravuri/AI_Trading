"""trading.strategies — all three strategy classes."""

from trading.strategies.base import BaseStrategy
from trading.strategies.trailing_stop import TrailingStopStrategy
from trading.strategies.copy_trading import CopyTradingStrategy
from trading.strategies.wheel import WheelStrategy

__all__ = [
    "BaseStrategy",
    "TrailingStopStrategy",
    "CopyTradingStrategy",
    "WheelStrategy",
]
