"""
trading/strategies/base.py
───────────────────────────
Abstract base class every strategy must implement.

Enforces a consistent interface so main.py can drive all three strategies
identically — schedule them, call run(), and retrieve status() uniformly.
"""

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Contract for all trading strategies.

    Concrete implementations must define:
      name     — human-readable identifier used in logs and the scheduler
      run()    — one full execution cycle (enter, monitor, exit, etc.)
      status() — serialisable snapshot of current state for logging / display
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name, e.g. 'TrailingStop[TSLA]'."""

    @abstractmethod
    def run(self) -> None:
        """
        Execute one complete cycle of the strategy.

        Implementations should be idempotent — calling run() twice in a row
        should not create duplicate orders or corrupt state.
        All exceptions raised here are caught by the scheduler in main.py
        and logged without crashing the scheduler loop.
        """

    @abstractmethod
    def status(self) -> dict:
        """
        Return a JSON-serialisable dict describing current strategy state.

        At minimum include:
          {'strategy': self.name, 'status': <str>, ...strategy-specific fields}
        """
