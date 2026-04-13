"""
trading/config.py
─────────────────
Centralised, validated configuration.

Usage anywhere in the package:
    from trading.config import get_settings
    settings = get_settings()
    print(settings.api_key)

The singleton is constructed once; subsequent calls return the cached instance.
All directories (logs/, state/) are created on first access.
"""

import os
import functools
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from trading.exceptions import ConfigurationError

# Resolve project root (two levels up from this file: trading/config.py → trading/ → root)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # ── Alpaca credentials ──────────────────────────────────────────────────
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets/v2"
    data_url: str = "https://data.alpaca.markets/v2"

    # ── Runtime directories ──────────────────────────────────────────────────
    log_dir: Path = field(default_factory=lambda: _ROOT / "logs")
    state_dir: Path = field(default_factory=lambda: _ROOT / "state")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_max_bytes: int = 10 * 1024 * 1024   # 10 MB per log file
    log_backup_count: int = 5

    # ── API client ────────────────────────────────────────────────────────────
    request_timeout: int = 15               # seconds per HTTP call
    retry_max_attempts: int = 3
    retry_backoff_base: float = 2.0         # wait = backoff_base ** attempt

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigurationError("ALPACA_API_KEY is missing or empty in .env")
        if not self.api_secret:
            raise ConfigurationError("ALPACA_API_SECRET is missing or empty in .env")

        # Create runtime directories (frozen dataclass — use object.__setattr__ bypass not needed;
        # mkdir has no side-effect on the immutable fields themselves)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            api_secret=os.getenv("ALPACA_API_SECRET", ""),
            base_url=os.getenv(
                "ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2"
            ),
        )


@functools.lru_cache(maxsize=None)
def get_settings() -> Settings:
    """Return the singleton Settings instance (constructed once, cached forever)."""
    return Settings.from_env()
