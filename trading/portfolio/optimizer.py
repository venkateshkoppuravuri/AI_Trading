"""
trading/portfolio/optimizer.py
───────────────────────────────
Hierarchical Risk Parity (HRP) portfolio optimizer.

HRP allocates capital so that each position contributes EQUAL RISK,
not equal dollar amount. A volatile stock gets less capital; a stable
stock gets more. This prevents one big mover from dominating the P&L.

Algorithm (López de Prado 2016):
  1. Compute correlation matrix from daily returns
  2. Cluster stocks by similarity (Ward linkage on correlation distance)
  3. Recursive bisection — split clusters, weight by inverse variance
  4. Scale weights to sum to 1.0

Fallback: if fewer than 3 stocks or insufficient history, returns
equal-weight allocation automatically.

Usage::

    from trading.portfolio.optimizer import HRPOptimizer
    opt = HRPOptimizer()
    weights = opt.allocate(["AXON", "CF", "CRL", "APP", "BKR"])
    # {"AXON": 0.24, "CF": 0.18, "CRL": 0.21, "APP": 0.19, "BKR": 0.18}

    dollars = opt.allocate_dollars(tickers, total_budget=2000)
    # {"AXON": 480, "CF": 360, ...}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading.logger import get_logger

logger = get_logger(__name__)

_BARS_DIR   = Path("state/bars")
_LOOKBACK   = 60    # trading days of returns used for covariance
_MIN_STOCKS = 2     # minimum tickers to run HRP (else equal-weight)


class HRPOptimizer:
    """
    Hierarchical Risk Parity optimizer.
    Uses cached OHLCV parquet files from the historical data fetcher.
    Falls back to equal-weight if data is insufficient.
    """

    # ── Public ────────────────────────────────────────────────────────────────

    def allocate(self, tickers: list[str]) -> dict[str, float]:
        """
        Return HRP portfolio weights that sum to 1.0.

        Parameters
        ----------
        tickers : list of stock symbols

        Returns
        -------
        dict mapping ticker → weight (float, 0–1)
        """
        if len(tickers) < _MIN_STOCKS:
            return self._equal_weight(tickers)

        returns = self._load_returns(tickers)
        if returns.empty or returns.shape[1] < _MIN_STOCKS:
            logger.warning("HRP: insufficient return data — falling back to equal weight")
            return self._equal_weight(tickers)

        try:
            weights = self._hrp(returns)
            logger.info(
                "HRP weights: " +
                " | ".join(f"{t} {w:.1%}" for t, w in sorted(weights.items(), key=lambda x: -x[1]))
            )
            return weights
        except Exception as exc:
            logger.warning(f"HRP failed ({exc}) — equal weight fallback")
            return self._equal_weight(tickers)

    def allocate_dollars(
        self,
        tickers: list[str],
        total_budget: float,
        prices: dict[str, float] | None = None,
    ) -> dict[str, dict]:
        """
        Return per-ticker dollar allocation and share counts.

        Parameters
        ----------
        tickers      : list of stock symbols
        total_budget : total USD to allocate
        prices       : optional {ticker: price} — loaded from cache if None

        Returns
        -------
        dict mapping ticker → {"dollars": float, "shares": int, "weight": float}
        """
        weights = self.allocate(tickers)

        if prices is None:
            prices = self._load_latest_prices(tickers)

        result: dict[str, dict] = {}
        for ticker, weight in weights.items():
            dollars = total_budget * weight
            price   = prices.get(ticker, 0)
            shares  = int(dollars / price) if price > 0 else 0
            result[ticker] = {
                "dollars": round(dollars, 2),
                "shares":  shares,
                "weight":  round(weight, 4),
                "price":   round(price, 2),
            }
        return result

    # ── HRP algorithm ─────────────────────────────────────────────────────────

    def _hrp(self, returns: pd.DataFrame) -> dict[str, float]:
        """Core HRP computation."""
        # Step 1: correlation + covariance
        cov  = returns.cov()
        corr = returns.corr()

        # Step 2: hierarchical clustering on correlation distance
        dist     = np.sqrt((1.0 - corr) / 2.0)
        clusters = self._cluster(dist)

        # Step 3: recursive bisection weighting
        weights  = pd.Series(1.0, index=cov.index)
        items    = [clusters]

        while items:
            items = [
                sub
                for cluster in items
                for sub in self._bisect(cluster)
                if len(sub) > 0
            ]
            for cluster in items:
                if len(cluster) <= 1:
                    continue
                left, right = self._bisect(cluster)
                if not left or not right:
                    continue

                var_l = self._cluster_var(cov, left)
                var_r = self._cluster_var(cov, right)
                alloc_l = 1.0 - var_l / (var_l + var_r) if (var_l + var_r) > 0 else 0.5

                weights[left]  *= alloc_l
                weights[right] *= (1.0 - alloc_l)

        # Normalise
        weights = weights / weights.sum()
        return weights.to_dict()

    @staticmethod
    def _cluster(dist: pd.DataFrame) -> list[str]:
        """Ward linkage → ordered list of tickers (quasi-diagonal reordering)."""
        from scipy.cluster.hierarchy import linkage, leaves_list
        dist_arr = dist.values
        np.fill_diagonal(dist_arr, 0)
        # Upper triangle condensed distance
        n    = len(dist)
        cond = []
        for i in range(n):
            for j in range(i + 1, n):
                cond.append(dist_arr[i, j])
        cond   = np.array(cond)
        Z      = linkage(cond, method="ward")
        order  = leaves_list(Z)
        return list(dist.index[order])

    @staticmethod
    def _bisect(items: list[str]) -> tuple[list[str], list[str]]:
        """Split a list in half."""
        mid = len(items) // 2
        return items[:mid], items[mid:]

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
        """Variance of an inverse-variance weighted sub-portfolio."""
        sub  = cov.loc[items, items]
        ivp  = 1.0 / np.diag(sub.values)
        ivp /= ivp.sum()
        return float(ivp @ sub.values @ ivp)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_returns(self, tickers: list[str]) -> pd.DataFrame:
        """Load daily close-to-close returns from cached parquet files."""
        frames: dict[str, pd.Series] = {}
        for ticker in tickers:
            path = _BARS_DIR / f"{ticker}.parquet"
            if not path.exists():
                logger.debug(f"HRP: no cached bars for {ticker}")
                continue
            try:
                df  = pd.read_parquet(path)
                ret = df["close"].pct_change().dropna()
                if len(ret) >= _LOOKBACK:
                    frames[ticker] = ret.iloc[-_LOOKBACK:]
            except Exception as exc:
                logger.debug(f"HRP: could not load {ticker}: {exc}")

        if not frames:
            return pd.DataFrame()
        return pd.DataFrame(frames).dropna()

    @staticmethod
    def _load_latest_prices(tickers: list[str]) -> dict[str, float]:
        """Load latest close prices from cached parquet files."""
        prices: dict[str, float] = {}
        for ticker in tickers:
            path = _BARS_DIR / f"{ticker}.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    prices[ticker] = float(df["close"].iloc[-1])
                except Exception:
                    pass
        return prices

    @staticmethod
    def _equal_weight(tickers: list[str]) -> dict[str, float]:
        w = 1.0 / len(tickers) if tickers else 0.0
        return {t: w for t in tickers}
