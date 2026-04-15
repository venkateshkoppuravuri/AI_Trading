"""
trading/backtest/result.py
───────────────────────────
BacktestResult — computes metrics and generates reports.

Metrics:
  total_return_pct      total % gain/loss over the backtest window
  annualized_return     CAGR (%)
  sharpe_ratio          annualised Sharpe (risk-free = 4.5 %)
  sortino_ratio         like Sharpe but only penalises downside volatility
  max_drawdown_pct      worst peak-to-trough drawdown (%)
  win_rate              % of trades that were profitable
  avg_holding_days      average trade duration
  profit_factor         gross profit / gross loss
  exit_breakdown        count per exit reason

Outputs (via save_report):
  state/backtest/YYYYMMDD_HHMMSS_metrics.json
  state/backtest/YYYYMMDD_HHMMSS_trades.csv
  state/backtest/YYYYMMDD_HHMMSS_equity.csv
  state/backtest/YYYYMMDD_HHMMSS_report.html   (interactive Plotly chart)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading.logger import get_logger

logger = get_logger(__name__)

_BACKTEST_DIR = Path("state/backtest")
_RISK_FREE    = 0.045          # 4.5 % annual risk-free rate
_TRADING_DAYS = 252


class BacktestResult:
    """
    Holds the output of one BacktestEngine.run() call and computes all metrics.
    """

    def __init__(
        self,
        equity_curve:    dict,
        trades:          list[dict],
        initial_capital: float,
        params:          dict,
    ) -> None:
        self.initial_capital = initial_capital
        self.params          = params

        # Equity curve as a sorted pandas Series
        self.equity = pd.Series(equity_curve).sort_index()

        # Trades DataFrame
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        if not self.trades.empty:
            for col in ("entry_date", "exit_date"):
                if col in self.trades.columns:
                    self.trades[col] = pd.to_datetime(self.trades[col])

    # ── Core metrics (properties) ─────────────────────────────────────────────

    @property
    def total_return_pct(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        return (self.equity.iloc[-1] / self.initial_capital - 1.0) * 100.0

    @property
    def annualized_return(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        days  = (self.equity.index[-1] - self.equity.index[0]).days
        years = days / 365.25
        if years <= 0:
            return 0.0
        return ((self.equity.iloc[-1] / self.initial_capital) ** (1.0 / years) - 1.0) * 100.0

    @property
    def sharpe_ratio(self) -> float:
        daily = self.equity.pct_change().dropna()
        if len(daily) < 2 or daily.std() == 0:
            return 0.0
        rf_daily = _RISK_FREE / _TRADING_DAYS
        excess   = daily - rf_daily
        return float((excess.mean() / excess.std()) * np.sqrt(_TRADING_DAYS))

    @property
    def sortino_ratio(self) -> float:
        daily = self.equity.pct_change().dropna()
        if len(daily) < 2:
            return 0.0
        rf_daily   = _RISK_FREE / _TRADING_DAYS
        excess     = daily - rf_daily
        downside   = excess[excess < 0].std()
        if downside == 0:
            return float("inf")
        return float((excess.mean() / downside) * np.sqrt(_TRADING_DAYS))

    @property
    def max_drawdown_pct(self) -> float:
        if len(self.equity) == 0:
            return 0.0
        rolling_max = self.equity.cummax()
        drawdown    = (self.equity - rolling_max) / rolling_max
        return float(drawdown.min() * 100.0)

    @property
    def win_rate(self) -> float:
        if self.trades.empty or "pnl" not in self.trades.columns:
            return 0.0
        return float((self.trades["pnl"] > 0).mean() * 100.0)

    @property
    def avg_holding_days(self) -> float:
        if self.trades.empty or "holding_days" not in self.trades.columns:
            return 0.0
        return float(self.trades["holding_days"].mean())

    @property
    def profit_factor(self) -> float:
        if self.trades.empty or "pnl" not in self.trades.columns:
            return 0.0
        gross_profit = self.trades[self.trades["pnl"] > 0]["pnl"].sum()
        gross_loss   = abs(self.trades[self.trades["pnl"] <= 0]["pnl"].sum())
        return round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")

    @property
    def exit_breakdown(self) -> dict[str, int]:
        if self.trades.empty or "exit_reason" not in self.trades.columns:
            return {}
        return self.trades["exit_reason"].value_counts().to_dict()

    @property
    def best_trade(self) -> dict:
        if self.trades.empty or "pnl" not in self.trades.columns:
            return {}
        row = self.trades.loc[self.trades["pnl"].idxmax()]
        return {"ticker": row.get("ticker", "?"), "pnl": round(float(row["pnl"]), 2),
                "pnl_pct": round(float(row.get("pnl_pct", 0)) * 100, 2)}

    @property
    def worst_trade(self) -> dict:
        if self.trades.empty or "pnl" not in self.trades.columns:
            return {}
        row = self.trades.loc[self.trades["pnl"].idxmin()]
        return {"ticker": row.get("ticker", "?"), "pnl": round(float(row["pnl"]), 2),
                "pnl_pct": round(float(row.get("pnl_pct", 0)) * 100, 2)}

    def metrics_dict(self) -> dict:
        """Return all metrics as a flat dict (JSON-serialisable)."""
        return {
            "initial_capital":   self.initial_capital,
            "final_equity":      round(float(self.equity.iloc[-1]), 2) if len(self.equity) else 0,
            "total_return_pct":  round(self.total_return_pct, 2),
            "annualized_return": round(self.annualized_return, 2),
            "sharpe_ratio":      round(self.sharpe_ratio, 3),
            "sortino_ratio":     round(self.sortino_ratio, 3),
            "max_drawdown_pct":  round(self.max_drawdown_pct, 2),
            "win_rate":          round(self.win_rate, 1),
            "profit_factor":     self.profit_factor,
            "total_trades":      len(self.trades),
            "avg_holding_days":  round(self.avg_holding_days, 1),
            "best_trade":        self.best_trade,
            "worst_trade":       self.worst_trade,
            "exit_breakdown":    self.exit_breakdown,
            "params":            self.params,
        }

    # ── Console output ────────────────────────────────────────────────────────

    def print_summary(self, benchmark_return: float | None = None) -> None:
        """Print a formatted summary to the console."""
        m = self.metrics_dict()

        _sep  = "=" * 58
        _line = lambda k, v: f"  {k:<26} {v}"

        sign  = lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%"
        dollar = lambda x: f"+${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"

        lines = [
            _sep,
            "  Backtest Results",
            f"  {m['params']['start_date']} → {m['params']['end_date']}",
            _sep,
            _line("Initial Capital",    f"${m['initial_capital']:,.2f}"),
            _line("Final Equity",       f"${m['final_equity']:,.2f}"),
            _line("Total Return",       sign(m["total_return_pct"])),
            _line("Annualised Return",  sign(m["annualized_return"])),
        ]

        if benchmark_return is not None:
            alpha = m["total_return_pct"] - benchmark_return
            lines.append(_line("SPY Return (same period)", sign(benchmark_return)))
            lines.append(_line("Alpha vs SPY",             sign(alpha)))

        lines += [
            _sep,
            _line("Sharpe Ratio",       f"{m['sharpe_ratio']:.2f}"),
            _line("Sortino Ratio",      f"{m['sortino_ratio']:.2f}"),
            _line("Max Drawdown",       f"{m['max_drawdown_pct']:.2f}%"),
            _line("Profit Factor",      str(m["profit_factor"])),
            _sep,
            _line("Total Trades",       str(m["total_trades"])),
            _line("Win Rate",           f"{m['win_rate']:.1f}%"),
            _line("Avg Holding Days",   f"{m['avg_holding_days']:.1f}"),
        ]

        bt = m["best_trade"]
        wt = m["worst_trade"]
        if bt:
            lines.append(_line("Best Trade",   f"{bt['ticker']}  {dollar(bt['pnl'])}  ({bt['pnl_pct']:+.1f}%)"))
        if wt:
            lines.append(_line("Worst Trade",  f"{wt['ticker']}  {dollar(wt['pnl'])}  ({wt['pnl_pct']:+.1f}%)"))

        if m["exit_breakdown"]:
            lines.append(_sep)
            lines.append("  Exit Breakdown:")
            for reason, count in sorted(m["exit_breakdown"].items(), key=lambda x: -x[1]):
                bar = "█" * count
                lines.append(f"    {reason:<20} {count:>3}  {bar}")

        lines.append(_sep)
        print("\n".join(lines))

    # ── Save reports ──────────────────────────────────────────────────────────

    def save_report(self, tag: str | None = None) -> Path:
        """
        Save equity CSV, trades CSV, metrics JSON, and interactive HTML chart.
        Returns the directory where files were saved.
        """
        _BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        pfx = f"{ts}_{tag}" if tag else ts

        # Equity CSV
        eq_path = _BACKTEST_DIR / f"{pfx}_equity.csv"
        self.equity.to_csv(eq_path, header=["portfolio_value"])

        # Trades CSV
        if not self.trades.empty:
            tr_path = _BACKTEST_DIR / f"{pfx}_trades.csv"
            self.trades.to_csv(tr_path, index=False)

        # Metrics JSON
        m_path = _BACKTEST_DIR / f"{pfx}_metrics.json"
        m_path.write_text(json.dumps(self.metrics_dict(), indent=2, default=str))

        # HTML chart
        try:
            html_path = _BACKTEST_DIR / f"{pfx}_report.html"
            self._save_html_chart(html_path)
            logger.info(f"Backtest report saved → {html_path}")
        except Exception as exc:
            logger.warning(f"Could not save HTML chart: {exc}")
            html_path = None

        logger.info(f"Backtest files saved to {_BACKTEST_DIR}/")
        return _BACKTEST_DIR

    def _save_html_chart(self, path: Path) -> None:
        """Generate a 3-panel Plotly chart: equity curve, drawdown, trade P&L."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Drawdown series
        rolling_max = self.equity.cummax()
        drawdown    = (self.equity - rolling_max) / rolling_max * 100

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.05,
            subplot_titles=("Portfolio Equity", "Drawdown %", "Trade P&L"),
        )

        final    = self.equity.iloc[-1]
        eq_color = "#00C851" if final >= self.initial_capital else "#FF4444"
        eq_fill  = "rgba(0,200,81,0.08)" if final >= self.initial_capital else "rgba(255,68,68,0.08)"

        # Panel 1: equity curve
        fig.add_trace(go.Scatter(
            x=list(self.equity.index), y=list(self.equity.values),
            mode="lines", name="Portfolio",
            line=dict(color=eq_color, width=2),
            fill="tozeroy", fillcolor=eq_fill,
            hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
        ), row=1, col=1)

        # Horizontal baseline at initial capital
        fig.add_hline(y=self.initial_capital, line_dash="dash",
                      line_color="gray", opacity=0.5, row=1, col=1)

        # Panel 2: drawdown
        fig.add_trace(go.Scatter(
            x=list(drawdown.index), y=list(drawdown.values),
            mode="lines", name="Drawdown",
            line=dict(color="#FF8800", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,136,0,0.12)",
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        ), row=2, col=1)

        # Panel 3: trade P&L scatter
        if not self.trades.empty and "pnl" in self.trades.columns:
            wins   = self.trades[self.trades["pnl"] > 0]
            losses = self.trades[self.trades["pnl"] <= 0]
            for subset, color, label in [
                (wins,   "#00C851", "Win"),
                (losses, "#FF4444", "Loss"),
            ]:
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset["exit_date"],
                        y=subset["pnl"],
                        mode="markers",
                        marker=dict(color=color, size=6, opacity=0.8),
                        name=label,
                        hovertemplate=(
                            "%{x}<br>"
                            + subset["ticker"].values[0] + "<br>"
                            if len(subset) == 1 else "%{x}<br>"
                        ) + "$%{y:+.2f}<extra></extra>",
                    ), row=3, col=1)

        fig.update_layout(
            title=dict(
                text=(
                    f"AI Signal Backtest — "
                    f"{self.params.get('start_date', '')} → {self.params.get('end_date', '')} | "
                    f"Return: {self.total_return_pct:+.1f}% | "
                    f"Sharpe: {self.sharpe_ratio:.2f} | "
                    f"Max DD: {self.max_drawdown_pct:.1f}%"
                ),
                font=dict(size=13),
            ),
            height=750,
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        fig.update_yaxes(tickprefix="$", row=1, col=1)
        fig.update_yaxes(ticksuffix="%", row=2, col=1)
        fig.update_yaxes(tickprefix="$", row=3, col=1)

        fig.write_html(str(path))


# ── SPY benchmark helper ──────────────────────────────────────────────────────

def compute_spy_return(start: object, end: object) -> float | None:
    """
    Return SPY total return % between start and end using cached bars.
    Returns None if SPY bars are not in the cache.
    """
    spy_path = Path("state/bars/SPY.parquet")
    if not spy_path.exists():
        return None
    try:
        df = pd.read_parquet(spy_path)
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index).date
        start_d = pd.to_datetime(start).date() if not isinstance(start, type(pd.Timestamp.now().date())) else start
        end_d   = pd.to_datetime(end).date()   if not isinstance(end,   type(pd.Timestamp.now().date())) else end
        df = df.loc[start_d:end_d]
        if len(df) < 2:
            return None
        return float((df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0)
    except Exception:
        return None
