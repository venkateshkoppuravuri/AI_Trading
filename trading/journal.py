"""
trading/journal.py
──────────────────
SQLite trade journal — every entry and exit is recorded with full context.

Tables
  trades      : one row per completed round-trip (entry + exit)
  open_trades : one row per currently open position (updated on entry/exit)

Usage::

    from trading.journal import TradeJournal
    j = TradeJournal()

    trade_id = j.open_trade(
        strategy="AISignal",
        ticker="AXON",
        shares=1,
        entry_price=380.0,
        thesis="LightGBM HIGH confidence: atr_pct↑ high_52w_pct↑ pred=+3.9%",
        pred_return=0.039,
        confidence="HIGH",
    )

    j.close_trade(
        trade_id=trade_id,
        exit_price=395.0,
        exit_reason="PROFIT_TARGET",
    )

    df = j.get_recent_trades(days=30)
    stats = j.get_performance_stats()
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from trading.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = Path("state/trade_journal.db")


class TradeJournal:
    """Persistent SQLite trade journal. Thread-safe via per-call connections."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db = db_path
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy        TEXT    NOT NULL,
                    ticker          TEXT    NOT NULL,
                    shares          INTEGER NOT NULL,
                    entry_price     REAL    NOT NULL,
                    exit_price      REAL,
                    entry_date      TEXT    NOT NULL,
                    exit_date       TEXT,
                    thesis          TEXT,
                    pred_return     REAL,
                    confidence      TEXT,
                    exit_reason     TEXT,
                    pnl             REAL,
                    pnl_pct         REAL,
                    holding_days    INTEGER,
                    outcome_grade   TEXT,
                    status          TEXT    NOT NULL DEFAULT 'OPEN'
                );

                CREATE TABLE IF NOT EXISTS open_trades (
                    ticker          TEXT    PRIMARY KEY,
                    trade_id        INTEGER NOT NULL,
                    strategy        TEXT    NOT NULL,
                    shares          INTEGER NOT NULL,
                    entry_price     REAL    NOT NULL,
                    entry_date      TEXT    NOT NULL,
                    thesis          TEXT,
                    pred_return     REAL,
                    confidence      TEXT,
                    peak_price      REAL,
                    trail_floor     REAL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_ticker   ON trades(ticker);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                CREATE INDEX IF NOT EXISTS idx_trades_status   ON trades(status);
            """)

    # ── Write ─────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        strategy:    str,
        ticker:      str,
        shares:      int,
        entry_price: float,
        thesis:      str  = "",
        pred_return: float = 0.0,
        confidence:  str  = "",
    ) -> int:
        """
        Record a new entry. Returns the trade_id.
        Also upserts open_trades so peak/floor tracking can start.
        """
        now = datetime.now().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trades
                   (strategy, ticker, shares, entry_price, entry_date,
                    thesis, pred_return, confidence, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
                (strategy, ticker, shares, entry_price, now,
                 thesis, pred_return, confidence),
            )
            trade_id = cur.lastrowid

            conn.execute(
                """INSERT OR REPLACE INTO open_trades
                   (ticker, trade_id, strategy, shares, entry_price, entry_date,
                    thesis, pred_return, confidence, peak_price, trail_floor)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, trade_id, strategy, shares, entry_price, now,
                 thesis, pred_return, confidence, entry_price, 0.0),
            )

        logger.info(f"Journal: OPEN #{trade_id} {strategy} {shares}x {ticker} @ ${entry_price:.2f}")
        return trade_id

    def close_trade(
        self,
        ticker:      str,
        exit_price:  float,
        exit_reason: str,
    ) -> dict | None:
        """
        Close the open trade for *ticker*.
        Calculates P&L, grades outcome, removes from open_trades.
        Returns the completed trade dict, or None if no open trade found.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM open_trades WHERE ticker = ?", (ticker,)
            ).fetchone()
            if not row:
                logger.warning(f"Journal: no open trade found for {ticker}")
                return None

            trade_id    = row["trade_id"]
            entry_price = row["entry_price"]
            shares      = row["shares"]
            entry_date  = datetime.fromisoformat(row["entry_date"])
            pred_return = row["pred_return"] or 0.0

            exit_date    = datetime.now()
            pnl          = (exit_price - entry_price) * shares
            pnl_pct      = (exit_price - entry_price) / entry_price
            holding_days = (exit_date - entry_date).days
            grade        = _grade_outcome(pnl_pct, pred_return)

            conn.execute(
                """UPDATE trades SET
                   exit_price=?, exit_date=?, exit_reason=?,
                   pnl=?, pnl_pct=?, holding_days=?,
                   outcome_grade=?, status='CLOSED'
                   WHERE id=?""",
                (exit_price, exit_date.isoformat(), exit_reason,
                 round(pnl, 2), round(pnl_pct, 4), holding_days,
                 grade, trade_id),
            )
            conn.execute("DELETE FROM open_trades WHERE ticker = ?", (ticker,))

        result = {
            "trade_id":     trade_id,
            "ticker":       ticker,
            "shares":       shares,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "pnl":          round(pnl, 2),
            "pnl_pct":      round(pnl_pct, 4),
            "holding_days": holding_days,
            "exit_reason":  exit_reason,
            "grade":        grade,
        }
        logger.info(
            f"Journal: CLOSE #{trade_id} {ticker} @ ${exit_price:.2f} "
            f"| P&L ${pnl:+.2f} ({pnl_pct*100:+.1f}%) | {grade}"
        )
        return result

    def update_peak(self, ticker: str, current_price: float) -> float:
        """
        Update peak_price and trail_floor for an open position.
        trail_floor = peak × 0.97 (3% trailing stop from peak).
        Returns the current trail_floor.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT peak_price FROM open_trades WHERE ticker=?", (ticker,)
            ).fetchone()
            if not row:
                return 0.0
            peak = max(row["peak_price"], current_price)
            floor = round(peak * 0.97, 2)
            conn.execute(
                "UPDATE open_trades SET peak_price=?, trail_floor=? WHERE ticker=?",
                (peak, floor, ticker),
            )
        return floor

    def get_open_trade(self, ticker: str) -> dict | None:
        """Return the open trade row for *ticker*, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM open_trades WHERE ticker=?", (ticker,)
            ).fetchone()
            return dict(row) if row else None

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_recent_trades(self, days: int = 30) -> pd.DataFrame:
        """Return closed trades from the last *days* days as a DataFrame."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM trades WHERE status='CLOSED' AND exit_date >= ? ORDER BY exit_date DESC",
                conn, params=(since,),
            )

    def get_open_positions(self) -> pd.DataFrame:
        """Return all currently open positions as a DataFrame."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM open_trades ORDER BY entry_date DESC", conn
            )

    def get_performance_stats(self, days: int = 30) -> dict:
        """
        Return summary stats for closed trades in the last *days* days.

        Returns::
            {
                total_trades, win_rate, avg_pnl, total_pnl,
                avg_holding_days, best_trade, worst_trade,
                grade_breakdown: {A: n, B: n, C: n, D: n, F: n}
            }
        """
        df = self.get_recent_trades(days)
        if df.empty:
            return {"total_trades": 0, "message": "No closed trades yet"}

        wins = df[df["pnl"] > 0]
        grades = df["outcome_grade"].value_counts().to_dict()

        return {
            "total_trades":    len(df),
            "win_rate":        round(len(wins) / len(df), 3),
            "avg_pnl":         round(df["pnl"].mean(), 2),
            "total_pnl":       round(df["pnl"].sum(), 2),
            "avg_pnl_pct":     round(df["pnl_pct"].mean() * 100, 2),
            "avg_holding_days": round(df["holding_days"].mean(), 1),
            "best_trade":      df.loc[df["pnl"].idxmax(), ["ticker", "pnl"]].to_dict(),
            "worst_trade":     df.loc[df["pnl"].idxmin(), ["ticker", "pnl"]].to_dict(),
            "grade_breakdown": {g: int(grades.get(g, 0)) for g in ["A", "B", "C", "D", "F"]},
        }

    def format_weekly_report(self) -> str:
        """Return a plain-text weekly performance report for Telegram."""
        stats = self.get_performance_stats(days=7)
        if stats.get("total_trades", 0) == 0:
            return "No closed trades in the last 7 days."

        lines = [
            "📊 Weekly Performance Report",
            f"Trades closed : {stats['total_trades']}",
            f"Win rate      : {stats['win_rate']*100:.1f}%",
            f"Total P&L     : ${stats['total_pnl']:+.2f}",
            f"Avg P&L/trade : ${stats['avg_pnl']:+.2f} ({stats['avg_pnl_pct']:+.1f}%)",
            f"Avg hold time : {stats['avg_holding_days']:.1f} days",
            f"Best trade    : {stats['best_trade']['ticker']} ${stats['best_trade']['pnl']:+.2f}",
            f"Worst trade   : {stats['worst_trade']['ticker']} ${stats['worst_trade']['pnl']:+.2f}",
            "",
            "Grade breakdown:",
        ]
        for grade in ["A", "B", "C", "D", "F"]:
            n = stats["grade_breakdown"].get(grade, 0)
            if n:
                lines.append(f"  {grade}: {'*' * n} ({n})")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _grade_outcome(pnl_pct: float, pred_return: float) -> str:
    """
    Grade a closed trade A–F based on P&L vs model prediction.

    A  profit >= pred_return            (beat the forecast)
    B  profit >= pred_return * 0.5      (half the forecast, still green)
    C  profit > 0                       (green but undershot)
    D  small loss  (< -1%)              (stopped out early)
    F  large loss  (>= -3%)             (stop-loss hit)
    """
    if pnl_pct >= pred_return:
        return "A"
    if pnl_pct >= pred_return * 0.5 and pnl_pct > 0:
        return "B"
    if pnl_pct > 0:
        return "C"
    if pnl_pct > -0.03:
        return "D"
    return "F"
