"""
streamlit_app.py
─────────────────
Streamlit web dashboard for the AI Trading Bot.

Run:
    streamlit run streamlit_app.py

Tabs:
  Portfolio   — live Alpaca positions with unrealized P&L, budget bar
  Journal     — open trade detail (trail stop / peak) + closed trades table
  Performance — cumulative P&L chart, win rate, grade breakdown bar chart
  Macro       — VIX gauge, yield curve, fed rate, regime explainer
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_STATE_FILE = Path("state/ai_signal_state.json")


def _load_ai_params() -> dict:
    """Load ai_signal section from params.yaml (never cached — always current)."""
    try:
        import yaml
        data = yaml.safe_load(Path("params.yaml").read_text())
        return data.get("ai_signal", {})
    except Exception:
        return {}


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _account() -> dict:
    from trading.client import AlpacaClient
    return AlpacaClient().get_account()


@st.cache_data(ttl=30)
def _alpaca_positions() -> list[dict]:
    from trading.client import AlpacaClient
    return AlpacaClient().get_positions()


@st.cache_data(ttl=300)
def _macro() -> dict:
    from trading.signals.macro import MacroData
    return MacroData().get_all()


@st.cache_data(ttl=30)
def _open_journal() -> pd.DataFrame:
    from trading.journal import TradeJournal
    return TradeJournal().get_open_positions()


@st.cache_data(ttl=30)
def _closed_trades(days: int = 30) -> pd.DataFrame:
    from trading.journal import TradeJournal
    return TradeJournal().get_recent_trades(days=days)


@st.cache_data(ttl=30)
def _perf_stats(days: int = 30) -> dict:
    from trading.journal import TradeJournal
    return TradeJournal().get_performance_stats(days=days)


def _load_state() -> dict:
    """Read ai_signal state file (un-cached — always current)."""
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {"positions": {}, "last_scored": None}


# ── Format helpers ────────────────────────────────────────────────────────────

def _fmt_dollar(val: float) -> str:
    return f"+${val:,.2f}" if val >= 0 else f"-${abs(val):,.2f}"


def _fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _grade_badge(grade: str) -> str:
    return {"A": "🟢 A", "B": "🟩 B", "C": "🟡 C",
            "D": "🟠 D", "F": "🔴 F"}.get(grade, f"⚪ {grade}")


# ── App entry point ───────────────────────────────────────────────────────────

def main() -> None:
    state     = _load_state()
    ai_params = _load_ai_params()

    # ── Header row ────────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([9, 1])
    with col_title:
        st.title("📈 AI Trading Bot")
    with col_btn:
        if st.button("↺ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Account data ──────────────────────────────────────────────────────────
    try:
        acct        = _account()
        equity      = float(acct.get("equity", 0))
        last_equity = float(acct.get("last_equity", equity))
        cash        = float(acct.get("cash", 0))
        day_pnl     = equity - last_equity
        day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0
    except Exception:
        equity = last_equity = cash = day_pnl = day_pnl_pct = 0.0

    # ── Dynamic budget + max_positions (mirrors ai_signal.py logic) ──────────
    fixed_budget = float(ai_params.get("budget", 0))
    budget_pct   = float(ai_params.get("budget_pct", 0.90))
    min_slot     = float(ai_params.get("min_position_dollars", 2_000))
    pos_cap      = int(ai_params.get("max_positions_cap", 20))
    fixed_pos    = int(ai_params.get("max_positions", 0))

    ai_budget = fixed_budget if fixed_budget > 0 else round(equity * budget_pct, 2)
    if fixed_pos > 0:
        max_pos = fixed_pos
    else:
        max_pos = min(max(1, int(ai_budget / min_slot)), pos_cap)

    # ── Macro headline ────────────────────────────────────────────────────────
    try:
        macro_data   = _macro()
        regime       = macro_data.get("regime", "UNKNOWN")
        regime_emoji = macro_data.get("regime_color", "⚪")
        vix_val      = macro_data.get("vix")
    except Exception:
        macro_data   = {}
        regime       = "UNKNOWN"
        regime_emoji = "⚪"
        vix_val      = None

    positions = state.get("positions", {})

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Portfolio Value", f"${equity:,.2f}")
    k2.metric("Day P&L",         _fmt_dollar(day_pnl), _fmt_pct(day_pnl_pct))
    k3.metric("Cash",            f"${cash:,.2f}")
    k4.metric("Macro Regime",    f"{regime_emoji} {regime}",
              f"VIX {vix_val:.1f}" if vix_val else None)
    k5.metric("AI Positions",    f"{len(positions)} / {max_pos}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_pos, tab_journal, tab_perf, tab_macro, tab_risk = st.tabs([
        "📊 Positions", "📒 Journal", "📈 Performance", "🌍 Macro", "🛡️ Risk",
    ])

    with tab_pos:
        _tab_positions(positions, ai_budget, max_pos)

    with tab_journal:
        _tab_journal()

    with tab_perf:
        _tab_performance()

    with tab_macro:
        _tab_macro(macro_data)

    with tab_risk:
        _tab_risk(equity, positions, ai_budget, max_pos)

    # ── Footer ────────────────────────────────────────────────────────────────
    last_scored = state.get("last_scored") or "Never"
    if last_scored != "Never":
        try:
            last_scored = datetime.fromisoformat(last_scored).strftime("%d %b %Y %H:%M")
        except Exception:
            pass
    st.caption(f"Last model score: {last_scored}  ·  Click ↺ to refresh all data")


# ── Tab: Positions ────────────────────────────────────────────────────────────

def _tab_positions(positions: dict, ai_budget: float, max_pos: int) -> None:
    st.subheader(f"Open Positions  ({len(positions)} / {max_pos} slots used)")

    if not positions:
        st.info("No open positions. Next cycle runs at 09:30 ET on the next trading day.")
        return

    # Live Alpaca data keyed by symbol
    try:
        live_map = {p["symbol"]: p for p in _alpaca_positions()}
    except Exception:
        live_map = {}

    rows = []
    for ticker, pos in positions.items():
        live    = live_map.get(ticker, {})
        entry   = float(pos.get("entry_price", 0))
        shares  = int(pos.get("shares", 0))
        current = float(live.get("current_price", entry))
        unr_pl  = float(live.get("unrealized_pl",
                                  (current - entry) * shares))
        unr_pct = float(live.get("unrealized_plpc",
                                  (current - entry) / entry if entry else 0)) * 100

        try:
            days_held = (datetime.now() - datetime.fromisoformat(pos["entry_date"])).days
        except Exception:
            days_held = "?"

        rows.append({
            "Ticker":      ticker,
            "Shares":      shares,
            "Entry $":     round(entry, 2),
            "Current $":   round(current, 2),
            "Unr. P&L":    _fmt_dollar(unr_pl),
            "P&L %":       _fmt_pct(unr_pct),
            "Days Held":   days_held,
            "Conf":        pos.get("confidence", "?"),
            "Pred Return": f"+{pos.get('pred_return', 0)*100:.1f}%",
            "Budget Used": f"${pos.get('budget_used', 0):.0f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Budget bar
    budget_used = sum(p.get("budget_used", 0) for p in positions.values())
    pct = min(budget_used / ai_budget, 1.0) if ai_budget > 0 else 0.0
    st.progress(pct, text=(f"AI Signal budget: ${budget_used:,.0f} / "
                            f"${ai_budget:,.0f}  ({pct*100:.0f}% deployed)"))


# ── Tab: Journal ─────────────────────────────────────────────────────────────

def _tab_journal() -> None:
    st.subheader("Open Trades — Trail Stop Detail")
    open_df = _open_journal()

    if open_df.empty:
        st.info("No open trades in journal.")
    else:
        cols = [c for c in [
            "ticker", "shares", "entry_price", "entry_date",
            "confidence", "pred_return", "peak_price", "trail_floor", "thesis",
        ] if c in open_df.columns]
        st.dataframe(open_df[cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Closed Trades — Last 30 Days")
    closed_df = _closed_trades(days=30)

    if closed_df.empty:
        st.info("No closed trades yet. Exits appear here after positions are sold.")
        return

    df = closed_df.copy()
    df["Grade"] = df["outcome_grade"].map(_grade_badge)
    df["P&L $"] = df["pnl"].map(_fmt_dollar)
    df["P&L %"] = (df["pnl_pct"] * 100).map(_fmt_pct)

    show = [c for c in [
        "ticker", "shares", "entry_price", "exit_price",
        "P&L $", "P&L %", "holding_days", "exit_reason", "Grade", "thesis",
    ] if c in df.columns]
    st.dataframe(df[show], use_container_width=True, hide_index=True)


# ── Tab: Performance ─────────────────────────────────────────────────────────

def _tab_performance() -> None:
    stats = _perf_stats(days=30)

    if stats.get("total_trades", 0) == 0:
        st.info("No closed trades yet — performance stats appear after first exits.")
        return

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades (30d)",    stats["total_trades"])
    c2.metric("Win Rate",        f"{stats['win_rate']*100:.1f}%")
    c3.metric("Total P&L",       _fmt_dollar(stats["total_pnl"]))
    c4.metric("Avg P&L / Trade", _fmt_dollar(stats["avg_pnl"]))

    # ── Cumulative P&L chart ──────────────────────────────────────────────────
    closed_df = _closed_trades(days=30)
    if not closed_df.empty and "pnl" in closed_df.columns:
        st.subheader("Cumulative P&L")
        df_s = closed_df.sort_values("exit_date").copy()
        df_s["cum_pnl"] = df_s["pnl"].cumsum()
        final_pnl = df_s["cum_pnl"].iloc[-1]
        clr = "#00C851" if final_pnl >= 0 else "#FF4444"
        fill_clr = "rgba(0,200,81,0.12)" if final_pnl >= 0 else "rgba(255,68,68,0.12)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_s["exit_date"], y=df_s["cum_pnl"],
            mode="lines+markers",
            line=dict(color=clr, width=2),
            marker=dict(size=6, color=clr),
            fill="tozeroy", fillcolor=fill_clr,
            hovertemplate="%{x}<br>$%{y:+.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Cumulative P&L ($)",
            xaxis_title="Exit Date",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Grade breakdown bar ───────────────────────────────────────────────────
    grades = stats.get("grade_breakdown", {})
    active = [(g, grades[g]) for g in ["A", "B", "C", "D", "F"] if grades.get(g, 0) > 0]
    if active:
        st.subheader("Trade Grade Breakdown")
        clr_map = {"A": "#00C851", "B": "#7BC67E", "C": "#FFD700",
                   "D": "#FF8800", "F": "#FF4444"}
        fig2 = go.Figure(go.Bar(
            x=[_grade_badge(g) for g, _ in active],
            y=[n for _, n in active],
            marker_color=[clr_map[g] for g, _ in active],
            text=[n for _, n in active], textposition="outside",
        ))
        fig2.update_layout(
            height=250, margin=dict(l=0, r=0, t=20, b=0),
            yaxis_title="Trades",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Best / worst
    best  = stats.get("best_trade",  {})
    worst = stats.get("worst_trade", {})
    if best or worst:
        cb, cw = st.columns(2)
        if best:
            cb.metric("Best Trade",  best.get("ticker", "?"),  _fmt_dollar(best.get("pnl", 0)))
        if worst:
            cw.metric("Worst Trade", worst.get("ticker", "?"), _fmt_dollar(worst.get("pnl", 0)))


# ── Tab: Macro ────────────────────────────────────────────────────────────────

def _tab_macro(macro_data: dict) -> None:
    if not macro_data:
        st.error("Could not load macro data. Check FRED_API_KEY in .env.")
        return

    regime       = macro_data.get("regime", "UNKNOWN")
    regime_emoji = macro_data.get("regime_color", "⚪")
    vix          = macro_data.get("vix")
    yc           = macro_data.get("yield_curve")
    fed          = macro_data.get("fed_rate")
    hy           = macro_data.get("hy_spread")
    unemp        = macro_data.get("unemployment")
    fetched      = macro_data.get("fetched_at", "unknown")

    st.subheader(f"{regime_emoji} Market Regime: {regime}")
    st.caption(f"FRED data as of {fetched}  ·  Cached for 1 hour  ·  Click ↺ to force refresh")

    # ── Indicator cards ───────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("VIX",
              f"{vix:.1f}" if vix else "N/A",
              help=">30 = HIGH_FEAR — bot skips new entries")
    m2.metric("Yield Curve (10Y-2Y)",
              f"{yc:+.2f}%" if yc else "N/A",
              help="Negative = inverted (recession signal)")
    m3.metric("Fed Funds Rate",
              f"{fed:.2f}%" if fed else "N/A")
    m4.metric("HY Spread",
              f"{hy:.0f} bps" if hy else "N/A",
              help=">500 bps = credit stress → bear signal")
    m5.metric("Unemployment",
              f"{unemp:.1f}%" if unemp else "N/A")

    # ── Regime impact banner ──────────────────────────────────────────────────
    _info = {
        "BULL":      ("1.0×",  "Full position sizes — favorable conditions."),
        "NEUTRAL":   ("0.75×", "Slightly reduced sizes — watch for change."),
        "BEAR":      ("0.5×",  "Half size — only enters if pred return > 2%."),
        "HIGH_FEAR": ("0.25×", "Minimal exposure — bot skips most new entries."),
    }
    mult, desc = _info.get(regime, ("?", "Unknown regime."))
    st.info(f"**Position Size Multiplier: {mult}** — {desc}")

    st.divider()

    # ── VIX gauge + reference table ───────────────────────────────────────────
    col_g, col_t = st.columns([1, 1])

    with col_g:
        if vix:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = vix,
                title = {"text": "VIX — Fear Index"},
                delta = {"reference": 20, "valueformat": ".1f"},
                gauge = {
                    "axis": {"range": [0, 50], "tickwidth": 1},
                    "bar":  {"color": "navy", "thickness": 0.25},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0,  18], "color": "#C8E6C9"},  # 🟢 calm
                        {"range": [18, 25], "color": "#FFF9C4"},  # 🟡 caution
                        {"range": [25, 30], "color": "#FFE0B2"},  # 🟠 elevated
                        {"range": [30, 50], "color": "#FFCDD2"},  # 🔴 fear
                    ],
                    "threshold": {
                        "line":      {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value":     30,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("VIX data unavailable.")

    with col_t:
        st.markdown("**VIX Regime Thresholds**")
        st.markdown("""
| Range | Zone | Multiplier | Bot Behavior |
|-------|------|-----------|--------------|
| < 18  | 🟢 Calm     | 1.0× | Full size, BULL mode |
| 18–25 | 🟡 Caution  | 0.75× | Slightly reduced, NEUTRAL |
| 25–30 | 🟠 Elevated | 0.5× | Half size, BEAR mode |
| > 30  | 🔴 Fear     | 0.25× | Minimal exposure, HIGH_FEAR |
""")
        if yc is not None:
            st.markdown("**Yield Curve (10Y – 2Y)**")
            if yc > 0.5:
                msg = "🟢 Normal — growth signal"
            elif yc > -0.2:
                msg = "🟡 Flat — watch for inversion"
            else:
                msg = "🔴 Inverted — recession warning"
            st.markdown(f"`{yc:+.3f}%` — {msg}")

        if hy is not None:
            st.markdown("**HY Credit Spread**")
            hy_msg = ("🟢 Low" if hy < 300
                      else "🟡 Moderate" if hy < 500
                      else "🔴 Stressed (>500 bps)")
            st.markdown(f"`{hy:.0f} bps` — {hy_msg}")


# ── Tab: Risk ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _risk_status(equity: float) -> dict:
    from trading.risk.manager import RiskManager
    import yaml
    try:
        params = yaml.safe_load(Path("params.yaml").read_text()).get("risk", {})
    except Exception:
        params = {}
    rm = RiskManager(**{k: v for k, v in params.items()
                        if k in ("daily_loss_limit", "max_drawdown",
                                 "max_position_pct", "correlation_threshold")})
    return rm.get_status(equity)


def _tab_risk(equity: float, positions: dict, ai_budget: float = 0, max_pos: int = 0) -> None:
    st.subheader("Portfolio Risk Monitor")

    if equity <= 0:
        st.warning("Could not load account equity — risk metrics unavailable.")
        return

    try:
        rs = _risk_status(equity)
    except Exception as e:
        st.error(f"Could not load risk status: {e}")
        return

    status     = rs["status"]
    status_map = {"GREEN": ("🟢", "success"), "YELLOW": ("🟡", "warning"), "RED": ("🔴", "error")}
    emoji, msg_type = status_map.get(status, ("⚪", "info"))

    banner_fn = {"success": st.success, "warning": st.warning, "error": st.error}.get(msg_type, st.info)
    messages = []
    if rs["halt_active"]:
        messages.append(f"MAX DRAWDOWN HALT — drawdown {rs['drawdown_pct']:.1%} exceeds {rs['max_drawdown']:.0%} limit")
    if rs["daily_blocked"]:
        messages.append(f"DAILY LOSS LIMIT — down {rs['day_loss_pct']:.1%} today (limit {rs['daily_loss_limit']:.0%})")
    if not messages:
        messages.append("All risk checks passing — trading active")
    banner_fn(f"{emoji} **Risk Status: {status}** — {messages[0]}")

    st.divider()

    # ── Metrics row ───────────────────────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Drawdown from Peak",
              f"{rs['drawdown_pct']:.1%}",
              help=f"Halt triggers at {rs['max_drawdown']:.0%}")
    r2.metric("Peak Equity",
              f"${rs['peak_equity']:,.2f}")
    r3.metric("Today's Loss",
              f"{rs['day_loss_pct']:.1%}",
              help=f"Block triggers at {rs['daily_loss_limit']:.0%}")
    r4.metric("Day-Open Equity",
              f"${rs['day_open_equity']:,.2f}")

    st.divider()

    col_dd, col_dl = st.columns(2)

    # ── Drawdown gauge ────────────────────────────────────────────────────────
    with col_dd:
        st.markdown("**Drawdown from Peak**")
        limit = rs["max_drawdown"]
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = rs["drawdown_pct"] * 100,
            number = {"suffix": "%", "valueformat": ".1f"},
            title  = {"text": f"Max allowed: {limit:.0%}"},
            gauge  = {
                "axis": {"range": [0, limit * 100 * 1.5], "ticksuffix": "%"},
                "bar":  {"color": "#FF4444" if rs["halt_active"] else
                         "#FF8800" if rs["drawdown_pct"] > limit * 0.5 else "#00C851",
                         "thickness": 0.3},
                "steps": [
                    {"range": [0,              limit * 50],  "color": "#E8F5E9"},
                    {"range": [limit * 50,     limit * 100], "color": "#FFF3E0"},
                    {"range": [limit * 100,    limit * 150], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line":      {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value":     limit * 100,
                },
            },
        ))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── Daily loss gauge ──────────────────────────────────────────────────────
    with col_dl:
        st.markdown("**Today's Loss**")
        dlimit = rs["daily_loss_limit"]
        fig2 = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = rs["day_loss_pct"] * 100,
            number = {"suffix": "%", "valueformat": ".1f"},
            title  = {"text": f"Block at: {dlimit:.0%}"},
            gauge  = {
                "axis": {"range": [0, dlimit * 100 * 1.5], "ticksuffix": "%"},
                "bar":  {"color": "#FF4444" if rs["daily_blocked"] else
                         "#FF8800" if rs["day_loss_pct"] > dlimit * 0.5 else "#00C851",
                         "thickness": 0.3},
                "steps": [
                    {"range": [0,               dlimit * 50],  "color": "#E8F5E9"},
                    {"range": [dlimit * 50,      dlimit * 100], "color": "#FFF3E0"},
                    {"range": [dlimit * 100,     dlimit * 150], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line":      {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value":     dlimit * 100,
                },
            },
        ))
        fig2.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Position concentration pie ────────────────────────────────────────────
    if positions:
        st.subheader("Position Concentration")
        try:
            live_map = {p["symbol"]: p for p in _alpaca_positions()}
        except Exception:
            live_map = {}

        labels, values = [], []
        for ticker, pos in positions.items():
            mkt_val = float(live_map.get(ticker, {}).get("market_value",
                            pos.get("budget_used", 0)))
            labels.append(ticker)
            values.append(mkt_val)

        # Add undeployed cash portion
        total_deployed = sum(values)
        cash_portion   = max(equity - total_deployed, 0)
        if cash_portion > 0:
            labels.append("Cash / Other")
            values.append(cash_portion)

        cap_pct = rs["max_position_pct"] * 100
        fig3 = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.4,
            textinfo="label+percent",
            hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>",
        ))
        fig3.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            annotations=[{"text": f"Max {cap_pct:.0f}%\nper pick",
                           "x": 0.5, "y": 0.5, "showarrow": False,
                           "font": {"size": 12}}],
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Config table ──────────────────────────────────────────────────────────
    st.subheader("Risk Configuration")
    cfg_df = pd.DataFrame([
        {"Rule":        "Daily Loss Limit",     "Threshold": f"{rs['daily_loss_limit']:.0%}",
         "Status": "🔴 ACTIVE" if rs["daily_blocked"] else "🟢 OK"},
        {"Rule":        "Max Drawdown Halt",    "Threshold": f"{rs['max_drawdown']:.0%}",
         "Status": "🔴 HALTED" if rs["halt_active"]   else "🟢 OK"},
        {"Rule":        "Concentration Cap",    "Threshold": f"{rs['max_position_pct']:.0%} per position",
         "Status": "🟢 Applied automatically"},
        {"Rule":        "Correlation Block",    "Threshold": f"{rs['corr_threshold']:.0%}",
         "Status": "🟢 Checked on entry"},
    ])
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
