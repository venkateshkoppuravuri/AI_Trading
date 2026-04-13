"""
dashboard.py — Live CLI dashboard for the AI Trading Bot.

Usage:
  python dashboard.py              # auto-refresh every 30 seconds
  python dashboard.py --interval 10
  python dashboard.py --once       # print snapshot and exit
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trading.client import AlpacaClient
from trading.config import get_settings

console = Console()


# ── Individual panels ─────────────────────────────────────────────────────────

def _header() -> Panel:
    now   = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    label = Text.assemble(
        ("AI Trading Bot", "bold green"),
        ("  |  ", "dim"),
        ("Paper Trading", "yellow"),
        ("  |  ", "dim"),
        (now, "cyan"),
    )
    label.justify = "center"
    return Panel(label, border_style="green", padding=(0, 1))


def _account_panel(client: AlpacaClient) -> Panel:
    try:
        acc       = client.get_account()
        portfolio = float(acc.get("portfolio_value", 0))
        equity    = float(acc.get("equity", 0))
        last_eq   = float(acc.get("last_equity", equity))
        buying_pw = float(acc.get("buying_power", 0))
        cash      = float(acc.get("cash", 0))
        day_pl    = equity - last_eq
        day_pl_pc = (day_pl / last_eq * 100) if last_eq else 0.0

        color = "green" if day_pl >= 0 else "red"
        sign  = "+" if day_pl >= 0 else ""

        t = Table.grid(padding=(0, 4))
        t.add_column(style="bold cyan",  min_width=18)
        t.add_column(min_width=18)
        t.add_column(style="bold cyan",  min_width=18)
        t.add_column(min_width=22)
        t.add_row(
            "Portfolio Value", f"${portfolio:>13,.2f}",
            "Day P/L",
            f"[{color}]{sign}${day_pl:>10,.2f}  ({sign}{day_pl_pc:.2f}%)[/{color}]",
        )
        t.add_row(
            "Buying Power",    f"${buying_pw:>13,.2f}",
            "Cash",            f"${cash:>13,.2f}",
        )
        return Panel(t, title="[bold white]Account", border_style="cyan")

    except Exception as exc:
        return Panel(f"[red]Error fetching account: {exc}", title="Account", border_style="red")


def _positions_panel(client: AlpacaClient) -> Panel:
    try:
        positions = client.get_positions()
    except Exception as exc:
        return Panel(f"[red]Error: {exc}", title="Positions", border_style="red")

    if not positions:
        return Panel(
            Text("No open positions", style="dim", justify="center"),
            title="[bold white]Positions (0)",
            border_style="blue",
        )

    t = Table(box=box.SIMPLE_HEAD, header_style="bold blue", show_edge=False)
    t.add_column("Symbol",    style="bold white", min_width=7)
    t.add_column("Qty",       justify="right", min_width=6)
    t.add_column("Avg Cost",  justify="right", min_width=10)
    t.add_column("Current",   justify="right", min_width=10)
    t.add_column("Mkt Value", justify="right", min_width=13)
    t.add_column("Unrealized P/L", justify="right", min_width=16)
    t.add_column("P/L %",     justify="right", min_width=9)

    total_unreal = 0.0
    for pos in sorted(positions, key=lambda p: p.get("symbol", "")):
        qty       = float(pos.get("qty", 0))
        avg_cost  = float(pos.get("avg_entry_price", 0))
        current   = float(pos.get("current_price", 0))
        mkt_val   = float(pos.get("market_value", 0))
        unreal_pl = float(pos.get("unrealized_pl", 0))
        unreal_pc = float(pos.get("unrealized_plpc", 0)) * 100
        total_unreal += unreal_pl

        color = "green" if unreal_pl >= 0 else "red"
        sign  = "+" if unreal_pl >= 0 else ""
        t.add_row(
            pos.get("symbol", "?"),
            f"{qty:.0f}",
            f"${avg_cost:.2f}",
            f"${current:.2f}",
            f"${mkt_val:,.2f}",
            f"[{color}]{sign}${unreal_pl:,.2f}[/{color}]",
            f"[{color}]{sign}{unreal_pc:.2f}%[/{color}]",
        )

    # Totals footer
    t.add_section()
    total_color = "green" if total_unreal >= 0 else "red"
    total_sign  = "+" if total_unreal >= 0 else ""
    t.add_row(
        "[bold]TOTAL", "", "", "", "",
        f"[bold {total_color}]{total_sign}${total_unreal:,.2f}[/bold {total_color}]",
        "",
    )

    return Panel(t, title=f"[bold white]Positions ({len(positions)})", border_style="blue")


def _orders_panel(client: AlpacaClient) -> Panel:
    try:
        orders = client.get_open_orders()
    except Exception as exc:
        return Panel(f"[red]Error: {exc}", title="Open Orders", border_style="red")

    if not orders:
        return Panel(
            Text("No open orders", style="dim", justify="center"),
            title="[bold white]Open Orders (0)",
            border_style="yellow",
        )

    t = Table(box=box.SIMPLE_HEAD, header_style="bold yellow", show_edge=False)
    t.add_column("Symbol",  style="bold", min_width=7)
    t.add_column("Side",    min_width=6)
    t.add_column("Type",    min_width=8)
    t.add_column("Qty",     justify="right", min_width=5)
    t.add_column("Status",  min_width=12)
    t.add_column("TIF",     min_width=5)

    for o in orders:
        side  = o.get("side", "?")
        color = "green" if side == "buy" else "red"
        t.add_row(
            o.get("symbol", "?"),
            f"[{color}]{side.upper()}[/{color}]",
            o.get("type", "?"),
            o.get("qty", "?"),
            o.get("status", "?"),
            o.get("time_in_force", "?"),
        )

    return Panel(t, title=f"[bold white]Open Orders ({len(orders)})", border_style="yellow")


def _strategy_panels() -> list[Panel]:
    """Read state JSON files and render one status panel per active strategy."""
    state_dir = get_settings().state_dir
    panels: list[Panel] = []

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    for f in sorted(state_dir.glob("trailing_stop_*.json")):
        sym = f.stem.replace("trailing_stop_", "")
        try:
            s      = json.loads(f.read_text())
            status = s.get("status", "?")
            sc     = {"IN_TRADE": "green", "STOPPED": "red", "IDLE": "dim"}.get(status, "white")

            g = Table.grid(padding=(0, 2))
            g.add_column(style="cyan",  min_width=14)
            g.add_column(min_width=18)
            g.add_row("Status",   f"[bold {sc}]{status}[/bold {sc}]")
            if status == "IN_TRADE":
                entry     = s.get("entry_price") or 0
                highest   = s.get("highest_price") or 0
                floor_p   = s.get("floor_price") or 0
                gain_pct  = ((highest - entry) / entry * 100) if entry else 0
                color_g   = "green" if gain_pct >= 0 else "red"
                sign_g    = "+" if gain_pct >= 0 else ""
                g.add_row("Entry",    f"${entry:.2f}")
                g.add_row("High",     f"${highest:.2f}  [{color_g}]({sign_g}{gain_pct:.1f}%)[/{color_g}]")
                g.add_row("Floor",    f"${floor_p:.2f}")
                g.add_row("Shares",   str(s.get("total_shares", 0)))
                g.add_row("Trailing", "[green]Active[/green]" if s.get("trailing_active") else "[dim]Not yet[/dim]")
                ladders   = s.get("ladder_triggered", [])
                g.add_row("Ladders", ", ".join(str(l) for l in ladders) if ladders else "[dim]none[/dim]")
            g.add_row("Updated", (s.get("last_updated") or "never")[:16])
            panels.append(Panel(g, title=f"[bold]Trailing Stop [{sym}]", border_style="green"))
        except Exception as exc:
            panels.append(Panel(f"[red]{exc}", title=f"Trailing Stop [{sym}]", border_style="red"))

    # ── Wheel ─────────────────────────────────────────────────────────────────
    for f in sorted(state_dir.glob("wheel_*.json")):
        sym = f.stem.replace("wheel_", "")
        try:
            s     = json.loads(f.read_text())
            stage = s.get("stage", "?")
            stage_colors = {
                "STAGE_1": "cyan", "WAITING_PUT": "yellow",
                "STAGE_2": "blue", "WAITING_CALL": "yellow", "IDLE": "dim",
            }
            sc = stage_colors.get(stage, "white")

            g = Table.grid(padding=(0, 2))
            g.add_column(style="cyan",  min_width=14)
            g.add_column(min_width=22)
            g.add_row("Stage",       f"[bold {sc}]{stage}[/bold {sc}]")
            g.add_row("Contract",    s.get("current_contract") or "[dim]—[/dim]")
            g.add_row("Shares Held", str(s.get("shares_held", 0)))
            g.add_row("Premium $",   f"${s.get('total_premium', 0):.2f}")
            g.add_row("Cycles Done", str(s.get("cycles", 0)))
            g.add_row("Last Run",    (s.get("last_run") or "never")[:16])
            panels.append(Panel(g, title=f"[bold]Wheel [{sym}]", border_style="magenta"))
        except Exception as exc:
            panels.append(Panel(f"[red]{exc}", title=f"Wheel [{sym}]", border_style="red"))

    # ── Copy Trading ──────────────────────────────────────────────────────────
    ct_file = state_dir / "copy_trading.json"
    if ct_file.exists():
        try:
            s          = json.loads(ct_file.read_text())
            following  = s.get("following") or []
            politicians = s.get("politicians", {})

            g = Table.grid(padding=(0, 2))
            g.add_column(style="cyan",  min_width=16)
            g.add_column(min_width=24)

            if following:
                for pol in following:
                    bucket  = politicians.get(pol, {})
                    tickers = list(bucket.get("positions", {}).keys())
                    seen    = len(bucket.get("seen_trades", []))
                    label   = pol.split()[-1]   # last name only to save space
                    ticker_str = ", ".join(tickers) if tickers else "[dim]—[/dim]"
                    g.add_row(
                        f"[bold]{label}[/bold]",
                        f"{len(tickers)} pos  [{ticker_str}]  seen={seen}",
                    )
            else:
                g.add_row("Following", "[dim]None yet[/dim]")

            g.add_row("Last Run", (s.get("last_run") or "never")[:16])
            panels.append(Panel(g, title="[bold]Copy Trading (top-3)", border_style="blue"))
        except Exception as exc:
            panels.append(Panel(f"[red]{exc}", title="Copy Trading", border_style="red"))

    return panels


def _log_tail(n: int = 10) -> Panel:
    log_dir   = get_settings().log_dir
    log_files = sorted(log_dir.glob("*.log"), key=os.path.getmtime, reverse=True)

    if not log_files:
        return Panel(
            Text("No log files found", style="dim"),
            title="[bold white]Recent Logs",
            border_style="dim",
        )
    try:
        lines = log_files[0].read_text(encoding="utf-8", errors="replace").splitlines()
        tail  = lines[-n:] if len(lines) >= n else lines

        content = Text()
        for line in tail:
            if "[ERROR" in line:
                content.append(line + "\n", style="bold red")
            elif "[WARNING" in line:
                content.append(line + "\n", style="yellow")
            else:
                content.append(line + "\n", style="dim white")

        return Panel(
            content,
            title=f"[bold white]Recent Logs  ({log_files[0].name})",
            border_style="dim",
        )
    except Exception as exc:
        return Panel(f"[red]{exc}", title="Recent Logs", border_style="red")


# ── Full render ───────────────────────────────────────────────────────────────

def _render(client: AlpacaClient) -> Group:
    strat_panels = _strategy_panels()

    items = [
        _header(),
        _account_panel(client),
        _positions_panel(client),
    ]
    if strat_panels:
        items.append(Columns(strat_panels, equal=True, expand=True))
    items.append(_orders_panel(client))
    items.append(_log_tail())

    return Group(*items)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="AI Trading Bot — live dashboard")
    ap.add_argument(
        "--interval", type=int, default=30, metavar="SEC",
        help="Refresh interval in seconds (default: 30)",
    )
    ap.add_argument("--once", action="store_true", help="Print snapshot and exit")
    args = ap.parse_args()

    client = AlpacaClient()

    if args.once:
        console.print(_render(client))
        return

    console.print("[bold green]Dashboard starting — press Ctrl+C to exit.[/bold green]\n")
    with Live(_render(client), console=console, refresh_per_second=1) as live:
        try:
            while True:
                time.sleep(args.interval)
                live.update(_render(client))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
