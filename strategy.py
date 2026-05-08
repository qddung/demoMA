"""
Moving Average Crossover Strategy
===================================
A simple but complete algorithmic trading strategy
built on Vietnam stock market data via vnstock.

Strategy Logic:
    - BUY  when MA_short crosses ABOVE MA_long  (golden cross)
    - SELL when MA_short crosses BELOW MA_long  (death cross)

Usage:
    python strategy.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from vnstock3 import Vnstock

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKER      = "VNM"        # Change to any HOSE/HNX ticker
START_DATE  = "2023-01-01"
END_DATE    = "2024-12-31"
MA_SHORT    = 20           # Fast moving average period
MA_LONG     = 50           # Slow moving average period
INITIAL_CAPITAL = 100_000_000   # 100 million VND
# ─────────────────────────────────────────────────────────────────────────────


def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data from vnstock."""
    print(f"Fetching data for {ticker} from {start} to {end}...")
    stock = Vnstock().stock(symbol=ticker, source="VCI")
    df = stock.quote.history(start=start, end=end, interval="1D")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    print(f"  Loaded {len(df)} trading days.\n")
    return df


def compute_signals(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """Add moving averages and crossover signals to the dataframe."""
    df = df.copy()
    df[f"MA{short}"] = df["close"].rolling(short).mean()
    df[f"MA{long}"]  = df["close"].rolling(long).mean()

    # 1 = MA_short above MA_long, -1 = below
    df["position"] = 0
    df.loc[df[f"MA{short}"] > df[f"MA{long}"], "position"] = 1
    df.loc[df[f"MA{short}"] < df[f"MA{long}"], "position"] = -1

    # Signal fires only when position CHANGES (the actual crossover)
    df["signal"] = df["position"].diff()
    df["buy"]  = df["signal"] ==  2   # -1 → +1 transition
    df["sell"] = df["signal"] == -2   # +1 → -1 transition

    return df.dropna()


def backtest(df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """
    Simple long-only backtest.
    We go fully invested on BUY and exit to cash on SELL.
    """
    cash   = capital
    shares = 0
    trades = []

    for date, row in df.iterrows():
        if row["buy"] and cash > 0:
            shares = cash / row["close"]
            cash   = 0
            trades.append({"date": date, "type": "BUY",
                           "price": row["close"], "shares": shares})

        elif row["sell"] and shares > 0:
            cash   = shares * row["close"]
            profit = cash - capital
            trades.append({"date": date, "type": "SELL",
                           "price": row["close"], "shares": shares,
                           "profit": profit})
            shares = 0

    # Close any open position at the last price
    if shares > 0:
        final_value = shares * df["close"].iloc[-1]
        trades.append({"date": df.index[-1], "type": "CLOSE (end)",
                       "price": df["close"].iloc[-1], "shares": shares})
        cash = final_value

    # Portfolio value over time
    df = df.copy()
    df["portfolio"] = df["close"].apply(
        lambda _: None  # placeholder; we'll calc below
    )

    # Rebuild daily portfolio value
    position_open = False
    entry_shares  = 0
    values = []
    cur_cash = capital

    for _, row in df.iterrows():
        if row["buy"] and not position_open:
            entry_shares  = cur_cash / row["close"]
            cur_cash      = 0
            position_open = True
        elif row["sell"] and position_open:
            cur_cash      = entry_shares * row["close"]
            entry_shares  = 0
            position_open = False

        if position_open:
            values.append(entry_shares * row["close"])
        else:
            values.append(cur_cash)

    df["portfolio"] = values
    return df, pd.DataFrame(trades), cash


def print_summary(df: pd.DataFrame, trades: pd.DataFrame,
                  final_cash: float, capital: float):
    """Print a performance summary to the console."""
    total_return  = (final_cash - capital) / capital * 100
    buy_hold_ret  = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0] * 100

    completed = trades[trades["type"] == "SELL"]
    n_trades  = len(completed)
    win_trades = (completed["profit"] > 0).sum() if n_trades > 0 else 0

    print("=" * 50)
    print(f"  BACKTEST SUMMARY  —  {TICKER}")
    print("=" * 50)
    print(f"  Period          : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Initial capital : {capital:>15,.0f} VND")
    print(f"  Final value     : {final_cash:>15,.0f} VND")
    print(f"  Strategy return : {total_return:>+.2f}%")
    print(f"  Buy & Hold      : {buy_hold_ret:>+.2f}%")
    print(f"  Total trades    : {n_trades}")
    if n_trades > 0:
        print(f"  Win rate        : {win_trades}/{n_trades} "
              f"({win_trades/n_trades*100:.0f}%)")
    print("=" * 50)
    print()

    if not trades.empty:
        print("Trade log:")
        print(trades.to_string(index=False))


def plot_results(df: pd.DataFrame, trades: pd.DataFrame, short: int, long: int):
    """Plot price + MAs + signals, and portfolio equity curve."""
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{TICKER} — MA{short}/MA{long} Crossover Strategy",
                 fontsize=14, fontweight="bold", y=0.98)

    gs  = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Price + MAs ──────────────────────────────────────────────────────────
    ax1.plot(df.index, df["close"],       color="#546e7a", lw=1.2,  label="Close")
    ax1.plot(df.index, df[f"MA{short}"],  color="#1565C0", lw=1.5,  label=f"MA{short}")
    ax1.plot(df.index, df[f"MA{long}"],   color="#EF6C00", lw=1.5,  label=f"MA{long}")

    buys  = trades[trades["type"] == "BUY"]
    sells = trades[trades["type"].isin(["SELL", "CLOSE (end)"])]

    ax1.scatter(buys["date"],  df.loc[buys["date"],  "close"],
                marker="^", color="#00897B", s=120, zorder=5, label="Buy signal")
    ax1.scatter(sells["date"], df.loc[sells["date"], "close"],
                marker="v", color="#E53935", s=120, zorder=5, label="Sell signal")

    ax1.set_ylabel("Price (VND)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax2.plot(df.index, df["portfolio"], color="#5E35B1", lw=1.5)
    ax2.fill_between(df.index, df["portfolio"], alpha=0.15, color="#5E35B1")
    ax2.axhline(INITIAL_CAPITAL, color="gray", lw=1, linestyle="--", label="Initial capital")
    ax2.set_ylabel("Portfolio Value (VND)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.savefig("result_chart.png", dpi=150, bbox_inches="tight")
    print("\nChart saved → result_chart.png")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df              = fetch_data(TICKER, START_DATE, END_DATE)
    df              = compute_signals(df, MA_SHORT, MA_LONG)
    df, trades, final_cash = backtest(df, INITIAL_CAPITAL)

    print_summary(df, trades, final_cash, INITIAL_CAPITAL)
    plot_results(df, trades, MA_SHORT, MA_LONG)