"""
Backtest engine: trend / breakout / RSI strategies vs buy-and-hold.

CLI:
    pip install yfinance pandas numpy matplotlib
    python backtest.py --ticker NVDA --start 2010-01-01 --split 2020-01-01
    python backtest.py --csv data.csv     # CSV: Date, Close (optional Open, High, Low, Volume)

Rules that keep the test honest:
  * Parameters are chosen on the TRAIN period only, then judged on TEST.
  * Signal is computed from the CLOSE of day t; the trade is executed at the
    OPEN of day t+1 (or the next close if the data has no Open column).
  * Fee + slippage are charged on every buy and every sell.
  * Long-only, all-in / all-out, no leverage, no shorting. Taxes not modelled.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_TRAIN_TRADES = 5


# ---------- data ----------
def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    if "Open" in df:
        df = df.dropna(subset=["Open"])
    return df.sort_index()


def load_prices(ticker=None, start=None, csv=None):
    if csv:
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf

        df = yf.download(ticker, start=str(start), auto_adjust=True, progress=False)
    return _clean(df)


# ---------- indicators / signals (1 = long, 0 = flat), computed from closes ----------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    if not {"High", "Low"} <= set(df.columns):
        return None
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def sig_sma_cross(close, fast, slow):
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)


def sig_breakout(close, n, exit_n):
    """Enter on a new n-day closing high, exit on a new exit_n-day closing low."""
    hi = close.rolling(n).max().shift(1)
    lo = close.rolling(exit_n).min().shift(1)
    pos = pd.Series(np.nan, index=close.index)
    pos[close > hi] = 1
    pos[close < lo] = 0
    return pos.ffill().fillna(0).astype(int)


def sig_rsi_revert(close, n, low, exit_level):
    """Buy when RSI < low, sell when RSI > exit_level."""
    r = rsi(close, n)
    pos = pd.Series(np.nan, index=close.index)
    pos[r < low] = 1
    pos[r > exit_level] = 0
    return pos.ffill().fillna(0).astype(int)


STRATEGIES = {
    "sma_cross": (sig_sma_cross, {"fast": [10, 20, 50], "slow": [50, 100, 200]}),
    "breakout": (sig_breakout, {"n": [20, 50, 100], "exit_n": [10, 20, 50]}),
    "rsi_revert": (sig_rsi_revert, {"n": [2, 14], "low": [10, 30], "exit_level": [50, 70]}),
}

STRATEGY_LABELS = {
    "sma_cross": "ตามเทรนด์ (SMA cross)",
    "breakout": "เบรกเอาต์ (Breakout)",
    "rsi_revert": "กลับตัว (RSI)",
}


def param_grid(space):
    keys = list(space)
    for vals in itertools.product(*space.values()):
        p = dict(zip(keys, vals))
        if "fast" in p and p["fast"] >= p["slow"]:
            continue
        if "n" in p and "exit_n" in p and p["exit_n"] >= p["n"]:
            continue
        yield p


# ---------- engine ----------
def _exec_price(df):
    return df["Open"] if "Open" in df.columns else df["Close"]


def run(df, signal, cost):
    """Daily strategy returns. Signal at close t -> trade at open t+1."""
    close, ex = df["Close"], _exec_price(df)
    s1 = signal.shift(1).fillna(0)  # position held during the day
    s2 = signal.shift(2).fillna(0)  # position held overnight into the day
    overnight = (ex / close.shift(1) - 1).fillna(0)
    intraday = (close / ex - 1).fillna(0)
    daily = s2 * overnight + s1 * intraday - (s1 - s2).abs() * cost
    return daily, s1


def trade_log(df, signal, cost):
    """One row per trade, net of costs. The last trade may still be open."""
    close, ex = df["Close"], _exec_price(df)
    s1 = signal.shift(1).fillna(0).astype(int)
    chg = s1.diff().fillna(s1)
    rows, entry = [], None
    for d, c in chg.items():
        if c == 1:
            entry = (d, float(ex[d]))
        elif c == -1 and entry is not None:
            xp = float(ex[d])
            rows.append((entry[0], entry[1], d, xp, xp / entry[1] - 1 - 2 * cost, "ปิดแล้ว"))
            entry = None
    if entry is not None:
        xp = float(close.iloc[-1])
        rows.append((entry[0], entry[1], close.index[-1], xp, xp / entry[1] - 1 - cost, "ถืออยู่"))
    out = pd.DataFrame(rows, columns=["entry_date", "entry_price", "exit_date", "exit_price",
                                      "return", "status"])
    # make dtypes explicit so an EMPTY table (no trades) still works with .dt
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"] = pd.to_datetime(out["exit_date"])
    for c in ["entry_price", "exit_price", "return"]:
        out[c] = out[c].astype(float)
    out["days"] = (out["exit_date"] - out["entry_date"]).dt.days
    return out


def metrics(daily, pos=None, trades=None):
    daily = daily.dropna()
    if len(daily) < 2:
        return {}
    eq = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    m = {
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "sharpe": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() > 0 else 0.0,
    }
    if pos is not None:
        m["time_in_market"] = float(pos.mean())
    if trades is not None:
        t = trades["return"].values if len(trades) else np.array([0.0])
        wins, losses = t[t > 0], t[t <= 0]
        m["trades"] = len(trades)
        m["win_rate"] = float((t > 0).mean()) if len(trades) else 0.0
        m["expectancy"] = float(t.mean()) if len(trades) else 0.0
        m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        m["avg_loss"] = float(losses.mean()) if len(losses) and len(trades) else 0.0
        m["profit_factor"] = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
    return m


def fit_strategy(df, split, name, cost):
    """Choose params on TRAIN (best Sharpe), then report TRAIN and TEST separately."""
    fn, space = STRATEGIES[name]
    split = pd.Timestamp(split)
    train = df[df.index < split]
    best, best_s = None, -np.inf
    for p in param_grid(space):
        sig = fn(train["Close"], **p)
        d, _ = run(train, sig, cost)
        s = metrics(d).get("sharpe", -np.inf)
        if len(trade_log(train, sig, cost)) < MIN_TRAIN_TRADES:
            s -= 10  # too few trades to trust
        if s > best_s:
            best, best_s = p, s

    signal = fn(df["Close"], **best)  # indicators warmed up on full history
    daily, pos = run(df, signal, cost)
    trades = trade_log(df, signal, cost)
    tr_mask, te_mask = daily.index < split, daily.index >= split
    m_tr = metrics(daily[tr_mask], pos[tr_mask], trades[trades["entry_date"] < split])
    m_te = metrics(daily[te_mask], pos[te_mask], trades[trades["entry_date"] >= split])
    return {"name": name, "params": best, "signal": signal, "daily": daily, "pos": pos,
            "trades": trades, "train": m_tr, "test": m_te}


def evaluate(df, split, cost):
    """Returns ({strategy: fit}, buy_hold_test_metrics, buy_hold_daily)."""
    split = pd.Timestamp(split)
    fits = {n: fit_strategy(df, split, n, cost) for n in STRATEGIES}
    bh = df["Close"].pct_change().fillna(0)
    bh_te = bh[bh.index >= split]
    m_bh = metrics(bh_te, pd.Series(1.0, index=bh_te.index), None)
    return fits, m_bh, bh


def current_action(signal):
    """What to do at the NEXT open, given the signal at the latest close."""
    now = int(signal.iloc[-1])
    prev = int(signal.iloc[-2]) if len(signal) > 1 else 0
    if now == 1 and prev == 0:
        return "BUY", "สัญญาณซื้อใหม่ → เข้าซื้อตอนเปิดตลาดวันถัดไป"
    if now == 0 and prev == 1:
        return "SELL", "สัญญาณขาย → ขายตอนเปิดตลาดวันถัดไป"
    if now == 1:
        return "HOLD", "ถืออยู่ต่อ (ยังไม่มีสัญญาณขาย)"
    return "WAIT", "อยู่นอกตลาด รอสัญญาณซื้อ"


def trigger_levels(name, params, close):
    """Human-readable distance to the next signal change, based on the latest close."""
    last = float(close.iloc[-1])
    out = []
    if name == "sma_cross":
        f, s = close.rolling(params["fast"]).mean().iloc[-1], close.rolling(params["slow"]).mean().iloc[-1]
        out.append((f"SMA{params['fast']} (เร็ว)", f"{f:,.2f}"))
        out.append((f"SMA{params['slow']} (ช้า)", f"{s:,.2f}"))
        out.append(("ช่องว่างเร็ว-ช้า", f"{(f / s - 1) * 100:+.2f}%  (ซื้อเมื่อ > 0)"))
    elif name == "breakout":
        up = close.iloc[-params["n"]:].max()
        dn = close.iloc[-params["exit_n"]:].min()
        out.append((f"ราคาปิดเหนือ (สูงสุด {params['n']} วัน) = ซื้อ", f"{up:,.2f}  ({(up / last - 1) * 100:+.1f}% จากราคาปิดล่าสุด)"))
        out.append((f"ราคาปิดต่ำกว่า (ต่ำสุด {params['exit_n']} วัน) = ขาย", f"{dn:,.2f}  ({(dn / last - 1) * 100:+.1f}%)"))
    elif name == "rsi_revert":
        r = rsi(close, params["n"]).iloc[-1]
        out.append((f"RSI({params['n']}) ล่าสุด", f"{r:.1f}"))
        out.append(("ซื้อเมื่อ RSI ต่ำกว่า", f"{params['low']}"))
        out.append(("ขายเมื่อ RSI สูงกว่า", f"{params['exit_level']}"))
    return out


# ---------- CLI ----------
COLS = ["total_return", "cagr", "max_drawdown", "sharpe", "time_in_market",
        "trades", "win_rate", "expectancy", "profit_factor"]
PCT = {"total_return", "cagr", "max_drawdown", "time_in_market", "win_rate", "expectancy"}


def _fmt(m, k):
    if k not in m:
        return ""
    v = m[k]
    if k in PCT:
        return f"{v * 100:.1f}%"
    return f"{int(v)}" if k == "trades" else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--split", default="2020-01-01", help="train < split <= test")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    cost = args.fee + args.slippage

    df = load_prices(args.ticker, args.start, args.csv)
    fits, m_bh, _ = evaluate(df, args.split, cost)
    print(f"{args.ticker}: {df.index[0].date()} -> {df.index[-1].date()}  cost/side {cost * 100:.2f}%\n")
    print("=== OUT-OF-SAMPLE (TEST) ===")
    print(f"{'strategy':<12}" + "".join(f"{c:>15}" for c in COLS))
    for n, f in fits.items():
        print(f"{n:<12}" + "".join(f"{_fmt(f['test'], c):>15}" for c in COLS))
    print(f"{'buy&hold':<12}" + "".join(f"{_fmt(m_bh, c):>15}" for c in COLS))
    print()
    for n, f in fits.items():
        act = current_action(f["signal"])[1]
        print(f"{n:<12} {f['params']}  train_sharpe={f['train'].get('sharpe', 0):.2f} "
              f"test_sharpe={f['test'].get('sharpe', 0):.2f}  -> {act}")


if __name__ == "__main__":
    main()
