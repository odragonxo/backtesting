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


# ---------- indicators / signals (1 = long, 0 = flat), computed from the CLOSE ----------
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


def macd_calc(close, fast, slow, sig):
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return m, m.ewm(span=sig, adjust=False).mean()


def supertrend(df, n, mult):
    """Classic Supertrend. Returns (trend series of +1/-1, trailing line)."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    a = atr(df, n).values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    k = len(c)
    fub, flb, tr = np.full(k, np.nan), np.full(k, np.nan), np.ones(k)
    start = n  # skip ATR warm-up
    if k <= start:
        return pd.Series(1, index=df.index), pd.Series(np.nan, index=df.index)
    fub[start], flb[start] = ub[start], lb[start]
    for i in range(start + 1, k):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if tr[i - 1] == 1 and c[i] < flb[i]:
            tr[i] = -1
        elif tr[i - 1] == -1 and c[i] > fub[i]:
            tr[i] = 1
        else:
            tr[i] = tr[i - 1]
    tr[:start] = 0
    line = np.where(tr == 1, flb, fub)
    return pd.Series(tr, index=df.index), pd.Series(line, index=df.index)


def _stateful(index, buy, sell):
    pos = pd.Series(np.nan, index=index)
    pos[buy] = 1
    pos[sell] = 0
    return pos.ffill().fillna(0).astype(int)


def sig_sma_cross(df, fast, slow):
    c = df["Close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(int)


def sig_sma_filter(df, n):
    """Long while price is above its n-day average (Faber-style trend filter)."""
    c = df["Close"]
    return (c > c.rolling(n).mean()).astype(int)


def sig_tsmom(df, n):
    """Time-series momentum: long while the n-day return is positive."""
    c = df["Close"]
    return (c / c.shift(n) - 1 > 0).astype(int)


def sig_breakout(df, n, exit_n):
    """Donchian-style: enter on a new n-day closing high, exit on a new exit_n-day low."""
    c = df["Close"]
    return _stateful(c.index, c > c.rolling(n).max().shift(1), c < c.rolling(exit_n).min().shift(1))


def sig_macd(df, fast, slow, sig):
    m, s_ = macd_calc(df["Close"], fast, slow, sig)
    out = (m > s_).astype(int)
    out.iloc[: slow * 2] = 0  # warm-up
    return out


def sig_supertrend(df, n, mult):
    tr, _ = supertrend(df, n, mult)
    return (tr == 1).astype(int)


def sig_bollinger(df, n, k):
    """Mean reversion: buy below the lower band, sell when price returns to the middle band."""
    c = df["Close"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    return _stateful(c.index, c < mid - k * sd, c > mid)


def sig_rsi_revert(df, n, low, exit_level):
    r = rsi(df["Close"], n)
    return _stateful(df.index, r < low, r > exit_level)


def vote_components(df):
    """Fixed, NOT optimised rules (so there is nothing to overfit)."""
    c = df["Close"]
    v = {
        "ราคา > SMA200": (c > c.rolling(200).mean()).astype(int),
        "ผลตอบแทน 12 เดือน > 0": (c / c.shift(252) - 1 > 0).astype(int),
        "MACD(12,26,9) > signal": sig_macd(df, 12, 26, 9),
        "Breakout 100/50": sig_breakout(df, 100, 50),
    }
    if {"High", "Low"} <= set(df.columns):
        v["Supertrend(10,3)"] = sig_supertrend(df, 10, 3.0)
    return v


def sig_trend_vote(df):
    v = vote_components(df)
    total = sum(v.values())
    return (total * 2 > len(v)).astype(int)  # strict majority


# name -> (function, param grid). Grids are kept small on purpose (less overfitting).
STRATEGIES = {
    "sma_cross": (sig_sma_cross, {"fast": [10, 20, 50], "slow": [50, 100, 200]}),
    "sma_filter": (sig_sma_filter, {"n": [50, 100, 150, 200]}),
    "tsmom": (sig_tsmom, {"n": [63, 126, 252]}),
    "breakout": (sig_breakout, {"n": [20, 50, 100], "exit_n": [10, 20, 50]}),
    "macd": (sig_macd, {"fast": [8, 12, 19], "slow": [21, 26, 39], "sig": [5, 9]}),
    "supertrend": (sig_supertrend, {"n": [7, 10, 14], "mult": [2.0, 3.0, 4.0]}),
    "bollinger": (sig_bollinger, {"n": [20, 50], "k": [2.0, 2.5]}),
    "rsi_revert": (sig_rsi_revert, {"n": [2, 14], "low": [10, 30], "exit_level": [50, 70]}),
    "trend_vote": (lambda df: sig_trend_vote(df), {}),
}
NEEDS_HL = {"supertrend"}

STRATEGY_LABELS = {
    "sma_cross": "ตามเทรนด์: SMA cross",
    "sma_filter": "ตามเทรนด์: ราคาเหนือเส้นค่าเฉลี่ย",
    "tsmom": "โมเมนตัมตามเวลา (Time-series momentum)",
    "breakout": "เบรกเอาต์ (Donchian)",
    "macd": "MACD",
    "supertrend": "Supertrend (ATR)",
    "bollinger": "กลับตัว: Bollinger Bands",
    "rsi_revert": "กลับตัว: RSI",
    "trend_vote": "โหวตหลายเทรนด์ (พารามิเตอร์คงที่)",
}

# (what it is, evidence / caveat)
STRATEGY_INFO = {
    "sma_cross": ("ซื้อเมื่อค่าเฉลี่ยเร็วตัดขึ้นเหนือค่าเฉลี่ยช้า ขายเมื่อกลับกัน",
                  "ตามเทรนด์แบบคลาสสิก ได้ผลเมื่อมีเทรนด์ยาว แต่โดนหลอกบ่อยในตลาดไซด์เวย์"),
    "sma_filter": ("ถือเมื่อราคาปิดอยู่เหนือค่าเฉลี่ย n วัน (เช่น 200 วัน) ไม่งั้นอยู่นอกตลาด",
                   "แนวคิดจากงานของ Faber (2007) เรื่อง tactical asset allocation เทรดน้อย ค่าใช้จ่ายต่ำ "
                   "ช่วยลด drawdown ในตลาดหมี แต่มักตามหลังตลาดในขาขึ้นแรง"),
    "tsmom": ("ถือเมื่อผลตอบแทน n วันที่ผ่านมาเป็นบวก",
              "อิงงาน Time series momentum (Moskowitz, Ooi, Pedersen 2012) ที่พบในหลายสินทรัพย์ "
              "แต่ผลในอดีตไม่การันตีว่ายังได้ผลต่อ"),
    "breakout": ("ซื้อเมื่อปิดทะลุสูงสุด n วัน ขายเมื่อหลุดต่ำสุด m วัน",
                 "แนวคิด Donchian channel / Turtle Traders ชนะน้อยครั้งแต่ได้กำไรต่อครั้งใหญ่ "
                 "ต้องทนแพ้ติดกันได้"),
    "macd": ("ซื้อเมื่อเส้น MACD ตัดขึ้นเหนือเส้นสัญญาณ",
             "ใช้แพร่หลายมาก แต่หลักฐานเชิงสถิติว่าชนะตลาดอย่างเสถียรค่อนข้างอ่อน "
             "โดยพื้นฐานเป็นตัวตามเทรนด์ที่ตอบสนองเร็ว"),
    "supertrend": ("เส้นตามหลังราคาโดยใช้ ATR ซื้อเมื่อราคาปิดเหนือเส้น ขายเมื่อหลุดเส้น",
                   "เป็น trailing stop ที่ปรับตามความผันผวน ช่วยให้ถือเทรนด์ได้นาน แต่ถูกเขย่าออกในตลาดแกว่ง "
                   "ต้องใช้ข้อมูล High/Low"),
    "bollinger": ("ซื้อเมื่อราคาปิดต่ำกว่าแถบล่าง ขายเมื่อกลับมาถึงเส้นกลาง",
                  "เป็นกลยุทธ์กลับตัว เหมาะตลาดแกว่งกรอบ แต่กับหุ้นเทรนด์แรงอย่างหุ้นเทค "
                  "อาจซื้อตอนราคายังร่วงต่อ ควรเทียบผลกับ buy & hold ให้ดี"),
    "rsi_revert": ("ซื้อเมื่อ RSI ต่ำมาก ขายเมื่อ RSI กลับมาสูง",
                   "แนวคิดกลับตัวระยะสั้นแบบ RSI(2) ของ Connors เทรดถี่ ต้นทุนมีผลมาก "
                   "และเสี่ยงโดนล้างเมื่อราคาตกต่อเนื่อง"),
    "trend_vote": ("รวมกฎเทรนด์ 4-5 ข้อ (SMA200, โมเมนตัม 12 เดือน, MACD, Breakout, Supertrend) "
                   "แล้วถือเมื่อเสียงข้างมากเป็นซื้อ",
                   "ไม่มีพารามิเตอร์ให้ปรับ จึงเสี่ยง overfit น้อยที่สุดในกลุ่ม และมักนิ่งกว่าตัวเดี่ยว "
                   "เป็นจุดเริ่มต้นที่สมเหตุสมผลสำหรับตั้งเป็นเกณฑ์อ้างอิง"),
}


def available_strategies(df):
    has_hl = {"High", "Low"} <= set(df.columns)
    return [n for n in STRATEGIES if has_hl or n not in NEEDS_HL]


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
        sig = fn(train, **p)
        d, _ = run(train, sig, cost)
        s = metrics(d).get("sharpe", -np.inf)
        if len(trade_log(train, sig, cost)) < MIN_TRAIN_TRADES:
            s -= 10  # too few trades to trust
        if s > best_s:
            best, best_s = p, s

    signal = fn(df, **best)  # indicators warmed up on full history
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
    fits = {n: fit_strategy(df, split, n, cost) for n in available_strategies(df)}
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


def trigger_levels(name, params, df):
    """Human-readable distance to the next signal change, based on the latest close."""
    close = df["Close"]
    last = float(close.iloc[-1])
    out = []
    if name == "sma_cross":
        f, s_ = close.rolling(params["fast"]).mean().iloc[-1], close.rolling(params["slow"]).mean().iloc[-1]
        out += [(f"SMA{params['fast']} (เร็ว)", f"{f:,.2f}"), (f"SMA{params['slow']} (ช้า)", f"{s_:,.2f}"),
                ("ช่องว่างเร็ว-ช้า", f"{(f / s_ - 1) * 100:+.2f}%  (ซื้อเมื่อ > 0)")]
    elif name == "sma_filter":
        m = close.rolling(params["n"]).mean().iloc[-1]
        out += [(f"SMA{params['n']}", f"{m:,.2f}"),
                ("ราคาห่างจากเส้น", f"{(last / m - 1) * 100:+.2f}%  (ถือเมื่อ > 0)")]
    elif name == "tsmom":
        n = params["n"]
        ref = float(close.iloc[-n])
        out += [(f"ผลตอบแทน {n} วัน", f"{(last / close.iloc[-n - 1] - 1) * 100:+.1f}%  (ถือเมื่อ > 0)"),
                ("ราคาปิดพรุ่งนี้ต้องสูงกว่า (เพื่อคงสถานะบวก)", f"{ref:,.2f}")]
    elif name == "breakout":
        up, dn = close.iloc[-params["n"]:].max(), close.iloc[-params["exit_n"]:].min()
        out += [(f"ปิดเหนือสูงสุด {params['n']} วัน = ซื้อ", f"{up:,.2f}  ({(up / last - 1) * 100:+.1f}% จากราคาปิด)"),
                (f"ปิดต่ำกว่าต่ำสุด {params['exit_n']} วัน = ขาย", f"{dn:,.2f}  ({(dn / last - 1) * 100:+.1f}%)")]
    elif name == "macd":
        m, s_ = macd_calc(close, params["fast"], params["slow"], params["sig"])
        out += [("MACD", f"{m.iloc[-1]:.3f}"), ("เส้นสัญญาณ", f"{s_.iloc[-1]:.3f}"),
                ("Histogram", f"{m.iloc[-1] - s_.iloc[-1]:+.3f}  (ซื้อเมื่อ > 0)")]
    elif name == "supertrend":
        tr, line = supertrend(df, params["n"], params["mult"])
        ln = float(line.iloc[-1])
        out += [("เทรนด์ปัจจุบัน", "ขาขึ้น" if tr.iloc[-1] == 1 else "ขาลง"),
                ("เส้น Supertrend", f"{ln:,.2f}  ({(ln / last - 1) * 100:+.1f}% จากราคาปิด)"),
                ("กติกา", "ปิดหลุดเส้นขณะขาขึ้น = ขาย · ปิดเหนือเส้นขณะขาลง = ซื้อ")]
    elif name == "bollinger":
        mid, sd = close.rolling(params["n"]).mean().iloc[-1], close.rolling(params["n"]).std().iloc[-1]
        lo = mid - params["k"] * sd
        out += [("แถบล่าง (ซื้อเมื่อปิดต่ำกว่า)", f"{lo:,.2f}  ({(lo / last - 1) * 100:+.1f}%)"),
                ("เส้นกลาง (ขายเมื่อปิดสูงกว่า)", f"{mid:,.2f}  ({(mid / last - 1) * 100:+.1f}%)"),
                ("หมายเหตุ", "ระดับแถบเป็นค่าโดยประมาณ จะขยับตามราคาวันถัดไป")]
    elif name == "rsi_revert":
        r = rsi(close, params["n"]).iloc[-1]
        out += [(f"RSI({params['n']}) ล่าสุด", f"{r:.1f}"), ("ซื้อเมื่อ RSI ต่ำกว่า", f"{params['low']}"),
                ("ขายเมื่อ RSI สูงกว่า", f"{params['exit_level']}")]
    elif name == "trend_vote":
        v = vote_components(df)
        for k, ser in v.items():
            out.append((k, "✅ ซื้อ" if ser.iloc[-1] == 1 else "❌ ไม่ซื้อ"))
        out.append(("รวม", f"{sum(int(x.iloc[-1]) for x in v.values())}/{len(v)} (ถือเมื่อเกินครึ่ง)"))
    return out


def overlay_lines(name, params, df):
    """Lines to draw on the price chart: list of (label, series, color, dash)."""
    c = df["Close"]
    if name == "sma_cross":
        return [(f"SMA{params['fast']}", c.rolling(params["fast"]).mean(), "blue", None),
                (f"SMA{params['slow']}", c.rolling(params["slow"]).mean(), "orange", None)]
    if name == "sma_filter":
        return [(f"SMA{params['n']}", c.rolling(params["n"]).mean(), "orange", None)]
    if name == "breakout":
        return [(f"สูงสุด {params['n']} วัน (ซื้อ)", c.rolling(params["n"]).max().shift(1), "green", "dot"),
                (f"ต่ำสุด {params['exit_n']} วัน (ขาย)", c.rolling(params["exit_n"]).min().shift(1), "red", "dot")]
    if name == "bollinger":
        mid, sd = c.rolling(params["n"]).mean(), c.rolling(params["n"]).std()
        return [("แถบบน", mid + params["k"] * sd, "grey", "dot"), ("เส้นกลาง (ขาย)", mid, "orange", None),
                ("แถบล่าง (ซื้อ)", mid - params["k"] * sd, "green", "dot")]
    if name == "supertrend":
        tr, line = supertrend(df, params["n"], params["mult"])
        return [("Supertrend ขาขึ้น", line.where(tr == 1), "green", None),
                ("Supertrend ขาลง", line.where(tr == -1), "red", None)]
    if name == "trend_vote":
        return [("SMA200", c.rolling(200).mean(), "orange", None)]
    return []  # tsmom, macd, rsi_revert: nothing on the price panel


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
