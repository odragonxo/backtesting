"""
Streamlit UI  (keep app.py, backtest.py, auth.py, settings_store.py, universe.py in one folder)

    pip install -r requirements.txt
    streamlit run app.py          # needs login secrets, see README_SETUP.md
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import auth
import backtest as bt
import settings_store as store
import universe

st.set_page_config(page_title="Backtest & สัญญาณซื้อขาย", page_icon="📈", layout="wide")
user = auth.require_login()  # stops the script here unless logged in

LABEL = bt.STRATEGY_LABELS
GREEN, RED, BLUE, ORANGE, GREY = "#26a69a", "#ef5350", "#42a5f5", "#ffa726", "#9e9e9e"
WINDOWS = {"3 เดือน": 63, "6 เดือน": 126, "1 ปี": 252, "3 ปี": 756, "ทั้งหมด": None}


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


# ---------- cached data / computation ----------
@st.cache_data(ttl=900, show_spinner="กำลังดึงข้อมูลราคา...")
def get_df(ticker, start):
    return bt.load_prices(ticker, str(start))


@st.cache_data(ttl=900, show_spinner="กำลังทดสอบกลยุทธ์...")
def get_eval(ticker, start, split, cost):
    df = get_df(ticker, start)
    fits, m_bh, bh = bt.evaluate(df, split, cost)
    return fits, m_bh, bh


@st.cache_data(ttl=900, show_spinner="กำลังฝึกซ้ำทีละช่วง (walk-forward)...")
def get_wf(ticker, start, name, cost, train_years, test_months):
    return bt.walk_forward(get_df(ticker, start), name, cost, train_years, test_months)


@st.cache_data(ttl=86400, show_spinner=False)
def _sp500_live():
    return universe.fetch_sp500()  # raises on failure -> failures are not cached


def get_universe():
    try:
        return _sp500_live(), True
    except Exception:
        return universe.fallback_df(), False


@st.cache_data(ttl=3600, show_spinner=False)
def download_chunk(tickers):
    """Batch-download ~2 years of daily bars for a tuple of tickers -> {ticker: DataFrame}."""
    import yfinance as yf

    raw = yf.download(list(tickers), period="2y", auto_adjust=True, group_by="ticker",
                      threads=True, progress=False)
    multi = isinstance(raw.columns, pd.MultiIndex)
    out = {}
    for t in tickers:
        try:
            d = raw[t] if multi else raw
            d = bt._clean(d.dropna(how="all").copy())
            if len(d) >= 260:
                out[t] = d
        except Exception:
            pass
    return out


def in_group(n, group):
    return group == "all" or bt.STRATEGY_GROUP[n] == group


def auto_pick(fits, group="all"):
    """Best TRAIN Sharpe within the style group (never pick by test results - that would be peeking)."""
    pool = [n for n in fits if in_group(n, group)] or list(fits)
    return max(pool, key=lambda n: fits[n]["train"].get("sharpe", -9))


# ---------- saved settings (per user) ----------
DEFAULTS = {
    "ticker": "NVDA", "start": "2010-01-01", "split": "2020-01-01",
    "fee_pct": 0.15, "slip_pct": 0.05, "group": "all", "strat_choice": "auto",
    "account": 10000.0, "risk_pct": 1.0, "atr_mult": 3.0,
    "wf_train_years": 5, "wf_test_months": 12, "fractional": True,
    "chart_layers": [], "lower_panel": "auto",
    "scan_universe": "sp500", "scan_custom": "NVDA, AMD, AVGO, MSFT, AAPL", "scan_group": "all",
    "scan_min_fresh": 1, "scan_uptrend": True, "scan_min_dollar_vol": 20.0, "scan_top_n": 10,
}
if st.session_state.get("_cfg_user") != user:  # first run after login: load what was saved before
    for k in [k for k in st.session_state if k.startswith("w_")]:
        del st.session_state[k]
    st.session_state["cfg"] = {**DEFAULTS, **{k: v for k, v in store.load(user).items() if k in DEFAULTS}}
    st.session_state["_cfg_user"] = user
    st.session_state["_saved_snapshot"] = json.dumps(st.session_state["cfg"], sort_keys=True, default=str)
cfg = st.session_state["cfg"]


def _date(v):
    return pd.Timestamp(v).date()


def persist():
    """Auto-save settings whenever they change."""
    snap = json.dumps(cfg, sort_keys=True, default=str)
    if snap != st.session_state.get("_saved_snapshot") and store.save(user, cfg):
        st.session_state["_saved_snapshot"] = snap


# ---------- sidebar ----------
with st.sidebar:
    if user != "local":
        st.caption(f"👤 {user}")
        if st.button("ออกจากระบบ"):
            auth.logout()
    st.header("ตั้งค่า")
    cfg["ticker"] = st.text_input("ชื่อหุ้น", value=str(cfg["ticker"]), key="w_ticker").strip().upper()
    cfg["start"] = st.date_input("เริ่มต้นข้อมูล", value=_date(cfg["start"]), key="w_start",
                                 min_value=dt.date(1980, 1, 1), max_value=dt.date.today())
    cfg["split"] = st.date_input("วันแบ่ง Train / Test", value=_date(cfg["split"]), key="w_split",
                                 min_value=dt.date(1980, 1, 1), max_value=dt.date.today(),
                                 help="ก่อนวันนี้ = ใช้หาพารามิเตอร์ | ตั้งแต่วันนี้ = ทดสอบจริง")
    c1, c2 = st.columns(2)
    cfg["fee_pct"] = c1.number_input("คอมมิชชัน/ข้าง (%)", min_value=0.0, max_value=2.0,
                                     value=float(cfg["fee_pct"]), step=0.01, key="w_fee")
    cfg["slip_pct"] = c2.number_input("สลิปเพจ/ข้าง (%)", min_value=0.0, max_value=2.0,
                                      value=float(cfg["slip_pct"]), step=0.01, key="w_slip")
    st.caption("เทรดสั้นเทรดถี่ ต้นทุนกินกำไรง่าย ใส่ค่าคอมมิชชันและสลิปเพจให้ตรงกับโบรกเกอร์จริง")
    g_opts = list(bt.GROUP_LABELS)
    cfg["group"] = st.radio("สไตล์การเทรด", g_opts, key="w_group",
                            index=g_opts.index(cfg["group"]) if cfg["group"] in g_opts else 0,
                            format_func=lambda g: bt.GROUP_LABELS[g])
    s_opts = ["auto"] + [n for n in bt.STRATEGIES if in_group(n, cfg["group"])]
    cfg["strat_choice"] = st.selectbox(
        "กลยุทธ์", s_opts, key=f"w_strat_{cfg['group']}",
        index=s_opts.index(cfg["strat_choice"]) if cfg["strat_choice"] in s_opts else 0,
        format_func=lambda x: "อัตโนมัติ (Sharpe ช่วง Train ดีสุด)" if x == "auto" else LABEL[x])

    st.header("จัดการความเสี่ยง")
    cfg["account"] = st.number_input("ขนาดพอร์ต (USD)", min_value=100.0, max_value=1e9,
                                     value=float(cfg["account"]), step=500.0, key="w_account")
    cfg["risk_pct"] = st.number_input("ยอมเสียต่อเทรด (% ของพอร์ต)", min_value=0.1, max_value=10.0,
                                      value=float(cfg["risk_pct"]), step=0.1, key="w_risk")
    cfg["atr_mult"] = st.number_input("Stop = ATR(14) x (ใช้กับกลยุทธ์ที่ไม่มี SL ของตัวเอง)", min_value=0.5,
                                      max_value=10.0, value=float(cfg["atr_mult"]), step=0.5, key="w_atr")
    cfg["fractional"] = st.checkbox("ซื้อหุ้นเศษส่วนได้ (เช่น Dime!)", value=bool(cfg["fractional"]), key="w_frac")

    with st.expander("💾 บันทึก / สำรองค่าที่ตั้งไว้"):
        st.caption("แอปบันทึกค่าให้อัตโนมัติทุกครั้งที่เปลี่ยน ถ้าโฮสต์บนคลาวด์ที่ไฟล์ถูกรีเซ็ตเมื่อรีสตาร์ต "
                   "ให้ดาวน์โหลดสำรองไว้")
        st.download_button("ดาวน์โหลดค่าที่ตั้งไว้ (JSON)",
                           json.dumps(cfg, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                           file_name=f"settings_{user}.json", mime="application/json")
        up = st.file_uploader("นำเข้าไฟล์ค่าที่เคยสำรองไว้", type="json", key="w_import")
        if up is not None and st.button("นำเข้าค่า"):
            try:
                data = json.loads(up.getvalue().decode("utf-8"))
                cfg.update({k: v for k, v in data.items() if k in DEFAULTS and type(v) is type(DEFAULTS[k])})
                for k in [k for k in st.session_state if k.startswith("w_")]:
                    del st.session_state[k]
                st.rerun()
            except Exception as e:
                st.error(f"ไฟล์ไม่ถูกต้อง: {e}")

ticker, start, split = cfg["ticker"], cfg["start"], cfg["split"]
fee, slip = cfg["fee_pct"] / 100, cfg["slip_pct"] / 100
group, strat_choice = cfg["group"], cfg["strat_choice"]
account, risk_pct, atr_mult = cfg["account"], cfg["risk_pct"], cfg["atr_mult"]
fractional = cfg["fractional"]
persist()

if not ticker:
    st.info("ใส่ชื่อหุ้นทางซ้ายเพื่อเริ่ม")
    st.stop()

cost = fee + slip
try:
    df = get_df(ticker, start)
    n_tr = int((df.index < pd.Timestamp(split)).sum())
    n_te = len(df) - n_tr
    if n_tr < 300 or n_te < 100:
        st.warning("ข้อมูลไม่พอสำหรับแบ่ง Train/Test ลองเลื่อนวันเริ่มหรือวันแบ่ง")
        st.stop()
    fits, m_bh, bh_daily = get_eval(ticker, start, split, cost)
except Exception as e:
    st.error(f"ดึงข้อมูลหรือคำนวณ {ticker} ไม่ได้: {e}")
    st.stop()

if strat_choice != "auto" and strat_choice not in fits:
    st.warning(f"{LABEL[strat_choice]} ใช้ไม่ได้กับข้อมูลชุดนี้ (ต้องมี High/Low) จึงใช้โหมดอัตโนมัติแทน")
name = strat_choice if strat_choice in fits else auto_pick(fits, group)
fit = fits[name]
close = df["Close"]
last_date, last_close = close.index[-1], float(close.iloc[-1])

st.title(f"📈 {ticker}")
st.caption(f"กลยุทธ์ที่ใช้: **{LABEL[name]}** · พารามิเตอร์ (เลือกจากช่วง Train) {fit['params'] or 'ไม่มี (ค่าคงที่)'}")
with st.expander("ℹ️ กลยุทธ์นี้ทำงานอย่างไร และควรระวังอะไร"):
    what, caveat = bt.STRATEGY_INFO[name]
    st.write(f"**หลักการ:** {what}")
    st.write(f"**ที่มา/ข้อควรระวัง:** {caveat}")

tab_sig, tab_chart, tab_res, tab_wf, tab_trades, tab_scan = st.tabs(
    ["🚦 สัญญาณวันนี้", "📊 กราฟราคา", "🧪 ผลทดสอบ", "🧠 ฝึกซ้ำ (Walk-forward)",
     "🧾 รายการเทรด", "🔍 สแกนหุ้น"])

# =====================================================================
# Tab 1: today's signal
# =====================================================================
with tab_sig:
    action, msg = bt.current_action_for(fit)
    banner = {"BUY": st.success, "SELL": st.error, "HOLD": st.info, "WAIT": st.warning}[action]
    banner(f"**{action}** · {msg}")

    now_et = pd.Timestamp.now(tz="America/New_York")
    if last_date.date() == now_et.date() and now_et.hour < 16:
        st.warning("แท่งราคาล่าสุดอาจยังไม่ปิด (ตลาดสหรัฐกำลังเปิดอยู่) สัญญาณอาจเปลี่ยนก่อนปิดตลาด "
                   "ควรใช้สัญญาณหลังตลาดปิด")
    st.caption(f"คำนวณจากราคาปิดวันที่ {last_date.date()} · ราคาปิด {last_close:,.2f} · "
               "การทดสอบสมมติว่าซื้อขายที่ราคาเปิดวันถัดไป")

    trades = fit["trades"]
    holding = len(trades) > 0 and trades.iloc[-1]["status"] == "ถืออยู่"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ราคาปิดล่าสุด", f"{last_close:,.2f}")
    if holding:
        t = trades.iloc[-1]
        m2.metric("สถานะ", "ถืออยู่")
        m3.metric("ราคาเข้า", f"{t['entry_price']:,.2f}", f"{(last_close / t['entry_price'] - 1) * 100:+.1f}%")
        m4.metric("ถือมาแล้ว", f"{int(t['days'])} วัน")
        if "max_hold" in fit["params"]:
            held = int((df.index >= t["entry_date"]).sum())
            mh = fit["params"]["max_hold"]
            st.info(f"⏱️ Time stop: ถือมาแล้ว {held}/{mh} วันทำการ — ครบกำหนดแล้วระบบจะให้สัญญาณขายเอง "
                    "แม้ยังไม่ถึงจุดออกตามราคา")
    else:
        m2.metric("สถานะ", "ไม่ได้ถือ")
        if len(trades):
            t = trades.iloc[-1]
            m3.metric("เทรดล่าสุดปิดเมื่อ", str(t["exit_date"].date()))
            m4.metric("ผลเทรดล่าสุด", pct(t["return"]))

    st.subheader("ใกล้เกิดสัญญาณแค่ไหน")
    for label, value in bt.trigger_levels(name, fit["params"], df):
        st.write(f"- {label}: **{value}**")

    st.subheader("ขนาดไม้และจุดตัดขาดทุน (แนวทางจัดการความเสี่ยง)")
    gst = fit.get("state")
    atr = bt.atr(df)
    use_gst = bool(gst) and (gst["status"] == "long" or gst["preview"] is not None)
    if not use_gst and atr is None:
        st.caption("ข้อมูลไม่มี High/Low จึงคำนวณ ATR ไม่ได้")
    else:
        if use_gst:
            ref_px = gst["entry_price"] if gst["status"] == "long" else gst["preview"]["entry"]
            stop = gst["sl"] if gst["status"] == "long" else gst["preview"]["sl"]
            a, risk_ps = gst["atr"], ref_px - stop
        else:
            a = float(atr.iloc[-1])
            risk_ps, stop = atr_mult * a, last_close - atr_mult * a
        if risk_ps <= 0:
            risk_ps, stop = atr_mult * a, last_close - atr_mult * a
        raw_sh = min(account * risk_pct / 100 / risk_ps, account / last_close)
        shares = round(raw_sh, 4) if fractional else int(raw_sh)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATR(14)", f"{a:,.2f}", f"{a / last_close * 100:.1f}% ของราคา", delta_color="off")
        c2.metric("ระดับ Stop", f"{stop:,.2f}", f"{(stop / last_close - 1) * 100:.1f}%", delta_color="off")
        c3.metric("จำนวนหุ้น", f"{shares:,.4f}".rstrip("0").rstrip(".") if fractional else f"{shares:,}",
                  f"มูลค่า {shares * last_close:,.0f}", delta_color="off")
        c4.metric("ขาดทุนสูงสุดถ้าโดน stop", f"{shares * risk_ps:,.0f}",
                  f"{shares * risk_ps / account * 100:.2f}% ของพอร์ต", delta_color="off")
        if use_gst:
            st.caption("Stop นี้คือ SL ของกลยุทธ์ ซึ่งรวมอยู่ในการทดสอบย้อนหลังแล้ว (แตะระหว่างวันออกที่ราคา SL "
                       "ถ้าเปิดช่องว่างทะลุจะออกที่ราคาเปิด) ไม่ใช้เลเวอเรจ: ขนาดไม้ไม่เกินมูลค่าพอร์ต")
        else:
            st.caption("⚠️ Stop นี้เป็นแนวทางจัดการความเสี่ยง **ไม่ได้อยู่ในการทดสอบย้อนหลัง** "
                       "ผลจริงอาจต่างจากตารางผลทดสอบ และราคาอาจกระโดดข้าม stop (gap) ได้ "
                       "ไม่ใช้เลเวอเรจ: ขนาดไม้ไม่เกินมูลค่าพอร์ต")

    st.subheader("แดชบอร์ดภาพรวม (แบบ GTR)")
    try:
        dash = bt.dashboard(df)
        dash["สถานะ"] = dash["state"].map({1: "🟢 บวก", -1: "🔴 ลบ", 0: "⚪ กลาง"})
        st.dataframe(dash[["ตัวชี้วัด", "ค่า", "สถานะ"]], hide_index=True)
        st.caption("ภาพรวมประกอบการตัดสินใจ ไม่ได้ทดสอบว่าช่วยให้ชนะ และไม่ได้อยู่ในสัญญาณของกลยุทธ์ที่เลือก "
                   "(Volume ซื้อ/ขาย เป็นค่าประมาณจากตำแหน่งราคาปิดในแท่ง)")
    except Exception as e:
        st.caption(f"สร้างแดชบอร์ดไม่ได้กับข้อมูลชุดนี้: {e}")

    st.subheader("เทรด 5 ครั้งล่าสุด")
    if len(trades):
        show = trades.tail(5).iloc[::-1].copy()
        show["return"] = show["return"].map(pct)
        show["entry_date"] = show["entry_date"].dt.date
        show["exit_date"] = show["exit_date"].dt.date
        st.dataframe(show.rename(columns={
            "entry_date": "วันเข้า", "entry_price": "ราคาเข้า", "exit_date": "วันออก/ล่าสุด",
            "exit_price": "ราคาออก/ล่าสุด", "return": "ผลตอบแทน", "status": "สถานะ", "days": "วัน",
            "reason": "เหตุผลออก"}),
            hide_index=True)

# =====================================================================
# Tab 2: price chart with signals
# =====================================================================
with tab_chart:
    win_label = st.radio("ช่วงที่แสดง", list(WINDOWS), index=2, horizontal=True)
    win = WINDOWS[win_label]
    view = df.iloc[-win:] if win else df
    p = fit["params"]
    LAYERS = {"ema9": "EMA9", "st": "Supertrend (10, 3)", "swing": "Major Swing (H/L)", "fib": "Fibonacci จากสวิงล่าสุด"}
    PANELS = {"auto": "อัตโนมัติตามกลยุทธ์", "rsi": "RSI(14)", "macd": "MACD (13,35,6)",
              "ewo": "Elliott Wave Osc.", "vol": "Volume ซื้อ/ขาย (ประมาณ)"}
    lc1, lc2 = st.columns([3, 2])
    cfg["chart_layers"] = lc1.multiselect("เลเยอร์เสริมจากชุดอินดิเคเตอร์ GTR", list(LAYERS),
                                          default=[x for x in cfg["chart_layers"] if x in LAYERS],
                                          format_func=lambda x: LAYERS[x], key="w_layers")
    cfg["lower_panel"] = lc2.selectbox("แผงล่าง", list(PANELS), key="w_panel", format_func=lambda x: PANELS[x],
                                       index=list(PANELS).index(cfg["lower_panel"]) if cfg["lower_panel"] in PANELS else 0)
    layers = cfg["chart_layers"]
    has_hl = {"High", "Low"} <= set(df.columns)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.72, 0.28])
    if {"Open", "High", "Low"} <= set(df.columns):
        fig.add_trace(go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="ราคา", increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="ราคาปิด", line=dict(color=BLUE)),
                      row=1, col=1)

    def line(series, label, color, dash=None):
        s = series.reindex(view.index)
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=label, mode="lines",
                                 line=dict(color=color, width=1.5, dash=dash)), row=1, col=1)

    COLORS = {"blue": BLUE, "orange": ORANGE, "green": GREEN, "red": RED, "grey": GREY}
    for label, series, ckey, dash in bt.overlay_lines(name, p, df):
        line(series, label, COLORS[ckey], dash)

    if "ema9" in layers:
        line(close.ewm(span=9, adjust=False).mean(), "EMA9", "#1E90FF")
    if "st" in layers and has_hl:
        td_, tl_ = bt.supertrend(df, 10, 3.0)
        line(tl_.where(td_ == 1), "Supertrend ขาขึ้น", GREEN)
        line(tl_.where(td_ == -1), "Supertrend ขาลง", RED)
    if ("swing" in layers or "fib" in layers) and has_hl:
        sw = bt.major_swings(df)
        if "swing" in layers and not sw.empty:
            swv = sw[(sw["bar_date"] >= view.index[0])]
            for kind, sym, col in (("High", "triangle-down", RED), ("Low", "triangle-up", GREEN)):
                q = swv[swv["kind"] == kind]
                fig.add_trace(go.Scatter(
                    x=q["bar_date"], y=q["price"], mode="markers", name=f"Swing {kind}",
                    marker=dict(symbol=sym, size=10, color=col, line=dict(width=1, color="white")),
                    customdata=q["confirm_date"].dt.strftime("%Y-%m-%d"),
                    hovertemplate=kind + " %{y:,.2f}<br>รู้ผลเมื่อ %{customdata}<extra></extra>"), row=1, col=1)
        if "fib" in layers:
            fb = bt.fib_levels(sw)
            if fb:
                for lv, px in fb["levels"].items():
                    fig.add_hline(y=px, line_dash="dot", line_color=GREY, row=1, col=1,
                                  annotation_text=f"Fib {lv} · {px:,.2f}", annotation_position="right")
    gs = fit.get("state")
    if gs:  # live trade levels of bracket-order strategies (GTR)
        lv_rows = []
        if gs["status"] == "long":
            lv_rows = [("Entry", gs["entry_price"], ORANGE), ("SL", gs["sl"], RED)] + \
                      [(f"TP{k}", px, GREEN) for k, px in enumerate(gs["tps"], 1)]
        elif gs["pending"] in ("entry", "entry_limit") and gs["preview"]:
            pv = gs["preview"]
            lv_rows = [("Entry (ประมาณ)", pv["entry"], ORANGE), ("SL (ประมาณ)", pv["sl"], RED)] + \
                      [(f"TP{k} (ประมาณ)", pv[f"tp{k}"], GREEN) for k in (1, 2, 3)]
        for lbl, px, col in lv_rows:
            fig.add_hline(y=px, line_dash="dash", line_color=col, row=1, col=1,
                          annotation_text=f"{lbl} {px:,.2f}", annotation_position="right")

    # in-position shading + entry/exit markers
    tv = fit["trades"]
    tv = tv[tv["exit_date"] >= view.index[0]]
    if len(tv) <= 80:
        for _, t in tv.iterrows():
            fig.add_vrect(x0=t["entry_date"], x1=t["exit_date"], fillcolor=GREEN, opacity=0.08,
                          line_width=0, layer="below", row=1, col=1)
    ent = tv[tv["entry_date"] >= view.index[0]]
    ext = tv[(tv["status"] == "ปิดแล้ว")]
    fig.add_trace(go.Scatter(
        x=ent["entry_date"], y=ent["entry_price"], mode="markers", name="เข้าซื้อ",
        marker=dict(symbol="triangle-up", size=13, color=GREEN, line=dict(width=1, color="white")),
        hovertemplate="ซื้อ %{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ext["exit_date"], y=ext["exit_price"], mode="markers", name="ขายออก",
        marker=dict(symbol="triangle-down", size=13, color=RED, line=dict(width=1, color="white")),
        customdata=ext["return"] * 100,
        hovertemplate="ขาย %{x|%Y-%m-%d}<br>%{y:,.2f}<br>ผลเทรด %{customdata:+.1f}%<extra></extra>"),
        row=1, col=1)

    # lower panel: user choice, or automatic by strategy
    panel = cfg["lower_panel"]
    if panel == "ewo" and has_hl:
        ev = bt.ewo(df).reindex(view.index)
        fig.add_trace(go.Bar(x=ev.index, y=ev.values, name="EWO",
                             marker_color=[GREEN if v >= 0 else RED for v in ev.fillna(0)]), row=2, col=1)
    elif panel == "vol" and "Volume" in df.columns and has_hl:
        sh_, _, _ = bt.buy_sell_volume(df, 10)
        sh_ = sh_.reindex(view.index)
        fig.add_trace(go.Scatter(x=sh_.index, y=sh_.values, name="ผู้ซื้อ % (10 แท่ง)",
                                 line=dict(color=BLUE, width=1.3)), row=2, col=1)
        fig.add_hline(y=55, line_dash="dot", line_color=GREEN, row=2, col=1)
        fig.add_hline(y=45, line_dash="dot", line_color=RED, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    elif panel == "rsi":
        r = bt.rsi(close, 14).reindex(view.index)
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name="RSI(14)", line=dict(color="#ab47bc", width=1.3)),
                      row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=GREEN, row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=RED, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    elif panel == "macd":
        m_, sg_ = bt.macd_calc(close, 13, 35, 6)
        m_, sg_ = m_.reindex(view.index), sg_.reindex(view.index)
        hist = m_ - sg_
        fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Histogram",
                             marker_color=[GREEN if v >= 0 else RED for v in hist.fillna(0)]), row=2, col=1)
        fig.add_trace(go.Scatter(x=m_.index, y=m_.values, name="MACD", line=dict(color=BLUE, width=1.3)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sg_.index, y=sg_.values, name="Signal", line=dict(color=ORANGE, width=1.3)),
                      row=2, col=1)
    elif name == "macd":
        m_, sg_ = bt.macd_calc(close, p["fast"], p["slow"], p["sig"])
        m_, sg_ = m_.reindex(view.index), sg_.reindex(view.index)
        hist = m_ - sg_
        fig.add_trace(go.Bar(x=hist.index, y=hist.values, name="Histogram",
                             marker_color=[GREEN if v >= 0 else RED for v in hist.fillna(0)]), row=2, col=1)
        fig.add_trace(go.Scatter(x=m_.index, y=m_.values, name="MACD", line=dict(color=BLUE, width=1.3)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=sg_.index, y=sg_.values, name="Signal", line=dict(color=ORANGE, width=1.3)),
                      row=2, col=1)
    elif name == "ibs":
        iv = bt.ibs(df).reindex(view.index)
        fig.add_trace(go.Scatter(x=iv.index, y=iv.values, name="IBS", mode="lines",
                                 line=dict(color="#ab47bc", width=1)), row=2, col=1)
        fig.add_hline(y=p["low"], line_dash="dot", line_color=GREEN, row=2, col=1)
        fig.update_yaxes(range=[0, 1], row=2, col=1)
    else:
        if name == "rsi_revert":
            rn, lo, hi = p["n"], p["low"], p["exit_level"]
        elif name == "rsi2_trend":
            rn, lo, hi = 2, p["low"], None
        else:
            rn, lo, hi = 14, 30, 70
        r = bt.rsi(close, rn).reindex(view.index)
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name=f"RSI({rn})", line=dict(color="#ab47bc", width=1.3)),
                      row=2, col=1)
        fig.add_hline(y=lo, line_dash="dot", line_color=GREEN, row=2, col=1)
        if hi is not None:
            fig.add_hline(y=hi, line_dash="dot", line_color=RED, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=680, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)
    st.caption("สามเหลี่ยมเขียว = จุดเข้า (ราคาเปิดวันถัดจากสัญญาณ) · สามเหลี่ยมแดง = จุดออก · "
               "แถบเขียวจาง = ช่วงที่ถือหุ้น · เส้นประ Entry/SL/TP = ระดับของเทรดที่ถืออยู่หรือที่กำลังจะเข้า")
    if "swing" in layers or "fib" in layers:
        st.caption("⚠️ Major Swing รู้ผลช้า: จุดสูง/ต่ำถูกยืนยันหลังจาก Stochastic ไปถึงโซนตรงข้ามแล้ว (ดูวันที่ใน hover) "
                   "กราฟย้อนหลังจึงดูแม่นกว่าที่เห็นตอนเทรดจริง และเส้น Fib จะถูกวาดใหม่เมื่อมีสวิงใหม่")

# =====================================================================
# Tab 3: results
# =====================================================================
with tab_res:
    st.markdown("**ผลช่วงทดสอบ (Out-of-sample) — ช่วงที่ไม่ได้ใช้เลือกพารามิเตอร์**")
    rows = []
    for n, f in fits.items():
        m = f["test"]
        rows.append({
            "กลยุทธ์": LABEL[n] + (" ◀" if n == name else ""),
            "ผลตอบแทนรวม": pct(m["total_return"]), "CAGR": pct(m["cagr"]),
            "Max drawdown": pct(m["max_drawdown"]), "Sharpe": round(m["sharpe"], 2),
            "เวลาอยู่ในตลาด": pct(m["time_in_market"], 0), "จำนวนเทรด": m["trades"],
            "ถือเฉลี่ย (วัน)": round(m["avg_days"], 1),
            "Win rate": pct(m["win_rate"], 0), "กำไรเฉลี่ย/เทรด": pct(m["expectancy"]),
            "Profit factor": "∞" if np.isinf(m["profit_factor"]) else round(m["profit_factor"], 2)})
    rows.append({
        "กลยุทธ์": "ซื้อถือ (buy & hold)", "ผลตอบแทนรวม": pct(m_bh["total_return"]),
        "CAGR": pct(m_bh["cagr"]), "Max drawdown": pct(m_bh["max_drawdown"]),
        "Sharpe": round(m_bh["sharpe"], 2), "เวลาอยู่ในตลาด": "100%", "จำนวนเทรด": "-", "ถือเฉลี่ย (วัน)": "-",
        "Win rate": "-", "กำไรเฉลี่ย/เทรด": "-", "Profit factor": "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    m_tr, m_te = fit["train"], fit["test"]
    notes = []
    notes.append("✅ Sharpe ชนะ buy & hold" if m_te["sharpe"] > m_bh["sharpe"] else "❌ Sharpe แพ้ buy & hold")
    notes.append("✅ drawdown ตื้นกว่า" if m_te["max_drawdown"] > m_bh["max_drawdown"] else "❌ drawdown ลึกกว่าหรือเท่ากัน")
    if m_te["total_return"] < m_bh["total_return"]:
        notes.append("ผลตอบแทนรวมน้อยกว่า buy & hold")
    if m_tr.get("sharpe", 0) > 0 and m_te["sharpe"] < 0.5 * m_tr["sharpe"]:
        notes.append("⚠️ Sharpe ตกจาก Train เกินครึ่ง (เสี่ยง overfit)")
    if m_te.get("trades", 0) < 20:
        notes.append("⚠️ เทรดน้อยกว่า 20 ครั้ง สถิติยังไม่น่าเชื่อถือ")
    st.write(f"**{LABEL[name]}**: " + " · ".join(notes))
    st.info("💡 ยิ่งลองกลยุทธ์และพารามิเตอร์มาก ก็ยิ่งมีโอกาสเจอตัวที่ \"บังเอิญดี\" ในช่วงทดสอบ "
            "ควรเชื่อกลยุทธ์ที่ดีทั้ง Train และ Test ใกล้เคียงกัน, ใช้ได้กับหุ้นหลายตัว (ลองแท็บสแกน) "
            "และมีเหตุผลรองรับ มากกว่าเลือกตัวที่ตัวเลข Test สูงสุดตัวเดียว "
            "ตัว \"โหวตหลายเทรนด์\" ไม่มีพารามิเตอร์ให้ปรับ จึงใช้เป็นเกณฑ์เทียบได้ดี")

    cmp_df = pd.DataFrame({
        "Train": [pct(m_tr["cagr"]), pct(m_tr["max_drawdown"]), round(m_tr["sharpe"], 2),
                  m_tr["trades"], pct(m_tr["win_rate"], 0), pct(m_tr["expectancy"])],
        "Test": [pct(m_te["cagr"]), pct(m_te["max_drawdown"]), round(m_te["sharpe"], 2),
                 m_te["trades"], pct(m_te["win_rate"], 0), pct(m_te["expectancy"])],
    }, index=["CAGR", "Max drawdown", "Sharpe", "จำนวนเทรด", "Win rate", "กำไรเฉลี่ย/เทรด"])
    st.markdown("**เทียบ Train กับ Test ของกลยุทธ์ที่เลือก** (ต่างกันมาก = สัญญาณว่าอาจ overfit)")
    st.dataframe(cmp_df)

    st.markdown("**ทดสอบความทนต่อต้นทุน** (ช่วง Test, ใช้สัญญาณเดิม แต่ค่าธรรมเนียม+สลิปเพจสูงขึ้น)")
    cs = bt.cost_stress(df, fit, split, cost)
    cs_show = pd.DataFrame({
        "ต้นทุนต่อข้าง": [f"{v:.2f}%  ({int(k)}x)" for v, k in zip(cs["cost_pct"], cs["mult"])],
        "ผลตอบแทนรวม": [pct(v) for v in cs["total_return"]],
        "CAGR": [pct(v) for v in cs["cagr"]],
        "Sharpe": cs["sharpe"].round(2),
        "กำไรเฉลี่ย/เทรด": [pct(v, 2) for v in cs["expectancy"]]})
    st.dataframe(cs_show, hide_index=True)
    st.caption("ถ้ากำไรหายหรือ Sharpe ติดลบเมื่อต้นทุนเพิ่มแค่ 2 เท่า แปลว่ากลยุทธ์บางเกินไปสำหรับใช้จริง "
               "โดยเฉพาะกลยุทธ์เทรดสั้นที่เทรดถี่")

    te = fit["trades"][fit["trades"]["entry_date"] >= pd.Timestamp(split)]
    if len(te):
        streak = best_streak = 0
        for v in te["return"]:
            streak = streak + 1 if v <= 0 else 0
            best_streak = max(best_streak, streak)
        a, b, c, d = st.columns(4)
        a.metric("กำไรเฉลี่ยเมื่อชนะ", pct(m_te["avg_win"]))
        b.metric("ขาดทุนเฉลี่ยเมื่อแพ้", pct(m_te["avg_loss"]))
        c.metric("แพ้ติดกันสูงสุด", f"{best_streak} ครั้ง")
        d.metric("เทรดที่ดี/แย่สุด", f"{pct(te['return'].max(), 0)} / {pct(te['return'].min(), 0)}")

    include_train = st.checkbox("แสดงรวมช่วง Train (ช่วงนี้ถูกปรับพารามิเตอร์ให้เข้ากับข้อมูล ไม่ควรเชื่อ)")
    sp = pd.Timestamp(split)
    d_s = fit["daily"] if include_train else fit["daily"][fit["daily"].index >= sp]
    d_b = bh_daily.reindex(d_s.index).fillna(0)
    eq_s, eq_b = (1 + d_s).cumprod(), (1 + d_b).cumprod()
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.68, 0.32])
    fig2.add_trace(go.Scatter(x=eq_s.index, y=eq_s.values, name=LABEL[name], line=dict(color=BLUE)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=eq_b.index, y=eq_b.values, name="ซื้อถือ", line=dict(color=GREY)), row=1, col=1)
    for eq, label, color in [(eq_s, LABEL[name], BLUE), (eq_b, "ซื้อถือ", GREY)]:
        dd = eq / eq.cummax() - 1
        fig2.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name=f"DD {label}", fill="tozeroy",
                                  line=dict(color=color, width=1), showlegend=False), row=2, col=1)
    if include_train:
        fig2.add_shape(type="line", x0=sp, x1=sp, y0=0, y1=1, xref="x", yref="paper",
                       line=dict(dash="dash", color=ORANGE))
    fig2.update_yaxes(type="log", title="มูลค่าพอร์ต (เริ่ม=1, log)", row=1, col=1)
    fig2.update_yaxes(title="drawdown (%)", row=2, col=1)
    fig2.update_layout(height=560, hovermode="x unified", legend=dict(orientation="h", y=1.06),
                       margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2)

# =====================================================================
# Tab: walk-forward (re-training)
# =====================================================================
with tab_wf:
    st.markdown(
        "**ฝึกซ้ำแบบเลื่อนช่วง (Walk-forward)** คือวิธีทำให้ระบบ \"ฉลาดขึ้นตามข้อมูลใหม่\" แบบไม่โกงตัวเอง: "
        "ทุกช่วง ระบบเลือกพารามิเตอร์จากข้อมูลย้อนหลังเท่านั้น แล้วเอาไปเทรดช่วงถัดไปที่ยังไม่เคยเห็น "
        "จากนั้นเลื่อนหน้าต่างไปข้างหน้าแล้วฝึกใหม่ ผลทุกช่วงที่ไม่เคยเห็นถูกต่อกันเป็นสถิติเดียว "
        "ซึ่งน่าเชื่อถือกว่าการแบ่ง Train/Test ครั้งเดียว เพราะผ่านหลายสภาวะตลาด")
    w1, w2 = st.columns(2)
    cfg["wf_train_years"] = int(w1.number_input("ฝึกจากข้อมูลย้อนหลัง (ปี)", min_value=2, max_value=15,
                                                value=int(cfg["wf_train_years"]), key="w_wf_ty"))
    cfg["wf_test_months"] = int(w2.number_input("ใช้ค่าที่ฝึกแล้วเทรดต่อ (เดือน) ก่อนฝึกใหม่", min_value=3,
                                                max_value=24, value=int(cfg["wf_test_months"]), key="w_wf_tm"))
    persist()
    try:
        wf = get_wf(ticker, start, name, cost, cfg["wf_train_years"], cfg["wf_test_months"])
    except Exception as e:
        wf = None
        st.warning(f"ทำ walk-forward ไม่ได้: {e}")

    if wf:
        wm, wb = wf["metrics"], wf["bh_metrics"]
        folds = pd.DataFrame(wf["folds"])
        n_beat = int((folds["test_return"] > folds["bh_return"]).sum())
        st.caption(f"กลยุทธ์: **{LABEL[name]}** · สถิติจริงนอกตัวอย่าง "
                   f"{folds['test_start'].iloc[0].date()} → {folds['test_end'].iloc[-1].date()} "
                   f"({len(folds)} ช่วง)")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("CAGR", pct(wm["cagr"]), f"ซื้อถือ {pct(wb['cagr'])}", delta_color="off")
        a2.metric("Sharpe", f"{wm['sharpe']:.2f}", f"ซื้อถือ {wb['sharpe']:.2f}", delta_color="off")
        a3.metric("Max drawdown", pct(wm["max_drawdown"]), f"ซื้อถือ {pct(wb['max_drawdown'])}", delta_color="off")
        a4.metric("ชนะซื้อถือ", f"{n_beat}/{len(folds)} ช่วง", f"เทรด {wm.get('trades', 0)} ครั้ง", delta_color="off")

        verdict = []
        verdict.append("✅ Sharpe ชนะ buy & hold" if wm["sharpe"] > wb["sharpe"] else "❌ Sharpe แพ้ buy & hold")
        verdict.append("✅ drawdown ตื้นกว่า" if wm["max_drawdown"] > wb["max_drawdown"] else "❌ drawdown ลึกกว่าหรือเท่ากัน")
        if n_beat < len(folds) / 2:
            verdict.append("⚠️ ชนะซื้อถือไม่ถึงครึ่งของช่วง")
        st.write(" · ".join(verdict))

        eq_s, eq_b = (1 + wf["daily"]).cumprod(), (1 + wf["bh_daily"]).cumprod()
        figw = go.Figure()
        figw.add_trace(go.Scatter(x=eq_s.index, y=eq_s.values, name=f"{LABEL[name]} (walk-forward)", line=dict(color=BLUE)))
        figw.add_trace(go.Scatter(x=eq_b.index, y=eq_b.values, name="ซื้อถือ", line=dict(color=GREY)))
        figw.update_yaxes(type="log", title="มูลค่าพอร์ต (เริ่ม=1, log)")
        figw.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.08),
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figw)

        st.markdown("**แต่ละช่วงเลือกพารามิเตอร์อะไร** (ถ้าเปลี่ยนไปมาก = ผลไม่นิ่ง น่าสงสัยว่า overfit)")
        show = pd.DataFrame({
            "ช่วงเทรด": [f"{a.date()} → {b.date()}" for a, b in zip(folds["test_start"], folds["test_end"])],
            "พารามิเตอร์ที่เลือก": [str(p) if p else "ค่าคงที่" for p in folds["params"]],
            "Sharpe ช่วงฝึก": folds["train_sharpe"].round(2),
            "ผลตอบแทนช่วงเทรด": [pct(v) for v in folds["test_return"]],
            "ซื้อถือช่วงเดียวกัน": [pct(v) for v in folds["bh_return"]],
            "ชนะ?": ["✅" if a > b else "❌" for a, b in zip(folds["test_return"], folds["bh_return"])]})
        st.dataframe(show, hide_index=True)

        st.subheader("สัญญาณวันนี้ด้วยพารามิเตอร์ที่ฝึกล่าสุด")
        lp = wf["latest_params"]
        act_l, msg_l = bt.current_action_for({"signal": wf["latest_signal"], "state": wf.get("latest_state")})
        act_f, _ = bt.current_action_for(fit)
        {"BUY": st.success, "SELL": st.error, "HOLD": st.info, "WAIT": st.warning}[act_l](
            f"**{act_l}** · {msg_l}")
        st.caption(f"พารามิเตอร์ล่าสุด (ฝึกจากข้อมูล {cfg['wf_train_years']} ปีล่าสุด): {lp or 'ค่าคงที่'} · "
                   f"เทียบกับพารามิเตอร์จากช่วง Train เดิม ({fit['params'] or 'ค่าคงที่'}) ซึ่งให้สัญญาณ **{act_f}**")
        for label, value in bt.trigger_levels(name, lp, df):
            st.write(f"- {label}: **{value}**")
        st.info("พารามิเตอร์ที่ฝึกใหม่ตามข้อมูลล่าสุดเหมาะใช้ดูสัญญาณจริงมากกว่าค่าที่ตรึงไว้ที่ปี 2020 "
                "แต่ยังเป็นการเรียนรู้จากอดีต ไม่ใช่การทำนายอนาคต")

# =====================================================================
# Tab 4: trade log
# =====================================================================
with tab_trades:
    only_test = st.checkbox("เฉพาะช่วงทดสอบ", value=True)
    tl = fit["trades"].copy()
    if only_test:
        tl = tl[tl["entry_date"] >= pd.Timestamp(split)]
    if tl.empty:
        st.info("ไม่มีเทรดในช่วงนี้")
    else:
        fig3 = go.Figure(go.Bar(
            x=list(range(1, len(tl) + 1)), y=tl["return"] * 100,
            marker_color=[GREEN if v > 0 else RED for v in tl["return"]],
            customdata=tl["entry_date"].dt.strftime("%Y-%m-%d"),
            hovertemplate="เทรด #%{x} (เข้า %{customdata})<br>%{y:+.1f}%<extra></extra>"))
        fig3.update_layout(height=300, xaxis_title="ลำดับเทรด", yaxis_title="ผลตอบแทน (%)",
                           margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig3)
        out = tl.copy()
        out["return"] = (out["return"] * 100).round(2)
        out["entry_date"], out["exit_date"] = out["entry_date"].dt.date, out["exit_date"].dt.date
        out = out.rename(columns={"entry_date": "วันเข้า", "entry_price": "ราคาเข้า", "exit_date": "วันออก",
                                  "exit_price": "ราคาออก", "return": "ผลตอบแทน (%)", "status": "สถานะ",
                                  "days": "วันที่ถือ", "reason": "เหตุผลออก"})
        st.dataframe(out, hide_index=True)
        st.download_button("ดาวน์โหลด CSV", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_{name}_trades.csv", mime="text/csv")

# =====================================================================
# Tab: stock scanner
# =====================================================================
with tab_scan:
    st.markdown("**สแกนหาหุ้นที่เพิ่งเกิดสัญญาณซื้อตามเทคนิค** (ขั้นที่ 1 คัดกรองเร็ว ขั้นที่ 2 ตรวจย้อนหลังเต็ม)")
    uni_opts = ["sp500", "custom"]
    cfg["scan_universe"] = st.radio(
        "กลุ่มหุ้นที่สแกน", uni_opts, horizontal=True, key="w_scan_uni",
        index=uni_opts.index(cfg["scan_universe"]) if cfg["scan_universe"] in uni_opts else 0,
        format_func=lambda x: "S&P 500 ทั้งดัชนี" if x == "sp500" else "รายชื่อที่ฉันกำหนด")
    if cfg["scan_universe"] == "custom":
        cfg["scan_custom"] = st.text_area("รายชื่อหุ้น (คั่นด้วยคอมมา)", value=cfg["scan_custom"], key="w_scan_custom")
    sg_opts = list(bt.GROUP_LABELS)
    cfg["scan_group"] = st.radio(
        "ชนิดสัญญาณที่ใช้คัดกรอง", sg_opts, horizontal=True, key="w_scan_group",
        index=sg_opts.index(cfg["scan_group"]) if cfg["scan_group"] in sg_opts else 0,
        format_func=lambda g: bt.GROUP_LABELS[g])
    f1, f2, f3 = st.columns(3)
    cfg["scan_min_fresh"] = int(f1.number_input("สัญญาณซื้อใหม่อย่างน้อย (ตัว)", min_value=0, max_value=6,
                                                value=int(cfg["scan_min_fresh"]), key="w_scan_minf"))
    cfg["scan_min_dollar_vol"] = float(f2.number_input("มูลค่าซื้อขายเฉลี่ย/วัน อย่างน้อย (ล้าน USD)", min_value=0.0,
                                                       value=float(cfg["scan_min_dollar_vol"]), step=5.0, key="w_scan_vol"))
    cfg["scan_uptrend"] = f3.checkbox("เฉพาะที่ราคาอยู่เหนือ SMA200", value=bool(cfg["scan_uptrend"]), key="w_scan_up")
    persist()

    scan_names = [n for n in bt.SCAN_SPECS if in_group(n, cfg["scan_group"])]
    with st.expander("สัญญาณที่ใช้และพารามิเตอร์ (ค่ามาตรฐาน ไม่ได้ปรับตามหุ้นแต่ละตัว)"):
        for n in scan_names:
            st.write(f"- `{n}` {LABEL[n]} {bt.SCAN_SPECS[n] or ''}")

    if st.button("🔍 สแกนตอนนี้", type="primary"):
        meta = pd.DataFrame(columns=["name", "sector"])
        if cfg["scan_universe"] == "sp500":
            uni, live_ok = get_universe()
            tks = list(uni["ticker"])
            meta = uni.set_index("ticker")[["name", "sector"]]
            if not live_ok:
                st.warning("ดึงรายชื่อ S&P 500 ล่าสุดไม่ได้ จึงใช้รายชื่อสำรอง (ไม่ครบ 500 ตัว)")
        else:
            tks = [t.strip().upper() for t in cfg["scan_custom"].split(",") if t.strip()]
        rows, failed = {}, 0
        prog = st.progress(0.0, text="เริ่มดึงข้อมูล...")
        for i in range(0, len(tks), 50):
            chunk = tuple(tks[i:i + 50])
            try:
                for t_, d_ in download_chunk(chunk).items():
                    rows[t_] = bt.scan_ticker(d_, scan_names)
            except Exception:
                failed += len(chunk)
            prog.progress(min(1.0, (i + 50) / len(tks)), text=f"ดึงและวิเคราะห์ {min(i + 50, len(tks))}/{len(tks)}")
        prog.empty()
        st.session_state.update(scan_rows=rows, scan_meta=meta, scan_names=scan_names,
                                scan_when=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), scan_failed=failed)
        st.session_state.pop("scan_deep", None)

    rows = st.session_state.get("scan_rows")
    if rows:
        res = bt.rank_scan(rows)
        n_all = len(res)
        if not res.empty:
            res = res[res["n_fresh"] >= cfg["scan_min_fresh"]]
            if cfg["scan_uptrend"]:
                res = res[res["above_sma200"]]
            res = res[res["dollar_vol"].fillna(0) >= cfg["scan_min_dollar_vol"] * 1e6]
        meta = st.session_state["scan_meta"]
        st.caption(f"สแกนเมื่อ {st.session_state['scan_when']} · วิเคราะห์ได้ {n_all} ตัว "
                   f"(ดึงไม่สำเร็จ {st.session_state.get('scan_failed', 0)}) · ผ่านตัวกรอง {len(res)} ตัว")
        if res.empty:
            st.info("ไม่มีหุ้นผ่านตัวกรองตอนนี้ ลองลดจำนวนสัญญาณขั้นต่ำหรือปิดตัวกรอง SMA200")
        else:
            show = pd.DataFrame({
                "หุ้น": res["ticker"],
                "บริษัท": res["ticker"].map(meta["name"]) if "name" in meta else "",
                "กลุ่ม": res["ticker"].map(meta["sector"]) if "sector" in meta else "",
                "ราคาปิด": res["close"].round(2),
                "สัญญาณซื้อใหม่": [", ".join(f) for f in res["fresh"]],
                "จำนวนใหม่": res["n_fresh"],
                "อยู่ในโหมดซื้อ (ตัว)": res["n_long"],
                "1M %": (res["ret_1m"] * 100).round(1), "3M %": (res["ret_3m"] * 100).round(1),
                "12M %": (res["ret_12m"] * 100).round(1),
                "ห่างจากสูงสุด 52 สัปดาห์ %": (res["from_high"] * 100).round(1),
                "RSI14": res["rsi14"].round(0), "ATR %": (res["atr_pct"] * 100).round(1),
                "ซื้อขาย/วัน (ล้าน$)": (res["dollar_vol"] / 1e6).round(0)})
            st.dataframe(show.head(100), hide_index=True)
            st.download_button("ดาวน์โหลดผลสแกน (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                               file_name="scan_results.csv", mime="text/csv")

            st.markdown("**ขั้นที่ 2: ตรวจย้อนหลังเต็ม (Train/Test) ของหุ้นอันดับต้น ๆ**")
            cfg["scan_top_n"] = int(st.slider("จำนวนหุ้นที่ตรวจ", 3, 20, int(cfg["scan_top_n"]), key="w_scan_topn"))
            if st.button("ตรวจย้อนหลังเต็ม"):
                deep, tops = [], res["ticker"].head(cfg["scan_top_n"]).tolist()
                prog = st.progress(0.0, text="กำลังทดสอบ...")
                for i, tk in enumerate(tops):
                    try:
                        d_ = get_df(tk, start)
                        f_, mb_, _ = get_eval(tk, start, split, cost)
                        nm = strat_choice if strat_choice in f_ else auto_pick(f_, group)
                        ff = f_[nm]
                        act, _ = bt.current_action_for(ff)
                        deep.append({
                            "หุ้น": tk, "กลยุทธ์ที่เลือก (จาก Train)": LABEL[nm], "สัญญาณวันนี้": act,
                            "Sharpe (Test)": round(ff["test"]["sharpe"], 2),
                            "Sharpe ซื้อถือ": round(mb_["sharpe"], 2),
                            "ชนะซื้อถือ?": "✅" if ff["test"]["sharpe"] > mb_["sharpe"] else "❌",
                            "Max DD (Test)": pct(ff["test"]["max_drawdown"]),
                            "เทรด (Test)": ff["test"]["trades"]})
                    except Exception as e:
                        deep.append({"หุ้น": tk, "กลยุทธ์ที่เลือก (จาก Train)": f"error: {e}"})
                    prog.progress((i + 1) / len(tops), text=f"{i + 1}/{len(tops)}")
                prog.empty()
                st.session_state["scan_deep"] = pd.DataFrame(deep)
            if "scan_deep" in st.session_state:
                st.dataframe(st.session_state["scan_deep"], hide_index=True)

            def _set_ticker(t):
                cfg["ticker"] = t
                st.session_state["w_ticker"] = t

            p1, p2 = st.columns([2, 1])
            pick = p1.selectbox("เปิดดูรายละเอียดหุ้นตัวนี้", res["ticker"].head(100).tolist(), key="w_scan_pick")
            p2.button("ตั้งเป็นหุ้นหลัก", on_click=_set_ticker, args=(pick,))
            st.caption("กดแล้วไปที่แท็บ 🚦 สัญญาณวันนี้ หรือ 📊 กราฟราคา ได้เลย")
        persist()

    st.warning(
        "⚠️ การสแกนบอกแค่ว่า **วันนี้เกิดสัญญาณตามกฎมาตรฐาน** ไม่ได้บอกว่ากลยุทธ์นั้นเคยได้ผลกับหุ้นตัวนั้น "
        "และไม่ใช่การทำนาย ควรใช้ขั้นที่ 2 และดูกราฟก่อนตัดสินใจ รายชื่อ S&P 500 เป็นสมาชิกปัจจุบัน "
        "ซึ่งเป็นกลุ่มที่ \"รอดมาได้\" (survivorship bias) ผลย้อนหลังของหุ้นกลุ่มนี้จึงดูดีเกินจริงเล็กน้อย")

persist()
st.divider()
st.caption("เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน ไม่ได้คิดภาษี และผลในอดีตไม่การันตีอนาคต")
