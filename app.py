"""
Streamlit UI for backtest.py  (keep both files in the same folder)

    pip install streamlit yfinance pandas numpy plotly
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import backtest as bt

st.set_page_config(page_title="Backtest & สัญญาณซื้อขาย", page_icon="📈", layout="wide")

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


def auto_pick(fits):
    """Best TRAIN Sharpe (never pick by test results - that would be peeking)."""
    return max(fits, key=lambda n: fits[n]["train"].get("sharpe", -9))


# ---------- sidebar ----------
with st.sidebar:
    st.header("ตั้งค่า")
    ticker = st.text_input("ชื่อหุ้น", "NVDA").strip().upper()
    start = st.date_input("เริ่มต้นข้อมูล", pd.Timestamp("2010-01-01"))
    split = st.date_input("วันแบ่ง Train / Test", pd.Timestamp("2020-01-01"),
                          help="ก่อนวันนี้ = ใช้หาพารามิเตอร์ | ตั้งแต่วันนี้ = ทดสอบจริง")
    c1, c2 = st.columns(2)
    fee = c1.number_input("คอมมิชชัน/ข้าง (%)", 0.0, 2.0, 0.10, 0.01) / 100
    slip = c2.number_input("สลิปเพจ/ข้าง (%)", 0.0, 2.0, 0.05, 0.01) / 100
    strat_choice = st.selectbox(
        "กลยุทธ์", ["auto"] + list(bt.STRATEGIES),
        format_func=lambda x: "อัตโนมัติ (Sharpe ช่วง Train ดีสุด)" if x == "auto" else LABEL[x])

    st.header("จัดการความเสี่ยง")
    account = st.number_input("ขนาดพอร์ต (USD)", 100.0, 1e9, 10000.0, 500.0)
    risk_pct = st.number_input("ยอมเสียต่อเทรด (% ของพอร์ต)", 0.1, 10.0, 1.0, 0.1)
    atr_mult = st.number_input("Stop = ATR(14) x", 0.5, 10.0, 3.0, 0.5)

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

name = auto_pick(fits) if strat_choice == "auto" else strat_choice
fit = fits[name]
close = df["Close"]
last_date, last_close = close.index[-1], float(close.iloc[-1])

st.title(f"📈 {ticker}")
st.caption(f"กลยุทธ์ที่ใช้: **{LABEL[name]}** · พารามิเตอร์ (เลือกจากช่วง Train) {fit['params']}")

tab_sig, tab_chart, tab_res, tab_trades, tab_scan = st.tabs(
    ["🚦 สัญญาณวันนี้", "📊 กราฟราคา", "🧪 ผลทดสอบ", "🧾 รายการเทรด", "🔍 สแกนหลายตัว"])

# =====================================================================
# Tab 1: today's signal
# =====================================================================
with tab_sig:
    action, msg = bt.current_action(fit["signal"])
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
    else:
        m2.metric("สถานะ", "ไม่ได้ถือ")
        if len(trades):
            t = trades.iloc[-1]
            m3.metric("เทรดล่าสุดปิดเมื่อ", str(t["exit_date"].date()))
            m4.metric("ผลเทรดล่าสุด", pct(t["return"]))

    st.subheader("ใกล้เกิดสัญญาณแค่ไหน")
    for label, value in bt.trigger_levels(name, fit["params"], close):
        st.write(f"- {label}: **{value}**")

    st.subheader("ขนาดไม้และจุดตัดขาดทุน (แนวทางจัดการความเสี่ยง)")
    atr = bt.atr(df)
    if atr is None:
        st.caption("ข้อมูลไม่มี High/Low จึงคำนวณ ATR ไม่ได้")
    else:
        a = float(atr.iloc[-1])
        risk_ps = atr_mult * a
        stop = last_close - risk_ps
        shares = int(min(account * risk_pct / 100 / risk_ps, account / last_close))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ATR(14)", f"{a:,.2f}", f"{a / last_close * 100:.1f}% ของราคา", delta_color="off")
        c2.metric("ระดับ Stop", f"{stop:,.2f}", f"{-risk_ps / last_close * 100:.1f}%", delta_color="off")
        c3.metric("จำนวนหุ้น", f"{shares:,}", f"มูลค่า {shares * last_close:,.0f}", delta_color="off")
        c4.metric("ขาดทุนสูงสุดถ้าโดน stop", f"{shares * risk_ps:,.0f}",
                  f"{shares * risk_ps / account * 100:.2f}% ของพอร์ต", delta_color="off")
        st.caption("⚠️ Stop นี้เป็นแนวทางจัดการความเสี่ยง **ไม่ได้อยู่ในการทดสอบย้อนหลัง** "
                   "ผลจริงอาจต่างจากตารางผลทดสอบ และราคาอาจกระโดดข้าม stop (gap) ได้ "
                   "ไม่ใช้เลเวอเรจ: ขนาดไม้ไม่เกินมูลค่าพอร์ต")

    st.subheader("เทรด 5 ครั้งล่าสุด")
    if len(trades):
        show = trades.tail(5).iloc[::-1].copy()
        show["return"] = show["return"].map(pct)
        show["entry_date"] = show["entry_date"].dt.date
        show["exit_date"] = show["exit_date"].dt.date
        st.dataframe(show.rename(columns={
            "entry_date": "วันเข้า", "entry_price": "ราคาเข้า", "exit_date": "วันออก/ล่าสุด",
            "exit_price": "ราคาออก/ล่าสุด", "return": "ผลตอบแทน", "status": "สถานะ", "days": "วัน"}),
            hide_index=True)

# =====================================================================
# Tab 2: price chart with signals
# =====================================================================
with tab_chart:
    win_label = st.radio("ช่วงที่แสดง", list(WINDOWS), index=2, horizontal=True)
    win = WINDOWS[win_label]
    view = df.iloc[-win:] if win else df
    p = fit["params"]

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

    if name == "sma_cross":
        line(close.rolling(p["fast"]).mean(), f"SMA{p['fast']}", BLUE)
        line(close.rolling(p["slow"]).mean(), f"SMA{p['slow']}", ORANGE)
    elif name == "breakout":
        line(close.rolling(p["n"]).max().shift(1), f"สูงสุด {p['n']} วัน (ซื้อ)", GREEN, "dot")
        line(close.rolling(p["exit_n"]).min().shift(1), f"ต่ำสุด {p['exit_n']} วัน (ขาย)", RED, "dot")

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

    # lower panel: RSI
    rn = p["n"] if name == "rsi_revert" else 14
    r = bt.rsi(close, rn).reindex(view.index)
    fig.add_trace(go.Scatter(x=r.index, y=r.values, name=f"RSI({rn})", line=dict(color="#ab47bc", width=1.3)),
                  row=2, col=1)
    lo, hi = (p["low"], p["exit_level"]) if name == "rsi_revert" else (30, 70)
    fig.add_hline(y=lo, line_dash="dot", line_color=GREEN, row=2, col=1)
    fig.add_hline(y=hi, line_dash="dot", line_color=RED, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=680, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.04), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)
    st.caption("สามเหลี่ยมเขียว = จุดเข้า (ราคาเปิดวันถัดจากสัญญาณ) · สามเหลี่ยมแดง = จุดออก · "
               "แถบเขียวจาง = ช่วงที่ถือหุ้น")

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
            "Win rate": pct(m["win_rate"], 0), "กำไรเฉลี่ย/เทรด": pct(m["expectancy"]),
            "Profit factor": "∞" if np.isinf(m["profit_factor"]) else round(m["profit_factor"], 2)})
    rows.append({
        "กลยุทธ์": "ซื้อถือ (buy & hold)", "ผลตอบแทนรวม": pct(m_bh["total_return"]),
        "CAGR": pct(m_bh["cagr"]), "Max drawdown": pct(m_bh["max_drawdown"]),
        "Sharpe": round(m_bh["sharpe"], 2), "เวลาอยู่ในตลาด": "100%", "จำนวนเทรด": "-",
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

    cmp_df = pd.DataFrame({
        "Train": [pct(m_tr["cagr"]), pct(m_tr["max_drawdown"]), round(m_tr["sharpe"], 2),
                  m_tr["trades"], pct(m_tr["win_rate"], 0), pct(m_tr["expectancy"])],
        "Test": [pct(m_te["cagr"]), pct(m_te["max_drawdown"]), round(m_te["sharpe"], 2),
                 m_te["trades"], pct(m_te["win_rate"], 0), pct(m_te["expectancy"])],
    }, index=["CAGR", "Max drawdown", "Sharpe", "จำนวนเทรด", "Win rate", "กำไรเฉลี่ย/เทรด"])
    st.markdown("**เทียบ Train กับ Test ของกลยุทธ์ที่เลือก** (ต่างกันมาก = สัญญาณว่าอาจ overfit)")
    st.dataframe(cmp_df)

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
                                  "days": "วันที่ถือ"})
        st.dataframe(out, hide_index=True)
        st.download_button("ดาวน์โหลด CSV", out.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_{name}_trades.csv", mime="text/csv")

# =====================================================================
# Tab 5: multi-ticker scan
# =====================================================================
with tab_scan:
    st.caption("ดูสัญญาณวันนี้ของหลายตัวพร้อมกัน ด้วยกลยุทธ์ที่เลือกทางซ้าย "
               "(พารามิเตอร์เลือกจากช่วง Train ของแต่ละตัว)")
    raw = st.text_area("รายชื่อหุ้น (คั่นด้วยคอมมา)", "NVDA, AMD, AVGO, MSFT, AAPL")
    if st.button("สแกน", type="primary"):
        rows = []
        prog = st.progress(0.0)
        tks = [t.strip().upper() for t in raw.split(",") if t.strip()]
        for i, tk in enumerate(tks):
            try:
                d_ = get_df(tk, start)
                f_, mb_, _ = get_eval(tk, start, split, cost)
                nm = auto_pick(f_) if strat_choice == "auto" else strat_choice
                ff = f_[nm]
                act, _ = bt.current_action(ff["signal"])
                rows.append({
                    "หุ้น": tk, "กลยุทธ์": LABEL[nm], "สัญญาณ": act,
                    "ราคาปิด": round(float(d_["Close"].iloc[-1]), 2),
                    "วันที่ข้อมูล": d_.index[-1].date(),
                    "Sharpe (Test)": round(ff["test"]["sharpe"], 2),
                    "Sharpe ซื้อถือ": round(mb_["sharpe"], 2),
                    "ชนะซื้อถือ?": "✅" if ff["test"]["sharpe"] > mb_["sharpe"] else "❌",
                    "เทรด (Test)": ff["test"]["trades"]})
            except Exception as e:
                rows.append({"หุ้น": tk, "กลยุทธ์": f"error: {e}"})
            prog.progress((i + 1) / len(tks))
        prog.empty()
        st.session_state["scan"] = pd.DataFrame(rows)
    if "scan" in st.session_state:
        st.dataframe(st.session_state["scan"], hide_index=True)
        st.caption("BUY = สัญญาณซื้อใหม่ · SELL = สัญญาณขาย · HOLD = ถือต่อ · WAIT = รอ")

st.divider()
st.caption("เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน ไม่ได้คิดภาษี และผลในอดีตไม่การันตีอนาคต")
