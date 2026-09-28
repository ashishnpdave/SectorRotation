"""NSE Sector Rotation tool: Relative Rotation Graph (RRG) built with Streamlit.

Educational tool, not investment advice.
Run: streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --------------------------------------------------------------------------
# CONFIG (edit freely)
# --------------------------------------------------------------------------
BENCHMARKS: dict[str, str] = {"Nifty 500": "^CRSLDX", "Nifty 50": "^NSEI"}

SECTOR_INDICES: dict[str, str] = {
    "Bank": "^NSEBANK", "IT": "^CNXIT", "Auto": "^CNXAUTO", "FMCG": "^CNXFMCG",
    "Pharma": "^CNXPHARMA", "Metal": "^CNXMETAL", "Realty": "^CNXREALTY",
    "Energy": "^CNXENERGY", "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "Media": "^CNXMEDIA", "PSU Bank": "^CNXPSUBANK", "Infra": "^CNXINFRA",
}

CONSTITUENTS: dict[str, list[str]] = {
    "Bank": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS"],
    "Auto": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY", "BOSCHLTD", "MRF"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "MARICO", "COLPAL", "TATACONSUM", "VBL"],
    "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ZYDUSLIFE", "ALKEM", "MANKIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "SAIL", "NMDC", "NATIONALUM", "APLAPOLLO", "HINDZINC"],
    "Realty": ["DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "PRESTIGE", "LODHA", "BRIGADE", "SOBHA", "SUNTECK", "MAHLIFE"],
    "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BPCL", "IOC", "GAIL", "TATAPOWER", "ADANIGREEN"],
    "Financial Services": ["HDFCBANK", "ICICIBANK", "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "CHOLAFIN", "SHRIRAMFIN", "MUTHOOTFIN", "PFC"],
    "Media": ["SUNTV", "ZEEL", "PVRINOX", "NETWORK18", "TV18BRDCST", "SAREGAMA", "NAZARA", "DBCORP", "HATHWAY", "TIPSMUSIC"],
    "PSU Bank": ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "IOB", "UCOBANK", "MAHABANK"],
    "Infra": ["LT", "ADANIPORTS", "ULTRACEMCO", "GRASIM", "IRB", "NCC", "KEC", "GMRAIRPORT", "CONCOR", "RVNL"],
}

RETURN_PERIODS: dict[str, int] = {"1W": 5, "1M": 21, "3M": 63, "6M": 126}  # trading days
HISTORY_PERIOD = "5y"
CACHE_TTL_SECONDS = 3600
DEFAULT_WINDOW = 14
DEFAULT_TAIL = 8
MOMENTUM_PERIOD = 4          # periods used for RS-Ratio rate of change
MIN_AXIS_SPAN = 1.0          # minimum half-width of RRG axes
AXIS_PADDING = 1.15
CENTER = 100.0

QUADRANT_COLORS = {"Leading": "#2e9e5b", "Weakening": "#d9a400", "Lagging": "#d64545", "Improving": "#3b7dd8"}
ARROWS = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Download adjusted closes for tickers; returns a DataFrame (columns = tickers)."""
    raw = yf.download(list(tickers), period=HISTORY_PERIOD, auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close = close.dropna(axis=1, how="all").sort_index()
    return close.ffill()  # forward-fill holidays / missing sessions


def fetch(labels_to_tickers: dict[str, str]) -> pd.DataFrame:
    """Fetch prices and rename columns to labels; warn and skip failed tickers."""
    prices = load_prices(tuple(labels_to_tickers.values()))
    out = {}
    for label, tkr in labels_to_tickers.items():
        if tkr in prices.columns and prices[tkr].notna().sum() > DEFAULT_WINDOW * 2:
            out[label] = prices[tkr]
        else:
            st.warning(f"No usable data for {label} ({tkr}); skipped.")
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# CALCULATIONS
# --------------------------------------------------------------------------
def compute_rrg(prices: pd.DataFrame, bench: pd.Series, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (RS-Ratio, RS-Momentum), both centred on 100."""
    rs = prices.div(bench, axis=0)
    ratio = CENTER + (rs - rs.rolling(window).mean()) / rs.rolling(window).std()
    roc = ratio.pct_change(MOMENTUM_PERIOD) * 100
    mom = CENTER + roc / roc.rolling(window).std()
    return ratio, mom


def classify(ratio: pd.Series, mom: pd.Series) -> pd.Series:
    """Vectorised quadrant classification."""
    conds = [(ratio >= CENTER) & (mom >= CENTER), (ratio >= CENTER) & (mom < CENTER),
             (ratio < CENTER) & (mom < CENTER)]
    return pd.Series(np.select(conds, ["Leading", "Weakening", "Lagging"], "Improving"), index=ratio.index)


def direction_arrow(dx: pd.Series, dy: pd.Series) -> pd.Series:
    """Map last movement vector to one of 8 arrows."""
    ang = np.degrees(np.arctan2(dy, dx))
    idx = (np.floor((ang + 22.5) / 45).astype(int)) % 8
    return pd.Series([ARROWS[i] for i in idx], index=dx.index)


def period_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """Trailing returns per period (rows = assets)."""
    last = daily.iloc[-1]
    return pd.DataFrame({k: last / daily.shift(n).iloc[-1] - 1 for k, n in RETURN_PERIODS.items()})


def build_rrg(daily: pd.DataFrame, bench_col: str, timeframe: str, window: int, lookback_m: int):
    """Resample, compute RRG on full history, then trim to the lookback window."""
    tf = daily.resample("W-FRI").last() if timeframe == "Weekly" else daily
    ratio, mom = compute_rrg(tf.drop(columns=bench_col), tf[bench_col], window)
    cutoff = tf.index[-1] - pd.DateOffset(months=lookback_m)
    ratio, mom = ratio.loc[ratio.index >= cutoff].dropna(how="all"), mom.loc[mom.index >= cutoff].dropna(how="all")
    return ratio, mom.reindex(ratio.index)


def summary_table(ratio: pd.DataFrame, mom: pd.DataFrame, tail: int, rets: pd.DataFrame) -> pd.DataFrame:
    """Ranking table with returns, RRG values, quadrant and rotation arrow."""
    r_now, m_now = ratio.ffill().iloc[-1], mom.ffill().iloc[-1]
    r_prev, m_prev = ratio.ffill().iloc[-2], mom.ffill().iloc[-2]
    df = rets.reindex(r_now.index).copy()
    df["RS-Ratio"], df["RS-Momentum"] = r_now, m_now
    df["Quadrant"] = classify(r_now, m_now)
    df["Direction"] = direction_arrow(r_now - r_prev, m_now - m_prev)
    df[f"RS chg ({tail}p)"] = r_now - ratio.ffill().iloc[-min(tail, len(ratio))]
    df = df.sort_values("RS-Ratio", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def style_table(df: pd.DataFrame):
    """Colour quadrants, arrows and returns."""
    pct = list(RETURN_PERIODS)
    sty = df.style.format({**{c: "{:.1%}" for c in pct}, "RS-Ratio": "{:.2f}", "RS-Momentum": "{:.2f}",
                           df.columns[-1]: "{:+.2f}"})
    sty = sty.apply(lambda s: [f"color:{QUADRANT_COLORS[q]};font-weight:600" for q in df["Quadrant"]], subset=["Quadrant"])
    sty = sty.apply(lambda s: [f"color:{QUADRANT_COLORS[q]};font-size:1.3em" for q in df["Quadrant"]], subset=["Direction"])
    return sty.map(lambda v: "color:#2e9e5b" if v > 0 else "color:#d64545", subset=pct)


# --------------------------------------------------------------------------
# CHARTS
# --------------------------------------------------------------------------
def rrg_figure(ratio: pd.DataFrame, mom: pd.DataFrame, tail: int, title: str) -> go.Figure:
    """RRG scatter with quadrant backgrounds and fading tails."""
    r, m = ratio.tail(tail), mom.tail(tail)
    dx = max(np.nanmax(np.abs(r.values - CENTER)), MIN_AXIS_SPAN) * AXIS_PADDING
    dy = max(np.nanmax(np.abs(m.values - CENTER)), MIN_AXIS_SPAN) * AXIS_PADDING
    fig = go.Figure()
    quads = [("Improving", CENTER - dx, CENTER, CENTER, CENTER + dy), ("Leading", CENTER, CENTER + dx, CENTER, CENTER + dy),
             ("Lagging", CENTER - dx, CENTER, CENTER - dy, CENTER), ("Weakening", CENTER, CENTER + dx, CENTER - dy, CENTER)]
    for name, x0, x1, y0, y1 in quads:
        col = QUADRANT_COLORS[name]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=col, opacity=0.12, line_width=0, layer="below")
        fig.add_annotation(x=x0 + 0.03 * dx if x0 < CENTER else x1 - 0.03 * dx, y=y1 - 0.04 * dy if y1 > CENTER else y0 + 0.04 * dy,
                           text=f"<b>{name}</b>", showarrow=False, font=dict(color=col, size=13),
                           xanchor="left" if x0 < CENTER else "right")
    palette = px.colors.qualitative.Dark24
    for i, col in enumerate(r.columns):
        ok = r[col].notna() & m[col].notna()
        x, y = r.loc[ok, col], m.loc[ok, col]
        n = len(x)
        if n == 0:
            continue
        c = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers+text", name=col, line=dict(width=1, color=c),
            marker=dict(size=np.linspace(4, 11, n), opacity=np.linspace(0.2, 1, n), color=c),
            text=[""] * (n - 1) + [col], textposition="top center",
            customdata=x.index.strftime("%d %b %Y"),
            hovertemplate=f"<b>{col}</b><br>%{{customdata}}<br>RS-Ratio %{{x:.2f}}<br>RS-Mom %{{y:.2f}}<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(title="RS-Ratio", range=[CENTER - dx, CENTER + dx], zeroline=False),
                      yaxis=dict(title="RS-Momentum", range=[CENTER - dy, CENTER + dy], zeroline=False),
                      height=620, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    fig.add_vline(x=CENTER, line_width=1, line_color="grey")
    fig.add_hline(y=CENTER, line_width=1, line_color="grey")
    return fig


# --------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="NSE Sector Rotation", layout="wide")
    st.title("NSE Sector Rotation (RRG)")
    st.caption("Educational tool, not investment advice.")

    with st.sidebar:
        bench_name = st.selectbox("Benchmark", list(BENCHMARKS))
        timeframe = st.radio("Timeframe", ["Weekly", "Daily"], horizontal=True)
        window = st.slider("Rolling window (periods)", 5, 40, DEFAULT_WINDOW)
        lookback = st.slider("Lookback (months)", 3, 36, 12)
        tail = st.slider("Tail length", 2, 20, DEFAULT_TAIL)
        sectors = st.multiselect("Sectors", list(SECTOR_INDICES), default=list(SECTOR_INDICES))

    if len(sectors) < 2:
        st.info("Select at least two sectors.")
        return

    try:
        with st.spinner("Downloading sector data..."):
            daily = fetch({"BENCH": BENCHMARKS[bench_name], **{s: SECTOR_INDICES[s] for s in sectors}})
        if "BENCH" not in daily or daily.shape[1] < 2:
            st.error("Benchmark or sector data unavailable. Try again later.")
            return
        ratio, mom = build_rrg(daily, "BENCH", timeframe, window, lookback)
        rets = period_returns(daily.drop(columns="BENCH"))
        table = summary_table(ratio, mom, tail, rets)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build sector view: {exc}")
        return

    t1, t2, t3, t4 = st.tabs(["RRG", "Ranking", "Drill-down", "Returns heatmap"])

    with t1:
        st.plotly_chart(rrg_figure(ratio, mom, tail, f"Sectors vs {bench_name}"), use_container_width=True)
        st.caption("Clockwise rotation: Improving → Leading → Weakening → Lagging. Tail shows recent path; the big dot is the latest.")

    with t2:
        st.dataframe(style_table(table), use_container_width=True)

    with t3:
        sector = st.selectbox("Sector", sectors)
        stocks = {s: f"{s}.NS" for s in CONSTITUENTS.get(sector, [])}
        try:
            with st.spinner(f"Loading {sector} stocks..."):
                sdata = fetch({"SECTOR": SECTOR_INDICES[sector], **stocks})
            if "SECTOR" not in sdata or sdata.shape[1] < 3:
                st.warning("Not enough data for drill-down.")
            else:
                sr, sm = build_rrg(sdata, "SECTOR", timeframe, window, lookback)
                stbl = summary_table(sr, sm, tail, period_returns(sdata.drop(columns="SECTOR")))
                st.plotly_chart(rrg_figure(sr, sm, tail, f"{sector} stocks vs {sector} index"), use_container_width=True)
                chg = stbl.columns[-1]
                c1, c2 = st.columns(2)
                c1.subheader("Top RS gainers")
                c1.dataframe(stbl.sort_values(chg, ascending=False).head(5)[["Quadrant", "Direction", "RS-Ratio", chg]])
                c2.subheader("Top RS laggards")
                c2.dataframe(stbl.sort_values(chg).head(5)[["Quadrant", "Direction", "RS-Ratio", chg]])
                st.dataframe(style_table(stbl), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Drill-down failed: {exc}")

    with t4:
        heat = rets.reindex(table.index) * 100
        fig = px.imshow(heat, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
                        color_continuous_midpoint=0, labels=dict(color="Return %"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Disclaimer: for education only. Not investment advice. Data via Yahoo Finance may be delayed or incomplete.")


if __name__ == "__main__":
    main()
