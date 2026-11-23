import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timezone

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BTCUSD Multi-Timeframe Confluence & Backtester",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Triggers a rerun every 30 seconds
refresh_count = st_autorefresh(interval=30000, limit=None, key="btcusd_30s_refresh")

st.caption(f"Auto-refresh active (30s) • Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} • Cycle #{refresh_count}")
# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Strategy & Risk Settings")
data_source = st.sidebar.selectbox(
    "Market Data Feed",
    ["Delta Exchange (BTCUSD Perp)", "Kraken Spot (BTC/USD)", "Coinbase Spot (BTC/USD)"],
    help="Delta Exchange pulls BTCUSD Perpetual directly; Kraken/Coinbase serve as cloud-friendly fallbacks."
)
account_balance = st.sidebar.number_input("Account Balance ($)", min_value=100, value=10000, step=500)
risk_per_trade_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
fee_pct = st.sidebar.number_input("Exchange Fee per Side (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
history_limit = st.sidebar.slider("Historical Candles (5m)", min_value=300, max_value=1500, value=1000, step=100)

if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
# --- CUSTOM TIMEFRAME CONFIGURATION & RESAMPLER ---
CUSTOM_INTERVALS = {
    "30 sec":   {"base_tf": "1m",  "rule": "30s",  "days_back": 1},
    "1 min":    {"base_tf": "1m",  "rule": None,   "days_back": 1},
    "3 mins":   {"base_tf": "1m",  "rule": "3min", "days_back": 2},
    "5 mins":   {"base_tf": "5m",  "rule": None,   "days_back": 4},
    "10 mins":  {"base_tf": "5m",  "rule": "10min","days_back": 7},
    "15 mins":  {"base_tf": "15m", "rule": None,   "days_back": 10},
    "30 mins":  {"base_tf": "30m", "rule": None,   "days_back": 15},
    "1 hr":     {"base_tf": "1h",  "rule": None,   "days_back": 30},
    "2 hrs":    {"base_tf": "1h",  "rule": "2h",   "days_back": 60},
    "3 hrs":    {"base_tf": "1h",  "rule": "3h",   "days_back": 90},
    "4 hrs":    {"base_tf": "4h",  "rule": None,   "days_back": 120},
    "1 day":    {"base_tf": "1d",  "rule": None,   "days_back": 365},
    "2 days":   {"base_tf": "1d",  "rule": "2D",   "days_back": 730},
    "3 days":   {"base_tf": "1d",  "rule": "3D",   "days_back": 1000},
    "5 days":   {"base_tf": "1d",  "rule": "5D",   "days_back": 1500},
    "10 days":  {"base_tf": "1d",  "rule": "10D",  "days_back": 2000},
    "1 month":  {"base_tf": "1d",  "rule": "1ME",  "days_back": 2500},
    "2 months": {"base_tf": "1d",  "rule": "2ME",  "days_back": 3000},
    "3 months": {"base_tf": "1d",  "rule": "3ME",  "days_back": 3650},
}

def fetch_custom_interval(source: str, label: str) -> pd.DataFrame:
    cfg = CUSTOM_INTERVALS[label]
    base_tf = cfg["base_tf"]
    rule = cfg["rule"]

    # Special handling for 30 sec using recent exchange trades (via CCXT)
    if label == "30 sec":
        try:
            ex_id = "coinbase" if "Coinbase" in source else "kraken"
            ex = getattr(ccxt, ex_id)()
            trades = ex.fetch_trades("BTC/USD", limit=1000)
            t_df = pd.DataFrame(trades)[["timestamp", "price", "amount"]]
            t_df["timestamp"] = pd.to_datetime(t_df["timestamp"], unit="ms")
            t_df.set_index("timestamp", inplace=True)
            ohlc = t_df["price"].resample("30s").ohlc()
            vol = t_df["amount"].resample("30s").sum().rename("volume")
            df_30s = pd.concat([ohlc, vol], axis=1).dropna()
            if len(df_30s) >= 20:
                return add_indicators(df_30s)
        except Exception:
            pass  # Falls back to 1m if trade history is too short

    # Update tf_seconds in fetch_ohlcv to support 1m, 15m, 4h, 1d
    df = fetch_ohlcv(source, base_tf, limit=1000)
    if rule and not df.empty:
        df = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

    return add_indicators(df)
# --- MULTI-SOURCE CLOUD-SAFE DATA FETCHER ---
@st.cache_data(ttl=20)
def fetch_ohlcv(source: str, timeframe: str = "5m", limit: int = 1000) -> pd.DataFrame:
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}

    # 1. Try Delta Exchange Public REST API first if selected
    if "Delta" in source:
        try:
            end_ts = int(datetime.now(timezone.utc).timestamp())
            start_ts = end_ts - (limit * tf_seconds[timeframe])
            url = "https://api.india.delta.exchange/v2/history/candles"
            params = {
                "resolution": timeframe,
                "symbol": "BTCUSD",
                "start": start_ts,
                "end": end_ts
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if data.get("success") and data.get("result"):
                df = pd.DataFrame(data["result"])
                df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                df = df.sort_values("timestamp").set_index("timestamp")
                df = df[["open", "high", "low", "close", "volume"]].astype(float)
                if len(df) > 50:
                    return df
        except Exception:
            pass

    # 2. Cloud-safe CCXT fallback (Kraken / Coinbase work globally without geo-blocks)
    exchange_id = "coinbase" if "Coinbase" in source else "kraken"
    exchange = getattr(ccxt, exchange_id)()
    bars = exchange.fetch_ohlcv(
        "BTC/USD",
        timeframe=timeframe,
        limit=min(limit, 720 if exchange_id == "kraken" else 1000)
    )
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.astype(float)

# --- PURE PANDAS/NUMPY INDICATOR ENGINE ---
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]

    # 1. Session VWAP (resets daily)
    tp = (high + low + close) / 3.0
    date_group = df.index.date
    cum_vol = vol.groupby(date_group).cumsum().replace(0, np.nan)
    df["VWAP"] = ((tp * vol).groupby(date_group).cumsum() / cum_vol).fillna(close)

    # 2. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 3. ATR & DMI / ADX (14, Wilder's smoothing)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_safe = df["ATR"].replace(0, np.nan)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean() / atr_safe)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)

    df["PLUS_DI"] = plus_di.fillna(0)
    df["MINUS_DI"] = minus_di.fillna(0)
    df["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)

    # 4. Williams %R (14)
    highest_high = high.rolling(14).max()
    lowest_low = low.rolling(14).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    df["WILLR"] = (-100 * ((highest_high - close) / denom)).fillna(-50)

    # 5. Ichimoku Tenkan (9) & Kijun (26)
    df["TENKAN"] = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    df["KIJUN"] = (high.rolling(26).max() + low.rolling(26).min()) / 2.0

    return df

# --- ALIGN TIMEFRAMES WITHOUT LOOKAHEAD BIAS ---
def align_timeframes(df_5m: pd.DataFrame, df_30m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_5m.empty or df_30m.empty or df_1h.empty:
        return pd.DataFrame()

    df_5m_clean = df_5m.copy()
    df_30m_clean = df_30m.shift(1).add_suffix("_30m")
    df_1h_clean = df_1h.shift(1).add_suffix("_1h")

    df_5m_clean.index = pd.to_datetime(df_5m_clean.index)
    df_30m_clean.index = pd.to_datetime(df_30m_clean.index)
    df_1h_clean.index = pd.to_datetime(df_1h_clean.index)

    merged = pd.merge_asof(
        df_5m_clean.sort_index().reset_index(),
        df_30m_clean.sort_index().reset_index(),
        on="timestamp",
        direction="backward"
    )
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        df_1h_clean.sort_index().reset_index(),
        on="timestamp",
        direction="backward"
    )
    merged.set_index("timestamp", inplace=True)

    required_cols = [
        "open", "high", "low", "close", "ATR", "WILLR", "TENKAN",
        "close_30m", "MACD_30m", "MACD_SIGNAL_30m", "PLUS_DI_30m", "MINUS_DI_30m",
        "close_1h", "VWAP_1h", "ADX_1h", "PLUS_DI_1h", "MINUS_DI_1h"
    ]
    existing_cols = [c for c in required_cols if c in merged.columns]
    return merged.dropna(subset=existing_cols)

# --- BACKTESTING ENGINE ---
def run_backtest(merged_df: pd.DataFrame, capital: float, risk_pct: float, fee_rate: float):
    equity = capital
    equity_curve = []
    trades = []
    position = None

    for ts, row in merged_df.iterrows():
        if position is not None:
            if position["side"] == "LONG":
                if row["low"] <= position["sl"]:
                    exit_price = position["sl"]
                    pnl = (exit_price - position["entry"]) * position["size"] - (exit_price * position["size"] * fee_rate)
                    equity += pnl
                    trades.append({**position, "exit_time": ts, "exit_price": exit_price, "pnl": pnl, "result": "LOSS"})
                    position = None
                elif row["high"] >= position["tp"]:
                    exit_price = position["tp"]
                    pnl = (exit_price - position["entry"]) * position["size"] - (exit_price * position["size"] * fee_rate)
                    equity += pnl
                    trades.append({**position, "exit_time": ts, "exit_price": exit_price, "pnl": pnl, "result": "WIN"})
                    position = None

            elif position["side"] == "SHORT":
                if row["high"] >= position["sl"]:
                    exit_price = position["sl"]
                    pnl = (position["entry"] - exit_price) * position["size"] - (exit_price * position["size"] * fee_rate)
                    equity += pnl
                    trades.append({**position, "exit_time": ts, "exit_price": exit_price, "pnl": pnl, "result": "LOSS"})
                    position = None
                elif row["low"] <= position["tp"]:
                    exit_price = position["tp"]
                    pnl = (position["entry"] - exit_price) * position["size"] - (exit_price * position["size"] * fee_rate)
                    equity += pnl
                    trades.append({**position, "exit_time": ts, "exit_price": exit_price, "pnl": pnl, "result": "WIN"})
                    position = None

        if position is None:
            bull_1h = (row["close_1h"] > row["VWAP_1h"]) and (row["ADX_1h"] > 20) and (row["PLUS_DI_1h"] > row["MINUS_DI_1h"])
            bear_1h = (row["close_1h"] < row["VWAP_1h"]) and (row["ADX_1h"] > 20) and (row["MINUS_DI_1h"] > row["PLUS_DI_1h"])

            bull_30m = (row["MACD_30m"] > row["MACD_SIGNAL_30m"]) and (row["PLUS_DI_30m"] > row["MINUS_DI_30m"])
            bear_30m = (row["MACD_30m"] < row["MACD_SIGNAL_30m"]) and (row["MINUS_DI_30m"] > row["PLUS_DI_30m"])

            bull_5m = (row["WILLR"] < -75) and (row["close"] > row["TENKAN"])
            bear_5m = (row["WILLR"] > -25) and (row["close"] < row["TENKAN"])

            atr = row["ATR"]
            entry_price = row["close"]

            if bull_1h and bull_30m and bull_5m and atr > 0:
                sl = entry_price - (1.5 * atr)
                tp = entry_price + (3.0 * atr)
                risk_amount = equity * (risk_pct / 100.0)
                size = risk_amount / (entry_price - sl)
                equity -= entry_price * size * fee_rate
                position = {"side": "LONG", "entry_time": ts, "entry": entry_price, "sl": sl, "tp": tp, "size": size}

            elif bear_1h and bear_30m and bear_5m and atr > 0:
                sl = entry_price + (1.5 * atr)
                tp = entry_price - (3.0 * atr)
                risk_amount = equity * (risk_pct / 100.0)
                size = risk_amount / (sl - entry_price)
                equity -= entry_price * size * fee_rate
                position = {"side": "SHORT", "entry_time": ts, "entry": entry_price, "sl": sl, "tp": tp, "size": size}

        equity_curve.append({"timestamp": ts, "equity": equity})

    if not equity_curve:
        empty_eq = pd.DataFrame(columns=["equity"])
        empty_eq.index.name = "timestamp"
        return pd.DataFrame(trades), empty_eq

    return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index("timestamp")

# --- LOAD & PROCESS DATA ---
with st.spinner("Fetching multi-timeframe data..."):
    df_5m = add_indicators(fetch_ohlcv(data_source, "5m", limit=history_limit))
    df_30m = add_indicators(fetch_ohlcv(data_source, "30m", limit=max(200, history_limit // 5)))
    df_1h = add_indicators(fetch_ohlcv(data_source, "1h", limit=max(200, history_limit // 10)))

# --- MAIN NAVIGATION TABS ---
tab_live, tab_backtest, tab_guide = st.tabs([
    "📡 Live Multi-Timeframe Scanner",
    "📊 Historical Backtest",
    "📘 Buy / Sell / Wait Guide"
])

# ==========================================
# TAB 1: LIVE SCANNER
# ==========================================
with tab_live:
    c5, c30, c1 = df_5m.iloc[-1], df_30m.iloc[-1], df_1h.iloc[-1]

    c1_vwap_bull, c1_vwap_bear = c1["close"] > c1["VWAP"], c1["close"] < c1["VWAP"]
    c1_adx_ok = c1["ADX"] > 20
    c1_dmi_bull, c1_dmi_bear = c1["PLUS_DI"] > c1["MINUS_DI"], c1["MINUS_DI"] > c1["PLUS_DI"]

    c30_macd_bull, c30_macd_bear = c30["MACD"] > c30["MACD_SIGNAL"], c30["MACD"] < c30["MACD_SIGNAL"]
    c30_dmi_bull, c30_dmi_bear = c30["PLUS_DI"] > c30["MINUS_DI"], c30["MINUS_DI"] > c30["PLUS_DI"]

    c5_willr_bull, c5_willr_bear = c5["WILLR"] < -75, c5["WILLR"] > -25
    c5_tenkan_bull, c5_tenkan_bear = c5["close"] > c5["TENKAN"], c5["close"] < c5["TENKAN"]

    bull_1h = c1_vwap_bull and c1_adx_ok and c1_dmi_bull
    bear_1h = c1_vwap_bear and c1_adx_ok and c1_dmi_bear
    bull_30m = c30_macd_bull and c30_dmi_bull
    bear_30m = c30_macd_bear and c30_dmi_bear
    bull_5m = c5_willr_bull and c5_tenkan_bull
    bear_5m = c5_willr_bear and c5_tenkan_bear

    price, atr = c5["close"], c5["ATR"]
    risk_usd = account_balance * (risk_per_trade_pct / 100.0)

    if bull_1h and bull_30m and bull_5m:
        signal, color = "STRONG BUY (LONG)", "green"
        sl, tp = price - (1.5 * atr), price + (3.0 * atr)
        pos_btc = risk_usd / abs(price - sl)
    elif bear_1h and bear_30m and bear_5m:
        signal, color = "STRONG SELL (SHORT)", "red"
        sl, tp = price + (1.5 * atr), price - (3.0 * atr)
        pos_btc = risk_usd / abs(sl - price)
    else:
        signal, color, sl, tp, pos_btc = "WAIT / NO CONFLUENCE", "orange", None, None, None

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("BTC Price", f"${price:,.2f}")
    col2.markdown(f"### :{color}[{signal}]")
    col3.metric("Stop Loss (1.5x ATR)", f"${sl:,.2f}" if sl else "—")
    col4.metric("Take Profit (3.0x ATR)", f"${tp:,.2f}" if tp else "—")
    col5.metric("Position Size (Delta Lots)", f"{int(pos_btc * 1000)} Lots ({pos_btc:.3f} BTC)" if pos_btc else "—")

    st.divider()
st.subheader("⏱️ Multi-Interval Technical Matrix (Williams %R, DMI/ADX, MACD)")

selected_intervals = st.multiselect(
    "Choose Time Intervals to Analyze",
    options=list(CUSTOM_INTERVALS.keys()),
    default=["1 min", "5 mins", "15 mins", "1 hr", "4 hrs", "1 day", "1 month"]
)

if selected_intervals:
    matrix_rows = []
    for tf_label in selected_intervals:
        tf_df = fetch_custom_interval(data_source, tf_label)
        if tf_df.empty:
            continue
        last = tf_df.iloc[-1]

        willr = last["WILLR"]
        willr_state = "🟢 Oversold" if willr < -75 else ("🔴 Overbought" if willr > -25 else "⚪ Neutral")
        dmi_state = "🟢 +DI > -DI" if last["PLUS_DI"] > last["MINUS_DI"] else "🔴 -DI > +DI"
        macd_state = "🟢 Bullish" if last["MACD"] > last["MACD_SIGNAL"] else "🔴 Bearish"

        matrix_rows.append({
            "Interval": tf_label,
            "Close ($)": f"${last['close']:,.2f}",
            "Williams %R": f"{willr:.2f} ({willr_state})",
            "+DI": f"{last['PLUS_DI']:.1f}",
            "-DI": f"{last['MINUS_DI']:.1f}",
            "ADX": f"{last['ADX']:.1f} ({dmi_state})",
            "MACD": f"{last['MACD']:.1f}",
            "MACD Signal": f"{last['MACD_SIGNAL']:.1f} ({macd_state})",
        })

    if matrix_rows:
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("Left Pane: 5m (Execution)")
        st.write(f"**Close:** ${c5['close']:,.2f} | **Tenkan:** ${c5['TENKAN']:,.2f}")
        st.write(f"**Williams %R:** `{c5['WILLR']:.2f}` | **ATR:** `${c5['ATR']:.2f}`")
        st.markdown(f"- Williams %R Trigger: {'🟢 Oversold (< -75)' if c5_willr_bull else ('🔴 Overbought (> -25)' if c5_willr_bear else '⚪ Neutral (-25 to -75)')}")
        st.markdown(f"- Ichimoku Tenkan Cross: {'🟢 Above Tenkan' if c5_tenkan_bull else '🔴 Below Tenkan'}")

    with p2:
        st.subheader("Middle Pane: 30m (Intraday)")
        st.write(f"**Close:** ${c30['close']:,.2f} | **VWAP:** ${c30['VWAP']:,.2f}")
        st.write(f"**DMI:** +DI `{c30['PLUS_DI']:.1f}` / -DI `{c30['MINUS_DI']:.1f}` | **MACD:** `{c30['MACD']:.1f}`")
        st.markdown(f"- MACD Momentum: {'🟢 Bullish (> Signal)' if c30_macd_bull else '🔴 Bearish (< Signal)'}")
        st.markdown(f"- DMI Trend Direction: {'🟢 +DI > -DI' if c30_dmi_bull else '🔴 -DI > +DI'}")

    with p3:
        st.subheader("Right Pane: 1h (Macro Bias)")
        st.write(f"**Close:** ${c1['close']:,.2f} | **VWAP:** ${c1['VWAP']:,.2f}")
        st.write(f"**DMI:** +DI `{c1['PLUS_DI']:.1f}` / -DI `{c1['MINUS_DI']:.1f}` (ADX: `{c1['ADX']:.1f}`)")
        st.markdown(f"- VWAP Position: {'🟢 Above VWAP' if c1_vwap_bull else '🔴 Below VWAP'}")
        adx_val = c1["ADX"]
        st.markdown(f"- Trend Strength (ADX > 20): {'✅ Active' if c1_adx_ok else '⚠️ Choppy'} (`{adx_val:.1f}`)")

    plot_df = df_5m.iloc[-80:]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            "5-Minute Execution Chart (Price, VWAP & Tenkan)",
            "Williams %R (14)",
            "DMI (+DI / -DI) & ADX (14)"
        )
    )

    # --- ROW 1: CANDLESTICKS + VWAP + TENKAN ---
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="BTC 5m"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["VWAP"], line=dict(color="cyan", width=1.5), name="VWAP"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["TENKAN"], line=dict(color="orange", width=1.2), name="Ichimoku Tenkan"),
        row=1, col=1
    )

    # --- ROW 2: WILLIAMS %R ---
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["WILLR"], line=dict(color="#bb86fc", width=1.6), name="Williams %R"),
        row=2, col=1
    )
    fig.add_hline(y=-25, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
    fig.add_hline(y=-75, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
    fig.update_yaxes(range=[-105, 5], title_text="Will %R", row=2, col=1)

    # --- ROW 3: +DI, -DI, ADX ---
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["PLUS_DI"], line=dict(color="#00e676", width=1.5), name="+DI"),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["MINUS_DI"], line=dict(color="#ff5252", width=1.5), name="-DI"),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["ADX"], line=dict(color="#ffd740", width=1.8), name="ADX"),
        row=3, col=1
    )
    fig.add_hline(y=20, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)
    fig.update_yaxes(title_text="DMI / ADX", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=760,
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=20, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
# ==========================================
# TAB 2: HISTORICAL BACKTEST
# ==========================================
with tab_backtest:
    merged_df = align_timeframes(df_5m, df_30m, df_1h)
    trades_df, eq_df = run_backtest(merged_df, account_balance, risk_per_trade_pct, fee_pct / 100.0)

    if trades_df.empty:
        st.warning("No trades triggered in this historical window. Try increasing the Historical Candles slider in the sidebar.")
    else:
        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        win_rate = (len(wins) / total_trades) * 100
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["drawdown_pct"] = ((eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]) * 100
        max_drawdown = eq_df["drawdown_pct"].min()
        net_return_pct = ((eq_df["equity"].iloc[-1] - account_balance) / account_balance) * 100

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Net Return", f"{net_return_pct:+.2f}%", f"${eq_df['equity'].iloc[-1] - account_balance:,.2f}")
        m2.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
        m3.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != np.inf else "∞")
        m4.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        m5.metric("Total Trades", f"{total_trades}")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_df.index, y=eq_df["equity"], mode="lines", name="Equity", line=dict(color="#00ffcc", width=2)))
        fig_eq.update_layout(template="plotly_dark", height=360, title="Simulated Equity Curve ($)", yaxis_title="Balance ($)")
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Executed Trade Log")
        display_trades = trades_df[["side", "entry_time", "exit_time", "entry", "exit_price", "pnl", "result"]].copy()
        display_trades["pnl"] = display_trades["pnl"].map(lambda x: f"${x:,.2f}")
        st.dataframe(display_trades, use_container_width=True)

# ==========================================
# TAB 3: BUY / SELL / WAIT EXECUTION GUIDE
# ==========================================
with tab_guide:
    st.subheader("📋 Signal Decision Matrix")
    st.markdown("""
    Only execute a trade when **all three timeframes** align. If even one filter disagrees, stay flat.

    | Signal | 1H Macro Bias (Filter) | 30M Intraday (Confirmation) | 5M Execution (Trigger) |
    | :--- | :--- | :--- | :--- |
    | **🟢 STRONG BUY (LONG)** | • Price **above** VWAP<br>• **+DI > -DI**<br>• **ADX > 20** | • **MACD > Signal** line<br>• **+DI > -DI** | • **Williams %R < -75** (oversold pullback)<br>• Price closes **above** Ichimoku Tenkan |
    | **🔴 STRONG SELL (SHORT)** | • Price **below** VWAP<br>• **-DI > +DI**<br>• **ADX > 20** | • **MACD < Signal** line<br>• **-DI > +DI** | • **Williams %R > -25** (overbought rally)<br>• Price closes **below** Ichimoku Tenkan |
    | **🟠 WAIT (NO TRADE)** | • Price chopping around VWAP<br>• **ADX < 20** (sideways chop) | • MACD & DMI conflict with 1H trend | • Williams %R between `-25` and `-75`<br>• No Tenkan crossover |
    """)

    st.divider()
    st.subheader("🛡️ Delta Exchange Execution & Risk Rules")
    st.markdown("""
    1. **Wait for 5M Candle Close:** Confirm the signal persists after the current 5-minute candle finishes closing before clicking Long or Short on Delta Exchange.
    2. **Attach Stop-Loss Immediately (`1.5x ATR`):** Enter the exact **Stop Loss** price shown on the Live Scanner tab. Never move your stop further away.
    3. **Set Take-Profit (`3.0x ATR`):** Place your limit Take-Profit target at a strict **1:2 Risk-to-Reward** ratio.
    4. **Respect the "Wait" Discipline:** High-confluence setups occur selectively. Never short an oversold market (`Williams %R < -80` on 5m)—wait for a relief bounce into resistance first.
    5. **Daily Circuit Breaker:** Stop trading for the day if you hit **3 consecutive Stop-Losses**.
    """)
