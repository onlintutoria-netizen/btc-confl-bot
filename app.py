import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BTCUSD Multi-Timeframe Confluence & Backtester",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ BTCUSD Multi-Timeframe Strategy & Backtester")
st.caption("1H Macro • 30M Intraday • 5M Execution | VWAP, Ichimoku, DMI/ADX, Williams %R & MACD")

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
selected_timeframes = st.sidebar.multiselect(
    "Candle Timeframes",
    ["5m", "30m", "1h"],
    default=["5m", "30m", "1h"],
)
if not selected_timeframes:
    st.sidebar.warning("Select at least one candle timeframe.")

if st.sidebar.button("🔄 Refresh Live Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# --- MULTI-SOURCE CLOUD-SAFE DATA FETCHER ---
@st.cache_data(ttl=60)
def fetch_ohlcv(source: str, timeframe: str = "5m", limit: int = 1000) -> pd.DataFrame:
    tf_seconds = {"5m": 300, "30m": 1800, "1h": 3600}

    # 1. Try Delta Exchange Public REST API first if selected
    if "Delta" in source:
        try:
            end_ts = int(datetime.utcnow().timestamp())
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
            pass  # Automatically fall back to CCXT if cloud IP is restricted

    # 2. Cloud-safe CCXT fallback (Kraken / Coinbase work globally without geo-blocks)
    exchange_id = "coinbase" if "Coinbase" in source else "kraken"
    exchange = getattr(ccxt, exchange_id)()
    bars = exchange.fetch_ohlcv("BTC/USD", timeframe=timeframe, limit=min(limit, 720 if exchange_id == "kraken" else 1000))
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.astype(float)

# --- PURE PANDAS/NUMPY INDICATOR ENGINE (NO EXTERNAL TA LIBS NEEDED) ---
def add_indicators(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
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

    # 6. Buy-signal confirmations (all lookbacks cover the last hour)
    hourly_window = {"5m": 12, "30m": 2, "1h": 1}.get(timeframe, 12)
    macd_histogram = df["MACD"] - df["MACD_SIGNAL"]
    df["MACD_HIST"] = macd_histogram

    hourly_macd_min = macd_histogram.rolling(hourly_window, min_periods=hourly_window).min()
    negative_macd = macd_histogram.where(macd_histogram < 0)
    df["MACD_NEG_AVG_1H"] = negative_macd.rolling(hourly_window, min_periods=hourly_window).mean().abs()
    df["MACD_MIN_1H"] = hourly_macd_min

    df["MACD_NEGATIVE_MIN_CROSS"] = (
        (macd_histogram.shift(1) < 0)
        & (macd_histogram > macd_histogram.shift(1))
        & (macd_histogram.shift(1) <= hourly_macd_min.shift(1))
    )

    macd_rise = macd_histogram.diff()
    df["MACD_RISE_ABOVE_AVERAGE"] = (macd_rise > 0) & (macd_rise > df["MACD_NEG_AVG_1H"])

    willr = df["WILLR"]
    hourly_willr_min = willr.rolling(hourly_window, min_periods=hourly_window).min()
    negative_willr = willr.where(willr < 0)
    df["WILLR_NEG_AVG_1H"] = negative_willr.rolling(hourly_window, min_periods=hourly_window).mean().abs()
    df["WILLR_MIN_1H"] = hourly_willr_min

    df["WILLR_NEGATIVE_MIN_CROSS"] = (
        (willr.shift(1) < 0)
        & (willr > willr.shift(1))
        & (willr.shift(1) <= hourly_willr_min.shift(1))
    )
    willr_rise = willr.diff()
    df["WILLR_RISE_ABOVE_AVERAGE"] = willr_rise > df["WILLR_NEG_AVG_1H"].replace(0, np.nan)

    df["MINUS_TO_PLUS_DI_CROSS"] = (
        (df["MINUS_DI"].shift(1) > df["PLUS_DI"].shift(1))
        & (df["PLUS_DI"] > df["MINUS_DI"])
        & (df["PLUS_DI"] > df["PLUS_DI"].shift(1))
        & (df["MINUS_DI"] < df["MINUS_DI"].shift(1))
    )

    df["GREEN_CANDLE"] = df["close"] > df["open"]
    df["VWAP_CROSS_UP"] = (df["close"] > df["VWAP"]) & (df["close"].shift(1) <= df["VWAP"].shift(1))
    df["RED_CANDLE"] = df["close"] < df["open"]
    df["VWAP_CROSS_DOWN"] = (df["close"] < df["VWAP"]) & (df["close"].shift(1) >= df["VWAP"].shift(1))

    macd_positive = macd_histogram.where(macd_histogram > 0)
    macd_max_30 = macd_histogram.rolling(30, min_periods=30).max()
    df["MACD_POS_MAX_30"] = macd_max_30
    df["MACD_POS_AVG_MAX_30"] = macd_positive.rolling(30, min_periods=30).mean().abs()
    macd_decrease = macd_histogram.shift(1) - macd_histogram
    df["MACD_POSITIVE_MAX_CROSS"] = (
        (macd_histogram.shift(1) > 0)
        & (macd_histogram < macd_histogram.shift(1))
        & (macd_histogram.shift(1) >= macd_max_30.shift(1))
    )
    df["MACD_DECREASE_ABOVE_AVERAGE"] = (macd_decrease > 0) & (
        macd_decrease > df["MACD_POS_AVG_MAX_30"]
    )

    willr_positive = willr.where(willr > -20)
    willr_max_30 = willr.rolling(30, min_periods=30).max()
    df["WILLR_POS_MAX_30"] = willr_max_30
    df["WILLR_POS_AVG_MAX_30"] = willr_positive.rolling(30, min_periods=30).mean().abs()
    willr_decrease = willr.shift(1) - willr
    df["WILLR_POSITIVE_MAX_CROSS"] = (
        (willr.shift(1) > -20)
        & (willr < willr.shift(1))
        & (willr.shift(1) >= willr_max_30.shift(1))
    )
    df["WILLR_DECREASE_ABOVE_AVERAGE"] = (willr_decrease > 0) & (
        willr_decrease > df["WILLR_POS_AVG_MAX_30"]
    )

    df["PLUS_TO_MINUS_DI_CROSS"] = (
        (df["PLUS_DI"].shift(1) > df["MINUS_DI"].shift(1))
        & (df["MINUS_DI"] > df["PLUS_DI"])
        & (df["PLUS_DI"] < df["PLUS_DI"].shift(1))
        & (df["MINUS_DI"] > df["MINUS_DI"].shift(1))
    )

    df["MACD_BULLISH_DIVERGENCE"] = (
        (df["low"] < df["low"].shift(3))
        & (df["MACD_HIST"] > df["MACD_HIST"].shift(3))
        & (df["MACD_HIST"] > df["MACD_HIST"].rolling(3, min_periods=3).min().shift(1))
    )
    df["WILLR_BULLISH_DIVERGENCE"] = (
        (df["low"] < df["low"].shift(3))
        & (df["WILLR"] > df["WILLR"].shift(3))
        & (df["WILLR"] > df["WILLR"].rolling(3, min_periods=3).min().shift(1))
    )
    df["LAST_3_CANDLES_DIVERGENCE"] = (
        df["MACD_BULLISH_DIVERGENCE"] & df["WILLR_BULLISH_DIVERGENCE"]
    )

    close_range_3 = df["close"].rolling(3, min_periods=3).max() - df["close"].rolling(3, min_periods=3).min()
    crossover_close_range = close_range_3.shift(3)
    df["LAST_3_CANDLES_CONVERGENCE"] = (
        close_range_3 < crossover_close_range
    ) & (
        df["close"].notna() & df["close"].shift(1).notna() & df["close"].shift(2).notna()
    )

    return df

# --- ALIGN TIMEFRAMES WITHOUT LOOKAHEAD BIAS ---
def align_timeframes(df_5m: pd.DataFrame, df_30m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    df_30m_closed = df_30m.shift(1).add_suffix("_30m").reset_index()
    df_1h_closed = df_1h.shift(1).add_suffix("_1h").reset_index()
    df_base = df_5m.reset_index()

    merged = pd.merge_asof(df_base, df_30m_closed, on="timestamp", direction="backward")
    merged = pd.merge_asof(merged, df_1h_closed, on="timestamp", direction="backward")
    merged.set_index("timestamp", inplace=True)
    return merged.dropna()

def is_buy_setup(row: pd.Series, suffix: str = "") -> bool:
    """Return True when one selected candle timeframe satisfies every buy condition."""
    return (
        bool(row[f"MACD_NEGATIVE_MIN_CROSS{suffix}"] and row[f"MACD_RISE_ABOVE_AVERAGE{suffix}"])
        and bool(row[f"WILLR_NEGATIVE_MIN_CROSS{suffix}"] and row[f"WILLR_RISE_ABOVE_AVERAGE{suffix}"])
        and bool(row[f"MINUS_TO_PLUS_DI_CROSS{suffix}"])
        and bool(row[f"GREEN_CANDLE{suffix}"])
        and bool(row[f"VWAP_CROSS_UP{suffix}"])
        and bool(row[f"LAST_3_CANDLES_DIVERGENCE{suffix}"])
    )

def is_sell_setup(row: pd.Series, suffix: str = "") -> bool:
    """Return True when one selected candle timeframe satisfies every sell condition."""
    return (
        bool(row[f"MACD_POSITIVE_MAX_CROSS{suffix}"] and row[f"MACD_DECREASE_ABOVE_AVERAGE{suffix}"])
        and bool(row[f"WILLR_POSITIVE_MAX_CROSS{suffix}"] and row[f"WILLR_DECREASE_ABOVE_AVERAGE{suffix}"])
        and bool(row[f"PLUS_TO_MINUS_DI_CROSS{suffix}"])
        and bool(row[f"RED_CANDLE{suffix}"])
        and bool(row[f"VWAP_CROSS_DOWN{suffix}"])
        and bool(row[f"LAST_3_CANDLES_CONVERGENCE{suffix}"])
    )

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
            timeframe_setups = {
                "5m": is_buy_setup(row),
                "30m": is_buy_setup(row, "_30m"),
                "1h": is_buy_setup(row, "_1h"),
            }
            sell_timeframe_setups = {
                "5m": is_sell_setup(row),
                "30m": is_sell_setup(row, "_30m"),
                "1h": is_sell_setup(row, "_1h"),
            }
            bull_selected = bool(selected_timeframes) and all(
                timeframe_setups[timeframe] for timeframe in selected_timeframes
            )
            sell_selected = bool(selected_timeframes) and all(
                sell_timeframe_setups[timeframe] for timeframe in selected_timeframes
            )

            atr = row["ATR"]
            entry_price = row["close"]

            if bull_selected and atr > 0:
                sl = entry_price - (1.5 * atr)
                tp = entry_price + (3.0 * atr)
                risk_amount = equity * (risk_pct / 100.0)
                size = risk_amount / (entry_price - sl)
                equity -= entry_price * size * fee_rate
                position = {"side": "LONG", "entry_time": ts, "entry": entry_price, "sl": sl, "tp": tp, "size": size}

            elif sell_selected and atr > 0:
                sl = entry_price + (1.5 * atr)
                tp = entry_price - (3.0 * atr)
                risk_amount = equity * (risk_pct / 100.0)
                size = risk_amount / (sl - entry_price)
                equity -= entry_price * size * fee_rate
                position = {"side": "SHORT", "entry_time": ts, "entry": entry_price, "sl": sl, "tp": tp, "size": size}

        equity_curve.append({"timestamp": ts, "equity": equity})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index("timestamp")

# --- LOAD & PROCESS DATA ---
with st.spinner("Fetching multi-timeframe data..."):
    df_5m = add_indicators(fetch_ohlcv(data_source, "5m", limit=history_limit), "5m")
    df_30m = add_indicators(fetch_ohlcv(data_source, "30m", limit=max(200, history_limit // 5)), "30m")
    df_1h = add_indicators(fetch_ohlcv(data_source, "1h", limit=max(200, history_limit // 10)), "1h")

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

    # Individual condition flags
    c1_vwap_bull, c1_vwap_bear = c1["close"] > c1["VWAP"], c1["close"] < c1["VWAP"]
    c1_adx_ok = c1["ADX"] > 20
    c1_dmi_bull, c1_dmi_bear = c1["PLUS_DI"] > c1["MINUS_DI"], c1["MINUS_DI"] > c1["PLUS_DI"]

    c30_macd_bull, c30_macd_bear = c30["MACD"] > c30["MACD_SIGNAL"], c30["MACD"] < c30["MACD_SIGNAL"]
    c30_dmi_bull, c30_dmi_bear = c30["PLUS_DI"] > c30["MINUS_DI"], c30["MINUS_DI"] > c30["PLUS_DI"]

    c5_willr_bull, c5_willr_bear = c5["WILLR"] < -75, c5["WILLR"] > -25
    c5_tenkan_bull, c5_tenkan_bear = c5["close"] > c5["TENKAN"], c5["close"] < c5["TENKAN"]
    timeframe_setups = {
        "5m": is_buy_setup(c5),
        "30m": is_buy_setup(c30),
        "1h": is_buy_setup(c1),
    }
    sell_timeframe_setups = {
        "5m": is_sell_setup(c5),
        "30m": is_sell_setup(c30),
        "1h": is_sell_setup(c1),
    }
    buy_macd_min_cross = bool(c5["MACD_NEGATIVE_MIN_CROSS"] and c5["MACD_RISE_ABOVE_AVERAGE"])
    buy_willr_min_cross = bool(c5["WILLR_NEGATIVE_MIN_CROSS"] and c5["WILLR_RISE_ABOVE_AVERAGE"])
    buy_dmi_cross = bool(c5["MINUS_TO_PLUS_DI_CROSS"])
    buy_green_vwap = bool(c5["GREEN_CANDLE"] and c5["VWAP_CROSS_UP"])
    buy_last_3_divergence = bool(c5["LAST_3_CANDLES_DIVERGENCE"])
    sell_macd_max_cross = bool(c5["MACD_POSITIVE_MAX_CROSS"] and c5["MACD_DECREASE_ABOVE_AVERAGE"])
    sell_willr_max_cross = bool(c5["WILLR_POSITIVE_MAX_CROSS"] and c5["WILLR_DECREASE_ABOVE_AVERAGE"])
    sell_dmi_cross = bool(c5["PLUS_TO_MINUS_DI_CROSS"])
    sell_red_vwap = bool(c5["RED_CANDLE"] and c5["VWAP_CROSS_DOWN"])
    sell_last_3_convergence = bool(c5["LAST_3_CANDLES_CONVERGENCE"])

    bull_selected = bool(selected_timeframes) and all(
        timeframe_setups[timeframe] for timeframe in selected_timeframes
    )
    sell_selected = bool(selected_timeframes) and all(
        sell_timeframe_setups[timeframe] for timeframe in selected_timeframes
    )

    price, atr = c5["close"], c5["ATR"]
    risk_usd = account_balance * (risk_per_trade_pct / 100.0)

    if bull_selected:
        signal, color = "STRONG BUY (LONG)", "green"
        sl, tp = price - (1.5 * atr), price + (3.0 * atr)
        pos_btc = risk_usd / abs(price - sl)
    elif sell_selected:
        signal, color = "STRONG SELL (SHORT)", "red"
        sl, tp = price + (1.5 * atr), price - (3.0 * atr)
        pos_btc = risk_usd / abs(sl - price)
    else:
        signal, color, sl, tp, pos_btc = "WAIT / NO CONFLUENCE", "orange", None, None, None

    # Top Recommendation Strip
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("BTC Price", f"${price:,.2f}")
    col2.markdown(f"### :{color}[{signal}]")
    col3.metric("Stop Loss (1.5x ATR)", f"${sl:,.2f}" if sl else "—")
    col4.metric("Take Profit (3.0x ATR)", f"${tp:,.2f}" if tp else "—")
    col5.metric("Position Size (Delta Lots)", f"{int(pos_btc * 1000)} Lots ({pos_btc:.3f} BTC)" if pos_btc else "—")

    selected_status = " | ".join(
        f"{timeframe}: {'BUY' if timeframe_setups[timeframe] else ('SELL' if sell_timeframe_setups[timeframe] else 'WAIT')}"
        for timeframe in selected_timeframes
    )
    st.markdown(f"**Selected timeframe status:** {selected_status}" if selected_status else "**Select at least one timeframe in the sidebar.**")

    st.divider()

    # 3-Pane Breakdown with Live Checklists
    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("Left Pane: 5m (Execution)")
        st.write(f"**Close:** ${c5['close']:,.2f} | **Tenkan:** ${c5['TENKAN']:,.2f}")
        st.write(f"**Williams %R:** `{c5['WILLR']:.2f}` | **ATR:** `${c5['ATR']:.2f}`")
        st.write(f"**DMI:** +DI `{c5['PLUS_DI']:.1f}` / -DI `{c5['MINUS_DI']:.1f}`")
        st.markdown(f"- MACD negative-minimum cross: {'🟢' if buy_macd_min_cross else '⚪'}")
        st.markdown(f"- Williams %R negative-minimum cross: {'🟢' if buy_willr_min_cross else '⚪'}")
        st.markdown(f"- -DI / +DI divergence cross: {'🟢' if buy_dmi_cross else '⚪'}")
        st.markdown(f"- Green candle / VWAP cross: {'🟢' if buy_green_vwap else '⚪'}")
        st.markdown(f"- 3-candle bullish divergence: {'🟢' if buy_last_3_divergence else '⚪'}")
        st.markdown(f"- MACD positive-maximum cross: {'🔴' if sell_macd_max_cross else '⚪'}")
        st.markdown(f"- Williams %R positive-maximum cross: {'🔴' if sell_willr_max_cross else '⚪'}")
        st.markdown(f"- +DI / -DI divergence cross: {'🔴' if sell_dmi_cross else '⚪'}")
        st.markdown(f"- Red candle / VWAP cross: {'🔴' if sell_red_vwap else '⚪'}")
        st.markdown(f"- 3-candle close convergence: {'🔴' if sell_last_3_convergence else '⚪'}")

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

    # Execution Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_5m.index[-80:], open=df_5m["open"][-80:], high=df_5m["high"][-80:],
        low=df_5m["low"][-80:], close=df_5m["close"][-80:], name="BTC 5m"
    ))
    fig.add_trace(go.Scatter(x=df_5m.index[-80:], y=df_5m["VWAP"][-80:], line=dict(color="cyan", width=1.5), name="VWAP"))
    fig.add_trace(go.Scatter(x=df_5m.index[-80:], y=df_5m["TENKAN"][-80:], line=dict(color="orange", width=1.2), name="Ichimoku Tenkan"))
    fig.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False, title="5-Minute Execution Chart")
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
    Execute only when **every selected timeframe** shows the complete BUY or SELL setup. If even one selected timeframe disagrees, stay flat.

    | Signal | 1H Macro Bias (Filter) | 30M Intraday (Confirmation) | 5M Execution (Trigger) |
    | :--- | :--- | :--- | :--- |
    | **🟢 STRONG BUY (LONG)** | • Price **above** VWAP<br>• **+DI > -DI**<br>• **ADX > 20** | • **MACD > Signal** line<br>• **+DI > -DI** | • MACD reaches a negative minimum and crosses upward, increasing above its 1-hour average negative minimum<br>• Williams %R reaches a negative minimum and crosses upward, increasing above its 1-hour average minimum<br>• **-DI crosses below +DI** with both DMI values diverging<br>• Current 5m candle is green and crosses above VWAP<br>• The last three candles show bullish divergence versus the crossover |
    | **🔴 STRONG SELL (SHORT)** | • Price **below** VWAP<br>• **-DI > +DI**<br>• **ADX > 20** | • **MACD < Signal** line<br>• **-DI > +DI** | • MACD reaches a positive maximum and crosses downward, decreasing below its 30-candle average maximum<br>• Williams %R reaches a positive maximum and crosses downward, decreasing below its 30-candle average maximum<br>• **+DI crosses below -DI** with both DMI values diverging<br>• Current candle is red and crosses below VWAP<br>• The last three closed values converge versus the crossover value |
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
