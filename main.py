import ccxt
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ============================================================
# TELEGRAM CONFIGURATION
# ============================================================
BOT_TOKEN = "8809200223:AAHmR969mpLw_jEuH2iVeLJ1RcGY8DeosXA"
CHAT_ID = "503404993"

def send_telegram_alert(message):
    """Send a Telegram alert."""
    if not BOT_TOKEN or not CHAT_ID:
        print("⚠️ Telegram alert skipped: credentials not configured.")
        return False

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": False,
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        print(f"❌ Telegram error: {response.text}")
        return False
    except Exception as exc:
        print(f"❌ Telegram error: {exc}")
        return False

# ============================================================
# SCANNER SETTINGS
# ============================================================
PRIORITY_COINS = [
    "BTC", "ETH", "SOL", "XRP", "AAVE", "XAUT", 
    "SPCXX", "GOOGLX", "TSLAX", "AMZNX", "NVDAX", "AAPLX", "METAX"
]

MACRO_TIMEFRAME = "4h"
ENTRY_TIMEFRAME = "15m"

# EMA Settings
EMA_FAST = 9
EMA_SLOW = 20
EMA_TREND = 200

# Filters
MIN_24H_VOLUME_USD = 200_000
USE_CLOSED_CANDLES = True

# ============================================================
# HELPERS
# ============================================================
def format_volume(volume):
    if volume is None: return "0.00"
    if volume >= 1_000_000_000: return f"{volume / 1_000_000_000:.2f}B"
    if volume >= 1_000_000: return f"{volume / 1_000_000:.2f}M"
    if volume >= 1_000: return f"{volume / 1_000:.2f}K"
    return f"{volume:.2f}"

def format_price(value):
    if value is None or pd.isna(value): return "N/A"
    value = float(value)
    if abs(value) >= 1000: return f"{value:,.2f}"
    if abs(value) >= 1: return f"{value:.4f}"
    if abs(value) >= 0.01: return f"{value:.6f}"
    return f"{value:.8f}"

def calculate_ema(df, period):
    return df["close"].ewm(span=period, adjust=False).mean()

def fetch_all_delta_india_tickers(exchange):
    raw_markets = exchange.fetch_markets()
    tradeable_pairs = []
    for market in raw_markets:
        is_derivative = market.get("swap", False) or market.get("future", False) or market.get("linear", False)
        is_active = market.get("active", True)
        is_option = market.get("option", False) or market.get("type") == "option"

        if is_derivative and is_active and not is_option:
            symbol = market.get("symbol")
            raw_id = market.get("id", "")
            clean_name = raw_id.replace("_", "").replace("-", "").replace("/", "").replace(":", "")
            item = (symbol, clean_name)
            if item not in tradeable_pairs:
                tradeable_pairs.append(item)
    return tradeable_pairs

def fetch_ohlcv_for_tf(symbol, exchange, timeframe, limit):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 50: return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        if USE_CLOSED_CANDLES and len(df) > 2:
            return df.iloc[:-1].copy()
        return df
    except Exception as exc:
        print(f"⚠️ Error fetching {timeframe} for {symbol}: {exc}")
        return None

# ============================================================
# STRATEGY ENGINE
# ============================================================
def scan_symbol(symbol, exchange, vol_24h):
    """Evaluates 4H Trend and 15m EMA Crossover Entry"""
    
    # 1. Macro Trend Filter (4H)
    df_4h = fetch_ohlcv_for_tf(symbol, exchange, MACRO_TIMEFRAME, 250)
    if df_4h is None or len(df_4h) < 205:
        return None

    df_4h['ema200'] = calculate_ema(df_4h, EMA_TREND)
    curr_4h = df_4h.iloc[-1]
    prev_4h = df_4h.iloc[-2]

    # Rule: Price must be on the right side of the EMA, and EMA must be sloping in that direction.
    bullish_trend = (curr_4h['close'] > curr_4h['ema200']) and (curr_4h['ema200'] > prev_4h['ema200'])
    bearish_trend = (curr_4h['close'] < curr_4h['ema200']) and (curr_4h['ema200'] < prev_4h['ema200'])

    if not bullish_trend and not bearish_trend:
        # Flat 200 EMA or mixed signals = Do Not Trade
        return None

    direction = "LONG" if bullish_trend else "SHORT"

    # 2. Entry Setup Phase (15m)
    df_15m = fetch_ohlcv_for_tf(symbol, exchange, ENTRY_TIMEFRAME, 100)
    if df_15m is None:
        return None

    df_15m['ema9'] = calculate_ema(df_15m, EMA_FAST)
    df_15m['ema20'] = calculate_ema(df_15m, EMA_SLOW)
    df_15m['ema200'] = calculate_ema(df_15m, EMA_TREND)
    df_15m['vol_ma'] = df_15m['volume'].rolling(20).mean()

    curr_15m = df_15m.iloc[-1]
    prev_15m = df_15m.iloc[-2]

    entry_triggered = False
    
    # Execution Rule: 9 EMA crosses 20 EMA in direction of trend, confirmed by volume
    if direction == "LONG":
        bullish_cross = (prev_15m['ema9'] <= prev_15m['ema20']) and (curr_15m['ema9'] > curr_15m['ema20'])
        vol_spike = curr_15m['volume'] > curr_15m['vol_ma']
        if bullish_cross and vol_spike:
            entry_triggered = True
    else:
        bearish_cross = (prev_15m['ema9'] >= prev_15m['ema20']) and (curr_15m['ema9'] < curr_15m['ema20'])
        vol_spike = curr_15m['volume'] > curr_15m['vol_ma']
        if bearish_cross and vol_spike:
            entry_triggered = True

    if not entry_triggered:
        return None

    # 3. Risk Management
    entry_price = float(curr_15m['close'])
    
    # Place stop just below/above the last 5 candles (swing low/high of the pullback)
    if direction == "LONG":
        stop_loss = float(df_15m['low'].tail(5).min())
    else:
        stop_loss = float(df_15m['high'].tail(5).max())

    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return None

    risk_pct = (risk / entry_price) * 100
    
    # Targets based on Risk Multiple
    target_1_5r = entry_price + (risk * 1.5) if direction == "LONG" else entry_price - (risk * 1.5)
    target_runner = entry_price + (risk * 3) if direction == "LONG" else entry_price - (risk * 3)

    return {
        "symbol": symbol,
        "direction": direction,
        "entry": entry_price,
        "stop_loss": stop_loss,
        "risk_pct": risk_pct,
        "target_1_5r": target_1_5r,
        "target_runner": target_runner,
        "vol_24h": vol_24h,
        "ema_4h_200": curr_4h['ema200'],
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ============================================================
# ALERT BUILDER
# ============================================================
def build_alert(setup, app_ticker):
    direction_icon = "🟢 LONG" if setup['direction'] == "LONG" else "🔴 SHORT"

    alert = (
        f"🚨 <b>MULTI-TF EMA STRATEGY ENTRY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>{app_ticker}</b>\n"
        f"📈 <b>Direction:</b> {direction_icon}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        
        f"🌊 <b>MACRO TREND (4H)</b>\n"
        f"   4H 200 EMA Filter Passed\n"
        f"   200 EMA Value: {format_price(setup['ema_4h_200'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"⚡ <b>ENTRY TRIGGER (15m)</b>\n"
        f"   9/20 EMA Crossover Confirmed\n"
        f"   Volume Spike Confirmed\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"💰 <b>Entry:</b> {format_price(setup['entry'])}\n"
        f"🛑 <b>Stop Loss:</b> {format_price(setup['stop_loss'])} ({setup['risk_pct']:.2f}% risk)\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"🎯 <b>TARGETS</b>\n"
        f"   TP1 (1.5R): {format_price(setup['target_1_5r'])} (Scale 50%)\n"
        f"   Runner (3.0R): {format_price(setup['target_runner'])}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"🕐 <b>Time:</b> {setup['time']} IST\n"
        f"🌊 <b>24H Volume:</b> ${format_volume(setup['vol_24h'])}\n"
    )
    return alert

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    exchange = ccxt.delta({
        "enableRateLimit": True,
        "urls": {
            "api": {
                "public": "https://api.india.delta.exchange",
                "private": "https://api.india.delta.exchange",
            }
        },
    })

    pairs = fetch_all_delta_india_tickers(exchange)
    print(f"📊 Total Active Futures Pairs Found: {len(pairs)}")

    send_telegram_alert(
        f"🤖 <b>EMA SCANNER ONLINE</b>\n"
        f"📊 Scanning: {len(pairs)} pairs\n"
        f"🌊 4H: 200 EMA Trend Filter\n"
        f"⚡ 15m: 9/20 EMA Cross + Volume"
    )

    detected_count = 0
    telegram_sent_count = 0

    for idx, (ccxt_symbol, app_ticker) in enumerate(pairs):
        
        # 1. Quick Volume Filter before processing history
        try:
            ticker = exchange.fetch_ticker(ccxt_symbol)
            vol_24h = ticker.get("quoteVolume") or ticker.get("baseVolume") or 0.0
        except:
            vol_24h = 0.0

        if vol_24h < MIN_24H_VOLUME_USD:
            continue
            
        time.sleep(0.1)  # Respect rate limits

        setup = scan_symbol(ccxt_symbol, exchange, vol_24h)

        if not setup:
            continue

        detected_count += 1
        alert_message = build_alert(setup, app_ticker)

        print("\n" + "=" * 80)
        print(alert_message.replace("<b>", "").replace("</b>", ""))
        print("=" * 80 + "\n")

        success = send_telegram_alert(alert_message)
        if success:
            telegram_sent_count += 1

    print("\n" + "=" * 80)
    print("🏁 SCAN COMPLETE")
    print(f"✅ Total Setups Found: {detected_count}")
    print(f"📱 Telegram Alerts Sent: {telegram_sent_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()