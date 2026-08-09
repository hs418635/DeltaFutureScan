import ccxt
import pandas as pd
import time
import requests
import numpy as np
from datetime import datetime

# ========== TELEGRAM CONFIGURATION ==========

BOT_TOKEN = "8809200223:AAHmR969mpLw_jEuH2iVeLJ1RcGY8DeosXA"
CHAT_ID = "503404993"

def send_telegram_alert(message):
    """
    Send alert via Telegram Bot API
    """
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_notification': False
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram alert sent successfully!")
            return True
        else:
            print(f"❌ Telegram error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

# ========== STRATEGY CONFIGURATION ==========

# ENTRY TIMEFRAME: 1HOUR ONLY
ENTRY_TIMEFRAME = '1h'

# EMA periods for entry timeframe
EMA_SHORT = 9
EMA_LONG = 20
EMA_TREND = 200

# ========== ENHANCED CONFIGURATIONS ==========

# Minimum signal strength (0-10)
MIN_SIGNAL_STRENGTH = 5

# Minimum 24h volume for liquidity
MIN_24H_VOLUME_USD = 200000

# Minimum candle volume
MIN_CANDLE_VOLUME = 500

# Minimum risk-reward ratio required
MIN_RR_RATIO = 2.0  # Minimum 1:2

# Crossover lookback period (number of candles to check)
CROSSOVER_LOOKBACK = 10

# Priority coins (these get scanned first)
PRIORITY_COINS = [
    # Major Cryptocurrencies
    'BTC', 'ETH', 'SOL', 'XRP', 'AAVE', 'XAUT',
    # xStock Tokens (Tokenized Stocks on Delta India)
    'SPCXX',    # SpaceX - $1.42M volume
    'GOOGLX',   # Alphabet/Google - $439K volume
    'TSLAX',    # Tesla - $363K volume
    'AMZNX',    # Amazon - $100K volume
    'NVDAX',    # NVIDIA - $95K volume
    'AAPLX',    # Apple - $86K volume
    'METAX',    # Meta/Facebook - $27K volume
]

# ========== HELPER FUNCTIONS ==========

def fetch_all_delta_india_tickers(exchange):
    """Fetches all active perpetual contracts hosted on the Delta India cluster."""
    raw_markets = exchange.fetch_markets()
    tradeable_pairs = []
    for m in raw_markets:
        is_derivative = (
            m.get('swap', False) or
            m.get('future', False) or
            m.get('linear', False) or
            (m.get('type') in ['swap', 'future', 'linear'])
        )
        is_active = m.get('active', True)
        is_option = m.get('option', False) or (m.get('type') == 'option')
        if is_derivative and is_active and not is_option:
            symbol = m.get('symbol')
            raw_id = m.get('id', '')
            clean_name = raw_id.replace('_', '').replace('-', '').replace('/', '').replace(':', '')
            if (symbol, clean_name) not in tradeable_pairs:
                tradeable_pairs.append((symbol, clean_name))
    return tradeable_pairs

def format_volume(volume):
    """Formats raw volume into human-readable K, M, B units."""
    if volume is None:
        return "0.00"
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.2f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.2f}K"
    else:
        return f"{volume:.2f}"

def fetch_ohlcv_for_tf(symbol, exchange, timeframe, limit=150):
    """Fetch OHLCV data for a specific timeframe."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if len(ohlcv) < 50:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"⚠️ Error fetching {timeframe} for {symbol}: {e}")
        return None

def calculate_ema(df, period):
    """Calculate EMA for a dataframe."""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculate ATR for a dataframe."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()

def calculate_zone_levels(df, lookback=20):
    """Calculate demand (support) and supply (resistance) zones."""
    # Find demand zones (areas of strong buying)
    demand_zones = []
    for i in range(len(df) - lookback, len(df) - 5):
        if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
            zone_low = df['low'].iloc[i]
            zone_high = df['close'].iloc[i] if df['close'].iloc[i] > df['open'].iloc[i] else df['open'].iloc[i]
            if df['volume'].iloc[i] > df['volume'].iloc[i-5:i+5].mean():
                demand_zones.append({'low': zone_low, 'high': zone_high})
    
    # Find supply zones (areas of strong selling)
    supply_zones = []
    for i in range(len(df) - lookback, len(df) - 5):
        if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
            zone_high = df['high'].iloc[i]
            zone_low = df['close'].iloc[i] if df['close'].iloc[i] < df['open'].iloc[i] else df['open'].iloc[i]
            if df['volume'].iloc[i] > df['volume'].iloc[i-5:i+5].mean():
                supply_zones.append({'low': zone_low, 'high': zone_high})
    
    # Get the most recent demand and supply zones
    current_price = df['close'].iloc[-1]
    nearest_demand = None
    nearest_supply = None
    
    for zone in demand_zones:
        if zone['high'] < current_price:
            if nearest_demand is None or zone['high'] > nearest_demand['high']:
                nearest_demand = zone
    
    for zone in supply_zones:
        if zone['low'] > current_price:
            if nearest_supply is None or zone['low'] < nearest_supply['low']:
                nearest_supply = zone
    
    return {
        'demand_zones': demand_zones,
        'supply_zones': supply_zones,
        'nearest_demand': nearest_demand,
        'nearest_supply': nearest_supply
    }

def calculate_trade_levels(direction, entry_price, stop_loss, target_multiplier=3):
    """
    Calculates 1:2, 1:3, 1:4, 1:5 risk-reward targets.
    """
    if direction == 'LONG':
        risk = entry_price - stop_loss
    else:
        risk = stop_loss - entry_price

    if risk is None or pd.isna(risk) or risk <= 0:
        return None

    risk_pct = (risk / entry_price) * 100
    
    targets = {}
    for i in range(2, 6):  # 1:2 to 1:5
        if direction == 'LONG':
            targets[f'target_{i}r'] = entry_price + (risk * i)
        else:
            targets[f'target_{i}r'] = entry_price - (risk * i)

    return {
        'entry': entry_price,
        'stop_loss': stop_loss,
        'risk_pct': risk_pct,
        'target_2r': targets['target_2r'],
        'target_3r': targets['target_3r'],
        'target_4r': targets['target_4r'],
        'target_5r': targets['target_5r'],
        'risk_reward': f"1:{target_multiplier}"
    }

# ========== ENHANCED EMA CROSSOVER DETECTION (10-CANDLE LOOKBACK) ==========

def check_ema_crossover(df_1h, lookback=10):
    """
    Check for EMA crossover using the last 'lookback' candles.
    Returns (golden_cross, death_cross, details, score)
    """
    if len(df_1h) < lookback + 5:
        return None, None, ["Insufficient data"], 0
    
    # Get the last N candles
    ema9_values = df_1h['ema9'].iloc[-lookback:].values
    ema20_values = df_1h['ema20'].iloc[-lookback:].values
    
    # Current and previous values
    curr_ema9 = ema9_values[-1]
    curr_ema20 = ema20_values[-1]
    prev_ema9 = ema9_values[-2]
    prev_ema20 = ema20_values[-2]
    
    # ---- Check if 9 EMA was consistently BELOW or ABOVE 20 EMA ----
    was_below_20 = sum(1 for i in range(len(ema9_values) - 6, len(ema9_values) - 2) if ema9_values[i] < ema20_values[i])
    was_above_20 = sum(1 for i in range(len(ema9_values) - 6, len(ema9_values) - 2) if ema9_values[i] > ema20_values[i])
    
    # ---- Check the trend of 9 EMA (consistently moving up/down) ----
    ema9_trend_up = all(ema9_values[i] > ema9_values[i-1] for i in range(len(ema9_values) - 4, len(ema9_values)))
    ema9_trend_down = all(ema9_values[i] < ema9_values[i-1] for i in range(len(ema9_values) - 4, len(ema9_values)))
    
    # ---- Check if 20 EMA is flattening ----
    ema20_change = abs(ema20_values[-1] - ema20_values[-5]) / ema20_values[-5] * 100 if ema20_values[-5] > 0 else 0
    ema20_flat = ema20_change < 0.1  # Less than 0.1% change
    
    # ---- Calculate ATR for distance check ----
    atr = calculate_atr(df_1h).iloc[-1] if 'atr' in df_1h.columns else 0
    distance = abs(curr_ema9 - curr_ema20)
    is_close = distance < (atr * 0.5) if atr > 0 else False
    
    # ---- Determine crossover ----
    golden_cross = None
    death_cross = None
    details = []
    score = 0
    
    # ---- GOLDEN CROSS: 9 EMA crosses ABOVE 20 EMA ----
    # Conditions:
    # 1. 9 EMA was BELOW 20 EMA for at least 4 of last 6 candles
    # 2. 9 EMA is NOW ABOVE 20 EMA (current cross)
    # 3. 9 EMA is moving UP (momentum)
    if was_below_20 >= 4 and curr_ema9 > curr_ema20 and ema9_trend_up:
        golden_cross = "confirmed"
        score = 5
        details.append(f"✅ Golden Cross CONFIRMED (10-candle validation)")
        details.append(f"   • 9 EMA ({curr_ema9:.6f}) > 20 EMA ({curr_ema20:.6f})")
        details.append(f"   • 9 EMA was below for {was_below_20}/6 candles")
        details.append(f"   • 9 EMA trend: UP ✅")
        if ema20_flat:
            details.append("   • 20 EMA is flattening (bullish confirmation)")
    
    # ---- DEATH CROSS: 9 EMA crosses BELOW 20 EMA ----
    elif was_above_20 >= 4 and curr_ema9 < curr_ema20 and ema9_trend_down:
        death_cross = "confirmed"
        score = 5
        details.append(f"✅ Death Cross CONFIRMED (10-candle validation)")
        details.append(f"   • 9 EMA ({curr_ema9:.6f}) < 20 EMA ({curr_ema20:.6f})")
        details.append(f"   • 9 EMA was above for {was_above_20}/6 candles")
        details.append(f"   • 9 EMA trend: DOWN ✅")
        if ema20_flat:
            details.append("   • 20 EMA is flattening (bearish confirmation)")
    
    # ---- IMMINENT CROSS (9 EMA is approaching 20 EMA) ----
    else:
        # Imminent Golden: 9 EMA below 20 but moving up and close
        if curr_ema9 < curr_ema20 and ema9_trend_up and is_close:
            golden_cross = "imminent"
            score = 3
            details.append(f"🟡 Golden Cross IMMINENT (10-candle validation)")
            details.append(f"   • 9 EMA ({curr_ema9:.6f}) approaching 20 EMA ({curr_ema20:.6f})")
            details.append(f"   • Distance: {distance:.6f} (ATR: {atr:.6f})")
            details.append(f"   • 9 EMA trend: UP ✅")
            if was_below_20 >= 3:
                details.append(f"   • 9 EMA was below for {was_below_20}/6 candles")
        
        # Imminent Death: 9 EMA above 20 but moving down and close
        elif curr_ema9 > curr_ema20 and ema9_trend_down and is_close:
            death_cross = "imminent"
            score = 3
            details.append(f"🟡 Death Cross IMMINENT (10-candle validation)")
            details.append(f"   • 9 EMA ({curr_ema9:.6f}) approaching 20 EMA ({curr_ema20:.6f})")
            details.append(f"   • Distance: {distance:.6f} (ATR: {atr:.6f})")
            details.append(f"   • 9 EMA trend: DOWN ✅")
            if was_above_20 >= 3:
                details.append(f"   • 9 EMA was above for {was_above_20}/6 candles")
        
        # ---- ALREADY CROSSED (trend established) ----
        elif curr_ema9 > curr_ema20:
            # Check if it's been above for a while
            if was_above_20 >= 5:
                golden_cross = "established"
                score = 2
                details.append(f"ℹ️ Golden Cross ESTABLISHED (trend active)")
                details.append(f"   • 9 EMA ({curr_ema9:.6f}) > 20 EMA ({curr_ema20:.6f})")
                details.append(f"   • Above for {was_above_20}/6 candles")
            else:
                details.append(f"ℹ️ 9 EMA above 20 EMA (recent cross, wait for confirmation)")
                score = 1
                
        elif curr_ema9 < curr_ema20:
            if was_below_20 >= 5:
                death_cross = "established"
                score = 2
                details.append(f"ℹ️ Death Cross ESTABLISHED (trend active)")
                details.append(f"   • 9 EMA ({curr_ema9:.6f}) < 20 EMA ({curr_ema20:.6f})")
                details.append(f"   • Below for {was_below_20}/6 candles")
            else:
                details.append(f"ℹ️ 9 EMA below 20 EMA (recent cross, wait for confirmation)")
                score = 1
        else:
            details.append(f"⚠️ No clear EMA relationship (9: {curr_ema9:.6f}, 20: {curr_ema20:.6f})")
    
    return golden_cross, death_cross, details, score

# ========== TOP-DOWN ANALYSIS ==========

def analyze_top_down_1h(symbol, exchange):
    """
    Complete top-down analysis for 1-hour entry with 10-candle EMA validation.
    """
    print(f"\n{'='*60}")
    print(f"📊 TOP-DOWN ANALYSIS: {symbol}")
    print(f"{'='*60}")
    
    results = {
        'direction': None,
        'strength_score': 0,
        'reasons': [],
        'details': {},
        'entry_ready': False
    }
    
    # ---- Fetch data ----
    print("\n⏰ Fetching data for all timeframes...")
    df_1h = fetch_ohlcv_for_tf(symbol, exchange, '1h', 150)
    df_4h = fetch_ohlcv_for_tf(symbol, exchange, '4h', 100)
    
    if df_1h is None or df_4h is None:
        print("❌ Failed to fetch all timeframes")
        return results
    
    # ---- Calculate EMAs ----
    print("\n📈 Calculating EMAs...")
    df_1h['ema9'] = calculate_ema(df_1h, 9)
    df_1h['ema20'] = calculate_ema(df_1h, 20)
    df_1h['ema200'] = calculate_ema(df_1h, 200)
    df_1h['atr'] = calculate_atr(df_1h)
    df_4h['ema200'] = calculate_ema(df_4h, 200)
    
    # ---- Current values ----
    current_1h = {
        'close': df_1h['close'].iloc[-1],
        'ema9': df_1h['ema9'].iloc[-1],
        'ema20': df_1h['ema20'].iloc[-1],
        'ema200': df_1h['ema200'].iloc[-1],
        'atr': df_1h['atr'].iloc[-1],
        'high': df_1h['high'].iloc[-1],
        'low': df_1h['low'].iloc[-1],
        'volume': df_1h['volume'].iloc[-1]
    }
    current_4h = {
        'close': df_4h['close'].iloc[-1],
        'ema200': df_4h['ema200'].iloc[-1]
    }
    
    # ---- Check EMA Crossover with 10-candle validation ----
    golden_cross, death_cross, cross_details, cross_score = check_ema_crossover(df_1h, lookback=CROSSOVER_LOOKBACK)
    
    # ---- LONG Conditions ----
    print("\n🔍 Checking LONG conditions...")
    long_conditions = []
    long_score = 0
    
    # 1. 4H > 200 EMA
    cond1 = current_4h['close'] > current_4h['ema200']
    long_conditions.append(f"✅ 4H > 200 EMA: {current_4h['close']:.6f} > {current_4h['ema200']:.6f}" if cond1 else f"❌ 4H > 200 EMA: {current_4h['close']:.6f} < {current_4h['ema200']:.6f}")
    if cond1:
        long_score += 4
    
    # 2. 1H > 200 EMA
    cond2 = current_1h['close'] > current_1h['ema200']
    long_conditions.append(f"✅ 1H > 200 EMA: {current_1h['close']:.6f} > {current_1h['ema200']:.6f}" if cond2 else f"❌ 1H > 200 EMA: {current_1h['close']:.6f} < {current_1h['ema200']:.6f}")
    if cond2:
        long_score += 3
    
    # 3. Golden Cross detection (from 10-candle validation)
    if golden_cross == "confirmed":
        long_conditions.append("✅ Golden Cross CONFIRMED (10-candle validation)")
        long_score += cross_score
        for detail in cross_details:
            long_conditions.append(f"   {detail}")
    elif golden_cross == "imminent":
        long_conditions.append("🟡 Golden Cross IMMINENT (10-candle validation)")
        long_score += cross_score
        for detail in cross_details:
            long_conditions.append(f"   {detail}")
    elif golden_cross == "established":
        long_conditions.append(f"ℹ️ Golden Cross ESTABLISHED (trend active)")
        long_score += cross_score
        for detail in cross_details:
            long_conditions.append(f"   {detail}")
    else:
        # Check if already crossed but not caught by our logic
        if current_1h['ema9'] > current_1h['ema20']:
            long_conditions.append(f"ℹ️ 9 EMA above 20 EMA (trend active, but cross not confirmed)")
            long_score += 1
        else:
            long_conditions.append(f"❌ No Golden Cross (9: {current_1h['ema9']:.6f}, 20: {current_1h['ema20']:.6f})")
    
    # 4. Near Demand Zone
    zones = calculate_zone_levels(df_1h)
    nearest_demand = zones['nearest_demand']
    if nearest_demand:
        distance_to_demand = (current_1h['close'] - nearest_demand['high']) / current_1h['close'] * 100
        cond4 = distance_to_demand < 2.0 and distance_to_demand > -1.0
        long_conditions.append(f"✅ Near Demand Zone: {distance_to_demand:.2f}% away" if cond4 else f"❌ Not near Demand Zone: {distance_to_demand:.2f}% away")
        if cond4:
            long_score += 2
    else:
        long_conditions.append("❓ No demand zone found")
        cond4 = False
    
    # Check if all main conditions met for LONG
    long_ready = cond1 and cond2 and (golden_cross in ["confirmed", "imminent"])
    
    # ---- SHORT Conditions ----
    print("\n🔍 Checking SHORT conditions...")
    short_conditions = []
    short_score = 0
    
    # 1. 4H < 200 EMA
    cond1 = current_4h['close'] < current_4h['ema200']
    short_conditions.append(f"✅ 4H < 200 EMA: {current_4h['close']:.6f} < {current_4h['ema200']:.6f}" if cond1 else f"❌ 4H < 200 EMA: {current_4h['close']:.6f} > {current_4h['ema200']:.6f}")
    if cond1:
        short_score += 4
    
    # 2. 1H < 200 EMA
    cond2 = current_1h['close'] < current_1h['ema200']
    short_conditions.append(f"✅ 1H < 200 EMA: {current_1h['close']:.6f} < {current_1h['ema200']:.6f}" if cond2 else f"❌ 1H < 200 EMA: {current_1h['close']:.6f} > {current_1h['ema200']:.6f}")
    if cond2:
        short_score += 3
    
    # 3. Death Cross detection (from 10-candle validation)
    if death_cross == "confirmed":
        short_conditions.append("✅ Death Cross CONFIRMED (10-candle validation)")
        short_score += cross_score
        for detail in cross_details:
            short_conditions.append(f"   {detail}")
    elif death_cross == "imminent":
        short_conditions.append("🟡 Death Cross IMMINENT (10-candle validation)")
        short_score += cross_score
        for detail in cross_details:
            short_conditions.append(f"   {detail}")
    elif death_cross == "established":
        short_conditions.append(f"ℹ️ Death Cross ESTABLISHED (trend active)")
        short_score += cross_score
        for detail in cross_details:
            short_conditions.append(f"   {detail}")
    else:
        if current_1h['ema9'] < current_1h['ema20']:
            short_conditions.append(f"ℹ️ 9 EMA below 20 EMA (trend active, but cross not confirmed)")
            short_score += 1
        else:
            short_conditions.append(f"❌ No Death Cross (9: {current_1h['ema9']:.6f}, 20: {current_1h['ema20']:.6f})")
    
    # 4. Near Supply Zone
    nearest_supply = zones['nearest_supply']
    if nearest_supply:
        distance_to_supply = (nearest_supply['low'] - current_1h['close']) / current_1h['close'] * 100
        cond4 = distance_to_supply < 2.0 and distance_to_supply > -1.0
        short_conditions.append(f"✅ Near Supply Zone: {distance_to_supply:.2f}% away" if cond4 else f"❌ Not near Supply Zone: {distance_to_supply:.2f}% away")
        if cond4:
            short_score += 2
    else:
        short_conditions.append("❓ No supply zone found")
        cond4 = False
    
    # Check if all main conditions met for SHORT
    short_ready = cond1 and cond2 and (death_cross in ["confirmed", "imminent"])
    
    # ---- Summary ----
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n🟢 LONG Conditions:")
    for cond in long_conditions:
        print(f"  {cond}")
    print(f"  Score: {long_score}/14")
    
    print("\n🔴 SHORT Conditions:")
    for cond in short_conditions:
        print(f"  {cond}")
    print(f"  Score: {short_score}/14")
    
    # ---- Decision ----
    if long_ready and long_score >= 6:
        results['direction'] = 'LONG'
        results['strength_score'] = min(long_score, 10)
        results['reasons'] = ["Top-down LONG confirmed", f"Score: {long_score}/14"]
        results['entry_ready'] = True
        results['details'] = {
            'long_conditions': long_conditions,
            'short_conditions': short_conditions,
            'long_score': long_score,
            'short_score': short_score,
            'zones': zones,
            'cross_details': cross_details
        }
        print(f"\n✅ DECISION: LONG (Score: {results['strength_score']}/10)")
        
    elif short_ready and short_score >= 6:
        results['direction'] = 'SHORT'
        results['strength_score'] = min(short_score, 10)
        results['reasons'] = ["Top-down SHORT confirmed", f"Score: {short_score}/14"]
        results['entry_ready'] = True
        results['details'] = {
            'long_conditions': long_conditions,
            'short_conditions': short_conditions,
            'long_score': long_score,
            'short_score': short_score,
            'zones': zones,
            'cross_details': cross_details
        }
        print(f"\n✅ DECISION: SHORT (Score: {results['strength_score']}/10)")
        
    else:
        if long_score >= 5 or short_score >= 5:
            print(f"\n⏳ Setup almost ready. Long: {long_score}/14, Short: {short_score}/14")
        else:
            print(f"\n❌ No setup confirmed. Long: {long_score}/14, Short: {short_score}/14")
    
    print("="*60 + "\n")
    return results

def get_entry_price_and_stop(df, direction, atr_multiplier=1.5):
    """
    Get entry price and stop loss based on recent price action and ATR.
    """
    current_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else calculate_atr(df).iloc[-1]
    
    if direction == 'LONG':
        entry = current_price
        recent_low = df['low'].iloc[-10:].min()
        stop = min(recent_low, df['ema9'].iloc[-1] - (atr * 0.5)) if 'ema9' in df.columns else recent_low
        stop = max(stop, current_price - (atr * atr_multiplier))
        
    else:  # SHORT
        entry = current_price
        recent_high = df['high'].iloc[-10:].max()
        stop = max(recent_high, df['ema9'].iloc[-1] + (atr * 0.5)) if 'ema9' in df.columns else recent_high
        stop = min(stop, current_price + (atr * atr_multiplier))
    
    return entry, stop

# ========== SCAN FUNCTION ==========

def scan_1h_entry(symbol, exchange):
    """
    Main scan function for 1-hour entry strategy.
    """
    try:
        print(f"\n🔍 Scanning {symbol}...")
        
        # ---- Fetch 1H data ----
        df_1h = fetch_ohlcv_for_tf(symbol, exchange, '1h', 150)
        if df_1h is None:
            return None
        
        # Calculate indicators
        df_1h['ema9'] = calculate_ema(df_1h, 9)
        df_1h['ema20'] = calculate_ema(df_1h, 20)
        df_1h['ema200'] = calculate_ema(df_1h, 200)
        df_1h['atr'] = calculate_atr(df_1h)
        
        current_price = df_1h['close'].iloc[-1]
        current_volume = df_1h['volume'].iloc[-1]
        
        # ---- Check volume & liquidity ----
        try:
            ticker = exchange.fetch_ticker(symbol)
            vol_24h = ticker.get('quoteVolume') or ticker.get('baseVolume', 0.0)
        except:
            vol_24h = 0.0
        
        if vol_24h < MIN_24H_VOLUME_USD:
            print(f"⏭️ Skipping {symbol}: 24h volume ${vol_24h:,.0f} < ${MIN_24H_VOLUME_USD:,}")
            return None
        
        if current_volume < MIN_CANDLE_VOLUME:
            print(f"⏭️ Skipping {symbol}: Candle volume {current_volume:.0f} < {MIN_CANDLE_VOLUME}")
            return None
        
        # ---- Run Top-Down Analysis ----
        analysis = analyze_top_down_1h(symbol, exchange)
        
        if not analysis['entry_ready']:
            return None
        
        direction = analysis['direction']
        strength_score = analysis['strength_score']
        reasons = analysis['reasons']
        details = analysis['details']
        
        # ---- Get entry and stop levels ----
        entry, stop = get_entry_price_and_stop(df_1h, direction)
        
        # ---- Calculate trade levels ----
        if strength_score >= 9:
            target_multiplier = 5
        elif strength_score >= 8:
            target_multiplier = 4
        else:
            target_multiplier = 3
        
        trade_levels = calculate_trade_levels(direction, entry, stop, target_multiplier)
        if trade_levels is None:
            return None
        
        # ---- Check if RR is acceptable ----
        risk_amount = abs(entry - stop)
        if direction == 'LONG':
            target_3r = entry + (risk_amount * 3)
        else:
            target_3r = entry - (risk_amount * 3)
        
        rr_ratio = abs(target_3r - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
        if rr_ratio < MIN_RR_RATIO:
            print(f"⏭️ Skipping {symbol}: RR ratio {rr_ratio:.1f} < {MIN_RR_RATIO}")
            return None
        
        # ---- Build result ----
        raw_ts = df_1h['timestamp'].iloc[-1]
        ist_time = pd.to_datetime(raw_ts, unit='ms', utc=True).tz_convert('Asia/Kolkata')
        formatted_ist = ist_time.strftime('%Y-%m-%d %H:%M:%S IST')
        
        zones = details.get('zones', {})
        nearest_demand = zones.get('nearest_demand')
        nearest_supply = zones.get('nearest_supply')
        
        zone_info = ""
        if direction == 'LONG' and nearest_demand:
            zone_info = f"Demand Zone: {nearest_demand['low']:.6f} - {nearest_demand['high']:.6f}"
        elif direction == 'SHORT' and nearest_supply:
            zone_info = f"Supply Zone: {nearest_supply['low']:.6f} - {nearest_supply['high']:.6f}"
        
        # Get cross details
        cross_details = details.get('cross_details', [])
        cross_info = "\n".join(cross_details) if cross_details else "N/A"
        
        result = {
            'symbol': symbol,
            'direction': direction,
            'entry': entry,
            'stop_loss': stop,
            'risk_pct': trade_levels['risk_pct'],
            'target_2r': trade_levels['target_2r'],
            'target_3r': trade_levels['target_3r'],
            'target_4r': trade_levels['target_4r'],
            'target_5r': trade_levels['target_5r'],
            'strength_score': strength_score,
            'reasons': reasons,
            'time': formatted_ist,
            'volume': current_volume,
            'vol_24h': vol_24h,
            'timeframe': '1h',
            'zone_info': zone_info,
            'rr_ratio': rr_ratio,
            'long_score': details.get('long_score', 0),
            'short_score': details.get('short_score', 0),
            'long_conditions': details.get('long_conditions', []),
            'short_conditions': details.get('short_conditions', []),
            'cross_info': cross_info,
        }
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error scanning {symbol}: {e}")
        return None

# ========== MAIN ORCHESTRATION ==========

def main():
    exchange = ccxt.delta({
        'enableRateLimit': True,
        'urls': {
            'api': {
                'public': 'https://api.india.delta.exchange',
                'private': 'https://api.india.delta.exchange',
            }
        }
    })
    
    pairs = fetch_all_delta_india_tickers(exchange)
    print(f"📊 Total Active Futures Pairs Found: {len(pairs)}")
    
    # Sort pairs: Priority coins first
    priority_pairs = []
    normal_pairs = []
    for ccxt_symbol, app_ticker in pairs:
        base_asset = ccxt_symbol.split('/')[0].upper() if '/' in ccxt_symbol else app_ticker
        if base_asset in PRIORITY_COINS:
            priority_pairs.append((ccxt_symbol, app_ticker))
        else:
            normal_pairs.append((ccxt_symbol, app_ticker))
    
    all_pairs = priority_pairs + normal_pairs
    
    print(f"⭐ Priority Coins: {len(priority_pairs)}")
    print(f"📊 Total Coins: {len(all_pairs)}")
    print(f"⏰ Entry Timeframe: 1H")
    print(f"📈 Min Strength Required: {MIN_SIGNAL_STRENGTH}/10")
    print(f"💰 Min 24h Volume: ${MIN_24H_VOLUME_USD:,}")
    print(f"📊 Crossover Lookback: {CROSSOVER_LOOKBACK} candles")
    print(f"🎯 Min RR Ratio: 1:{MIN_RR_RATIO:.0f}")
    print("\n" + "="*60)
    print("🚀 Starting 1H Entry Scanner...")
    print("="*60 + "\n")
    
    # Send startup message
    send_telegram_alert(f"🤖 <b>1H ENTRY SCANNER ONLINE</b>\n"
                        f"📊 Scanning {len(all_pairs)} pairs\n"
                        f"⭐ Priority: {len(priority_pairs)} pairs\n"
                        f"⏰ Entry: 1H\n"
                        f"📈 Top-Down: 4H + 1H\n"
                        f"📊 Crossover Lookback: {CROSSOVER_LOOKBACK} candles\n"
                        f"🎯 Min RR: 1:{MIN_RR_RATIO:.0f}\n"
                        f"💪 Min Strength: {MIN_SIGNAL_STRENGTH}/10")
    
    detected_count = 0
    telegram_sent_count = 0
    
    for idx, (ccxt_symbol, app_ticker) in enumerate(all_pairs):
        if idx % 20 == 0 and idx > 0:
            print(f"📊 Progress: {idx}/{len(all_pairs)} pairs scanned...")
        
        time.sleep(0.1)
        
        setup = scan_1h_entry(ccxt_symbol, exchange)
        
        if setup:
            detected_count += 1
            
            reasons_str = " | ".join(setup.get('reasons', [])) if setup.get('reasons') else "N/A"
            
            alert_message = (
                f"🚨 <b>1H ENTRY ALERT</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>{app_ticker}</b>\n"
                f"📈 <b>Direction:</b> {'🟢 LONG' if setup['direction'] == 'LONG' else '🔴 SHORT'}\n"
                f"💪 <b>Strength:</b> {setup['strength_score']}/10\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 <b>Entry:</b> {setup['entry']:.6f}\n"
                f"🛑 <b>Stop Loss:</b> {setup['stop_loss']:.6f} ({setup['risk_pct']:.2f}% risk)\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 <b>Targets (1:{setup['rr_ratio']:.1f} RR):</b>\n"
                f"   🟢 1:2 → {setup['target_2r']:.6f}\n"
                f"   🟢 1:3 → {setup['target_3r']:.6f}\n"
                f"   🟢 1:4 → {setup['target_4r']:.6f}\n"
                f"   🟢 1:5 → {setup['target_5r']:.6f}\n"
            )
            
            if setup.get('zone_info'):
                alert_message += f"━━━━━━━━━━━━━━━━━━━━\n"
                alert_message += f"📍 <b>{setup['zone_info']}</b>\n"
            
            # Add crossover details
            if setup.get('cross_info') and setup['cross_info'] != "N/A":
                alert_message += f"━━━━━━━━━━━━━━━━━━━━\n"
                alert_message += f"📊 <b>EMA Crossover:</b>\n"
                for line in setup['cross_info'].split('\n'):
                    alert_message += f"   {line}\n"
            
            alert_message += (
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>Top-Down Scores:</b>\n"
                f"   🟢 Long: {setup['long_score']}/14\n"
                f"   🔴 Short: {setup['short_score']}/14\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ <b>Vol:</b> {format_volume(setup['volume'])}\n"
                f"🌊 <b>24h Vol:</b> ${format_volume(setup['vol_24h'])}\n"
                f"🕐 <b>Time:</b> {setup['time']}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Reasons:</b> {reasons_str}"
            )
            
            print("\n" + "="*60)
            print(alert_message.replace('<b>', '').replace('</b>', ''))
            print("="*60 + "\n")
            
            success = send_telegram_alert(alert_message)
            if success:
                telegram_sent_count += 1
    
    # Final summary
    print("\n" + "="*60)
    print("🏁 SCAN COMPLETE")
    print("="*60)
    print(f"✅ Total Setups Found: {detected_count}")
    print(f"📱 Telegram Alerts Sent: {telegram_sent_count}")
    print("="*60)
    
    send_telegram_alert(
        f"✅ <b>1H Entry Scan Complete</b>\n"
        f"📊 Found: {detected_count} setups\n"
        f"📱 Sent: {telegram_sent_count} alerts\n"
        f"📊 Crossover Lookback: {CROSSOVER_LOOKBACK} candles\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
    print("\n🏁 Master execution finished. Exiting.")