"""Delta India liquidity-sweep / EMA-reclaim research scanner. No order execution."""
import os
import time
import html
import argparse
from datetime import datetime, timezone

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
PRIORITY = {'BTC','ETH','SOL','XRP','AAVE','XAUT','SPCXX','GOOGLX','TSLAX','AMZNX','NVDAX','AAPLX','METAX'}
API_RETRIES = 2
SWING_LEFT = 2
SWING_RIGHT = 2
LOOKBACK = 100
MAX_SWEEP_TO_RECLAIM = 3
MAX_SWEEP_TO_EMA = 8
EQUAL_TOLERANCE = 0.0025  # 0.25% tolerance for grouping equal lows/highs
MIN_24H_QUOTE_VOLUME = 200_000
SL_BUFFER = 0.001


def ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def api_call(fn, *args, **kwargs):
    for attempt in range(API_RETRIES):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as exc:
            print(f'  API retry {attempt+1}/{API_RETRIES}: {type(exc).__name__}')
            if attempt + 1 == API_RETRIES:
                return None
            time.sleep(1)
        except ccxt.ExchangeError as exc:
            print(f'  Exchange error: {exc}')
            return None
    return None


def candles(exchange, symbol, tf, limit):
    rows = api_call(exchange.fetch_ohlcv, symbol, tf, limit=limit)
    if not rows or len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    # Only retain completed bars, rather than blindly removing the final bar.
    duration_ms = exchange.parse_timeframe(tf) * 1000
    now_ms = exchange.milliseconds()
    df = df[df.timestamp + duration_ms <= now_ms].reset_index(drop=True)
    return df if len(df) >= 60 else None


def ema(df, periods):
    df = df.copy()
    for p in periods:
        df[f'ema{p}'] = df.close.ewm(span=p, adjust=False).mean()
    return df


def confirmed_swings(df, direction):
    """A pivot is usable only after SWING_RIGHT subsequent CLOSED bars."""
    field = 'low' if direction == 'LONG' else 'high'
    values = df[field].to_numpy()
    pivots = []
    for i in range(SWING_LEFT, len(df)-SWING_RIGHT):
        left = values[i-SWING_LEFT:i]
        right = values[i+1:i+SWING_RIGHT+1]
        v = values[i]
        valid = (v < left.min() and v <= right.min()) if direction == 'LONG' else (v > left.max() and v >= right.max())
        if valid:
            pivots.append({'pivot': i, 'available': i+SWING_RIGHT, 'level': float(v)})
    return pivots


def liquidity_candidates(pivots, sweep_i, direction):
    """Pre-sweep known pivots, including clustered equal highs/lows."""
    known = [p for p in pivots if p['available'] < sweep_i and sweep_i-p['pivot'] <= LOOKBACK]
    candidates = []
    for p in known:
        group = [q for q in known if q['pivot'] <= p['pivot'] and abs(q['level']-p['level'])/max(abs(p['level']), 1e-12) <= EQUAL_TOLERANCE]
        # Only count distinct, separated pivots. Avoid using later pivots to define earlier levels.
        group = sorted(group, key=lambda q:q['pivot'])
        if len(group) >= 2:
            level = min(q['level'] for q in group) if direction == 'LONG' else max(q['level'] for q in group)
            kind = 'equal lows' if direction == 'LONG' else 'equal highs'
        else:
            level = p['level']
            kind = 'swing low' if direction == 'LONG' else 'swing high'
        candidates.append((level, kind, p['pivot']))
    # Prefer clustered zones, then the most recent pivot.
    return sorted(candidates, key=lambda x:(not x[1].startswith('equal'), -x[2]))


def evaluate(df, direction, scan_i, macro_ok=True):
    """No look-ahead: all information used is available by scan_i."""
    if not macro_ok:
        return None, '4H trend filter failed'
    if scan_i < 55:
        return None, 'insufficient history'
    # Compute pivots using data only through this scan candle.
    history = df.iloc[:scan_i+1]
    pivots = confirmed_swings(history, direction)
    if not pivots:
        return None, 'no confirmed swing pivots'
    saw_sweep = False
    saw_reclaim = False
    # Latest sweep first; only setups with the FIRST qualifying EMA confirmation now.
    for sweep_i in range(scan_i, max(45, scan_i-MAX_SWEEP_TO_EMA)-1, -1):
        bar = df.iloc[sweep_i]
        for level, kind, pivot_i in liquidity_candidates(pivots, sweep_i, direction):
            swept = bar.low < level if direction == 'LONG' else bar.high > level
            if not swept:
                continue
            saw_sweep = True
            reclaim_i = None
            for j in range(sweep_i, min(scan_i, sweep_i+MAX_SWEEP_TO_RECLAIM)+1):
                close = df.iloc[j].close
                if (close > level if direction == 'LONG' else close < level):
                    reclaim_i = j
                    break
            if reclaim_i is None:
                continue
            saw_reclaim = True
            # Invalidated if sweep extreme is violated before confirmation.
            segment = df.iloc[sweep_i:scan_i+1]
            extreme = float(segment.low.min() if direction == 'LONG' else segment.high.max())
            # For a clean setup, don't allow a second break of the original sweep extreme
            initial_extreme = float(df.iloc[sweep_i:reclaim_i+1].low.min() if direction == 'LONG' else df.iloc[sweep_i:reclaim_i+1].high.max())
            later = df.iloc[reclaim_i+1:scan_i+1]
            if len(later) and ((later.low < initial_extreme).any() if direction == 'LONG' else (later.high > initial_extreme).any()):
                continue
            # Require first EMA reclaim to happen on the evaluated candle.
            for j in range(reclaim_i, min(scan_i, sweep_i+MAX_SWEEP_TO_EMA)+1):
                b = df.iloc[j]
                qualifies = (b.close > b.ema9 and b.close > b.ema20) if direction == 'LONG' else (b.close < b.ema9 and b.close < b.ema20)
                if not qualifies:
                    continue
                if j != scan_i:
                    break
                entry = float(b.close)
                stop = initial_extreme * (1-SL_BUFFER if direction == 'LONG' else 1+SL_BUFFER)
                risk = entry-stop if direction == 'LONG' else stop-entry
                if risk <= 0:
                    break
                sign = 1 if direction == 'LONG' else -1
                return {
                    'direction':direction,'kind':kind,'level':level,'sweep_time':ts(df.iloc[sweep_i].timestamp),
                    'confirmation_time':ts(b.timestamp),'entry':entry,'stop':stop,'risk_pct':risk/entry*100,
                    'tp1':entry+sign*1.5*risk,'tp2':entry+sign*3*risk,
                    'ema9':float(b.ema9),'ema20':float(b.ema20),'sweep_extreme':initial_extreme,
                    'sweep_index':sweep_i,'confirmation_index':scan_i
                }, 'confirmed'
    if saw_reclaim:
        return None, 'liquidity reclaimed; awaiting first 9/20 EMA close (or expired)'
    if saw_sweep:
        return None, 'liquidity swept; reclaim not confirmed'
    return None, 'no recent swing/equal-liquidity sweep'


def macro_trend(df, timestamp, strict_slope):
    # As-of alignment: only 4H candles that had CLOSED by the 15M signal close.
    tf_ms = 4*60*60*1000
    closed = df[df.timestamp+tf_ms <= timestamp+15*60*1000]
    if len(closed) < 2:
        return []
    a, b = closed.iloc[-2], closed.iloc[-1]
    long_ok = b.close > b.ema200 and (not strict_slope or b.ema200 > a.ema200)
    short_ok = b.close < b.ema200 and (not strict_slope or b.ema200 < a.ema200)
    return (['LONG'] if long_ok else []) + (['SHORT'] if short_ok else [])


def volume_usd(ticker):
    # baseVolume is NOT USD. Estimate with last price if quoteVolume unavailable.
    q = ticker.get('quoteVolume')
    if q is not None:
        return float(q)
    base, last = ticker.get('baseVolume'), ticker.get('last')
    return float(base)*float(last) if base is not None and last is not None else None


def notify(signal, symbol):
    if not BOT_TOKEN or not CHAT_ID:
        print('  Telegram not configured; console alert only')
        return
    msg = (f"🚨 <b>{html.escape(symbol)} {signal['direction']}</b>\n"
           f"Liquidity: {signal['kind']} @ {signal['level']:.6g}\n"
           f"Sweep: {signal['sweep_time']}\nConfirmation: {signal['confirmation_time']}\n"
           f"Entry (signal close): {signal['entry']:.6g}\nSL: {signal['stop']:.6g}\n"
           f"Risk distance: {signal['risk_pct']:.2f}%\nTP1: {signal['tp1']:.6g}\nTP2: {signal['tp2']:.6g}")
    try:
        r = requests.post(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage',json={'chat_id':CHAT_ID,'text':msg,'parse_mode':'HTML'},timeout=10)
        if not r.ok:
            print(f'  Telegram HTTP {r.status_code}: {r.text[:150]}')
    except requests.RequestException as exc:
        print(f'  Telegram error: {exc}')


def market_base(m):
    return (m.get('base') or '').upper()


def run(args):
    exchange = ccxt.delta({'enableRateLimit':True,'timeout':10000,'urls':{'api':{'public':'https://api.india.delta.exchange','private':'https://api.india.delta.exchange'}}})
    markets = api_call(exchange.fetch_markets)
    if not markets:
        print('Could not load Delta markets'); return
    selected = []
    for m in markets:
        if not (m.get('swap') or m.get('future')) or m.get('option') or m.get('active') is False:
            continue
        if args.priority_only and market_base(m) not in PRIORITY:
            continue
        if m.get('symbol'):
            selected.append(m)
    selected.sort(key=lambda m:(market_base(m) not in PRIORITY,market_base(m),m['symbol']))
    print(f'Scanning {len(selected)} markets | mode={args.mode} | 4H slope={args.strict_slope}')
    seen = set()
    results = []
    for idx, market in enumerate(selected,1):
        symbol = market['symbol']
        print(f'[{idx}/{len(selected)}] {symbol}')
        try:
            ticker = api_call(exchange.fetch_ticker,symbol)
            if not ticker:
                print('  SKIP: ticker unavailable'); continue
            vol = volume_usd(ticker)
            if vol is None:
                print('  SKIP: USD volume unknown'); continue
            if vol < args.min_volume:
                print(f'  SKIP: low quote volume ${vol:,.0f}'); continue
            macro = candles(exchange,symbol,'4h',260)
            entry = candles(exchange,symbol,'15m',args.candles)
            if macro is None or entry is None or len(macro)<205:
                print('  SKIP: insufficient 4H/15M history'); continue
            macro = ema(macro,[200]); entry = ema(entry,[9,20])
            indices = [len(entry)-1] if args.mode=='live' else range(max(55,len(entry)-args.history_bars),len(entry))
            reasons = {}
            for i in indices:
                directions = macro_trend(macro,int(entry.iloc[i].timestamp),args.strict_slope)
                if not directions:
                    reasons['4H filter failed']=reasons.get('4H filter failed',0)+1
                for direction in directions:
                    signal, reason = evaluate(entry,direction,i)
                    reasons[reason] = reasons.get(reason,0)+1
                    if signal:
                        key = (symbol,direction,signal['confirmation_time'],signal['sweep_time'])
                        if key in seen: continue
                        seen.add(key)
                        signal['symbol']=symbol
                        results.append(signal)
                        print(f"  {direction} {signal['kind']} | sweep {signal['sweep_time']} | confirm {signal['confirmation_time']} | entry {signal['entry']:.6g} SL {signal['stop']:.6g}")
                        if args.mode=='live': notify(signal,symbol)
            if args.debug:
                print('  Diagnostics:',reasons)
        except KeyboardInterrupt:
            print('Interrupted'); break
        except Exception as exc:
            print(f'  SKIP {type(exc).__name__}: {exc}')
    if results:
        path = args.output
        pd.DataFrame(results).to_csv(path,index=False)
        print(f'Saved {len(results)} signal(s) to {path}')
    else:
        print('No matching signals in selected window. Try historical mode and inspect diagnostics.')
    print('Signals are research candidates, not validated trades. Backtest and account for slippage/fees.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Delta liquidity sweep / EMA scanner')
    parser.add_argument('--mode',choices=['live','historical'],default='historical')
    parser.add_argument('--priority-only',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--strict-slope',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--min-volume',type=float,default=MIN_24H_QUOTE_VOLUME)
    parser.add_argument('--candles',type=int,default=500)
    parser.add_argument('--history-bars',type=int,default=200)
    parser.add_argument('--debug',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--output',default='liquidity_signals.csv')
    run(parser.parse_args())
