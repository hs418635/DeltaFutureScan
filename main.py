"""
Delta India 15M entry scanner  (v3)

Universe : EVERY active perpetual / future on Delta India (no volume filter by default).
Bias     : 4H + 1H EMA state (used as a hard filter for the CROSS pattern, as information for SWEEP).
Triggers (15M, two patterns, both on by default):

  CROSS  – EMA9/EMA20 cross in the bias direction, scored on structure (pullback, trigger bar,
           higher-low/lower-high, not over-extended). Needs 4H and 1H to agree.

  SWEEP  – the "liquidity sweep + impulse" entry:
             1. a wick takes out the prior swing low (long) / swing high (short) inside the last
                --sweep-window bars,
             2. the trigger bar is an impulse: body >= --min-body-atr x ATR, body >= --min-body-ratio
                of its range, volume >= --vol-mult x 20-bar average,
             3. the trigger bar closes back on the right side of the swept level,
             4. EMA9 has crossed EMA20 in the trade direction (or is about to: close beyond EMA20
                with EMA9 turning) – reported as "crossed" / "pending".
           Entry = trigger close, SL = sweep wick +/- ATR buffer, targets 1:1 / 1:2 / 1:3.
           Setups whose stop is wider than --max-risk-pct of price are skipped (1:2 unrealistic).

Timing   : scans right after warm-up, then every --poll seconds on the FORMING 15M bar (EARLY alert)
           and at every 15M close (CONFIRMED alert). One alert per symbol / pattern / bar / stage.
Alerts   : Telegram (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID in .env). No order execution.
"""
import os
import json
import time
import html
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
print(f"Telegram bot token: {'set' if BOT_TOKEN else 'not set'}, chat ID: {'set' if CHAT_ID else 'not set'}")

PRIORITY = {"BTC", "ETH", "SOL", "XRP", "AAVE", "XAUT", "SPCXX", "GOOGLX",
            "TSLAX", "AMZNX", "NVDAX", "AAPLX", "METAX"}

TIMEFRAMES  = ["4h", "1h", "15m"]
EMA_PERIODS = [9, 20, 100, 200]
ATR_PERIOD  = 14
API_RETRIES = 2
IST = timezone(timedelta(hours=5, minutes=30))


# ------------------------------------------------------------------ helpers
def api_call(fn, *args, retries=API_RETRIES, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as exc:
            print(f"  API retry {attempt+1}/{retries}: {type(exc).__name__}")
            if attempt + 1 == retries:
                return None
            time.sleep(1)
        except ccxt.ExchangeError as exc:
            print(f"  Exchange error: {exc}")
            return None
    return None


def build_exchange():
    return ccxt.delta({
        "enableRateLimit": True,
        "timeout": 15000,
        "urls": {"api": {"public":  "https://api.india.delta.exchange",
                         "private": "https://api.india.delta.exchange"}},
    })


def fetch_candles(exchange, symbol, tf, limit, keep_forming=False):
    rows = api_call(exchange.fetch_ohlcv, symbol, tf, limit=limit)
    if not rows or len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if not keep_forming:
        tf_ms = exchange.parse_timeframe(tf) * 1000
        df = df[df.timestamp + tf_ms <= exchange.milliseconds()].reset_index(drop=True)
    return df if len(df) >= 60 else None


def add_indicators(df):
    df = df.copy()
    for p in EMA_PERIODS:
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df["vol_avg20"] = df["volume"].rolling(20).mean().shift(1)   # average of the 20 bars BEFORE each bar
    return df


def fmt_ts(ms):
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return f"{dt.astimezone(IST):%d-%b %H:%M IST}"


# ------------------------------------------------------------------ bias
def ema_bias(row, mode):
    if mode == "920100":
        if row["close"] > row["ema200"] and row["ema9"] > row["ema20"] > row["ema100"]:
            return "LONG"
        if row["close"] < row["ema200"] and row["ema9"] < row["ema20"] < row["ema100"]:
            return "SHORT"
        return None
    if row["close"] > row["ema200"] and row["ema9"] > row["ema20"]:
        return "LONG"
    if row["close"] < row["ema200"] and row["ema9"] < row["ema20"]:
        return "SHORT"
    return None


# ------------------------------------------------------------------ CROSS pattern
def find_cross(df, direction, lookback):
    n = len(df) - 1
    lo = max(1, n - lookback + 1)
    for i in range(n, lo - 1, -1):
        prev, cur = df.iloc[i - 1], df.iloc[i]
        if direction == "LONG" and prev["ema9"] <= prev["ema20"] and cur["ema9"] > cur["ema20"] and cur["close"] > cur["ema20"]:
            return i
        if direction == "SHORT" and prev["ema9"] >= prev["ema20"] and cur["ema9"] < cur["ema20"] and cur["close"] < cur["ema20"]:
            return i
    return None


def swing_points(df, left=2, right=2):
    highs, lows = [], []
    h, l = df["high"].values, df["low"].values
    for i in range(left, len(df) - right):
        if h[i] == h[i - left:i + right + 1].max():
            highs.append((i, float(h[i])))
        if l[i] == l[i - left:i + right + 1].min():
            lows.append((i, float(l[i])))
    return highs, lows


def score_cross(df, direction, cross_idx, mode, pullback_lookback, max_ext_atr):
    score, reasons = 0, []
    cur, trig = df.iloc[-1], df.iloc[cross_idx]
    sign = 1 if direction == "LONG" else -1

    ok_trend = sign * (cur["close"] - cur["ema200"]) > 0
    if mode == "920100":
        ok_trend = ok_trend and sign * (cur["close"] - cur["ema100"]) > 0
    if ok_trend: score += 20; reasons.append("15m trend ok")
    else:        reasons.append("15m against ema200")

    before = df.iloc[max(0, cross_idx - pullback_lookback):cross_idx]
    touched = ((before["low"] <= before["ema20"]) if direction == "LONG" else (before["high"] >= before["ema20"])).any()
    if touched: score += 15; reasons.append("pullback to ema20")
    else:       reasons.append("no pullback")

    rng = trig["high"] - trig["low"]
    body_ratio = abs(trig["close"] - trig["open"]) / rng if rng > 0 else 0
    if sign * (trig["close"] - trig["open"]) > 0 and body_ratio >= 0.5 and sign * (trig["close"] - trig["ema9"]) > 0:
        score += 20; reasons.append(f"strong trigger bar ({body_ratio:.0%})")
    else:
        reasons.append(f"weak trigger bar ({body_ratio:.0%})")

    highs, lows = swing_points(df.tail(80).reset_index(drop=True))
    if direction == "LONG":
        if len(lows) >= 2 and lows[-1][1] > lows[-2][1]:   score += 20; reasons.append("higher low")
        else:                                              reasons.append("no higher low")
        if len(highs) >= 2 and highs[-1][1] > highs[-2][1]: score += 10; reasons.append("higher high")
    else:
        if len(highs) >= 2 and highs[-1][1] < highs[-2][1]: score += 20; reasons.append("lower high")
        else:                                               reasons.append("no lower high")
        if len(lows) >= 2 and lows[-1][1] < lows[-2][1]:   score += 10; reasons.append("lower low")

    if cur["atr"] > 0:
        ext = abs(cur["close"] - cur["ema20"]) / cur["atr"]
        if ext <= max_ext_atr: score += 15; reasons.append(f"ext {ext:.1f} ATR ok")
        else:                  reasons.append(f"over-extended {ext:.1f} ATR")
    return score, reasons


def structure_stop(df, direction, entry, stop_lookback, atr_buffer):
    atr = float(df.iloc[-1]["atr"])
    highs, lows = swing_points(df.tail(80).reset_index(drop=True))
    if direction == "LONG":
        c = [p for _, p in lows if p < entry]
        return (c[-1] if c else float(df["low"].iloc[-stop_lookback:].min())) - atr_buffer * atr
    c = [p for _, p in highs if p > entry]
    return (c[-1] if c else float(df["high"].iloc[-stop_lookback:].max())) + atr_buffer * atr


# ------------------------------------------------------------------ SWEEP pattern
def find_sweep(df, direction, a):
    """
    Liquidity sweep + impulse on the LAST bar of df (closed, or forming in early mode).
    Returns a dict with the levels and quality notes, or None.
    """
    n = len(df) - 1
    if n < a.level_lookback + a.sweep_window + 2:
        return None
    trig = df.iloc[n]
    atr = float(df.iloc[n - 1]["atr"])          # ATR *before* the impulse, so the spike doesn't inflate its own yardstick
    if atr <= 0 or pd.isna(trig["vol_avg20"]) or trig["vol_avg20"] <= 0:
        return None

    ref    = df.iloc[n - a.sweep_window - a.level_lookback: n - a.sweep_window]   # bars that define the level
    window = df.iloc[n - a.sweep_window: n + 1]                                   # bars where the sweep may happen
    sign = 1 if direction == "LONG" else -1

    # 1. sweep of the prior swing level
    if direction == "LONG":
        level = float(ref["low"].min());  wick = float(window["low"].min());  swept = wick < level
        sweep_ts = int(window.loc[window["low"].idxmin(), "timestamp"])
    else:
        level = float(ref["high"].max()); wick = float(window["high"].max()); swept = wick > level
        sweep_ts = int(window.loc[window["high"].idxmax(), "timestamp"])
    if not swept:
        return None

    # 3. close back on the right side of the level
    if sign * (trig["close"] - level) <= 0:
        return None

    # 2. impulse bar
    body = sign * (trig["close"] - trig["open"])
    rng  = trig["high"] - trig["low"]
    if body <= 0 or rng <= 0:
        return None
    body_atr   = body / atr
    body_ratio = body / rng
    vol_ratio  = trig["volume"] / trig["vol_avg20"]
    if body_atr < a.min_body_atr or body_ratio < a.min_body_ratio or vol_ratio < a.vol_mult:
        return None

    # 4. EMA 9/20 state
    recent = df.iloc[max(0, n - 3): n]           # the 3 bars before the trigger
    crossed_now = sign * (trig["ema9"] - trig["ema20"]) > 0
    was_other   = (sign * (recent["ema9"] - recent["ema20"]) <= 0).any()
    ema9_turning = sign * (trig["ema9"] - df.iloc[n - 1]["ema9"]) > 0
    if crossed_now and was_other:
        cross_state = "crossed"
    elif crossed_now:
        cross_state = "already above" if direction == "LONG" else "already below"
    elif sign * (trig["close"] - trig["ema20"]) > 0 and ema9_turning:
        cross_state = "pending"
    else:
        return None
    if a.require_cross and cross_state == "pending":
        return None

    # trade plan
    entry = float(trig["close"])
    stop  = wick - sign * a.atr_buffer * atr
    risk  = abs(entry - stop)
    if risk <= 0 or risk / entry * 100 > a.max_risk_pct:
        return None

    notes = [f"sweep of {level:.6g} (wick {wick:.6g})",
             f"body {body_atr:.1f} ATR, {body_ratio:.0%} of range",
             f"volume {vol_ratio:.1f}x avg",
             f"ema9/20 {cross_state}"]
    if sign * (trig["close"] - trig["ema200"]) > 0:
        notes.append("closed through/above ema200" if direction == "LONG" else "closed through/below ema200")
    else:
        notes.append("still below ema200" if direction == "LONG" else "still above ema200")

    return {"level": level, "wick": wick, "sweep_ts": sweep_ts, "entry": entry, "stop": stop,
            "risk": risk, "cross_state": cross_state, "notes": notes,
            "body_atr": body_atr, "vol_ratio": vol_ratio}


# ------------------------------------------------------------------ telegram
def notify_telegram(sig):
    if not BOT_TOKEN or not CHAT_ID:
        print("  Telegram not configured; console alert only")
        return
    arrow = "🟢 BUY" if sig["direction"] == "LONG" else "🔴 SELL"
    stage = "⚡ EARLY – 15M bar still forming" if sig["early"] else "✅ CONFIRMED – 15M bar closed"
    head  = f"{arrow} <b>{html.escape(sig['symbol'])}</b>  [{sig['pattern']}]"
    if sig["pattern"] == "CROSS":
        head += f"  score {sig['score']}/100"
    lines = [head, stage,
             f"Trigger bar: {fmt_ts(sig['trigger_bar_ts'])}"]
    if sig["pattern"] == "SWEEP":
        lines.append(f"Sweep wick : {fmt_ts(sig['sweep_ts'])}  level {sig['level']:.6g}")
    lines += [f"Entry (mkt): {sig['entry']:.6g}",
              f"Entry zone : {sig['entry_zone'][0]:.6g} – {sig['entry_zone'][1]:.6g}  (ema9..ema20 pullback)",
              f"Stop loss  : {sig['stop']:.6g}  (risk {sig['risk_pct']:.2f}%)",
              f"Target 1:1 : {sig['tp1']:.6g}",
              f"Target 1:2 : {sig['tp2']:.6g}",
              f"Target 1:3 : {sig['tp3']:.6g}",
              f"Checks: {html.escape('; '.join(sig['reasons']))}",
              f"Bias 4H={sig['bias4h'] or '-'}  1H={sig['bias1h'] or '-'}"]
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": "\n".join(lines), "parse_mode": "HTML"}, timeout=10)
        if not r.ok:
            print(f"  Telegram HTTP {r.status_code}: {r.text[:150]}")
    except requests.RequestException as exc:
        print(f"  Telegram error: {exc}")


# ------------------------------------------------------------------ scanner
class Scanner:
    def __init__(self, exchange, symbols, args):
        self.exchange = exchange
        self.symbols  = symbols
        self.a        = args
        self.cache    = {s: {} for s in symbols}
        self.alerted  = {}      # (symbol, pattern, early) -> trigger_bar_ts

    def refresh_tf(self, symbol, tf, keep_forming=False):
        df = fetch_candles(self.exchange, symbol, tf, self.a.candles, keep_forming)
        if df is not None:
            self.cache[symbol][tf] = add_indicators(df)
        return df

    def status(self, symbol):
        c = self.cache.get(symbol, {})
        if any(tf not in c for tf in TIMEFRAMES):
            return f"{symbol:<16} data missing"
        b4 = ema_bias(c["4h"].iloc[-1], self.a.ema_mode) or "-"
        b1 = ema_bias(c["1h"].iloc[-1], self.a.ema_mode) or "-"
        r = c["15m"].iloc[-1]
        return f"{symbol:<16} 4H={b4:<5} 1H={b1:<5} 15m {'9>20' if r['ema9'] > r['ema20'] else '9<20'}  close={r['close']:.6g}"

    def _plan(self, symbol, pattern, direction, df15, entry, stop, trigger_ts, reasons, early, extra=None, score=None):
        last = df15.iloc[-1]
        risk = abs(entry - stop)
        sign = 1 if direction == "LONG" else -1
        c = self.cache[symbol]
        sig = {"symbol": symbol, "pattern": pattern, "direction": direction, "early": early,
               "score": score, "reasons": reasons, "trigger_bar_ts": trigger_ts,
               "entry": entry, "entry_zone": sorted([float(last["ema9"]), float(last["ema20"])]),
               "stop": stop, "risk_pct": risk / entry * 100,
               "tp1": entry + sign * 1 * risk, "tp2": entry + sign * 2 * risk, "tp3": entry + sign * 3 * risk,
               "bias4h": ema_bias(c["4h"].iloc[-1], self.a.ema_mode),
               "bias1h": ema_bias(c["1h"].iloc[-1], self.a.ema_mode),
               "ts": datetime.now(timezone.utc).isoformat()}
        if extra:
            sig.update(extra)
        return sig

    def evaluate(self, symbol, early=False):
        a, c = self.a, self.cache.get(symbol, {})
        
        # Fetch missing timeframes on the fly
        missing = [tf for tf in TIMEFRAMES if tf not in c or len(c[tf]) < 60]
        if missing:
            for tf in missing:
                self.refresh_tf(symbol, tf, keep_forming=early)
            c = self.cache.get(symbol, {})

        if any(tf not in c for tf in TIMEFRAMES):
            return []
        df15 = c["15m"]
        b4 = ema_bias(c["4h"].iloc[-1], a.ema_mode)
        b1 = ema_bias(c["1h"].iloc[-1], a.ema_mode)
        out = []

        # ---- CROSS: needs 4H and 1H to agree
        if a.pattern in ("cross", "both") and b4 and b4 == b1:
            idx = find_cross(df15, b4, a.cross_lookback)
            if idx is not None:
                ts = int(df15.iloc[idx]["timestamp"])
                key = (symbol, "CROSS", early)
                if self.alerted.get(key) != ts:
                    score, reasons = score_cross(df15, b4, idx, a.ema_mode, a.pullback_lookback, a.max_ext_atr)
                    if score >= a.min_score:
                        entry = float(df15.iloc[-1]["close"])
                        stop  = structure_stop(df15, b4, entry, a.stop_lookback, a.atr_buffer)
                        if abs(entry - stop) > 0:
                            self.alerted[key] = ts
                            out.append(self._plan(symbol, "CROSS", b4, df15, entry, stop, ts, reasons, early, score=score))
                    else:
                        print(f"  {symbol} CROSS {b4} score {score} < {a.min_score}: {', '.join(reasons)}")

        # ---- SWEEP: bias is optional (reversal pattern)
        if a.pattern in ("sweep", "both"):
            for direction in ("LONG", "SHORT"):
                if a.sweep_bias == "1h" and b1 != direction:
                    continue
                if a.sweep_bias == "both" and not (b1 == direction and b4 == direction):
                    continue
                res = find_sweep(df15, direction, a)
                if not res:
                    continue
                ts = int(df15.iloc[-1]["timestamp"])
                key = (symbol, "SWEEP", early)
                if self.alerted.get(key) == ts:
                    continue
                self.alerted[key] = ts
                out.append(self._plan(symbol, "SWEEP", direction, df15, res["entry"], res["stop"], ts,
                                      res["notes"], early,
                                      extra={"level": res["level"], "sweep_ts": res["sweep_ts"]}))
        return out


# ------------------------------------------------------------------ universe
def select_symbols(exchange, priority_only, min_volume):
    markets = api_call(exchange.fetch_markets)
    if not markets:
        print("Could not load Delta markets")
        return []
    pool = []
    for m in markets:
        if not (m.get("swap") or m.get("future")) or m.get("option"):
            continue
        if m.get("active") is False or not m.get("symbol"):
            continue
        base = (m.get("base") or "").upper()
        if priority_only and base not in PRIORITY:
            continue
        pool.append((base, m["symbol"]))
    pool.sort(key=lambda x: (x[0] not in PRIORITY, x[0], x[1]))
    print(f"Active futures/perpetuals on Delta India: {len(pool)}")

    if min_volume <= 0:
        return [s for _, s in pool]

    tickers = api_call(exchange.fetch_tickers) or {}
    out = []
    for _, sym in pool:
        t = tickers.get(sym) or api_call(exchange.fetch_ticker, sym)
        if not t:
            continue
        q = t.get("quoteVolume")
        if q is None and t.get("baseVolume") is not None and t.get("last") is not None:
            q = float(t["baseVolume"]) * float(t["last"])
        if q is not None and float(q) >= min_volume:
            out.append(sym)
    print(f"After volume filter (>= {min_volume:,.0f}): {len(out)}")
    return out


# ------------------------------------------------------------------ scheduling
def next_15m_boundary():
    now = datetime.now(timezone.utc)
    return now.replace(second=0, microsecond=0, minute=0) + timedelta(minutes=(now.minute // 15 + 1) * 15)


def wait_until(dt):
    while (dt - datetime.now(timezone.utc)).total_seconds() > 0:
        time.sleep(min((dt - datetime.now(timezone.utc)).total_seconds(), 5))


# ------------------------------------------------------------------ run
def scan_all(scanner, symbols, dry_run, early=False, verbose=False):
    hits = 0
    t0 = time.time()
    for i, s in enumerate(symbols, 1):
        if verbose:
            print(scanner.status(s))
        try:
            sigs = scanner.evaluate(s, early=early)
        except Exception as exc:
            print(f"  evaluate {s}: {type(exc).__name__}: {exc}")
            continue
        for sig in sigs:
            hits += 1
            print("SIGNAL:", json.dumps(sig, default=str))
            if not dry_run:
                notify_telegram(sig)
        if i % 25 == 0:
            print(f"  ... {i}/{len(symbols)} scanned ({time.time() - t0:.0f}s)")
    return hits


def refresh_many(scanner, symbols, tfs, keep_forming=False):
    t0 = time.time()
    for i, s in enumerate(symbols, 1):
        for tf in tfs:
            try:
                scanner.refresh_tf(s, tf, keep_forming)
            except Exception as exc:
                print(f"  {s} {tf}: {type(exc).__name__}: {exc}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(symbols)} ({time.time() - t0:.0f}s)")
    return time.time() - t0


def early_poll(scanner, symbols, args, until):
    while datetime.now(timezone.utc) < until:
        took = refresh_many(scanner, symbols, ["15m"], keep_forming=True)
        hits = scan_all(scanner, symbols, args.dry_run, early=True)
        print(f"{datetime.now(IST):%H:%M:%S IST} early poll: {took:.0f}s, early signals={hits}")
        remaining = (until - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            break
        time.sleep(max(0, min(args.poll - took, remaining)))


# def run(args):
#     exchange = build_exchange()
#     symbols  = select_symbols(exchange, args.priority_only, args.min_volume)
#     if not symbols:
#         print("No symbols selected")
#         return
#     print(f"Scanning {len(symbols)} symbol(s) | pattern={args.pattern} | ema_mode={args.ema_mode} | sweep_bias={args.sweep_bias}")

#     scanner = Scanner(exchange, symbols, args)
#     print("Warming up (4h, 1h, 15m for every symbol) ...")
#     took = refresh_many(scanner, symbols, TIMEFRAMES)
#     print(f"Warm-up done in {took:.0f}s")

#     hits = scan_all(scanner, symbols, args.dry_run, verbose=args.verbose)
#     print(f"Initial scan: signals={hits}")
#     if args.once:
#         return

#     print("Live loop started.")
#     while True:
#         boundary = next_15m_boundary()
#         wake = boundary + timedelta(seconds=args.buffer)
#         print(f"Next 15M close: {boundary.astimezone(IST):%H:%M IST}")
#         if args.poll > 0:
#             early_poll(scanner, symbols, args, until=wake)
#         else:
#             wait_until(wake)

#         tfs = ["15m"]
#         if boundary.minute == 0:
#             tfs.append("1h")
#         if boundary.minute == 0 and boundary.hour % 4 == 0:
#             tfs.append("4h")
#         took = refresh_many(scanner, symbols, tfs)
#         hits = scan_all(scanner, symbols, args.dry_run)
#         print(f"{datetime.now(IST):%H:%M:%S IST} confirmed scan ({', '.join(tfs)}): {took:.0f}s, signals={hits}")

def run(args):
    exchange = build_exchange()
    symbols  = select_symbols(exchange, args.priority_only, args.min_volume)
    if not symbols:
        print("No symbols selected")
        return
    print(f"Scanning {len(symbols)} symbol(s) | pattern={args.pattern} | ema_mode={args.ema_mode} | sweep_bias={args.sweep_bias}")

    scanner = Scanner(exchange, symbols, args)
    
    # Start scanning immediately (data fetched on the fly)
    print("Starting live scan (fetching data on the fly)...")
    hits = scan_all(scanner, symbols, args.dry_run, verbose=args.verbose)
    print(f"Scan complete: signals={hits}")
    return

# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Delta India 15M entry scanner: EMA cross + liquidity sweep/impulse")
    p.add_argument("--once", action="store_true", help="One scan and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print signals, no Telegram.")
    p.add_argument("--verbose", action="store_true", help="Print one status line per symbol on the initial scan.")
    # universe
    p.add_argument("--priority-only", action=argparse.BooleanOptionalAction, default=False,
                   help="Only the PRIORITY list. Default: every active futures pair.")
    p.add_argument("--min-volume", type=float, default=0,
                   help="Skip pairs under this 24h quote volume. 0 = no filter (all pairs).")
    p.add_argument("--candles", type=int, default=300)
    # timing
    p.add_argument("--poll", type=int, default=90, help="Seconds between forming-bar checks (EARLY alerts). 0 = off.")
    p.add_argument("--buffer", type=int, default=8, help="Seconds after a 15M close before the confirmed fetch.")
    # patterns
    p.add_argument("--pattern", choices=["cross", "sweep", "both"], default="both")
    p.add_argument("--ema-mode", choices=["920", "920100"], default="920")
    # cross
    p.add_argument("--cross-lookback", type=int, default=2)
    p.add_argument("--min-score", type=int, default=60)
    p.add_argument("--pullback-lookback", type=int, default=12)
    p.add_argument("--max-ext-atr", type=float, default=1.5)
    p.add_argument("--stop-lookback", type=int, default=10)
    # sweep
    p.add_argument("--sweep-bias", choices=["off", "1h", "both"], default="off",
                   help="Require 1H (or 4H+1H) bias to agree with the sweep direction. Default off: it is a reversal pattern.")
    p.add_argument("--level-lookback", type=int, default=30, help="Bars that define the swing level being swept.")
    p.add_argument("--sweep-window", type=int, default=6, help="Recent bars in which the sweep wick must occur.")
    p.add_argument("--min-body-atr", type=float, default=1.0, help="Impulse bar body must be >= this x ATR.")
    p.add_argument("--min-body-ratio", type=float, default=0.55, help="Impulse bar body must be >= this share of its range.")
    p.add_argument("--vol-mult", type=float, default=1.5, help="Impulse bar volume must be >= this x 20-bar average.")
    p.add_argument("--max-risk-pct", type=float, default=5.0,
                   help="Skip if the stop (entry to sweep wick) is wider than this %% of price - 1:2 unrealistic.")
    p.add_argument("--require-cross", action="store_true", help="SWEEP only when EMA9/20 has already crossed (no 'pending').")
    p.add_argument("--atr-buffer", type=float, default=0.25)
    run(p.parse_args())