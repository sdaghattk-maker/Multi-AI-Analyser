#!/usr/bin/env python3
"""
trend_regime.py v1 - System A: daily trend/regime module (spot, long-only, buy-hold-sell).

PARAMETERS ARE FROZEN (pre-registered; do not tune on past data):
  per-asset signal  = 0.5*[mom20d>0] + 0.5*[mom30d>0]
  regime gate       = 0.5*[BTC>MA150] + 0.5*[BTC>MA200]
  target weight_i   = signal_i * gate * (25% vol target / n_assets) / vol_i, gross capped at 1.0 (no leverage)
  rebalance         = executed every 5th day (day-number mod 5 == 0); other days: look only
  costs (paper/backtest) = 10 bps per unit of turnover

  python trend_regime.py                 today's state (+ paper results if started)
  python trend_regime.py --init          start the paper portfolio from the latest closed daily candle
  python trend_regime.py --backtest      history check of this exact frozen ensemble (bt_cache reused)
  python trend_regime.py --ai-flag medium   log an AI risk flag (low/medium/high). LOGGED ONLY, never changes weights.
  python trend_regime.py --synthetic trend  offline smoke test

From an app:  from trend_regime import get_regime_state ; st = get_regime_state()
Not financial advice; research tool. Needs numpy, pandas, requests.
"""
import argparse, os, json, math, time
import numpy as np
import pandas as pd
import requests

HOSTS = ["https://data-api.binance.vision", "https://data.binance.com", "https://api.binance.com", "https://api.binance.us"]
COLS = ["t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "ig"]
DEFAULT = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,DOTUSDT,LTCUSDT,TRXUSDT,ATOMUSDT,UNIUSDT,NEARUSDT"
PPY = 365
TARGET_VOL, MAX_LEV, REBAL_DAYS, COST = 0.25, 1.0, 5, 10 / 1e4
STATE, LOG = "trend_state.json", "trend_log.csv"


def _get(symbol, start_ms):
    rows, cur = [], start_ms
    end = int(time.time() * 1000)
    while cur < end:
        data, err = None, ""
        for h in HOSTS:
            try:
                r = requests.get(f"{h}/api/v3/klines", timeout=20,
                                 params=dict(symbol=symbol, interval="1d", startTime=cur, limit=1000))
                if r.status_code == 200:
                    data = r.json(); break
                err = f"{h}: {r.status_code}"
            except Exception as e:
                err = f"{h}: {e}"
        if data is None:
            raise RuntimeError(f"fetch failed for {symbol}: {err}")
        if not data:
            break
        rows += data; cur = data[-1][0] + 86_400_000
        if len(data) < 1000:
            break
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=COLS)[["t", "ct", "close"]].astype(float).drop_duplicates("t")
    return df[df.ct < time.time() * 1000]


def load_prices(symbols, days=None, years=None, cache="bt_cache", synthetic=None):
    cols = {}
    if synthetic:
        n = 2200; g = np.random.default_rng(7); idx = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n, freq="D")
        for i in range(15):
            x = g.normal(0, 0.03, n)
            if synthetic == "trend":
                mu = np.zeros(n)
                for t in range(1, n):
                    mu[t] = 0.97 * mu[t - 1] + g.normal(0, 0.0012)
                x = x + mu
            cols["BTCUSDT" if i == 0 else f"S{i}USDT"] = pd.Series(100 * np.exp(np.cumsum(x - 0.5 * 0.03 ** 2)), index=idx)
        P = pd.DataFrame(cols)
        return P.iloc[-(days or n):] if days else P
    for s in symbols:
        try:
            if years:
                os.makedirs(cache, exist_ok=True); path = os.path.join(cache, f"{s}_1d_{years}y.csv")
                if os.path.exists(path):
                    d = pd.read_csv(path)
                else:
                    d = _get(s, int(time.time() * 1000) - int(years * 365.25 * 86_400_000))[["t", "close"]]; d.to_csv(path, index=False)
            else:
                d = _get(s, int(time.time() * 1000) - int((days or 460) * 86_400_000))
        except Exception as e:
            print("skip", s, e); continue
        cols[s] = pd.Series(d.close.values, index=pd.to_datetime(d.t, unit="ms").dt.normalize())
    P = pd.DataFrame(cols).sort_index()
    return P[~P.index.duplicated()]


def compute_weights(P):
    ret = P.pct_change()
    vol = ret.ewm(span=30, min_periods=30).std() * math.sqrt(PPY)
    s = 0.5 * ((P / P.shift(20) - 1) > 0) + 0.5 * ((P / P.shift(30) - 1) > 0)
    btc = P["BTCUSDT"] if "BTCUSDT" in P else P.iloc[:, 0]
    ma150, ma200 = btc.rolling(150).mean(), btc.rolling(200).mean()
    g = 0.5 * (btc > ma150) + 0.5 * (btc > ma200)
    n_act = vol.notna().sum(axis=1).clip(lower=1)
    raw = s.astype(float).mul(g.astype(float), axis=0).mul(TARGET_VOL / n_act, axis=0) / vol
    raw = raw.replace([np.inf, -np.inf], 0).fillna(0.0)
    gross = raw.abs().sum(axis=1)
    w_daily = raw.mul(np.minimum(1.0, MAX_LEV / gross.replace(0, np.nan)).fillna(1.0), axis=0)
    dayno = (P.index - pd.Timestamp("1970-01-01")).days
    reb = pd.Series(np.asarray(dayno) % REBAL_DAYS == 0, index=P.index)
    w = w_daily.where(reb, axis=0).ffill().fillna(0.0)
    return dict(ret=ret, vol=vol, s=s, g=g, w_daily=w_daily, w=w, reb=reb, btc=btc, ma150=ma150, ma200=ma200)


def simulate(ret, w, start_after=None):
    pos = w.shift(1).fillna(0.0)
    if start_after is not None:
        pos.loc[pos.index <= start_after] = 0.0
    gross = (pos * ret.fillna(0.0)).sum(axis=1)
    to = pos.diff().abs().sum(axis=1).fillna(0.0)
    return gross - to * COST


def next_reb(d):
    for k in range(1, REBAL_DAYS + 1):
        x = d + pd.Timedelta(days=k)
        if ((x - pd.Timestamp("1970-01-01")).days % REBAL_DAYS) == 0:
            return x


def get_regime_state(symbols=None, days=460, synthetic=None):
    syms = (symbols or DEFAULT).split(",") if isinstance(symbols, (str, type(None))) else list(symbols)
    P = load_prices(syms, days=days, synthetic=synthetic)
    c = compute_weights(P); d = P.index[-1]
    btc = c["btc"]
    return dict(date=d, regime_g=float(c["g"].iloc[-1]), btc=float(btc.iloc[-1]),
                btc_vs_ma150=float(btc.iloc[-1] / c["ma150"].iloc[-1] - 1) if c["ma150"].iloc[-1] == c["ma150"].iloc[-1] else None,
                btc_vs_ma200=float(btc.iloc[-1] / c["ma200"].iloc[-1] - 1) if c["ma200"].iloc[-1] == c["ma200"].iloc[-1] else None,
                signal=c["s"].iloc[-1].to_dict(), vol=c["vol"].iloc[-1].to_dict(),
                target_today=c["w_daily"].iloc[-1].to_dict(), effective=c["w"].iloc[-1].to_dict(),
                is_rebalance_day=bool(c["reb"].iloc[-1]), next_rebalance=next_reb(d), _calc=c, _P=P)


def stats(r):
    r = r.dropna(); eq = (1 + r).cumprod()
    return dict(sharpe=r.mean() / r.std() * math.sqrt(PPY) if r.std() > 0 else float("nan"),
                ann=r.mean() * PPY, vol=r.std() * math.sqrt(PPY), mdd=float((eq / eq.cummax() - 1).min()),
                total=float(eq.iloc[-1] - 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=DEFAULT); p.add_argument("--init", action="store_true")
    p.add_argument("--backtest", action="store_true"); p.add_argument("--years", type=float, default=6)
    p.add_argument("--ai-flag", choices=["low", "medium", "high"], default=None)
    p.add_argument("--synthetic", choices=["random", "trend"], default=None)
    a = p.parse_args()
    print("trend_regime.py v1 (frozen parameters; System A)")
    syms = a.symbols.split(",")

    if a.backtest:
        P = load_prices(syms, years=a.years, synthetic=a.synthetic)
        c = compute_weights(P); net = simulate(c["ret"], c["w"])[P.index[260]:]
        ew = c["ret"].mean(axis=1)[net.index]; bt = c["ret"]["BTCUSDT"][net.index]
        print(f"\nFrozen ensemble, net of {COST * 1e4:.0f} bps/turnover, from {net.index[0].date()} to {net.index[-1].date()} ({P.shape[1]} assets)")
        rows = {"STRATEGY": stats(net), "EW buy&hold": stats(ew), "BTC buy&hold": stats(bt)}
        t = pd.DataFrame(rows).T
        for k in ("ann", "vol", "mdd", "total"): t[k] = (t[k] * 100).round(1).astype(str) + "%"
        t["sharpe"] = t["sharpe"].astype(float).round(2); print(t[["sharpe", "ann", "vol", "mdd", "total"]].to_string())
        print("\nYearly return / worst drawdown (first/last years partial):")
        out = []
        for y in sorted(set(net.index.year)):
            row = {"year": y}
            for n_, r in (("STRAT", net), ("EW_BH", ew), ("BTC_BH", bt)):
                x = r[r.index.year == y]; eq = (1 + x).cumprod()
                row[n_ + " ret"] = f"{(eq.iloc[-1] - 1) * 100:.0f}%"; row[n_ + " dd"] = f"{(eq / eq.cummax() - 1).min() * 100:.0f}%"
            out.append(row)
        print(pd.DataFrame(out).to_string(index=False))
        print("\nExpectation check: the parameter-sweep plateau had Sharpe ~0.9-1.2 and max drawdown ~12-22% in-sample; "
              "expect LOWER out-of-sample (second-half Sharpe was 0.2-0.8).")
        return

    st = get_regime_state(a.symbols, synthetic=a.synthetic)
    P, c, d = st.pop("_P"), st.pop("_calc"), st["date"]
    g = st["regime_g"]
    label = {1.0: "ریسک‌پذیر: بیت‌کوین بالای هر دو میانگین ۱۵۰ و ۲۰۰ روزه", 0.5: "نیمه‌محتاط: بیت‌کوین فقط بالای یکی از دو میانگین",
             0.0: "نقد: بیت‌کوین زیر هر دو میانگین"}[g]
    print(f"\nآخرین کندل بسته‌شده: {d.date()}   قیمت BTC: {st['btc']:.0f}")
    if st["btc_vs_ma150"] is not None:
        print(f"فاصله BTC از MA150: {st['btc_vs_ma150'] * 100:+.1f}%   از MA200: {st['btc_vs_ma200'] * 100:+.1f}%")
    print(f"وضعیت رژیم: {label}   (gate={g})")
    rows = []
    for k in P.columns:
        if st["effective"][k] > 0 or st["target_today"][k] > 0 or st["signal"][k] > 0:
            rows.append((k, st["signal"][k], f"{st['vol'][k] * 100:.0f}%", f"{st['target_today'][k] * 100:.1f}%", f"{st['effective'][k] * 100:.1f}%"))
    if rows:
        print("\nنماد  | سیگنال(۰/۰٫۵/۱) | نوسان سالانه | وزن هدف امروز | وزن اجرایی (آخرین بازتنظیم)")
        for r in rows: print("  ".join(str(x) for x in r))
    cash_t = 1 - sum(st["target_today"].values()); cash_e = 1 - sum(st["effective"].values())
    print(f"\nنقد: هدف امروز {cash_t * 100:.0f}%  |  اجرایی {cash_e * 100:.0f}%")
    print(f"امروز روز بازتنظیم است؟ {'بله: وزن‌های «هدف امروز» را اجرا کن' if st['is_rebalance_day'] else 'خیر: فقط نگاه کن'}   | بازتنظیم بعدی: {st['next_rebalance'].date()}")

    nav = None
    if a.init:
        if os.path.exists(STATE):
            print(f"\n{STATE} already exists (start {json.load(open(STATE))['start']}); delete it to restart.")
        else:
            json.dump({"start": str(d.date())}, open(STATE, "w")); print(f"\nPaper portfolio started after close of {d.date()} (starts flat; first positions from the weights above).")
    if os.path.exists(STATE):
        s0 = pd.Timestamp(json.load(open(STATE))["start"])
        net = simulate(c["ret"], c["w"], start_after=s0)[lambda x: x.index > s0]
        if len(net):
            nav = float((1 + net).cumprod().iloc[-1]); sb = (1 + c["ret"]["BTCUSDT"][net.index]).prod() - 1
            eq = (1 + net).cumprod(); dd = float((eq / eq.cummax() - 1).min())
            print(f"\nمعامله کاغذی از {s0.date()}: NAV={nav:.4f} ({(nav - 1) * 100:+.2f}%)  بیشترین افت {dd * 100:.1f}%  |  BTC در همان مدت {sb * 100:+.1f}%  ({len(net)} روز)")
    row = dict(date=str(d.date()), gate=g, btc_vs_ma150=st["btc_vs_ma150"], btc_vs_ma200=st["btc_vs_ma200"], cash_target=round(cash_t, 4),
               cash_effective=round(cash_e, 4), reb_day=st["is_rebalance_day"], nav=nav, ai_flag=a.ai_flag,
               weights_effective=json.dumps({k: round(v, 4) for k, v in st["effective"].items() if v > 0}))
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(old):
        old = old[old.date != row["date"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)
    print(f"\nثبت شد: {LOG} (یک ردیف برای هر روز؛ اجرای دوباره همان روز، ردیف را به‌روز می‌کند)")


if __name__ == "__main__":
    main()
