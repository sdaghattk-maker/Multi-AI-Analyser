#!/usr/bin/env python3
"""
Backtest of the app's technical engine (analyze_market from app.py) on historical Binance candles.

Usage examples:
  python backtest.py --app app.py --tfs 1h --years 1 --step 6 --out bt_1h.csv
  python backtest.py --app app.py --tfs 15m --years 1 --step 12 --out bt_15m.csv
  python backtest.py --app app.py --tfs 5m --years 1 --step 24 --out bt_5m.csv
  python backtest.py --app app.py --synthetic            # offline smoke test

Needs: python 3.9+, numpy, pandas, requests, plotly, matplotlib (same as the app).
"""
import argparse, os, sys, types, time, zlib
import numpy as np
import pandas as pd
import requests

HOSTS = ["https://data-api.binance.vision", "https://data.binance.com",
         "https://api.binance.com", "https://api.binance.us"]
IV_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
         "4h": 14_400_000, "1d": 86_400_000}
COLS = ["t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tb", "tq", "ig"]


def load_engine(app_path):
    src = open(app_path, encoding="utf-8").read()
    cut = src.index("import streamlit.components.v1 as components")
    stub = types.ModuleType("streamlit")
    stub.session_state, stub.secrets = {}, {}
    def deco(*a, **k):
        def d(f):
            f.clear = lambda *x, **y: None
            return f
        return d(a[0]) if len(a) == 1 and callable(a[0]) and not k else d
    stub.cache_data = deco; stub.cache_resource = deco
    sys.modules["streamlit"] = stub
    ns = {"__name__": "engine"}
    exec(compile(src[:cut], app_path, "exec"), ns)
    return ns["analyze_market"]


def fetch(symbol, tf, years, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{tf}_{years}y.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    end = int(time.time() * 1000)
    cur = end - int(years * 365.25 * 86_400_000)
    rows = []
    while cur < end:
        data, err = None, ""
        for h in HOSTS:
            try:
                r = requests.get(f"{h}/api/v3/klines", timeout=20,
                                 params=dict(symbol=symbol, interval=tf, startTime=cur, limit=1000))
                if r.status_code == 200:
                    data = r.json(); break
                err = f"{h}: {r.status_code}"
            except Exception as e:
                err = f"{h}: {e}"
        if data is None:
            raise RuntimeError(f"fetch failed for {symbol} {tf}: {err}")
        if not data:
            break
        rows += data
        cur = data[-1][0] + IV_MS[tf]
        if len(data) < 1000:
            break
        time.sleep(0.12)
    df = pd.DataFrame(rows, columns=COLS)[["t", "open", "high", "low", "close", "volume"]].astype(float)
    df = df.drop_duplicates("t").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def synthetic(n, seed):
    r = np.random.default_rng(seed)
    ret = r.normal(0, 0.004, n)
    c = 100 * np.exp(np.cumsum(ret)); o = np.r_[100, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.002, n)))
    l = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.002, n)))
    return pd.DataFrame(dict(t=np.arange(n) * 3_600_000.0, open=o, high=h, low=l, close=c,
                             volume=np.abs(r.normal(100, 30, n))))


def prep(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["t"], unit="ms")
    return df


def simulate(df, ti, plan, side, wait, maxbar, fee):
    d = 1 if side == "long" else -1
    entry, sl, tp = plan["entry"], plan["sl"], plan["tp1"]
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    H, L, C = df.high.values, df.low.values, df.close.values
    n = len(df); f = None
    if plan["dist_atr"] == 0:
        f = ti
    else:
        for k in range(ti + 1, min(n, ti + 1 + wait)):
            if (d == 1 and L[k] <= entry) or (d == -1 and H[k] >= entry):
                f = k; break
    if f is None:
        return dict(filled=False, R=np.nan, exit_bar=ti + wait, exit="NOFILL")
    cost = 2 * fee * entry / risk
    start = f if f != ti else ti + 1
    for k in range(start, min(n, f + maxbar)):
        hit_sl = (L[k] <= sl) if d == 1 else (H[k] >= sl)
        hit_tp = (H[k] >= tp) if d == 1 else (L[k] <= tp)
        if hit_sl:
            return dict(filled=True, R=-1.0 - cost, exit_bar=k, exit="SL")
        if hit_tp:
            return dict(filled=True, R=abs(tp - entry) / risk - cost, exit_bar=k, exit="TP")
    k = min(n, f + maxbar) - 1
    return dict(filled=True, R=float((C[k] - entry) * d / risk) - cost, exit_bar=k, exit="TIMEOUT")


def walk(analyze, df, symbol, tf, a):
    rows, busy, labels = [], -1, None
    sh = 1 if a.drops_last else 0
    for ti in range(a.window - 1, len(df) - 2, a.step):
        if not a.overlap and ti <= busy:
            continue
        w = df.iloc[ti - a.window + 1 + sh: ti + 1 + sh].reset_index(drop=True)
        try:
            A = analyze(w)
        except Exception:
            continue
        side = A["side"]; other = "short" if side == "long" else "long"
        plan = A["plans"][side]
        if plan is None:
            continue
        sim = simulate(df, ti, plan, side, a.wait, a.maxbar, a.fee_bps / 1e4)
        if sim is None:
            continue
        ctrl = np.nan
        if A["plans"][other] is not None:
            cs = simulate(df, ti, A["plans"][other], other, a.wait, a.maxbar, a.fee_bps / 1e4)
            if cs and cs["filled"]:
                ctrl = cs["R"]
        items = A["scores"][side]["items"]
        labels = labels or [i["label"] for i in items]
        row = dict(symbol=symbol, tf=tf, idx=ti, t=df.t.iloc[ti], verdict=A["verdict"], side=side,
                   pct=A["scores"][side]["pct"], opct=A["scores"][other]["pct"], rr=plan["rr"],
                   dist_atr=plan["dist_atr"], kind=plan["kind"], filled=sim["filled"], R=sim["R"],
                   exit=sim["exit"], ctrl_R=ctrl)
        for j, it in enumerate(items):
            row[f"it{j}"] = int(it["ok"])
        rows.append(row)
        busy = sim["exit_bar"] if sim["filled"] else ti + a.wait
    return rows, labels


def st(R, boot=2000, seed=0):
    R = np.asarray([x for x in R if x == x], float)
    if len(R) == 0:
        return dict(n=0)
    g = np.random.default_rng(seed)
    m = [g.choice(R, len(R)).mean() for _ in range(boot)] if len(R) > 2 else [R.mean()]
    gains, losses = R[R > 0].sum(), -R[R < 0].sum()
    eq = np.cumsum(R); dd = float((np.maximum.accumulate(eq) - eq).max()) if len(R) else 0
    return dict(n=len(R), win=round(float((R > 0).mean()), 3), avgR=round(float(R.mean()), 3),
                ci_lo=round(float(np.percentile(m, 2.5)), 3), ci_hi=round(float(np.percentile(m, 97.5)), 3),
                PF=round(float(gains / losses), 2) if losses > 0 else np.inf, maxDD_R=round(dd, 1))


def table(df, by, name):
    out = []
    for k, g in df.groupby(by, observed=True):
        s = st(g.loc[g.filled, "R"])
        label = " / ".join(map(str, k if isinstance(k, tuple) else (k,)))
        out.append({"group": label, "signals": len(g), "fill%": round(100 * g.filled.mean(), 0), **s})
    res = pd.DataFrame(out)
    print(f"\n=== {name} ===")
    print(res.to_string(index=False) if len(res) else "(no data)")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="app.py")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--tfs", default="1h")
    p.add_argument("--years", type=float, default=1)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--step", type=int, default=6)
    p.add_argument("--wait", type=int, default=24)
    p.add_argument("--maxbar", type=int, default=150)
    p.add_argument("--fee-bps", dest="fee_bps", type=float, default=7.0, help="per side, basis points (fee+slippage)")
    p.add_argument("--drops-last", dest="drops_last", type=int, default=1,
                   help="1 if analyze_market discards the last candle (df.iloc[:-1]); 0 otherwise")
    p.add_argument("--overlap", action="store_true", help="allow overlapping trades (default: one at a time)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n", type=int, default=3000, help="candles per series for --synthetic")
    p.add_argument("--cache", default="bt_cache")
    p.add_argument("--out", default="bt_trades.csv")
    a = p.parse_args()

    analyze = load_engine(a.app)
    all_rows, labels = [], None
    for sym in a.symbols.split(","):
        for tf in a.tfs.split(","):
            t0 = time.time()
            df = prep(synthetic(a.n, zlib.crc32((sym + tf).encode()) % 10_000) if a.synthetic else fetch(sym, tf, a.years, a.cache))
            rows, lab = walk(analyze, df, sym, tf, a)
            labels = labels or lab
            all_rows += rows
            print(f"{sym} {tf}: {len(df)} candles, {len(rows)} signals, {time.time() - t0:.0f}s")
    if not all_rows:
        print("no signals"); return
    D = pd.DataFrame(all_rows).sort_values("t").reset_index(drop=True)
    D.to_csv(a.out, index=False)
    cut = D.t.quantile(0.6)
    D["split"] = np.where(D.t <= cut, "train(60%)", "test(40%)")

    table(D, ["verdict"], "By verdict (all periods)")
    table(D, ["split", "verdict"], "By verdict and period (the TEST rows are the honest ones)")
    D["pct_bucket"] = pd.cut(D.pct, [0, 0.5, 0.65, 0.8, 1.01], right=False)
    table(D, ["pct_bucket"], "By rule-score bucket (should rise from left to right if the score means anything)")
    table(D[D.verdict == "ENTER"], ["symbol", "tf"], "ENTER by symbol/timeframe")

    F = D[D.filled]
    print("\n=== Control: engine side vs the OPPOSITE side plan (same windows) ===")
    for v in ("ENTER", "WATCH", "NO_TRADE"):
        g = F[F.verdict == v]
        if len(g):
            c = g.ctrl_R.dropna()
            print(f"{v:9s} chosen side avgR={g.R.mean():+.3f} (n={len(g)}) | opposite side avgR={c.mean():+.3f} (n={len(c)})")

    print("\n=== Checklist items: avg R when the item passed vs failed (all filled trades) ===")
    print("labels in order:", labels)
    rows = []
    for j, lab in enumerate(labels or []):
        ok, no = F[F[f"it{j}"] == 1].R, F[F[f"it{j}"] == 0].R
        if len(ok) > 20 and len(no) > 20:
            se = np.sqrt(ok.var() / len(ok) + no.var() / len(no))
            rows.append(dict(item=lab[:48], n_pass=len(ok), n_fail=len(no), avgR_pass=round(ok.mean(), 3),
                             avgR_fail=round(no.mean(), 3), diff=round(ok.mean() - no.mean(), 3),
                             t=round((ok.mean() - no.mean()) / se, 1) if se > 0 else np.nan))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nTrades saved to {a.out}. fee={a.fee_bps} bps/side, one-trade-at-a-time={not a.overlap}.")
    print("Reading guide: an edge needs avgR>0 with the CI (ci_lo..ci_hi) above 0 in the TEST period, "
          "beating the opposite-side control, on more than one symbol.")


if __name__ == "__main__":
    main()
