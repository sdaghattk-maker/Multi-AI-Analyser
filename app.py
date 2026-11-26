
import base64, json, time, sqlite3, os, re, concurrent.futures as cf
import numpy as np, pandas as pd
import requests, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
]
JUDGE_DEFAULT = "nvidia/nemotron-3-ultra-550b-a55b:free"

BAD_WORDS = ("lyria", "tts", "audio", "music", "image", "video", "embed",
             "whisper", "speech", "voice", "sora", "veo", "imagen", "flux",
             "diffusion", "moderation", "guard", "router", "openrouter/")

BINANCE_HOSTS = ["https://data-api.binance.vision", "https://data.binance.com",
                 "https://api.binance.com", "https://api.binance.us"]
TF_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1H": "1h",
          "4H": "4h", "1D": "1d", "1W": "1w", "1M": "1M"}
NO_TF = "— انتخاب نشده —"

ANALYST_PROMPT = """تو یک تحلیلگر تکنیکال دقیق و محتاط هستی. فقط بر اساس اطلاعات قابل مشاهده در تصویر تحلیل کن.
فقط یک JSON معتبر برگردان (بدون متن اضافه) با این کلیدها:
{
 "symbol": str|null, "timeframe": str|null, "last_price": number|null,
 "trend": "bullish|bearish|sideways|unclear",
 "market_structure": str,
 "support": [{"price": number, "reason": str, "confidence": 0-1}],
 "resistance": [{"price": number, "reason": str, "confidence": 0-1}],
 "patterns": [{"name": str, "evidence": str, "confidence": 0-1}],
 "indicators_seen": [{"name": str, "reading": str}],
 "scenarios": {
   "bullish": {"trigger": str, "target": str, "invalidation": str},
   "bearish": {"trigger": str, "target": str, "invalidation": str},
   "neutral": {"description": str}},
 "claims": [{"text": str, "type": "observed|assumed"}],
 "unknowns": [str],
 "overall_confidence": 0-1
}
قوانین: هر ادعا را observed یا assumed علامت بزن. اگر قیمتی خوانا نیست حدس نزن و null بگذار.
قیمت‌ها فقط عدد باشند. توصیه قطعی خرید/فروش یا وعده سود نده. متن‌ها فارسی و کلیدها انگلیسی باشند."""

JUDGE_PROMPT = """چند تحلیل JSON از یک چارت واحد داری. کار تو داور است:
1) موارد مورد توافق (حداقل دو مدل) را جدا کن.
2) اختلافات را با ذکر نام مدل‌ها بنویس.
3) ادعاهای assumed که فقط یک مدل گفته را «ضعیف» علامت بزن.
4) سناریوهای نهایی مشروط با ابطال (invalidation) بده، نه توصیه قطعی.
5) چک‌لیست تأیید دستی با داده زنده بده.
خروجی فارسی و به‌صورت Markdown با جدول باشد. این محتوا مشاوره مالی نیست."""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');

html, body, .stApp, .stMarkdown, label, p, h1, h2, h3, h4, button, input, textarea {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stCaption, h1, h2, h3, label, [data-testid="stWidgetLabel"] p,
[data-testid="stAlert"] p {
    direction: rtl;
    text-align: right;
}

[data-testid="stSidebar"] { min-width: 300px !important; max-width: 340px !important; }

.block-container { max-width: 1100px; padding-top: 2rem; }
h1 { font-size: 1.7rem !important; }
h2 { font-size: 1.3rem !important; margin-top: 1.5rem; }

.stButton > button {
    width: 100%; border-radius: 12px; padding: 0.7rem 1rem;
    font-weight: 700; border: 0; background: #2563eb; color: #fff;
}
.stButton > button:hover { background: #1d4ed8; color: #fff; }

[data-testid="stExpander"], [data-testid="stFileUploader"] section { border-radius: 12px; }

pre, code, [data-testid="stCodeBlock"], .js-plotly-plot, [data-testid="stDataFrame"],
input[type="password"] { direction: ltr !important; text-align: left !important; }
</style>
"""

# ------------------------- AI -------------------------
def b64(file):
    return base64.b64encode(file.getvalue()).decode()

def call(key, model, messages, timeout=120):
    t = time.time()
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "temperature": 0.2}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"], round(time.time()-t, 1)

def analyze(key, model, img_b64, mime, note):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": ANALYST_PROMPT + ("\nاطلاعات واردشده توسط کاربر: " + note if note else "")},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}]}]
    try:
        out, sec = call(key, model, msgs)
        return model, out, sec, None
    except Exception as e:
        return model, None, 0, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_free_models():
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    r.raise_for_status()
    vision, text = [], []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        p = m.get("pricing", {}) or {}
        try:
            free = float(p.get("prompt", 1)) == 0 and float(p.get("completion", 1)) == 0
        except Exception:
            free = False
        if not free or any(w in mid.lower() for w in BAD_WORDS):
            continue
        arch = m.get("architecture", {}) or {}
        ins = arch.get("input_modalities", []) or []
        outs = arch.get("output_modalities", ["text"]) or ["text"]
        if outs != ["text"]:
            continue
        if (m.get("context_length") or 0) < 8000:
            continue
        (vision if "image" in ins else text).append(mid)
    return sorted(vision), sorted(text)

def parse_json(s):
    if not s:
        return None
    m = re.search(r"\{.*\}", s.strip(), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def to_num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def short(m):
    return m.split("/")[-1].replace(":free", "")[:22]

def consensus(levels, tol=0.003):
    levels = sorted(levels, key=lambda x: x[0])
    groups = []
    for price, model in levels:
        if groups and abs(price - groups[-1]["p"][-1]) / groups[-1]["p"][-1] <= tol:
            groups[-1]["p"].append(price); groups[-1]["m"].add(model)
        else:
            groups.append({"p": [price], "m": {model}})
    return [(sum(g["p"]) / len(g["p"]), len(g["m"])) for g in groups if len(g["m"]) >= 2]

def ai_summary(parsed):
    if not parsed:
        return None, []
    bull = sum(1 for j in parsed.values() if j.get("trend") == "bullish")
    bear = sum(1 for j in parsed.values() if j.get("trend") == "bearish")
    trend = 1 if bull > bear else (-1 if bear > bull else 0)
    levels = []
    for m, j in parsed.items():
        for key in ("support", "resistance"):
            for it in j.get(key, []) or []:
                p = to_num(it.get("price")) if isinstance(it, dict) else None
                if p is not None:
                    levels.append((p, m))
    return trend, [p for p, _ in consensus(levels)]

def build_figure(parsed):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28],
        subplot_titles=("سطوح حمایت/مقاومت هر مدل", "اطمینان کلی مدل‌ها"))
    levels, names, conf, colors = [], [], [], []
    tcol = {"bullish": "#16a34a", "bearish": "#dc2626", "sideways": "#6b7280", "unclear": "#9ca3af"}
    for m, j in parsed.items():
        n = short(m); names.append(n)
        conf.append(to_num(j.get("overall_confidence")) or 0)
        colors.append(tcol.get(str(j.get("trend")), "#9ca3af"))
        for key, color, sym in (("support", "#16a34a", "triangle-up"), ("resistance", "#dc2626", "triangle-down")):
            for it in j.get(key, []) or []:
                p = to_num(it.get("price")) if isinstance(it, dict) else None
                if p is None:
                    continue
                levels.append((p, m))
                fig.add_trace(go.Scatter(x=[n], y=[p], mode="markers",
                    marker=dict(size=14, color=color, symbol=sym),
                    hovertext=str(it.get("reason", ""))[:200], showlegend=False), row=1, col=1)
    for p, c in consensus(levels):
        fig.add_hline(y=p, line_dash="dot", line_color="#2563eb", row=1, col=1,
            annotation_text=f"توافق {c} مدل: {p:.5g}", annotation_position="top left")
    fig.add_trace(go.Bar(x=names, y=conf, marker_color=colors, showlegend=False,
        text=[f"{c:.0%}" for c in conf], textposition="outside"), row=1, col=2)
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def scenario_table(parsed):
    rows = []
    for m, j in parsed.items():
        sc = j.get("scenarios", {}) or {}
        b, d = sc.get("bullish", {}) or {}, sc.get("bearish", {}) or {}
        rows.append({"مدل": short(m), "روند": j.get("trend"), "نماد": j.get("symbol"),
            "تایم‌فریم": j.get("timeframe"), "قیمت آخر": j.get("last_price"),
            "صعودی: ورود": b.get("trigger"), "صعودی: ابطال": b.get("invalidation"),
            "نزولی: ورود": d.get("trigger"), "نزولی: ابطال": d.get("invalidation"),
            "ناشناخته‌ها": "؛ ".join(map(str, j.get("unknowns", []) or []))})
    return rows

# ------------------------- Market data -------------------------
def normalize_symbol(s):
    s = s.upper().strip().replace("/", "").replace("-", "").replace(" ", "").replace(".P", "")
    if s.endswith("USD"):
        s += "T"
    return s

@st.cache_data(ttl=60, show_spinner=False)
def fetch_klines(symbol, interval, limit=500):
    last = "unknown"
    for h in BINANCE_HOSTS:
        try:
            r = requests.get(f"{h}/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
            if r.status_code == 200:
                df = pd.DataFrame(r.json(), columns=["t", "open", "high", "low", "close", "volume",
                    "ct", "qv", "n", "tb", "tq", "ig"])
                df = df[["t", "open", "high", "low", "close", "volume"]].astype(float)
                df["time"] = pd.to_datetime(df["t"], unit="ms")
                return df.reset_index(drop=True), h
            last = f"{h}: {r.status_code} {r.text[:120]}"
        except Exception as e:
            last = f"{h}: {e}"
    raise RuntimeError(last)

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi_calc(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr_calc(df, n=14):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def find_swings(d, n=3):
    hi, lo = d.high.values, d.low.values
    sh, sl = [], []
    for i in range(n, len(d) - n):
        if hi[i] > hi[i-n:i].max() and hi[i] >= hi[i+1:i+n+1].max():
            sh.append((i, float(hi[i])))
        if lo[i] < lo[i-n:i].min() and lo[i] <= lo[i+1:i+n+1].min():
            sl.append((i, float(lo[i])))
    return sh, sl

def structure(d, sh, sl, n=3):
    cl = d.close.values
    shm, slm = dict(sh), dict(sl)
    events, trend = [], 0
    last_h = last_l = None
    for i in range(len(d)):
        j = i - n
        if j in shm: last_h = [j, shm[j], False]
        if j in slm: last_l = [j, slm[j], False]
        if last_h and not last_h[2] and cl[i] > last_h[1]:
            events.append(dict(i=i, dir=1, kind="BOS" if trend >= 0 else "CHoCH",
                               level=last_h[1], origin=last_h[0]))
            last_h[2] = True; trend = 1
        if last_l and not last_l[2] and cl[i] < last_l[1]:
            events.append(dict(i=i, dir=-1, kind="BOS" if trend <= 0 else "CHoCH",
                               level=last_l[1], origin=last_l[0]))
            last_l[2] = True; trend = -1
    return events, trend

def find_obs(d, events, at):
    o, h, l, c = d.open.values, d.high.values, d.low.values, d.close.values
    a = at.values
    obs = {}
    for ev in events:
        i, dr, a0 = ev["i"], ev["dir"], ev["origin"]
        if dr == 1:
            m = a0 + int(np.argmin(l[a0:i+1]))
            if h[m:i+1].max() - l[m] < 1.5 * a[i]:
                continue
            k = next((x for x in range(m, max(m-10, -1), -1) if c[x] < o[x]), None)
            if k is None: continue
            top, bot = float(h[k]), float(l[k])
            if (c[k+1:] < bot).any(): continue
            tested = bool((l[i+1:] <= top).any()) if i + 1 < len(l) else False
        else:
            m = a0 + int(np.argmax(h[a0:i+1]))
            if h[m] - l[m:i+1].min() < 1.5 * a[i]:
                continue
            k = next((x for x in range(m, max(m-10, -1), -1) if c[x] > o[x]), None)
            if k is None: continue
            top, bot = float(h[k]), float(l[k])
            if (c[k+1:] > top).any(): continue
            tested = bool((h[i+1:] >= bot).any()) if i + 1 < len(h) else False
        obs[k] = dict(i=k, dir=dr, top=top, bottom=bot, tested=tested, event=ev["kind"])
    return sorted(obs.values(), key=lambda x: x["i"])

def find_fvg(d, at, min_atr=0.15):
    h, l, c = d.high.values, d.low.values, d.close.values
    a = at.values
    out = []
    for i in range(2, len(d)):
        if l[i] > h[i-2] and (l[i] - h[i-2]) >= min_atr * a[i]:
            bot, top, dr = float(h[i-2]), float(l[i]), 1
            if (c[i+1:] < bot).any(): continue
            tested = bool((l[i+1:] <= top).any())
        elif h[i] < l[i-2] and (l[i-2] - h[i]) >= min_atr * a[i]:
            top, bot, dr = float(l[i-2]), float(h[i]), -1
            if (c[i+1:] > top).any(): continue
            tested = bool((h[i+1:] >= bot).any())
        else:
            continue
        out.append(dict(i=i, dir=dr, top=top, bottom=bot, tested=tested))
    return out

def liquidity(d, sh, sl, n=3, tol=0.001):
    def eq(sw):
        rec, res = sw[-8:], []
        for a in range(len(rec)):
            for b in range(a + 1, len(rec)):
                pa, pb = rec[a][1], rec[b][1]
                if abs(pa - pb) / pa <= tol:
                    res.append(max(pa, pb) if sw is sh else min(pa, pb))
        return sorted(set(res))
    hi, lo, cl = d.high.values, d.low.values, d.close.values
    N = len(d)
    sweep = {"sell_side": False, "buy_side": False}
    for b in range(max(N - 8, 0), N):
        for idx, p in sl:
            if idx + n < b and lo[idx+n+1:b].min(initial=np.inf) > p and lo[b] < p < cl[b]:
                sweep["sell_side"] = True
        for idx, p in sh:
            if idx + n < b and hi[idx+n+1:b].max(initial=-np.inf) < p and hi[b] > p > cl[b]:
                sweep["buy_side"] = True
    return {"eqh": eq(sh), "eql": eq(sl), "sweep": sweep}

def premium_discount(d, sh, sl, price):
    N = len(d)
    hs = [p for i, p in sh if i >= N - 120] or [float(d.high.tail(100).max())]
    ls = [p for i, p in sl if i >= N - 120] or [float(d.low.tail(100).min())]
    hi, lo = max(hs), min(ls)
    pos = min(max((price - lo) / (hi - lo), 0), 1) if hi > lo else 0.5
    return dict(high=hi, low=lo, eq=(hi + lo) / 2, pos=pos,
                zone="premium" if pos > 0.5 else "discount")

def build_plan(side, price, atr_v, obs, fvgs, sh, sl):
    d = 1 if side == "long" else -1
    def dist(z):
        if d == 1:
            return max(0.0, price - z["top"]) if z["bottom"] <= price else None
        return max(0.0, z["bottom"] - price) if z["top"] >= price else None
    cand = [(dist(z), 0, z, "OB") for z in obs if z["dir"] == d]
    cand += [(dist(z), 1, z, "FVG") for z in fvgs if z["dir"] == d]
    cand = [c for c in cand if c[0] is not None and c[0] <= 6 * atr_v]
    if not cand:
        return None
    cand.sort(key=lambda x: (x[0] > atr_v, x[1], x[0]))
    dd, _, z, kind = cand[0]
    entry = price if dd == 0 else (z["top"] + z["bottom"]) / 2
    slp = z["bottom"] - 0.5 * atr_v if d == 1 else z["top"] + 0.5 * atr_v
    risk = abs(entry - slp)
    if risk <= 0:
        return None
    targets = sorted([p for _, p in sh if p > entry]) if d == 1 else sorted([p for _, p in sl if p < entry], reverse=True)
    tp1 = next((p for p in targets if abs(p - entry) / risk >= 1.5), None)
    src1 = "swing"
    if tp1 is None:
        tp1, src1 = entry + d * 2 * risk, "2R"
    tp2 = next((p for p in targets if abs(p - entry) > abs(tp1 - entry) * 1.2), None)
    src2 = "swing"
    if tp2 is None:
        tp2, src2 = entry + d * 3 * risk, "3R"
    return dict(side=side, zone=z, kind=kind, entry=entry, sl=slp, tp1=tp1, tp2=tp2,
                rr=abs(tp1 - entry) / risk, dist_atr=dd / atr_v, src1=src1, src2=src2)

def score_side(side, A, plan):
    d = 1 if side == "long" else -1
    items = []
    def add(label, pts, ok, detail="", na=False):
        items.append(dict(label=label, pts=pts, ok=bool(ok) and not na, detail=detail, na=na))
    ai_t, ai_l = A["ai_trend"], A["ai_levels"]
    add("ساختار بازار هم‌جهت (آخرین BOS/CHoCH)", 2, A["trend"] == d,
        {1: "صعودی", -1: "نزولی", 0: "نامشخص"}[A["trend"]])
    add("EMA50 نسبت به EMA200 هم‌جهت", 1, (A["ema50"] > A["ema200"]) if d == 1 else (A["ema50"] < A["ema200"]))
    add("قیمت در سمت درست EMA50", 1, (A["price"] > A["ema50"]) if d == 1 else (A["price"] < A["ema50"]))
    add("موقعیت در Discount (لانگ) / Premium (شورت)", 1,
        A["pd"]["zone"] == ("discount" if d == 1 else "premium"), f'{A["pd"]["pos"]:.0%} از رنج')
    add("Order Block معتبر هم‌جهت نزدیک قیمت", 2, plan and plan["kind"] == "OB",
        "" if plan else "ناحیه‌ای پیدا نشد")
    ov = bool(plan) and any(f["dir"] == d and f["bottom"] <= plan["zone"]["top"] and
                            f["top"] >= plan["zone"]["bottom"] for f in A["fvgs"])
    add("FVG پرنشده هم‌پوشان با ناحیه ورود", 1, ov)
    sw = A["liq"]["sweep"]
    add("جمع‌آوری نقدینگی اخیر در جهت مخالف (Sweep)", 1, sw["sell_side"] if d == 1 else sw["buy_side"])
    add("RSI در محدوده سالم", 1, (30 <= A["rsi"] <= 65) if d == 1 else (35 <= A["rsi"] <= 70), f'{A["rsi"]:.0f}')
    add("حجم اخیر بالاتر از میانگین", 1, A["rv"] >= 1.2, f'{A["rv"]:.1f}x')
    add("اجماع مدل‌های AI هم‌جهت", 1, ai_t == d, "AI اجرا نشده" if ai_t is None else "", na=ai_t is None)
    near = bool(plan) and any(abs(x - plan["entry"]) <= A["atr"] for x in ai_l)
    add("سطح اجماع AI نزدیک ناحیه ورود", 1, near, "AI اجرا نشده" if ai_t is None else "", na=ai_t is None)
    add("ریسک به ریوارد ≥ ۲", 1, plan and plan["rr"] >= 2, f'{plan["rr"]:.1f}' if plan else "")
    mx = sum(i["pts"] for i in items if not i["na"])
    sc = sum(i["pts"] for i in items if i["ok"])
    return dict(items=items, score=sc, max=mx, pct=sc / mx if mx else 0)

def analyze_market(df, ai_trend=None, ai_levels=None):
    d = df.iloc[:-1].reset_index(drop=True)
    price = float(df.close.iloc[-1])
    at = atr_calc(d)
    atr_v = float(at.iloc[-1])
    sh, sl = find_swings(d, 3)
    events, trend = structure(d, sh, sl, 3)
    obs = find_obs(d, events, at)
    fvgs = find_fvg(d, at)
    liq = liquidity(d, sh, sl, 3)
    pdz = premium_discount(d, sh, sl, price)
    e20, e50, e200 = ema(d.close, 20), ema(d.close, 50), ema(d.close, 200)
    rs = rsi_calc(d.close)
    vsma = d.volume.rolling(20).mean()
    rv = float(d.volume.tail(3).max() / vsma.iloc[-1]) if vsma.iloc[-1] else 1.0
    A = dict(d=d, price=price, atr=atr_v, sh=sh, sl=sl, events=events, trend=trend,
             obs=obs, fvgs=fvgs, liq=liq, pd=pdz, ema20=float(e20.iloc[-1]),
             ema50=float(e50.iloc[-1]), ema200=float(e200.iloc[-1]), rsi=float(rs.iloc[-1]),
             rv=rv, ai_trend=ai_trend, ai_levels=ai_levels or [],
             series=dict(e20=e20, e50=e50, e200=e200, rsi=rs))
    plans = {s: build_plan(s, price, atr_v, obs, fvgs, sh, sl) for s in ("long", "short")}
    scores = {s: score_side(s, A, plans[s]) for s in ("long", "short")}
    side = "long" if scores["long"]["pct"] >= scores["short"]["pct"] else "short"
    other = "short" if side == "long" else "long"
    pct, opct, plan = scores[side]["pct"], scores[other]["pct"], plans[side]
    if plan is None or pct < 0.5:
        verdict = "NO_TRADE"
    elif pct >= 0.65 and pct - opct >= 0.2 and plan["rr"] >= 2 and plan["dist_atr"] <= 1.5:
        verdict = "ENTER"
    else:
        verdict = "WATCH"
    A.update(plans=plans, scores=scores, side=side, verdict=verdict)
    return A

def fmt(p):
    if p is None: return "—"
    return f"{p:,.2f}" if p >= 100 else (f"{p:.4f}" if p >= 1 else f"{p:.6f}")

def card_html(M):
    A = M["A"]; side = A["side"]; plan = A["plans"][side]; sc = A["scores"][side]
    fa = {"long": "لانگ", "short": "شورت"}[side]
    v = A["verdict"]
    color = {"ENTER": "#16a34a" if side == "long" else "#dc2626", "WATCH": "#d97706", "NO_TRADE": "#6b7280"}[v]
    title = {"ENTER": f"سناریوی {fa} — شرایط ورود مشروط", "WATCH": f"در انتظار تأیید — گرایش {fa}",
             "NO_TRADE": "معامله نکن — همگرایی کافی نیست"}[v]
    icon = {"ENTER": "🟢" if side == "long" else "🔴", "WATCH": "🟡", "NO_TRADE": "⚪"}[v]
    def bar(label, s, c):
        w = int(s["pct"] * 100)
        return (f'<div style="margin:6px 0"><div style="font-size:.85rem">{label}: {s["score"]}/{s["max"]} ({w}%)</div>'
                f'<div style="background:#e5e7eb;border-radius:8px;height:10px"><div style="width:{w}%;background:{c};'
                f'height:10px;border-radius:8px"></div></div></div>')
    def cell(k, val, c="#111827"):
        return (f'<div style="flex:1 1 30%;min-width:120px;background:#f9fafb;border-radius:10px;padding:8px;margin:4px">'
                f'<div style="font-size:.75rem;color:#6b7280">{k}</div><div style="font-weight:700;color:{c};direction:ltr">{val}</div></div>')
    if plan:
        z = plan["zone"]
        cells = "".join([
            cell("ناحیه ورود", f'{fmt(z["bottom"])} – {fmt(z["top"])} ({plan["kind"]})'),
            cell("ورود", fmt(plan["entry"]), "#2563eb"),
            cell("حد ضرر", fmt(plan["sl"]), "#dc2626"),
            cell(f'هدف ۱ ({plan["src1"]})', fmt(plan["tp1"]), "#16a34a"),
            cell(f'هدف ۲ ({plan["src2"]})', fmt(plan["tp2"]), "#16a34a"),
            cell("ریسک به ریوارد", f'1 : {plan["rr"]:.1f}'),
            cell("فاصله قیمت تا ناحیه", f'{plan["dist_atr"]:.1f} ATR')])
    else:
        cells = '<div style="padding:8px">ناحیه معتبر (OB/FVG) نزدیک قیمت پیدا نشد؛ طرح معامله ساخته نشد.</div>'
    rows = ""
    for it in sc["items"]:
        mark = "➖" if it["na"] else ("✅" if it["ok"] else "❌")
        rows += (f'<tr><td style="padding:3px 6px">{mark}</td><td style="padding:3px 6px">{it["label"]}</td>'
                 f'<td style="padding:3px 6px;color:#6b7280;direction:ltr">{it["detail"]}</td>'
                 f'<td style="padding:3px 6px">{it["pts"]}</td></tr>')
    return (f'<div style="border:2px solid {color};border-radius:16px;padding:16px;direction:rtl;text-align:right">'
            f'<div style="font-size:1.35rem;font-weight:700;color:{color}">{icon} {title}</div>'
            f'<div style="color:#6b7280;margin:4px 0 8px;direction:ltr;text-align:right">{M["symbol"]} · {M["tf"]} · قیمت: {fmt(A["price"])} · ATR: {fmt(A["atr"])}</div>'
            f'{bar("امتیاز لانگ", A["scores"]["long"], "#16a34a")}{bar("امتیاز شورت", A["scores"]["short"], "#dc2626")}'
            f'<div style="display:flex;flex-wrap:wrap;margin-top:8px">{cells}</div>'
            f'<div style="margin-top:10px;font-weight:700">چک‌لیست تأیید ({fa})</div>'
            f'<table style="width:100%;font-size:.85rem;border-collapse:collapse">{rows}</table>'
            f'<div style="margin-top:10px;font-size:.78rem;color:#6b7280">این خروجی قانون‌محور و آموزشی است، نه توصیه مالی. '
            f'قبل از هر معامله با قیمت زنده، اخبار، کارمزد و مدیریت سرمایه خودت تأیید کن.</div></div>')

def build_market_chart(M, view=220):
    A = M["A"]; df = M["df"]; d = A["d"]
    off = max(0, len(d) - view)
    v = d.iloc[off:].reset_index(drop=True)
    t = v["time"]
    step = (t.iloc[-1] - t.iloc[-2]) if len(t) > 1 else pd.Timedelta(hours=1)
    x_end = t.iloc[-1] + step * 12
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.14, 0.24], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=t, open=v.open, high=v.high, low=v.low, close=v.close,
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626", showlegend=False), row=1, col=1)
    live = df.iloc[-1]
    fig.add_trace(go.Candlestick(x=[live["time"]], open=[live.open], high=[live.high], low=[live.low],
        close=[live.close], increasing_line_color="#86efac", decreasing_line_color="#fca5a5", showlegend=False), row=1, col=1)
    for k, c in (("e20", "#f59e0b"), ("e50", "#2563eb"), ("e200", "#7c3aed")):
        fig.add_trace(go.Scatter(x=t, y=A["series"][k].iloc[off:], mode="lines", line=dict(width=1.2, color=c),
            name=k.upper().replace("E", "EMA"), showlegend=False), row=1, col=1)
    def rect(z, color, label):
        x0 = t.iloc[max(z["i"] - off, 0)] if z["i"] >= off else t.iloc[0]
        fig.add_shape(type="rect", x0=x0, x1=x_end, y0=z["bottom"], y1=z["top"], fillcolor=color,
                      line=dict(width=0), layer="below", row=1, col=1)
        fig.add_annotation(x=x_end, y=(z["top"] + z["bottom"]) / 2, text=label, showarrow=False,
                           xanchor="right", font=dict(size=10), row=1, col=1)
    for z in A["obs"][-6:]:
        if z["i"] >= off - 80:
            rect(z, "rgba(22,163,74,0.20)" if z["dir"] == 1 else "rgba(220,38,38,0.20)",
                 ("OB+" if z["dir"] == 1 else "OB-") + (" (تست‌شده)" if z["tested"] else " (تازه)"))
    for z in A["fvgs"][-8:]:
        if z["i"] >= off - 80:
            rect(z, "rgba(37,99,235,0.18)" if z["dir"] == 1 else "rgba(234,88,12,0.18)", "FVG+" if z["dir"] == 1 else "FVG-")
    for ev in A["events"][-6:]:
        if ev["i"] >= off:
            x0 = t.iloc[max(ev["origin"] - off, 0)]; x1 = t.iloc[ev["i"] - off]
            fig.add_shape(type="line", x0=x0, x1=x1, y0=ev["level"], y1=ev["level"],
                line=dict(color="#6b7280", width=1, dash="dot"), row=1, col=1)
            fig.add_annotation(x=x1, y=ev["level"], text=ev["kind"] + ("↑" if ev["dir"] == 1 else "↓"),
                showarrow=False, yshift=10 if ev["dir"] == 1 else -10, font=dict(size=10, color="#374151"), row=1, col=1)
    for key, name in (("eqh", "EQH"), ("eql", "EQL")):
        for p in A["liq"][key]:
            fig.add_shape(type="line", x0=t.iloc[0], x1=x_end, y0=p, y1=p,
                line=dict(color="#a855f7", width=1, dash="dash"), row=1, col=1)
            fig.add_annotation(x=t.iloc[0], y=p, text=name, showarrow=False, xanchor="left",
                font=dict(size=10, color="#a855f7"), row=1, col=1)
    for p in M.get("ai_levels", []):
        fig.add_shape(type="line", x0=t.iloc[0], x1=x_end, y0=p, y1=p,
            line=dict(color="#0ea5e9", width=1.2, dash="dot"), row=1, col=1)
        fig.add_annotation(x=t.iloc[0], y=p, text="AI", showarrow=False, xanchor="left",
            font=dict(size=10, color="#0ea5e9"), row=1, col=1)
    pdz = A["pd"]
    fig.add_shape(type="line", x0=t.iloc[0], x1=x_end, y0=pdz["eq"], y1=pdz["eq"],
        line=dict(color="#9ca3af", width=1, dash="longdash"), row=1, col=1)
    fig.add_annotation(x=x_end, y=pdz["eq"], text="EQ 50%", showarrow=False, xanchor="right",
        yshift=8, font=dict(size=10, color="#6b7280"), row=1, col=1)
    plan = A["plans"][A["side"]]
    if plan:
        for key, c, name in (("entry", "#2563eb", "ورود"), ("sl", "#dc2626", "SL"),
                             ("tp1", "#16a34a", "TP1"), ("tp2", "#15803d", "TP2")):
            fig.add_shape(type="line", x0=t.iloc[-1], x1=x_end, y0=plan[key], y1=plan[key],
                line=dict(color=c, width=2), row=1, col=1)
            fig.add_annotation(x=x_end, y=plan[key], text=f"{name} {fmt(plan[key])}", showarrow=False,
                xanchor="left", font=dict(size=10, color=c), row=1, col=1)
    cols = ["#16a34a" if c >= o else "#dc2626" for o, c in zip(v.open, v.close)]
    fig.add_trace(go.Bar(x=t, y=v.volume, marker_color=cols, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=A["series"]["rsi"].iloc[off:], mode="lines",
        line=dict(color="#7c3aed", width=1.5), showlegend=False), row=3, col=1)
    for lv in (30, 70):
        fig.add_hline(y=lv, line_dash="dot", line_color="#9ca3af", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=760, margin=dict(l=5, r=60, t=20, b=10), dragmode="pan",
        xaxis_rangeslider_visible=False, hovermode="x unified", plot_bgcolor="#ffffff")
    return fig

def load_market(symbol, tf_label, parsed):
    sym = normalize_symbol(symbol)
    df, host = fetch_klines(sym, TF_MAP[tf_label], 500)
    ai_t, ai_l = ai_summary(parsed)
    A = analyze_market(df, ai_t, ai_l)
    return dict(df=df, host=host, symbol=sym, tf=tf_label, A=A, ai_levels=ai_l)

def db():
    c = sqlite3.connect("analyses.db")
    c.execute("""CREATE TABLE IF NOT EXISTS a(id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT,
      timeframe TEXT, image BLOB, outputs TEXT, verdict TEXT, outcome TEXT)""")
    return c

# ------------------------- UI -------------------------
st.set_page_config(page_title="Chart AI Hub", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("هاب تحلیل چارت چندمدلی")
st.caption("خروجی ابزار آموزشی/پژوهشی است و مشاوره مالی نیست.")

with st.sidebar:
    key = st.text_input("OpenRouter API key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    if st.button("🔄 به‌روزرسانی لیست مدل‌های رایگان"):
        fetch_free_models.clear()
    try:
        vision_free, text_free = fetch_free_models()
        st.success(f"{len(vision_free)} مدل رایگان با ورودی تصویر | {len(text_free)} مدل فقط‌متن")
    except Exception as e:
        vision_free, text_free = DEFAULT_MODELS, [JUDGE_DEFAULT]
        st.warning(f"دریافت لیست ناموفق بود: {e}")
    pick = [m for m in DEFAULT_MODELS if m in vision_free] or vision_free[:3]
    models = st.multiselect("مدل‌های تحلیلگر (رایگان، دارای تصویر)", vision_free, default=pick[:3])
    judge_opts = text_free + vision_free
    judge = st.selectbox("مدل داور", judge_opts, index=0) if judge_opts else JUDGE_DEFAULT

tab1, tab2 = st.tabs(["تحلیل", "آرشیو"])
with tab1:
    f = st.file_uploader("تصویر چارت (برای تحلیل AI)", type=["png", "jpg", "jpeg", "webp"])

    c1, c2, c3 = st.columns(3)
    symbol_in = c1.text_input("نماد بایننس", placeholder="مثلاً BTCUSDT")
    tf_in = c2.selectbox("تایم‌فریم", [NO_TF] + list(TF_MAP.keys()))
    use_date = c3.checkbox("تاریخ را وارد می‌کنم")
    date_in = c3.date_input("تاریخ چارت") if use_date else None
    extra_in = st.text_input("توضیح دیگر (اختیاری)", placeholder="مثلاً منبع داده یا اندیکاتورها")

    parts = []
    if symbol_in.strip():
        parts.append(f"نماد: {symbol_in.strip()}")
    if tf_in != NO_TF:
        parts.append(f"تایم‌فریم: {tf_in}")
    if date_in:
        parts.append(f"تاریخ چارت: {date_in.isoformat()}")
    if extra_in.strip():
        parts.append(extra_in.strip())
    note = " | ".join(parts)

    if f: st.image(f, use_container_width=True)
    can_mkt = bool(symbol_in.strip()) and tf_in != NO_TF
    b1, b2 = st.columns(2)
    run_ai = b1.button("تحلیل AI + نمودار و کارت", disabled=not (f and key and models))
    run_mkt = b2.button("فقط نمودار و کارت تصمیم", disabled=not can_mkt)

    if run_mkt:
        try:
            with st.spinner("دریافت داده بایننس..."):
                st.session_state["mkt"] = load_market(symbol_in, tf_in, {})
            st.session_state["ai"] = None
        except Exception as e:
            st.session_state["mkt"] = None
            st.error(f"دریافت داده ناموفق بود. نماد را بررسی کن (مثل BTCUSDT). جزئیات: {e}")

    if run_ai:
        img, mime = b64(f), f.type
        with st.spinner("در حال تحلیل..."), cf.ThreadPoolExecutor(len(models)) as ex:
            results = list(ex.map(lambda m: analyze(key, m, img, mime, note), models))
        outs = {m: out for m, out, sec, err in results if not err}
        parsed = {m: j for m, o in outs.items() if (j := parse_json(o))}
        verdict = None
        if len(outs) >= 2:
            with st.spinner("داور..."):
                joined = "\n\n".join(f"### {m}\n{o}" for m, o in outs.items())
                try:
                    verdict, _ = call(key, judge, [{"role": "user", "content": JUDGE_PROMPT + "\n\n" + joined}])
                except Exception as e:
                    verdict = f"خطای داور: {e}"
            c = db()
            first = next(iter(parsed.values()), {})
            c.execute("INSERT INTO a(ts,symbol,timeframe,image,outputs,verdict) VALUES(datetime('now'),?,?,?,?,?)",
                      (symbol_in.strip() or first.get("symbol"), (tf_in if tf_in != NO_TF else first.get("timeframe")),
                       f.getvalue(), json.dumps(outs, ensure_ascii=False), verdict))
            c.commit()
        st.session_state["ai"] = dict(results=results, outs=outs, parsed=parsed, verdict=verdict)
        if can_mkt:
            try:
                with st.spinner("دریافت داده بایننس..."):
                    st.session_state["mkt"] = load_market(symbol_in, tf_in, parsed)
            except Exception as e:
                st.session_state["mkt"] = None
                st.error(f"دریافت داده بایننس ناموفق بود: {e}")
        else:
            st.session_state["mkt"] = None
            st.info("برای نمودار واقعی و کارت تصمیم، نماد بایننس (مثل BTCUSDT) و تایم‌فریم را انتخاب کن.")

    M = st.session_state.get("mkt")
    if M:
        A = M["A"]
        st.markdown("## کارت تصمیم‌گیری")
        st.markdown(card_html(M), unsafe_allow_html=True)
        plan = A["plans"][A["side"]]
        if plan:
            r1, r2 = st.columns(2)
            bal = r1.number_input("موجودی حساب (USDT)", min_value=0.0, value=1000.0, step=100.0)
            rp = r2.number_input("ریسک هر معامله (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            risk_usd = bal * rp / 100
            unit = abs(plan["entry"] - plan["sl"])
            qty = risk_usd / unit if unit else 0
            st.info(f"ریسک: {risk_usd:,.2f} USDT | حجم پوزیشن: {qty:,.6f} واحد | ارزش پوزیشن: {qty*plan['entry']:,.2f} USDT")
        st.markdown("## نمودار حرفه‌ای (داده بایننس)")
        st.plotly_chart(build_market_chart(M), use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False})
        st.caption(f"منبع داده: {M['host']} | سبز/قرمز روشن = OB | آبی/نارنجی = FVG | بنفش = نقدینگی مساوی (EQH/EQL) | آبی نقطه‌چین = سطح اجماع AI | آخرین کندل ناقص است و در تشخیص‌ها استفاده نمی‌شود.")
        with st.expander("جزئیات تشخیص‌ها"):
            st.write(f"ساختار: {'صعودی' if A['trend']==1 else 'نزولی' if A['trend']==-1 else 'نامشخص'} | تعداد OB معتبر: {len(A['obs'])} | تعداد FVG پرنشده: {len(A['fvgs'])}")
            st.write(f"EQH: {[fmt(x) for x in A['liq']['eqh']]} | EQL: {[fmt(x) for x in A['liq']['eql']]}")
            st.write(f"Sweep اخیر: {A['liq']['sweep']}")

    AI = st.session_state.get("ai")
    if AI:
        st.markdown("## خروجی مدل‌ها")
        for m, out, sec, err in AI["results"]:
            with st.expander(f"{m} ({sec}s)", expanded=bool(err)):
                if err: st.error(err)
                else: st.code(out, language="json")
        if AI["parsed"]:
            st.markdown("## تحلیل گرافیکی مدل‌ها")
            st.plotly_chart(build_figure(AI["parsed"]), use_container_width=True)
            st.dataframe(scenario_table(AI["parsed"]), use_container_width=True)
            st.caption("خط آبی نقطه‌چین = سطحی که حداقل دو مدل (در ۰٫۳٪ فاصله) روی آن توافق دارند.")
        elif AI["outs"]:
            st.info("خروجی هیچ مدلی JSON معتبر نبود؛ نمودار ساخته نشد.")
        if AI["verdict"]:
            st.markdown("## جمع‌بندی داور"); st.markdown(AI["verdict"])

with tab2:
    c = db()
    rows = c.execute("SELECT id,ts,symbol,timeframe,outcome FROM a ORDER BY id DESC").fetchall()
    st.dataframe(rows)
    rid = st.number_input("id", min_value=0, step=1)
    out = st.text_input("نتیجه واقعی (مثلاً: سناریوی صعودی فعال شد / ابطال)")
    if st.button("ثبت نتیجه") and rid:
        c.execute("UPDATE a SET outcome=? WHERE id=?", (out, rid)); c.commit(); st.rerun()
