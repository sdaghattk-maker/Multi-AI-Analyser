
import base64, io, json, time, sqlite3, os, re, datetime, concurrent.futures as cf
import numpy as np, pandas as pd
import requests, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------- Providers -------------------------
PROVIDERS = {
    "gemini": dict(label="Gemini", base="https://generativelanguage.googleapis.com/v1beta/openai",
                   secret="GEMINI_API_KEY"),
    "groq": dict(label="Groq", base="https://api.groq.com/openai/v1", secret="GROQ_API_KEY"),
    "openrouter": dict(label="OpenRouter", base="https://openrouter.ai/api/v1", secret="OPENROUTER_API_KEY"),
}
ORDER = ["gemini", "groq", "openrouter"]
KEYS = {}

PREF = {
    ("gemini", "any"): ["gemini-3.8-flash", "gemini-3-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"],
    ("groq", "text"): ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
    ("groq", "vision"): ["qwen/qwen3.8-27b"],
    ("openrouter", "vision"): ["google/gemma-4-26b-a4b-it:free", "nvidia/nemotron-3.5-super-vl:free",
                               "google/gemma-4-31b-it:free"],
    ("openrouter", "text"): ["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free",
                             "nvidia/nemotron-3.5-super-vl:free", "nvidia/nemotron-3-ultra-550b-a55b:free"],
}
STATIC_MODELS = {
    "gemini": dict(vision=["gemini-3.8-flash", "gemini-3-flash", "gemini-3.1-flash-lite"],
                   text=["gemini-3.8-flash", "gemini-3-flash", "gemini-3.1-flash-lite"]),
    "groq": dict(vision=["qwen/qwen3.8-27b"],
                 text=["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]),
    "openrouter": dict(vision=PREF[("openrouter", "vision")], text=PREF[("openrouter", "text")]),
}
GEM_BAD = ("embedding", "image", "imagen", "veo", "tts", "live", "audio", "omni", "robotics",
           "computer-use", "learnlm", "aqa", "nano-banana", "lyria", "gemma")
GROQ_BAD = ("whisper", "orpheus", "guard", "safeguard", "tts", "embed", "playai", "muse-spark")
GROQ_VISION = ("qwen3.8", "vision", "scout", "maverick")
MIN_TEXT_B = 7
OR_LIMIT = 50
PRICE_TOL = 0.02
WEIGHTS = {"rules": 0.5, "data": 0.3, "vision": 0.2}
LWC_URLS = [
    "https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js",
    "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js",
    "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js",
]

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
4) مدل‌هایی که اعتبارشان «نامعتبر» علامت خورده را در قضاوت کم‌اهمیت بگیر و دلیلش را بنویس.
5) سناریوهای نهایی مشروط با ابطال (invalidation) بده، نه توصیه قطعی.
6) چک‌لیست تأیید دستی با داده زنده بده.
خروجی فارسی و به‌صورت Markdown با جدول باشد. این محتوا مشاوره مالی نیست."""

DATA_PROMPT = """تو تحلیلگر داده‌خوان هستی. فقط اعداد و واقعیت‌های JSON زیر را داری که از داده واقعی بایننس محاسبه شده‌اند.
قوانین: هیچ سطح یا قیمتی خارج از این JSON اختراع نکن. تناقض‌ها را پیدا کن، طرح‌های long و short را نقد کن و یکی از سه حالت long/short/wait را انتخاب کن.
اگر شواهد ضعیف یا متناقض است، wait بده. توصیه قطعی یا وعده سود نده.
فقط یک JSON معتبر برگردان (بدون توضیح اضافه و بدون فکر کردن طولانی):
{"bias": "long|short|wait", "confidence": 0-1,
 "reasons": [str, str, str], "risks": [str], "invalidations": [str],
 "plan_check": {"agree_with_plan": true|false, "comment": str},
 "contradictions": [str]}
متن‌ها فارسی و کوتاه باشند."""

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

.stTabs [data-baseweb="tab-list"] { direction: rtl; justify-content: flex-start; gap: 8px; }
.stTabs [data-baseweb="tab"] { direction: rtl; }

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

# ------------------------- DB / usage -------------------------
def db():
    c = sqlite3.connect("analyses.db", timeout=10)
    c.execute("""CREATE TABLE IF NOT EXISTS a(id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT,
      timeframe TEXT, image BLOB, outputs TEXT, verdict TEXT, outcome TEXT)""")
    c.execute("CREATE TABLE IF NOT EXISTS usage2(day TEXT, provider TEXT, n INTEGER, PRIMARY KEY(day, provider))")
    return c

def today():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")

def bump_usage(provider, k=1):
    try:
        c = db()
        c.execute("INSERT INTO usage2(day,provider,n) VALUES(?,?,?) ON CONFLICT(day,provider) DO UPDATE SET n=n+?",
                  (today(), provider, k, k))
        c.commit(); c.close()
    except Exception:
        pass

def usage_by_provider():
    try:
        c = db()
        rows = c.execute("SELECT provider,n FROM usage2 WHERE day=?", (today(),)).fetchall()
        c.close()
        return {p: n for p, n in rows}
    except Exception:
        return {}

# ------------------------- Model ids / keys -------------------------
def fid(provider, model):
    return f"{provider}::{model}"

def split_id(f):
    p, m = f.split("::", 1)
    return p, m

def label_of(f):
    p, m = split_id(f)
    return f"{PROVIDERS[p]['label']} · {m.split('/')[-1].replace(':free', '')}"

def short(f):
    return label_of(f) if "::" in f else f.split("/")[-1].replace(":free", "")[:22]

def get_secret(name):
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return (v or os.getenv(name) or "").strip()

def model_size_b(mid):
    m = re.search(r"(\d+(?:\.\d+)?)b", mid.lower())
    return float(m.group(1)) if m else None

def pick_best(ids, prefs):
    for p in prefs:
        for i in ids:
            if i == p or i.startswith(p):
                return i
    return ids[0] if ids else None

def rank_ids(ids, prefs):
    def k(i):
        for n, p in enumerate(prefs):
            if i == p or i.startswith(p):
                return (n, i)
        return (99, i)
    return sorted(ids, key=k)

def _or_models():
    r = requests.get(f"{PROVIDERS['openrouter']['base']}/models", timeout=30)
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
        if outs != ["text"] or (m.get("context_length") or 0) < 8000:
            continue
        sz = model_size_b(mid)
        if sz is not None and sz < MIN_TEXT_B:
            continue
        text.append(mid)
        if "image" in ins:
            vision.append(mid)
    return dict(vision=rank_ids(vision, PREF[("openrouter", "vision")]),
                text=rank_ids(text, PREF[("openrouter", "text")]))

def _compat_models(provider, key):
    r = requests.get(f"{PROVIDERS[provider]['base']}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    ids = []
    for m in r.json().get("data", []):
        mid = (m.get("id") or m.get("name") or "")
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
        if mid:
            ids.append(mid)
    return ids

@st.cache_data(ttl=3600, show_spinner=False)
def list_models(provider, _key):
    if provider == "openrouter":
        return _or_models()
    ids = _compat_models(provider, _key)
    if provider == "gemini":
        ids = [i for i in ids if i.lower().startswith("gemini") and not any(w in i.lower() for w in GEM_BAD)]
        ids = rank_ids(ids, PREF[("gemini", "any")])
        return dict(vision=ids, text=ids)
    ids = [i for i in ids if not any(w in i.lower() for w in GROQ_BAD)]
    vis = [i for i in ids if any(w in i.lower() for w in GROQ_VISION) and "muse-spark" not in i.lower()]
    return dict(vision=rank_ids(vis, PREF[("groq", "vision")]), text=rank_ids(ids, PREF[("groq", "text")]))

# ------------------------- AI calls -------------------------
def b64(file):
    return base64.b64encode(file.getvalue()).decode()

def extras_for(provider, model, json_mode):
    ex = {}
    if provider == "gemini":
        ex["reasoning_effort"] = "low"
    elif provider == "groq" and "gpt-oss" in model:
        ex["reasoning_effort"] = "low"
    if json_mode and provider in ("gemini", "groq"):
        ex["response_format"] = {"type": "json_object"}
    return ex

def call(f, messages, timeout=120, max_tokens=None, json_mode=False):
    prov, model = split_id(f)
    key = KEYS.get(prov)
    if not key:
        raise RuntimeError("401: کلید این ارائه‌دهنده تنظیم نشده")
    base = PROVIDERS[prov]["base"]
    t = time.time()
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    if max_tokens:
        mt = max_tokens * 2 if prov == "gemini" else max_tokens
        payload["max_completion_tokens" if prov == "groq" else "max_tokens"] = mt
    extras = extras_for(prov, model, json_mode)
    def send(pl):
        try:
            return requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                                 json=pl, timeout=timeout)
        finally:
            bump_usage(prov)
    r = send({**payload, **extras})
    if r.status_code == 400 and extras:
        r = send(payload)
    if r.status_code == 400 and max_tokens:
        p2 = {k: v for k, v in payload.items() if k not in ("max_tokens", "max_completion_tokens")}
        r = send(p2)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    data = r.json()
    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"پاسخ نامعتبر: {str(data)[:200]}")
    return (data["choices"][0]["message"].get("content") or ""), round(time.time()-t, 1)

def explain(err):
    e = (err or "").lower()
    if "per-day" in e or "per day" in e or "daily" in e or "quota" in e:
        return "سقف روزانه/سهمیه تمام شده"
    if "429" in e or "rate" in e and "limit" in e:
        return "شلوغ/محدودیت نرخ (موقت)"
    if "404" in e:
        return "مدل در دسترس نیست یا شناسه تغییر کرده"
    if "413" in e or "too large" in e or "tokens per minute" in e:
        return "درخواست از سقف توکن بزرگ‌تر است"
    if "empty" in e:
        return "پاسخ خالی (مدل استدلالی احتمالاً توکن را مصرف کرد)"
    if "invalid json" in e:
        return "خروجی JSON معتبر نبود"
    if "401" in e or "403" in e or "api key" in e:
        return "مشکل کلید API"
    if "400" in e:
        return "ورودی/پارامتر نامعتبر برای این مدل"
    if "timeout" in e or "timed out" in e:
        return "زمان پاسخ تمام شد"
    return (err or "")[:120]

def tiny_png_b64():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (96, 96), "white")
    ImageDraw.Draw(im).line((5, 80, 45, 30, 90, 10), fill="green", width=4)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def ping(f, png):
    t = time.time()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Reply with the single word OK."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}]
    try:
        call(f, msgs, timeout=45, max_tokens=16)
        return f, dict(ok=True, sec=round(time.time()-t, 1), reason="")
    except Exception as e:
        return f, dict(ok=False, sec=round(time.time()-t, 1), reason=explain(str(e)))

def analyze(f, img_b64, mime, note):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": ANALYST_PROMPT + ("\nاطلاعات واردشده توسط کاربر: " + note if note else "")},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}]}]
    try:
        out, sec = call(f, msgs, max_tokens=3000, json_mode=True)
        return f, out, sec, None
    except Exception as e:
        return f, None, 0, str(e)

def try_models(models, prompt, max_tokens, want_json=False):
    errs = []
    for m in models:
        try:
            out, sec = call(m, [{"role": "user", "content": prompt}], max_tokens=max_tokens, json_mode=want_json)
            if not out.strip():
                raise RuntimeError("empty response")
            j = None
            if want_json:
                j = parse_json(out)
                if not j:
                    raise RuntimeError("invalid json")
            return dict(ok=True, model=m, out=out, j=j, sec=sec, errs=errs)
        except Exception as e:
            why = explain(str(e))
            errs.append((m, why))
            if "سقف روزانه" in why:
                continue
    return dict(ok=False, errs=errs)

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

def consensus(levels, tol=0.003):
    levels = sorted(levels, key=lambda x: x[0])
    groups = []
    for price, model in levels:
        if groups and abs(price - groups[-1]["p"][-1]) / groups[-1]["p"][-1] <= tol:
            groups[-1]["p"].append(price); groups[-1]["m"].add(model)
        else:
            groups.append({"p": [price], "m": {model}})
    return [(sum(g["p"]) / len(g["p"]), len(g["m"])) for g in groups if len(g["m"]) >= 2]

def ai_levels_from(parsed):
    levels = []
    for m, j in parsed.items():
        for key in ("support", "resistance"):
            for it in j.get(key, []) or []:
                p = to_num(it.get("price")) if isinstance(it, dict) else None
                if p is not None:
                    levels.append((p, m))
    return [p for p, _ in consensus(levels)]

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

def scenario_table(parsed, validation=None):
    rows = []
    for m, j in parsed.items():
        sc = j.get("scenarios", {}) or {}
        b, d = sc.get("bullish", {}) or {}, sc.get("bearish", {}) or {}
        v = (validation or {}).get(m, {})
        rows.append({"مدل": short(m), "اعتبار": v.get("label", "—"), "روند": j.get("trend"),
            "نماد": j.get("symbol"), "تایم‌فریم": j.get("timeframe"), "قیمت آخر": j.get("last_price"),
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
    def eq(sw, is_high):
        rec, res = sw[-8:], []
        for a in range(len(rec)):
            for b in range(a + 1, len(rec)):
                pa, pb = rec[a][1], rec[b][1]
                if abs(pa - pb) / pa <= tol:
                    res.append(max(pa, pb) if is_high else min(pa, pb))
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
    return {"eqh": eq(sh, True), "eql": eq(sl, False), "sweep": sweep}

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
    def add(label, pts, ok, detail=""):
        items.append(dict(label=label, pts=pts, ok=bool(ok), detail=detail))
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
    add("ریسک به ریوارد ≥ ۲", 1, plan and plan["rr"] >= 2, f'{plan["rr"]:.1f}' if plan else "")
    mx = sum(i["pts"] for i in items)
    sc = sum(i["pts"] for i in items if i["ok"])
    return dict(items=items, score=sc, max=mx, pct=sc / mx if mx else 0)

def analyze_market(df):
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
             rv=rv, series=dict(e20=e20, e50=e50, e200=e200, rsi=rs))
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

def load_market(symbol, tf_label):
    sym = normalize_symbol(symbol)
    df, host = fetch_klines(sym, TF_MAP[tf_label], 500)
    return dict(df=df, host=host, symbol=sym, tf=tf_label, A=analyze_market(df), ai_levels=[])

# ------------------------- Voters & final decision -------------------------
def market_facts(M):
    A = M["A"]; p = A["price"]; atr_v = A["atr"]
    def z2(z, kind):
        inside = z["bottom"] <= p <= z["top"]
        return dict(kind=kind, dir="صعودی" if z["dir"] == 1 else "نزولی", bottom=round(z["bottom"], 6),
                    top=round(z["top"], 6), tested=z["tested"],
                    dist_atr=0 if inside else round(min(abs(p - z["top"]), abs(p - z["bottom"])) / atr_v, 2))
    obs = sorted(A["obs"], key=lambda z: min(abs(p - z["top"]), abs(p - z["bottom"])))[:4]
    fv = sorted(A["fvgs"], key=lambda z: min(abs(p - z["top"]), abs(p - z["bottom"])))[:4]
    def plan_d(pl):
        if not pl: return None
        d0 = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in pl.items() if k != "zone"}
        d0["zone"] = [round(pl["zone"]["bottom"], 6), round(pl["zone"]["top"], 6)]
        return d0
    return dict(symbol=M["symbol"], timeframe=M["tf"], price=round(p, 6), atr=round(atr_v, 6),
        structure={1: "bullish", -1: "bearish", 0: "unclear"}[A["trend"]],
        last_events=[dict(kind=e["kind"], dir="up" if e["dir"] == 1 else "down", level=round(e["level"], 6))
                     for e in A["events"][-3:]],
        order_blocks=[z2(z, "OB") for z in obs], fvgs=[z2(z, "FVG") for z in fv],
        liquidity=dict(eqh=[round(x, 6) for x in A["liq"]["eqh"]], eql=[round(x, 6) for x in A["liq"]["eql"]], sweep=A["liq"]["sweep"]),
        premium_discount=dict(zone=A["pd"]["zone"], position=round(A["pd"]["pos"], 2), eq=round(A["pd"]["eq"], 6)),
        ema=dict(e20=round(A["ema20"], 6), e50=round(A["ema50"], 6), e200=round(A["ema200"], 6)),
        rsi=round(A["rsi"], 1), relative_volume=round(A["rv"], 2),
        plans=dict(long=plan_d(A["plans"]["long"]), short=plan_d(A["plans"]["short"])),
        rule_scores=dict(long=round(A["scores"]["long"]["pct"], 2), short=round(A["scores"]["short"]["pct"], 2)))

def run_data_ai(models, M):
    prompt = DATA_PROMPT + "\n\n" + json.dumps(market_facts(M), ensure_ascii=False)
    r = try_models(models, prompt, 1500, want_json=True)
    if r["ok"]:
        return dict(ok=True, j=r["j"], raw=r["out"], sec=r["sec"], model=r["model"], errs=r["errs"])
    return dict(ok=False, errs=r["errs"], raw="", error="؛ ".join(f"{short(m)}: {w}" for m, w in r["errs"]))

def validate_vision(parsed, real_price):
    res = {}
    for m, j in parsed.items():
        lp = to_num(j.get("last_price"))
        if real_price is None:
            res[m] = dict(status="unverified", label="⚪ بدون داده بایننس", w=0.5)
        elif lp is None:
            res[m] = dict(status="unverified", label="⚪ قیمت خوانده نشد", w=0.5)
        elif abs(lp - real_price) / real_price <= PRICE_TOL:
            res[m] = dict(status="valid", label="✅ قیمت مطابق بایننس", w=1.0)
        else:
            res[m] = dict(status="invalid", label=f"❌ قیمت خوانده‌شده {lp:g} با واقعی {real_price:g} فرق دارد (یا اسکرین‌شات قدیمی است)", w=0.0)
    return res

def vision_vote(parsed, validation):
    num = den = 0.0; nv = 0
    for m, j in parsed.items():
        w = validation[m]["w"]
        if w <= 0: continue
        d = {"bullish": 1, "bearish": -1}.get(j.get("trend"), 0)
        c = to_num(j.get("overall_confidence"))
        c = 0.5 if c is None else min(max(c, 0), 1)
        num += w * d * c; den += w; nv += 1
    if den == 0:
        return None
    return dict(signed=num / den, n_valid=nv, n_total=len(parsed))

def sgn(x, th=0.1):
    return 1 if x >= th else (-1 if x <= -th else 0)

def make_final(M, vis, dat):
    A = M["A"]
    lp, sp = A["scores"]["long"]["pct"], A["scores"]["short"]["pct"]
    voters = [dict(key="rules", name="موتور قانون", w=WEIGHTS["rules"], s=lp - sp,
                   note=f"لانگ {lp:.0%} | شورت {sp:.0%}")]
    if dat and dat.get("ok"):
        j = dat["j"]
        s = {"long": 1, "short": -1}.get(j.get("bias"), 0) * (to_num(j.get("confidence")) or 0.5)
        reasons0 = j.get("reasons") or [""]
        voters.append(dict(key="data", name="AI داده‌خوان", w=WEIGHTS["data"], s=s, note=str(reasons0[0])[:90]))
    if vis:
        voters.append(dict(key="vision", name="AI بینایی", w=WEIGHTS["vision"], s=vis["signed"],
                           note=f'{vis["n_valid"]} از {vis["n_total"]} مدل معتبر'))
    tw = sum(v["w"] for v in voters)
    fs = sum(v["w"] * v["s"] for v in voters) / tw
    direction = sgn(fs, 0.25)
    dirs = [sgn(v["s"]) for v in voters]
    conflict = (1 in dirs) and (-1 in dirs)
    side = ("long" if direction == 1 else "short") if direction else A["side"]
    plan = A["plans"][side]
    k = sum(1 for v in voters if direction and sgn(v["s"]) == direction)
    fa = {"long": "لانگ", "short": "شورت"}[side]
    agree = f"{k} از {len(voters)} رأی‌دهنده {fa}" if direction else "رأی‌دهنده‌ها به جهت واحد نرسیدند"
    ai_strong_opp = any(v["key"] != "rules" and direction and sgn(v["s"], 0.5) == -direction for v in voters)
    if direction == 0 or plan is None:
        verdict = "NO_TRADE"
    elif len(voters) == 1:
        verdict = A["verdict"] if A["side"] == side else "WATCH"
    elif A["scores"][side]["pct"] >= 0.5 and plan["rr"] >= 2 and plan["dist_atr"] <= 1.5 and k >= 2 and not ai_strong_opp:
        verdict = "ENTER"
    else:
        verdict = "WATCH"
    j = dat["j"] if dat and dat.get("ok") else {}
    passed = [i["label"] for i in A["scores"][side]["items"] if i["ok"]]
    reasons = [str(x) for x in (j.get("reasons") or [])][:3] or passed[:3]
    risks = [str(x) for x in (j.get("risks") or [])][:3]
    if A["rv"] < 0.8: risks.append("حجم اخیر ضعیف است")
    if A["rsi"] > 70 or A["rsi"] < 30: risks.append(f'RSI در ناحیه افراطی ({A["rsi"]:.0f})')
    if plan and plan["dist_atr"] > 1.5: risks.append("قیمت از ناحیه ورود دور است؛ باید منتظر برگشت به ناحیه بود")
    inval = [str(x) for x in (j.get("invalidations") or [])][:2]
    if plan:
        inval.append(f'بسته شدن کندل فراتر از حد ضرر ({fmt(plan["sl"])})')
    inval.append("CHoCH در خلاف جهت سناریو روی همین تایم‌فریم")
    return dict(direction=direction, side=side, verdict=verdict, final=fs, voters=voters, agree=agree,
                conflict=conflict, reasons=reasons, risks=risks[:4], invalidations=inval[:4],
                contradictions=[str(x) for x in (j.get("contradictions") or [])][:3])

def card_html(M, F):
    A = M["A"]; side = F["side"]; plan = A["plans"][side]
    fa = {"long": "لانگ", "short": "شورت"}[side]
    v = F["verdict"]
    color = {"ENTER": "#16a34a" if side == "long" else "#dc2626", "WATCH": "#d97706", "NO_TRADE": "#6b7280"}[v]
    if v == "NO_TRADE":
        title = "تضاد بین رأی‌دهنده‌ها — معامله نکن" if F["conflict"] else "معامله نکن — همگرایی کافی نیست"
    elif v == "ENTER":
        title = f"سناریوی {fa} — شرایط ورود مشروط"
    else:
        title = f"در انتظار تأیید — گرایش {fa}"
    icon = {"ENTER": "🟢" if side == "long" else "🔴", "WATCH": "🟡", "NO_TRADE": "⚪"}[v]
    def badge(s):
        d = sgn(s)
        t, c = ("لانگ", "#16a34a") if d == 1 else (("شورت", "#dc2626") if d == -1 else ("صبر", "#6b7280"))
        return f'<span style="background:{c};color:#fff;border-radius:8px;padding:2px 10px;font-weight:700">{t}</span>'
    vcols = ""
    for x in F["voters"]:
        vcols += (f'<div style="flex:1 1 30%;min-width:140px;background:#f9fafb;border-radius:12px;padding:10px;margin:4px">'
                  f'<div style="font-size:.8rem;color:#6b7280">{x["name"]} · وزن {int(x["w"]*100)}٪</div>'
                  f'<div style="margin:6px 0">{badge(x["s"])} <span style="direction:ltr;display:inline-block">{abs(x["s"]):.0%}</span></div>'
                  f'<div style="font-size:.78rem;color:#374151">{x["note"]}</div></div>')
    pos = int((F["final"] + 1) / 2 * 100)
    meter = (f'<div style="position:relative;background:linear-gradient(90deg,#fecaca,#e5e7eb,#bbf7d0);border-radius:8px;height:12px;margin:6px 0">'
             f'<div style="position:absolute;left:{pos}%;top:-3px;width:4px;height:18px;background:#111827;border-radius:2px"></div></div>'
             f'<div style="display:flex;justify-content:space-between;font-size:.72rem;color:#6b7280;direction:ltr"><span>شورت</span><span>امتیاز نهایی {F["final"]:+.2f}</span><span>لانگ</span></div>')
    def cell(k, val, c="#111827"):
        return (f'<div style="flex:1 1 30%;min-width:120px;background:#f9fafb;border-radius:10px;padding:8px;margin:4px">'
                f'<div style="font-size:.75rem;color:#6b7280">{k}</div><div style="font-weight:700;color:{c};direction:ltr">{val}</div></div>')
    if plan:
        z = plan["zone"]
        cells = "".join([
            cell("ناحیه ورود", f'{fmt(z["bottom"])} – {fmt(z["top"])} ({plan["kind"]})'),
            cell("ورود", fmt(plan["entry"]), "#2563eb"), cell("حد ضرر", fmt(plan["sl"]), "#dc2626"),
            cell(f'هدف ۱ ({plan["src1"]})', fmt(plan["tp1"]), "#16a34a"),
            cell(f'هدف ۲ ({plan["src2"]})', fmt(plan["tp2"]), "#16a34a"),
            cell("ریسک به ریوارد", f'1 : {plan["rr"]:.1f}'),
            cell("فاصله قیمت تا ناحیه", f'{plan["dist_atr"]:.1f} ATR')])
    else:
        cells = '<div style="padding:8px">ناحیه معتبر (OB/FVG) نزدیک قیمت پیدا نشد؛ طرح معامله ساخته نشد.</div>'
    def ul(items):
        return "<ul style='margin:4px 0;padding-right:18px'>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>" if items else "<div style='color:#6b7280'>—</div>"
    contra = (f'<div style="margin-top:8px;font-weight:700">⚠️ تناقض‌های یافته‌شده</div>{ul(F["contradictions"])}' if F["contradictions"] else "")
    return (f'<div style="border:2px solid {color};border-radius:16px;padding:16px;direction:rtl;text-align:right">'
            f'<div style="font-size:1.35rem;font-weight:700;color:{color}">{icon} {title}</div>'
            f'<div style="margin:4px 0;font-weight:600">{F["agree"]}</div>'
            f'<div style="color:#6b7280;margin-bottom:6px;direction:ltr;text-align:right">{M["symbol"]} · {M["tf"]} · قیمت: {fmt(A["price"])} · ATR: {fmt(A["atr"])}</div>'
            f'{meter}<div style="display:flex;flex-wrap:wrap;margin-top:8px">{vcols}</div>'
            f'<div style="display:flex;flex-wrap:wrap;margin-top:8px">{cells}</div>'
            f'<div style="margin-top:10px;font-weight:700">چرا؟</div>{ul(F["reasons"])}'
            f'<div style="margin-top:6px;font-weight:700">خطرها</div>{ul(F["risks"])}'
            f'<div style="margin-top:6px;font-weight:700">چه چیزی نظر را عوض می‌کند؟</div>{ul(F["invalidations"])}{contra}'
            f'<div style="margin-top:10px;font-size:.78rem;color:#6b7280">این خروجی قانون‌محور و آموزشی است، نه توصیه مالی. '
            f'قبل از هر معامله با قیمت زنده، اخبار، کارمزد و مدیریت سرمایه خودت تأیید کن.</div></div>')

# ------------------------- Zones helpers -------------------------
def zone_dist(z, price):
    if z["bottom"] <= price <= z["top"]:
        return 0.0
    return min(abs(price - z["top"]), abs(price - z["bottom"]))

def zone_rows(M, n=6):
    A = M["A"]; price, atr_v = A["price"], A["atr"]
    rows = []
    for kind, lst in (("OB", A["obs"]), ("FVG", A["fvgs"])):
        for z in sorted(lst, key=lambda z: zone_dist(z, price))[:n]:
            rows.append({"نوع": kind, "جهت": "صعودی" if z["dir"] == 1 else "نزولی",
                         "محدوده": f'{fmt(z["bottom"])} – {fmt(z["top"])}',
                         "وضعیت": "تست‌شده" if z["tested"] else "تازه",
                         "فاصله (ATR)": round(zone_dist(z, price) / atr_v, 2)})
    rows.sort(key=lambda r: r["فاصله (ATR)"])
    return rows

# ------------------------- Fast chart (Lightweight Charts, one iframe) -------------------------
@st.cache_resource(show_spinner=False)
def _lwc_store():
    return {"js": None}

def get_lwc_js():
    store = _lwc_store()
    if store["js"]:
        return store["js"]
    for u in LWC_URLS:
        try:
            r = requests.get(u, timeout=20)
            if r.status_code == 200 and len(r.text) > 50000 and "createChart" in r.text:
                store["js"] = r.text
                return r.text
        except Exception:
            pass
    return None

def lwc_payload(M, main_h=460, rsi_h=150):
    A = M["A"]; df = M["df"]; d = A["d"]; price = A["price"]; atr_v = A["atr"]
    prec = 2 if price >= 100 else (4 if price >= 1 else (5 if price >= 0.1 else 6))
    ts = (df["t"] / 1000).astype(int).tolist()
    n = len(ts)
    candles = [dict(time=ts[i], open=float(df.open.iloc[i]), high=float(df.high.iloc[i]),
                    low=float(df.low.iloc[i]), close=float(df.close.iloc[i])) for i in range(n)]
    volume = [dict(time=ts[i], value=float(df.volume.iloc[i]),
                   color="rgba(22,163,74,0.45)" if df.close.iloc[i] >= df.open.iloc[i] else "rgba(220,38,38,0.45)")
              for i in range(n)]
    def line(series):
        out = []
        for i, v in enumerate(series.tolist()):
            if v == v and i < len(ts):
                out.append(dict(time=ts[i], value=round(float(v), prec + 2)))
        return out
    S = A["series"]
    zones = []
    for kind, lst in (("OB", A["obs"]), ("FVG", A["fvgs"])):
        for z in sorted(lst, key=lambda z: zone_dist(z, price))[:8]:
            zones.append(dict(kind=kind, dir=z["dir"], top=z["top"], bottom=z["bottom"], t0=ts[z["i"]],
                              tested=bool(z["tested"]), dist=round(zone_dist(z, price) / atr_v, 2)))
    events = [dict(t=ts[e["i"]], t0=ts[e["origin"]], level=e["level"], dir=e["dir"], kind=e["kind"])
              for e in A["events"][-8:]]
    plan = M.get("plan")
    step = ts[-1] - ts[-2] if n > 1 else 3600
    return dict(symbol=M["symbol"], tf=M["tf"], price=price, prec=prec, minMove=10 ** (-prec),
                mainH=main_h, rsiH=rsi_h, candles=candles, volume=volume,
                e20=line(S["e20"]), e50=line(S["e50"]), e200=line(S["e200"]), rsi=line(S["rsi"]),
                zones=zones, events=events, eqh=A["liq"]["eqh"], eql=A["liq"]["eql"], eq50=A["pd"]["eq"],
                ai=[float(x) for x in M.get("ai_levels", [])],
                plan=(dict(entry=plan["entry"], sl=plan["sl"], tp1=plan["tp1"], tp2=plan["tp2"]) if plan else None),
                tEnd=ts[-1] + step * 3)

def build_lwc_html(M, lib):
    data = json.dumps(lwc_payload(M), ensure_ascii=False)
    return LWC_TEMPLATE.replace("__DATA__", data).replace("__LIB__", lib)

# ------------------------- Fallback chart (Plotly, simple) -------------------------
def build_price_chart(M, view=150):
    A = M["A"]; df = M["df"]; d = A["d"]
    off = max(0, len(d) - view)
    v = d.iloc[off:].reset_index(drop=True)
    t = v["time"]
    step = (t.iloc[-1] - t.iloc[-2]) if len(t) > 1 else pd.Timedelta(hours=1)
    x_end = t.iloc[-1] + step * 10
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=t, open=v.open, high=v.high, low=v.low, close=v.close,
        increasing_line_color="#16a34a", decreasing_line_color="#dc2626", showlegend=False))
    price, atr_v = A["price"], A["atr"]
    for lst, kind in ((A["obs"], "OB"), (A["fvgs"], "FVG")):
        near = [z for z in sorted(lst, key=lambda z: zone_dist(z, price)) if zone_dist(z, price) <= 3 * atr_v][:2]
        for z in near:
            x0 = t.iloc[max(z["i"] - off, 0)] if z["i"] >= off else t.iloc[0]
            col = "rgba(22,163,74,0.22)" if z["dir"] == 1 else "rgba(220,38,38,0.22)"
            fig.add_shape(type="rect", x0=x0, x1=x_end, y0=z["bottom"], y1=z["top"], fillcolor=col,
                          line=dict(width=0), layer="below")
            fig.add_annotation(x=x_end, y=(z["top"] + z["bottom"]) / 2, text=kind, showarrow=False,
                               xanchor="right", font=dict(size=11))
    extra = []
    plan = M.get("plan")
    if plan:
        for key, c, name in (("entry", "#2563eb", "Entry"), ("sl", "#dc2626", "SL"),
                             ("tp1", "#16a34a", "TP1"), ("tp2", "#15803d", "TP2")):
            extra.append(plan[key])
            fig.add_shape(type="line", x0=t.iloc[-1], x1=x_end, y0=plan[key], y1=plan[key], line=dict(color=c, width=2))
            fig.add_annotation(x=x_end, y=plan[key], text=f"{name} {fmt(plan[key])}", showarrow=False,
                               xanchor="left", font=dict(size=11, color=c))
    lo_, hi_ = min([v.low.min()] + extra), max([v.high.max()] + extra)
    pad = (hi_ - lo_) * 0.05
    fig.update_yaxes(range=[lo_ - pad, hi_ + pad], side="right")
    fig.update_xaxes(range=[t.iloc[0], x_end], rangeslider_visible=False)
    fig.update_layout(height=620, margin=dict(l=5, r=70, t=10, b=10), dragmode="pan",
        plot_bgcolor="#ffffff", showlegend=False, hovermode="x")
    return fig

# ------------------------- Role helpers -------------------------
def best_of(p, kind, ids):
    prefs = PREF[("gemini", "any")] if p == "gemini" else PREF[(p, kind)]
    return pick_best(ids, prefs)

def chain_pool(enabled, LM, kind):
    out = []
    for p in enabled:
        b = best_of(p, kind, LM.get(p, {}).get(kind, []))
        if b:
            out.append(fid(p, b))
    return out

def vision_backup_pool(enabled, LM):
    out = []
    for p in enabled:
        for m in LM.get(p, {}).get("vision", [])[:2]:
            out.append(fid(p, m))
    return out

def candidates(chosen, pool, k):
    return [chosen] + [m for m in pool if m != chosen][:k]

def default_vision(enabled, LM, bad=()):
    d = []
    for p in enabled:
        if p in ("gemini", "groq"):
            b = best_of(p, "vision", LM.get(p, {}).get("vision", []))
            if b and fid(p, b) not in bad:
                d.append(fid(p, b))
    if not d and "openrouter" in enabled:
        for m in LM.get("openrouter", {}).get("vision", [])[:2]:
            if fid("openrouter", m) not in bad:
                d.append(fid("openrouter", m))
    return d[:3]

def default_text(enabled, LM, prefer):
    for p in prefer + [x for x in enabled if x not in prefer]:
        if p in enabled:
            b = best_of(p, "text", LM.get(p, {}).get("text", []))
            if b:
                return fid(p, b)
    return None

LWC_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{margin:0;font-family:Tahoma,Arial,sans-serif;background:#fff}
#bar{display:flex;flex-wrap:wrap;gap:6px;padding:6px 4px;direction:rtl}
.sep{width:100%;height:0}
.chip{border:1px solid #d1d5db;border-radius:16px;padding:5px 11px;font-size:12px;background:#f9fafb;color:#111827;cursor:pointer;user-select:none;-webkit-user-select:none}
.chip.on{background:#2563eb;color:#fff;border-color:#2563eb}
.chip.pre{font-weight:700}
#info{font-size:12px;padding:2px 8px 4px;direction:ltr;color:#374151;min-height:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#main,#rsi{width:100%;direction:ltr}
</style></head><body>
<div id="bar"></div><div id="info"></div><div id="main"></div><div id="rsi"></div>
<script>__LIB__</script>
<script>
(function(){
const infoEl = document.getElementById('info');
try{
const D = __DATA__;
const LW = LightweightCharts;
const W = Math.max(document.body.clientWidth || window.innerWidth || 360, 300);
function mk(el,h){
  return LW.createChart(el,{width:W,height:h,
    layout:{background:{type:'solid',color:'#ffffff'},textColor:'#374151',fontSize:11},
    grid:{vertLines:{color:'#f3f4f6'},horzLines:{color:'#f3f4f6'}},
    rightPriceScale:{borderColor:'#e5e7eb',minimumWidth:72},
    timeScale:{borderColor:'#e5e7eb',timeVisible:true,secondsVisible:false,rightOffset:10},
    crosshair:{mode:0},
    handleScroll:{vertTouchDrag:false,horzTouchDrag:true,mouseWheel:true,pressedMouseMove:true},
    handleScale:{pinch:true,mouseWheel:true,axisPressedMouseMove:true}});
}
const main = mk(document.getElementById('main'), D.mainH);
const rsiC = mk(document.getElementById('rsi'), D.rsiH);

const candle = main.addCandlestickSeries({upColor:'#16a34a',downColor:'#dc2626',borderVisible:false,
  wickUpColor:'#16a34a',wickDownColor:'#dc2626',
  priceFormat:{type:'price',precision:D.prec,minMove:D.minMove}});
candle.setData(D.candles);

const vol = main.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'vol',
  lastValueVisible:false,priceLineVisible:false,visible:false});
main.priceScale('vol').applyOptions({scaleMargins:{top:0.82,bottom:0}});
vol.setData(D.volume);

const rsiS = rsiC.addLineSeries({color:'#7c3aed',lineWidth:2,priceLineVisible:false,
  priceFormat:{type:'price',precision:1,minMove:0.1},
  autoscaleInfoProvider:()=>({priceRange:{minValue:0,maxValue:100}})});
rsiS.setData(D.rsi);
[30,70].forEach(v=>rsiS.createPriceLine({price:v,color:'#9ca3af',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:''}));

let syncing=false;
function link(a,b){a.timeScale().subscribeVisibleLogicalRangeChange(r=>{
  if(syncing||!r) return; syncing=true; b.timeScale().setVisibleLogicalRange(r); syncing=false;});}
link(main,rsiC); link(rsiC,main);

const rsiMap=new Map(D.rsi.map(r=>[r.time,r.value]));
const volMap=new Map(D.candles.map((c,i)=>[c.time,D.volume[i]?D.volume[i].value:0]));
function f(x){return (x===undefined||x===null)?'—':Number(x).toFixed(D.prec);}
function defInfo(){
  const lr=D.rsi.length?D.rsi[D.rsi.length-1].value.toFixed(0):'—';
  return D.symbol+' · '+D.tf+' · قیمت '+f(D.price)+' · RSI '+lr;
}
infoEl.textContent=defInfo();
main.subscribeCrosshairMove(p=>{
  if(!p||!p.time||!p.seriesData){infoEl.textContent=defInfo();return;}
  const c=p.seriesData.get(candle);
  if(!c){infoEl.textContent=defInfo();return;}
  const r=rsiMap.get(p.time), v=volMap.get(p.time);
  infoEl.textContent='O '+f(c.open)+'  H '+f(c.high)+'  L '+f(c.low)+'  C '+f(c.close)+
    '  | Vol '+(v===undefined?'—':Math.round(v))+'  | RSI '+(r===undefined?'—':r.toFixed(0));
});

const state={focus:true,n:2};
const L={};
function layer(name,label,addFn){L[name]={on:false,h:[],add:addFn,label:label};}
function setLayer(name,on){
  const l=L[name]; if(!l||on===l.on) return;
  if(on){l.h=l.add();}else{l.h.forEach(fn=>{try{fn();}catch(e){}});l.h=[];}
  l.on=on;
}
function pline(price,color,title,style,width,axis){
  const p=candle.createPriceLine({price:price,color:color,lineWidth:width||1,
    lineStyle:(style===undefined?2:style),axisLabelVisible:axis!==false,title:title||''});
  return ()=>candle.removePriceLine(p);
}
function band(z,fill){
  const s=main.addBaselineSeries({baseValue:{type:'price',price:z.bottom},
    topLineColor:'rgba(0,0,0,0)',bottomLineColor:'rgba(0,0,0,0)',
    topFillColor1:fill,topFillColor2:fill,bottomFillColor1:'rgba(0,0,0,0)',bottomFillColor2:'rgba(0,0,0,0)',
    lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
  s.setData([{time:z.t0,value:z.top},{time:D.tEnd,value:z.top}]);
  return ()=>main.removeSeries(s);
}
function pickZones(){
  const out=[];
  ['OB','FVG'].forEach(k=>{
    let a=D.zones.filter(z=>z.kind===k).sort((x,y)=>x.dist-y.dist);
    if(state.focus) a=a.filter(z=>z.dist<=3);
    if(state.n) a=a.slice(0,state.n);
    out.push.apply(out,a);
  });
  return out;
}
layer('zones','OB / FVG',()=>{
  const rm=[];
  pickZones().forEach(z=>{
    const bull=z.dir===1;
    const fill=z.kind==='OB'?(bull?'rgba(22,163,74,0.30)':'rgba(220,38,38,0.30)'):(bull?'rgba(37,99,235,0.25)':'rgba(234,88,12,0.25)');
    const lc=z.kind==='OB'?(bull?'#16a34a':'#dc2626'):(bull?'#2563eb':'#ea580c');
    rm.push(band(z,fill));
    rm.push(pline(z.top,lc,z.kind+(bull?'+':'-')+(z.tested?' tested':' fresh'),1,1,true));
    rm.push(pline(z.bottom,lc,'',1,1,false));
  });
  return rm;
});
layer('structure','ساختار BOS/CHoCH',()=>{
  const rm=[],ms=[];
  D.events.forEach(e=>{
    const s=main.addLineSeries({color:'#6b7280',lineWidth:1,lineStyle:2,priceLineVisible:false,
      lastValueVisible:false,crosshairMarkerVisible:false});
    s.setData([{time:e.t0,value:e.level},{time:e.t,value:e.level}]);
    rm.push(()=>main.removeSeries(s));
    ms.push({time:e.t,position:e.dir===1?'belowBar':'aboveBar',color:e.dir===1?'#16a34a':'#dc2626',
      shape:e.dir===1?'arrowUp':'arrowDown',text:e.kind});
  });
  ms.sort((a,b)=>a.time-b.time);
  candle.setMarkers(ms);
  rm.push(()=>candle.setMarkers([]));
  return rm;
});
layer('liq','نقدینگی',()=>{
  const rm=[];
  D.eqh.forEach(p=>rm.push(pline(p,'#a855f7','EQH',2,1,true)));
  D.eql.forEach(p=>rm.push(pline(p,'#a855f7','EQL',2,1,true)));
  return rm;
});
layer('eq','تعادل 50%',()=>[pline(D.eq50,'#9ca3af','EQ 50%',3,1,true)]);
layer('ai','سطوح AI',()=>D.ai.map(p=>pline(p,'#0ea5e9','AI',1,1,true)));
layer('ema','EMA',()=>{
  const rm=[];
  [['e20','#f59e0b'],['e50','#2563eb'],['e200','#7c3aed']].forEach(x=>{
    const s=main.addLineSeries({color:x[1],lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
    s.setData(D[x[0]]); rm.push(()=>main.removeSeries(s));
  });
  return rm;
});
layer('vol','حجم',()=>{vol.applyOptions({visible:true});return [()=>vol.applyOptions({visible:false})];});
layer('plan','طرح معامله',()=>{
  const rm=[]; const p=D.plan; if(!p) return rm;
  rm.push(pline(p.entry,'#2563eb','Entry',0,2,true));
  rm.push(pline(p.sl,'#dc2626','SL',0,2,true));
  rm.push(pline(p.tp1,'#16a34a','TP1',0,2,true));
  rm.push(pline(p.tp2,'#15803d','TP2',0,2,true));
  return rm;
});

const PRE={
  'ساده':{layers:['plan','zones','vol'],focus:true,n:2},
  'SMC':{layers:['plan','zones','structure','liq','eq','ai','vol'],focus:true,n:3},
  'کامل':{layers:['plan','zones','structure','liq','eq','ai','ema','vol'],focus:false,n:0}};
let cur='ساده'; const chips=[];
function refresh(){chips.forEach(c=>c.el.classList.toggle('on',c.on()));}
function applyPreset(name){
  const p=PRE[name]; state.focus=p.focus; state.n=p.n; cur=name;
  Object.keys(L).forEach(k=>setLayer(k,false));
  p.layers.forEach(k=>setLayer(k,true));
  refresh();
}
const bar=document.getElementById('bar');
function chip(txt,fn,on,cls){
  const el=document.createElement('div'); el.className='chip'+(cls?' '+cls:''); el.textContent=txt;
  el.onclick=()=>{fn();refresh();}; bar.appendChild(el); chips.push({el:el,on:on||(()=>false)}); return el;
}
function sep(){const s=document.createElement('div'); s.className='sep'; bar.appendChild(s);}
Object.keys(PRE).forEach(n=>chip(n,()=>applyPreset(n),()=>cur===n,'pre'));
sep();
['plan','zones','structure','liq','eq','ai','ema','vol'].forEach(k=>
  chip(L[k].label,()=>{setLayer(k,!L[k].on);},()=>L[k].on));
sep();
chip('فقط زون‌های نزدیک',()=>{
  state.focus=!state.focus; if(L.zones.on){setLayer('zones',false);setLayer('zones',true);}},()=>state.focus);
const N=D.candles.length;
function setRange(n){
  if(!n){main.timeScale().fitContent();return;}
  main.timeScale().setVisibleLogicalRange({from:Math.max(N-n,0),to:N+8});
}
[80,150,300].forEach(n=>chip(n+' کندل',()=>setRange(n)));
chip('همه',()=>setRange(0));
chip('تمام‌صفحه',()=>{try{(document.documentElement.requestFullscreen||function(){}).call(document.documentElement);}catch(e){}});

applyPreset('ساده');
setRange(150);
window.addEventListener('resize',()=>{
  const w=Math.max(document.body.clientWidth||360,300);
  main.applyOptions({width:w}); rsiC.applyOptions({width:w});});
}catch(e){infoEl.textContent='خطا در نمودار: '+(e&&e.message?e.message:e);}
})();
</script></body></html>
"""
# ------------------------- UI -------------------------
import streamlit.components.v1 as components

st.set_page_config(page_title="Chart AI Hub", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("health", {})
st.session_state.setdefault("bad", set())

st.title("هاب تحلیل چارت چندمدلی")
st.caption("خروجی ابزار آموزشی/پژوهشی است و مشاوره مالی نیست.")

with st.sidebar:
    st.markdown("**🔑 کلیدها**")
    with st.expander("جایگزینی موقت کلید (اختیاری)"):
        ovr = {p: st.text_input(f"{PROVIDERS[p]['label']} API key", type="password", key=f"ovr_{p}") for p in ORDER}
    for p in ORDER:
        KEYS[p] = (ovr.get(p) or "").strip() or get_secret(PROVIDERS[p]["secret"])
    have = [p for p in ORDER if KEYS[p]]
    st.caption(" | ".join(f"{'✅' if KEYS[p] else '❌'} {PROVIDERS[p]['label']}" for p in ORDER))
    if not have:
        st.error("هیچ کلیدی تنظیم نشده. کلیدها را در Secrets اپ بگذار (GEMINI_API_KEY، GROQ_API_KEY، OPENROUTER_API_KEY).")
        st.stop()
    sel = st.multiselect("ارائه‌دهنده‌های فعال (اولویت: Gemini ← Groq ← OpenRouter)", have, default=have,
                         format_func=lambda p: PROVIDERS[p]["label"])
    enabled = [p for p in ORDER if p in sel]
    if not enabled:
        st.warning("حداقل یک ارائه‌دهنده را فعال کن.")
        st.stop()

    if st.button("🔄 به‌روزرسانی لیست مدل‌ها"):
        list_models.clear()
    LM, list_err = {}, {}
    for p in enabled:
        try:
            LM[p] = list_models(p, KEYS[p])
            if not LM[p]["vision"] and not LM[p]["text"]:
                raise RuntimeError("لیست خالی بود")
        except Exception as e:
            list_err[p] = explain(str(e))
            LM[p] = STATIC_MODELS[p]
    st.caption(" | ".join(f"{PROVIDERS[p]['label']}: {len(LM[p]['vision'])} بینایی، {len(LM[p]['text'])} متنی" for p in enabled))
    for p, w in list_err.items():
        st.warning(f"لیست مدل‌های {PROVIDERS[p]['label']} گرفته نشد ({w})؛ لیست پیش‌فرض استفاده می‌شود. اگر مشکل کلید است، همین‌جا معلوم می‌شود.")

    u = usage_by_provider()
    st.caption(f"درخواست‌های امروز (UTC، تقریبی) — Gemini: {u.get('gemini', 0)} | Groq: {u.get('groq', 0)} | OpenRouter: {u.get('openrouter', 0)}/{OR_LIMIT}")
    if u.get("openrouter", 0) >= OR_LIMIT:
        st.warning("سقف رایگان روزانه OpenRouter (حدود ۵۰) تمام شده.")

    vis_options = [fid(p, m) for p in enabled for m in LM[p]["vision"]]
    with st.expander("🧪 تست سلامت مدل‌های بینایی"):
        st.caption("هر مدل تستی ۱ درخواست از سهمیه همان ارائه‌دهنده مصرف می‌کند (حتی اگر خطا بدهد).")
        dflt = default_vision(enabled, LM) or vis_options[:2]
        test_pool = st.multiselect("مدل‌های برای تست (حداکثر ۸)", vis_options, default=dflt, max_selections=8,
                                   format_func=label_of)
        st.caption(f"هزینه تست: {len(test_pool)} درخواست")
        if st.button("اجرای تست", disabled=not test_pool):
            png = tiny_png_b64()
            with st.spinner("در حال تست..."), cf.ThreadPoolExecutor(3) as ex:
                for f_, r in ex.map(lambda x: ping(x, png), test_pool):
                    st.session_state["health"][f_] = r
                    if r["ok"]:
                        st.session_state["bad"].discard(f_)
                    else:
                        st.session_state["bad"].add(f_)
        if st.session_state["health"]:
            st.dataframe([{"مدل": label_of(m), "وضعیت": "✅ سالم" if r["ok"] else "❌ خطا",
                           "ثانیه": r["sec"], "علت": r["reason"]}
                          for m, r in st.session_state["health"].items()], use_container_width=True)
        if st.button("پاک کردن نتایج تست و لیست خراب‌ها"):
            st.session_state["health"] = {}; st.session_state["bad"] = set(); st.rerun()

    healthy = [m for m, r in st.session_state["health"].items() if r["ok"] and m in vis_options]
    only_healthy = st.checkbox("فقط مدل‌های سالم (تست‌شده)", value=bool(healthy), disabled=not healthy)
    options = healthy if (only_healthy and healthy) else vis_options
    base_default = healthy if healthy else default_vision(enabled, LM, st.session_state["bad"])
    default_sel = [m for m in base_default if m in options and m not in st.session_state["bad"]][:3]
    models = st.multiselect("مدل‌های بینایی (تحلیل عکس، حداکثر ۳)", options, default=default_sel,
                            max_selections=3, format_func=label_of)
    use_backup = st.checkbox("اگر مدل بینایی شکست خورد، پشتیبان امتحان شود", value=True)
    if st.session_state["bad"]:
        st.caption("مدل‌های خراب‌شده در این نشست: " + "، ".join(label_of(m) for m in st.session_state["bad"]))

    st.markdown("**مدل‌های متنی**")
    st.caption("اگر مدل انتخابی خطا بدهد، مدل بعدی از ارائه‌دهنده بعدی امتحان می‌شود (هر تلاش ۱ درخواست).")
    text_options = [fid(p, m) for p in enabled for m in LM[p]["text"]]
    j_def = default_text(enabled, LM, ["gemini", "groq", "openrouter"])
    d_def = default_text(enabled, LM, ["groq", "gemini", "openrouter"])
    judge = st.selectbox("مدل داور", text_options, index=text_options.index(j_def) if j_def in text_options else 0,
                         format_func=label_of, key="judge_model")
    data_model = st.selectbox("مدل AI داده‌خوان", text_options,
                              index=text_options.index(d_def) if d_def in text_options else 0,
                              format_func=label_of, key="data_model")
    fb_data = st.slider("تعداد مدل جایگزین برای داده‌خوان و داور", 0, 2, 1)

text_chain = chain_pool(enabled, LM, "text")

tab1, tab2 = st.tabs(["تحلیل", "آرشیو"])
with tab1:
    f = st.file_uploader("تصویر چارت (برای تحلیل AI بینایی)", type=["png", "jpg", "jpeg", "webp"])

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
    n_v = len(models) if f else 0
    need_full = n_v + (1 if can_mkt else 0) + (1 if n_v >= 2 else 0)
    st.caption(f"هزینه تقریبی (بدون تلاش مجدد): تحلیل کامل {need_full} درخواست (پخش‌شده بین ارائه‌دهنده‌ها) | بدون عکس (فقط داده‌خوان) 1 درخواست | فقط نمودار صفر.")
    or_planned = sum(1 for m in models if m.startswith("openrouter::")) + sum(1 for m in (data_model, judge) if m.startswith("openrouter::"))
    if or_planned and u.get("openrouter", 0) + or_planned > OR_LIMIT:
        st.warning("سهمیه تخمینی OpenRouter برای این تحلیل کافی نیست.")

    b1, b2, b3 = st.columns(3)
    run_ai = b1.button("تحلیل کامل (AI + نمودار + کارت)", disabled=not (f and models))
    run_data = b2.button("بدون عکس (AI داده‌خوان)", disabled=not can_mkt)
    run_mkt = b3.button("فقط نمودار (رایگان)", disabled=not can_mkt)

    def finish(M, vis, dat, ai_state):
        F = make_final(M, vis, dat)
        M["plan"] = M["A"]["plans"][F["side"]]
        st.session_state["mkt"] = M
        st.session_state["final"] = F
        st.session_state["ai"] = ai_state
        st.session_state["dat"] = dat

    def data_models():
        return candidates(data_model, text_chain, fb_data)

    if run_mkt:
        try:
            with st.spinner("دریافت داده بایننس..."):
                M = load_market(symbol_in, tf_in)
            finish(M, None, None, None)
        except Exception as e:
            st.session_state["mkt"] = None; st.session_state["final"] = None
            st.error(f"دریافت داده ناموفق بود. نماد را بررسی کن (مثل BTCUSDT). جزئیات: {e}")

    if run_data:
        try:
            with st.spinner("دریافت داده بایننس..."):
                M = load_market(symbol_in, tf_in)
            with st.spinner("AI داده‌خوان در حال بررسی..."):
                dat = run_data_ai(data_models(), M)
            if not dat["ok"]:
                st.warning("AI داده‌خوان جواب نداد: " + dat["error"] + " — فقط رأی موتور قانون نشان داده می‌شود.")
            finish(M, None, dat, None)
        except Exception as e:
            st.session_state["mkt"] = None; st.session_state["final"] = None
            st.error(f"دریافت داده ناموفق بود: {e}")

    if run_ai:
        M = None
        if can_mkt:
            try:
                with st.spinner("دریافت داده بایننس..."):
                    M = load_market(symbol_in, tf_in)
            except Exception as e:
                st.error(f"دریافت داده بایننس ناموفق بود: {e}")
        img, mime = b64(f), f.type
        with st.spinner("مدل‌های بینایی در حال تحلیل..."), cf.ThreadPoolExecutor(len(models)) as ex:
            results = list(ex.map(lambda m: analyze(m, img, mime, note), models))
        if use_backup:
            pool = [b for b in vision_backup_pool(enabled, LM) if b not in models and b not in st.session_state["bad"]]
            for m_, o_, s_, e_ in list(results):
                if e_ and pool:
                    with st.spinner("پشتیبان بینایی..."):
                        results.append(analyze(pool.pop(0), img, mime, note))
        for m, out, sec, err in results:
            if err:
                st.session_state["bad"].add(m)
                st.session_state["health"][m] = dict(ok=False, sec=sec, reason=explain(err))
        outs = {m: out for m, out, sec, err in results if not err}
        parsed = {m: j for m, o in outs.items() if (j := parse_json(o))}
        real_price = M["A"]["price"] if M else None
        validation = validate_vision(parsed, real_price)
        vis = vision_vote(parsed, validation) if parsed else None
        good = {m: j for m, j in parsed.items() if validation[m]["status"] != "invalid"}
        dat = None
        if M:
            M["ai_levels"] = ai_levels_from(good)
            with st.spinner("AI داده‌خوان در حال بررسی..."):
                dat = run_data_ai(data_models(), M)
            if not dat["ok"]:
                st.warning("AI داده‌خوان جواب نداد: " + dat["error"])
        verdict = None
        if len(outs) >= 2:
            with st.spinner("داور..."):
                joined = "\n\n".join(f"### {label_of(m)}\nاعتبار: {validation.get(m, {}).get('label', '—')}\n{o}" for m, o in outs.items())
                r = try_models(candidates(judge, text_chain, fb_data), JUDGE_PROMPT + "\n\n" + joined, 3000)
                if r["ok"]:
                    verdict = r["out"] + (f"\n\n_(مدل داور: {label_of(r['model'])})_" if r["model"] != judge else "")
                else:
                    verdict = "داور جواب نداد: " + "؛ ".join(f"{label_of(m)}: {w}" for m, w in r["errs"])
            c = db()
            first = next(iter(parsed.values()), {})
            c.execute("INSERT INTO a(ts,symbol,timeframe,image,outputs,verdict) VALUES(datetime('now'),?,?,?,?,?)",
                      (symbol_in.strip() or first.get("symbol"), (tf_in if tf_in != NO_TF else first.get("timeframe")),
                       f.getvalue(), json.dumps(outs, ensure_ascii=False), verdict))
            c.commit(); c.close()
        ai_state = dict(results=results, outs=outs, parsed=parsed, verdict=verdict, validation=validation)
        failed = [m for m, o, s, e in results if e]
        if failed:
            st.warning("این مدل‌ها خطا دادند و از انتخاب‌های بعدی کنار گذاشته شدند: " + "، ".join(label_of(m) for m in failed))
        if M:
            finish(M, vis, dat, ai_state)
        else:
            st.session_state["mkt"] = None; st.session_state["final"] = None
            st.session_state["ai"] = ai_state
            if not can_mkt:
                st.info("برای نمودار واقعی و کارت تصمیم، نماد بایننس (مثل BTCUSDT) و تایم‌فریم را انتخاب کن.")

    M = st.session_state.get("mkt"); F = st.session_state.get("final")
    if M and F:
        A = M["A"]
        st.markdown("## کارت تصمیم‌گیری")
        st.markdown(card_html(M, F), unsafe_allow_html=True)
        plan = M.get("plan")
        if plan:
            r1, r2 = st.columns(2)
            bal = r1.number_input("موجودی حساب (USDT)", min_value=0.0, value=1000.0, step=100.0)
            rp = r2.number_input("ریسک هر معامله (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            risk_usd = bal * rp / 100
            unit = abs(plan["entry"] - plan["sl"])
            qty = risk_usd / unit if unit else 0
            st.info(f"ریسک: {risk_usd:,.2f} USDT | حجم پوزیشن: {qty:,.6f} واحد | ارزش پوزیشن: {qty*plan['entry']:,.2f} USDT")
        with st.expander("چک‌لیست موتور قانون"):
            st.dataframe([{"": "✅" if i["ok"] else "❌", "مورد": i["label"], "جزئیات": i["detail"], "امتیاز": i["pts"]}
                          for i in A["scores"][F["side"]]["items"]], use_container_width=True)
        dat = st.session_state.get("dat")
        if dat:
            with st.expander("خروجی و خطاهای AI داده‌خوان"):
                if dat.get("ok"):
                    st.caption(f"مدل استفاده‌شده: {label_of(dat['model'])} ({dat['sec']}s)")
                    st.code(dat["raw"], language="json")
                for m, w in dat.get("errs", []):
                    st.write(f"❌ {label_of(m)}: {w}")

        st.markdown("## نمودار")
        lib = get_lwc_js()
        if lib:
            components.html(build_lwc_html(M, lib), height=800, scrolling=False)
            st.caption(f"منبع داده: {M['host']} | همه کنترل‌ها (حالت نمایش، لایه‌ها، بازه) داخل خود نمودار است. حجم روی نمودار قیمت و RSI همگام زیر آن است؛ با لمس نمودار مقادیر کندل، حجم و RSI همان لحظه بالا نوشته می‌شود. آخرین کندل ناقص است و در تشخیص‌ها استفاده نمی‌شود.")
        else:
            st.warning("بارگذاری کتابخانه نمودار سریع از سرور ممکن نشد؛ نمودار ساده جایگزین نشان داده می‌شود.")
            st.plotly_chart(build_price_chart(M), use_container_width=True)
        with st.expander("نمودار Plotly (جایگزین)"):
            st.plotly_chart(build_price_chart(M), use_container_width=True)
        with st.expander("جدول زون‌ها"):
            st.dataframe(zone_rows(M), use_container_width=True)
        with st.expander("جزئیات تشخیص‌ها"):
            st.write(f"ساختار: {'صعودی' if A['trend']==1 else 'نزولی' if A['trend']==-1 else 'نامشخص'} | OB معتبر: {len(A['obs'])} | FVG پرنشده: {len(A['fvgs'])}")
            st.write(f"EQH: {[fmt(x) for x in A['liq']['eqh']]} | EQL: {[fmt(x) for x in A['liq']['eql']]}")
            st.write(f"Sweep اخیر: {A['liq']['sweep']}")

    AI = st.session_state.get("ai")
    if AI:
        st.markdown("## خروجی مدل‌های بینایی")
        for m, out, sec, err in AI["results"]:
            lab = AI["validation"].get(m, {}).get("label", "")
            with st.expander(f"{label_of(m)} ({sec}s) {lab}", expanded=bool(err)):
                if err: st.error(f"{explain(err)} — {err[:200]}")
                else: st.code(out, language="json")
        if AI["parsed"]:
            st.markdown("## تحلیل گرافیکی مدل‌ها")
            st.plotly_chart(build_figure(AI["parsed"]), use_container_width=True)
            st.dataframe(scenario_table(AI["parsed"], AI["validation"]), use_container_width=True)
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
