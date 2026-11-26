
import base64, json, time, sqlite3, os, re, concurrent.futures as cf
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

def db():
    c = sqlite3.connect("analyses.db")
    c.execute("""CREATE TABLE IF NOT EXISTS a(id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT,
      timeframe TEXT, image BLOB, outputs TEXT, verdict TEXT, outcome TEXT)""")
    return c

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
    f = st.file_uploader("تصویر چارت", type=["png", "jpg", "jpeg", "webp"])

    c1, c2, c3 = st.columns(3)
    symbol_in = c1.text_input("نماد (اختیاری)", placeholder="مثلاً EURUSD یا BTCUSDT")
    tf_in = c2.selectbox("تایم‌فریم (اختیاری)",
        ["— انتخاب نشده —", "1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W", "1M"])
    use_date = c3.checkbox("تاریخ را وارد می‌کنم")
    date_in = c3.date_input("تاریخ چارت") if use_date else None
    extra_in = st.text_input("توضیح دیگر (اختیاری)", placeholder="مثلاً منبع داده یا اندیکاتورها")

    parts = []
    if symbol_in.strip():
        parts.append(f"نماد: {symbol_in.strip()}")
    if tf_in != "— انتخاب نشده —":
        parts.append(f"تایم‌فریم: {tf_in}")
    if date_in:
        parts.append(f"تاریخ چارت: {date_in.isoformat()}")
    if extra_in.strip():
        parts.append(extra_in.strip())
    note = " | ".join(parts)

    if f: st.image(f, use_container_width=True)
    if st.button("تحلیل کن", disabled=not (f and key and models)):
        img, mime = b64(f), f.type
        with st.spinner("در حال تحلیل..."), cf.ThreadPoolExecutor(len(models)) as ex:
            results = list(ex.map(lambda m: analyze(key, m, img, mime, note), models))
        outs = {}
        for m, out, sec, err in results:
            with st.expander(f"{m} ({sec}s)", expanded=bool(err)):
                if err: st.error(err)
                else:
                    outs[m] = out; st.code(out, language="json")
        parsed = {m: j for m, o in outs.items() if (j := parse_json(o))}
        if parsed:
            st.markdown("## تحلیل گرافیکی")
            st.plotly_chart(build_figure(parsed), use_container_width=True)
            st.dataframe(scenario_table(parsed), use_container_width=True)
            st.caption("خط آبی نقطه‌چین = سطحی که حداقل دو مدل (در ۰٫۳٪ فاصله) روی آن توافق دارند. روی نمادها بزن تا دلیل مدل را ببینی.")
        elif outs:
            st.info("خروجی هیچ مدلی JSON معتبر نبود؛ نمودار ساخته نشد.")
        if len(outs) >= 2:
            with st.spinner("داور..."):
                joined = "\n\n".join(f"### {m}\n{o}" for m, o in outs.items())
                try:
                    verdict, _ = call(key, judge, [{"role": "user", "content": JUDGE_PROMPT + "\n\n" + joined}])
                except Exception as e:
                    verdict = f"خطای داور: {e}"
            st.markdown("## جمع‌بندی داور"); st.markdown(verdict)
            c = db()
            first = next(iter(parsed.values()), {})
            c.execute("INSERT INTO a(ts,symbol,timeframe,image,outputs,verdict) VALUES(datetime('now'),?,?,?,?,?)",
                      (symbol_in.strip() or first.get("symbol"),
                       (tf_in if tf_in != "— انتخاب نشده —" else first.get("timeframe")),
                       f.getvalue(), json.dumps(outs, ensure_ascii=False), verdict))
            c.commit(); st.success("ذخیره شد.")
with tab2:
    c = db()
    rows = c.execute("SELECT id,ts,symbol,timeframe,outcome FROM a ORDER BY id DESC").fetchall()
    st.dataframe(rows)
    rid = st.number_input("id", min_value=0, step=1)
    out = st.text_input("نتیجه واقعی (مثلاً: سناریوی صعودی فعال شد / ابطال)")
    if st.button("ثبت نتیجه") and rid:
        c.execute("UPDATE a SET outcome=? WHERE id=?", (out, rid)); c.commit(); st.rerun()
