
import base64, json, time, sqlite3, os, concurrent.futures as cf
import requests, streamlit as st

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash-001",
]
JUDGE_DEFAULT = "openai/gpt-4o"

ANALYST_PROMPT = """تو یک تحلیلگر تکنیکال دقیق و محتاط هستی. فقط بر اساس اطلاعات قابل مشاهده در تصویر تحلیل کن.
فقط یک JSON معتبر برگردان با این کلیدها:
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
توصیه قطعی خرید/فروش یا وعده سود نده. خروجی فارسی باشد (کلیدها انگلیسی)."""

JUDGE_PROMPT = """چند تحلیل JSON از یک چارت واحد داری. کار تو داور است:
1) موارد مورد توافق (حداقل دو مدل) را جدا کن.
2) اختلافات را با ذکر نام مدل‌ها بنویس.
3) ادعاهای assumed که فقط یک مدل گفته را «ضعیف» علامت بزن.
4) سناریوهای نهایی مشروط با ابطال (invalidation) بده، نه توصیه قطعی.
5) چک‌لیست تأیید دستی با داده زنده بده.
خروجی فارسی و به‌صورت Markdown با جدول باشد. این محتوا مشاوره مالی نیست."""

def b64(file):
    return base64.b64encode(file.getvalue()).decode()

def call(key, model, messages, timeout=120):
    t = time.time()
    r = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "temperature": 0.2}, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], round(time.time()-t, 1)

def analyze(key, model, img_b64, mime, note):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": ANALYST_PROMPT + ("\nتوضیح کاربر: " + note if note else "")},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}]}]
    try:
        out, sec = call(key, model, msgs)
        return model, out, sec, None
    except Exception as e:
        return model, None, 0, str(e)

def db():
    c = sqlite3.connect("analyses.db")
    c.execute("""CREATE TABLE IF NOT EXISTS a(id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT,
      timeframe TEXT, image BLOB, outputs TEXT, verdict TEXT, outcome TEXT)""")
    return c

st.set_page_config(page_title="Chart AI Hub", layout="wide")
st.title("هاب تحلیل چارت چندمدلی")
st.caption("خروجی ابزار آموزشی/پژوهشی است و مشاوره مالی نیست.")

with st.sidebar:
    key = st.text_input("OpenRouter API key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    models = st.text_area("مدل‌ها (هر خط یکی)", "\n".join(DEFAULT_MODELS)).split()
    judge = st.text_input("مدل داور", JUDGE_DEFAULT)

tab1, tab2 = st.tabs(["تحلیل", "آرشیو"])
with tab1:
    f = st.file_uploader("تصویر چارت", type=["png", "jpg", "jpeg", "webp"])
    note = st.text_input("اطلاعات اضافه (نماد، تایم‌فریم، تاریخ، منبع داده)")
    if f: st.image(f, use_container_width=True)
    if st.button("تحلیل کن", disabled=not (f and key)):
        img, mime = b64(f), f.type
        with st.spinner("در حال تحلیل..."), cf.ThreadPoolExecutor(len(models)) as ex:
            results = list(ex.map(lambda m: analyze(key, m, img, mime, note), models))
        outs = {}
        cols = st.columns(len(results))
        for col, (m, out, sec, err) in zip(cols, results):
            with col:
                st.subheader(m); st.caption(f"{sec}s")
                if err: st.error(err)
                else:
                    outs[m] = out; st.code(out, language="json")
        if len(outs) >= 2:
            with st.spinner("داور..."):
                joined = "\n\n".join(f"### {m}\n{o}" for m, o in outs.items())
                verdict, _ = call(key, judge, [{"role": "user", "content": JUDGE_PROMPT + "\n\n" + joined}])
            st.markdown("## جمع‌بندی داور"); st.markdown(verdict)
            c = db()
            sym = tf = None
            try:
                j = json.loads(next(iter(outs.values())).strip().strip("`").removeprefix("json"))
                sym, tf = j.get("symbol"), j.get("timeframe")
            except Exception: pass
            c.execute("INSERT INTO a(ts,symbol,timeframe,image,outputs,verdict) VALUES(datetime('now'),?,?,?,?,?)",
                      (sym, tf, f.getvalue(), json.dumps(outs, ensure_ascii=False), verdict))
            c.commit(); st.success("ذخیره شد.")
with tab2:
    c = db()
    rows = c.execute("SELECT id,ts,symbol,timeframe,outcome FROM a ORDER BY id DESC").fetchall()
    st.dataframe(rows, column_config={0: "id", 1: "زمان", 2: "نماد", 3: "تایم‌فریم", 4: "نتیجه واقعی"})
    rid = st.number_input("id", min_value=0, step=1)
    out = st.text_input("نتیجه واقعی (مثلاً: سناریوی صعودی فعال شد / ابطال)")
    if st.button("ثبت نتیجه") and rid:
        c.execute("UPDATE a SET outcome=? WHERE id=?", (out, rid)); c.commit(); st.rerun()
