# pages/1_Investment.py  - System A card for the Streamlit app (does NOT modify app.py)
import os, json, re, sqlite3, datetime
import numpy as np, pandas as pd, requests, streamlit as st
import trend_regime as tr

st.set_page_config(page_title="سرمایه‌گذاری روندی (آزمایشی)", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
html, body, .stApp, .stMarkdown, label, p, h1, h2, h3, h4, button, input, textarea {font-family: Vazirmatn, Tahoma, sans-serif !important;}
.stMarkdown p, .stMarkdown li, h1, h2, h3, label, .stCaption, [data-testid="stAlert"] p {direction: rtl; text-align: right;}
.block-container {max-width: 900px; padding-top: 1.5rem;}
pre, code, [data-testid="stDataFrame"] {direction: ltr !important; text-align: left !important;}
.stButton button {width: 100%; border-radius: 12px; font-weight: 700;}
</style>""", unsafe_allow_html=True)

DIR = os.getenv("TREND_DATA_DIR", ".")
STATE_P = os.path.join(DIR, "trend_state.json")
LOG_P = os.path.join(DIR, "trend_log.csv")
DB_P = os.getenv("ANALYSES_DB_PATH", "analyses.db")
SYN = os.getenv("TREND_SYNTHETIC") or None
LABELS = {1.0: ("ریسک‌پذیر", "بیت‌کوین بالای هر دو میانگین ۱۵۰ و ۲۰۰ روزه است.", "success"),
          0.5: ("نیمه‌محتاط", "بیت‌کوین فقط بالای یکی از دو میانگین است.", "warning"),
          0.0: ("نقد", "بیت‌کوین زیر هر دو میانگین است؛ خرید جدید نداریم.", "error")}
PROV = {"gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"]),
        "groq": ("https://api.groq.com/openai/v1", ["GROQ_API_KEY", "GROQ_KEY"]),
        "openrouter": ("https://openrouter.ai/api/v1", ["OPENROUTER_API_KEY", "OPENROUTER_KEY"])}


def secret(names):
    for n in names:
        try:
            v = st.secrets[n]
            if v: return str(v).strip().strip('"').strip("'")
        except Exception:
            pass
        v = os.getenv(n)
        if v: return v.strip().strip('"').strip("'")
    return ""


def db():
    c = sqlite3.connect(DB_P, timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS ai_risk_log(id INTEGER PRIMARY KEY, ts TEXT, date TEXT, flag TEXT, reasons TEXT, model TEXT, btc REAL, gate REAL, headlines TEXT)")
    return c


@st.cache_data(ttl=900, show_spinner=False)
def load():
    s = tr.get_regime_state(synthetic=SYN)
    c = s.pop("_calc"); P = s.pop("_P")
    paper = None
    if os.path.exists(STATE_P):
        s0 = pd.Timestamp(json.load(open(STATE_P))["start"])
        net = tr.simulate(c["ret"], c["w"], start_after=s0)
        net = net[net.index > s0]
        if len(net):
            eq = (1 + net).cumprod(); b = (1 + c["ret"]["BTCUSDT"].reindex(net.index).fillna(0)).cumprod()
            paper = dict(start=s0, nav=float(eq.iloc[-1]), dd=float((eq / eq.cummax() - 1).min()), days=len(net), btc=float(b.iloc[-1] - 1),
                         series=pd.DataFrame({"استراتژی": eq, "بیت‌کوین": b}))
    return s, paper, P["BTCUSDT"] if "BTCUSDT" in P else P.iloc[:, 0]


def log_today(s, paper):
    cash_t = 1 - sum(s["target_today"].values()); cash_e = 1 - sum(s["effective"].values())
    row = dict(date=str(s["date"].date()), gate=s["regime_g"], btc_vs_ma150=s["btc_vs_ma150"], btc_vs_ma200=s["btc_vs_ma200"],
               cash_target=round(cash_t, 4), cash_effective=round(cash_e, 4), reb_day=s["is_rebalance_day"],
               nav=paper["nav"] if paper else None, ai_flag=None,
               weights_effective=json.dumps({k: round(v, 4) for k, v in s["effective"].items() if v > 0}))
    old = pd.read_csv(LOG_P) if os.path.exists(LOG_P) else pd.DataFrame()
    if len(old):
        keep = old[old.date == row["date"]]
        if len(keep) and "ai_flag" in keep: row["ai_flag"] = keep["ai_flag"].iloc[-1]
        old = old[old.date != row["date"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG_P, index=False)
    return cash_t, cash_e


def ai_call(prov, model, prompt):
    base, names = PROV[prov]; key = secret(names)
    if not key: raise RuntimeError("کلید API برای این ارائه‌دهنده تنظیم نشده است.")
    if not model:
        r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        ids = [(m.get("id") or "").replace("models/", "") for m in r.json().get("data", [])] if r.status_code == 200 else []
        pref = [i for i in ids if ("flash" in i and prov == "gemini" and not any(w in i for w in ("image", "tts", "live", "audio", "embed")))] if prov == "gemini" else \
               [i for i in ids if any(w in i for w in ("llama-3.3-70b", "gpt-oss-120b"))]
        model = (pref or ids or [""])[0]
        if not model: raise RuntimeError("مدلی پیدا نشد؛ نام مدل را دستی وارد کن.")
    msgs = [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "temperature": 0.2, "max_tokens": 600}
    if prov in ("gemini", "groq"): body["response_format"] = {"type": "json_object"}
    r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
    if r.status_code == 400 and "response_format" in body:
        body.pop("response_format"); r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
    if r.status_code != 200: raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    txt = r.json()["choices"][0]["message"]["content"] or ""
    m = re.search(r"\{.*\}", txt, re.S)
    j = json.loads(m.group(0)) if m else {}
    flag = str(j.get("risk_flag", "")).lower()
    if flag not in ("low", "medium", "high"): raise RuntimeError("خروجی نامعتبر از مدل")
    return flag, [str(x)[:200] for x in (j.get("reasons") or [])][:3], model


st.title("سرمایه‌گذاری روندی (آزمایشی)")
st.caption("خرید و نگهداری و فروش روزانه، فقط لانگ و اسپات. پارامترها قفل‌شده‌اند. این ابزار پژوهشی است و توصیه مالی نیست.")
try:
    s, paper, btc = load()
except Exception as e:
    st.error(f"دریافت داده ناموفق بود: {e}")
    st.stop()

title, text, kind = LABELS[s["regime_g"]]
getattr(st, kind)(f"وضعیت: {title} — {text}")
cash_t, cash_e = log_today(s, paper)
c1, c2, c3 = st.columns(3)
c1.metric("BTC از MA150", f"{s['btc_vs_ma150'] * 100:+.1f}%" if s["btc_vs_ma150"] is not None else "-")
c2.metric("BTC از MA200", f"{s['btc_vs_ma200'] * 100:+.1f}%" if s["btc_vs_ma200"] is not None else "-")
c3.metric("نقد (هدف)", f"{cash_t * 100:.0f}%")
st.caption(f"آخرین کندل بسته‌شده: {s['date'].date()}  |  قیمت BTC: {s['btc']:,.0f}  |  بازتنظیم بعدی: {s['next_rebalance'].date()}  |  "
           f"{'امروز روز بازتنظیم است' if s['is_rebalance_day'] else 'امروز فقط نگاه کن'}")

rows = [dict(نماد=k.replace("USDT", ""), سیگنال=s["signal"][k], نوسان=f"{s['vol'][k] * 100:.0f}%",
             هدف=f"{s['target_today'][k] * 100:.1f}%", اجرایی=f"{s['effective'][k] * 100:.1f}%")
        for k in s["signal"] if s["target_today"][k] > 0 or s["effective"][k] > 0]
if rows:
    st.subheader("وزن‌های پیشنهادی")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("«هدف» وزن امروز است و «اجرایی» وزن آخرین روز بازتنظیم؛ فقط در روز بازتنظیم تغییرات را اجرا کن.")
else:
    st.info("هیچ دارایی‌ای انتخاب نشده؛ ۱۰۰٪ نقد.")

st.subheader("معامله کاغذی")
if paper:
    a, b, c = st.columns(3)
    a.metric("NAV", f"{paper['nav']:.4f}", f"{(paper['nav'] - 1) * 100:+.2f}%")
    b.metric("بیشترین افت", f"{paper['dd'] * 100:.1f}%")
    c.metric("بیت‌کوین همان مدت", f"{paper['btc'] * 100:+.1f}%")
    st.line_chart(paper["series"], height=200)
    st.caption(f"از {paper['start'].date()} ({paper['days']} روز). معیار توقف از پیش‌ثبت‌شده: افت بیش از ۲۵٪ یا ارزیابی ضعیف بعد از ۶ ماه.")
elif os.path.exists(STATE_P):
    st.info("معامله کاغذی شروع شده؛ هنوز روز کامل جدیدی ثبت نشده است.")
elif st.button("شروع معامله کاغذی از امروز"):
    json.dump({"start": str(s["date"].date())}, open(STATE_P, "w")); st.cache_data.clear(); st.rerun()

with st.expander("پرچم ریسک هوش مصنوعی (فقط ثبت می‌شود، وزن‌ها را تغییر نمی‌دهد)"):
    st.caption("مدل به اینترنت دسترسی ندارد؛ فقط با اعداد بالا و خبرهایی که خودت پیست می‌کنی قضاوت می‌کند. دقتش را با ثبت نتیجه‌ی ۷ روز بعد می‌سنجیم.")
    prov = st.selectbox("ارائه‌دهنده", [p for p in PROV if secret(PROV[p][1])] or list(PROV))
    model = st.text_input("نام مدل (خالی = انتخاب خودکار)", value=os.getenv("TREND_AI_MODEL", ""))
    news = st.text_area("تیترها و خبرهای امروز (اختیاری)", height=100)
    if st.button("ثبت پرچم ریسک امروز"):
        prompt = ("You are a risk reviewer for a long-only crypto daily trend system. You cannot browse the web. "
                  "Use ONLY the data and headlines below. Do NOT predict prices. Answer in JSON: "
                  '{"risk_flag":"low|medium|high","reasons":["max 3 short Persian strings"]}. '
                  "Use high only if data or headlines show exchange, stablecoin, regulatory or systemic stress.\n"
                  f"date={s['date'].date()} btc={s['btc']:.0f} btc_vs_ma150={s['btc_vs_ma150']} btc_vs_ma200={s['btc_vs_ma200']} "
                  f"regime_gate={s['regime_g']} cash_target={cash_t:.2f}\nheadlines:\n{news.strip() or '(none provided)'}")
        try:
            flag, reasons, used = ai_call(prov, model.strip(), prompt)
            c = db(); c.execute("INSERT INTO ai_risk_log(ts,date,flag,reasons,model,btc,gate,headlines) VALUES(?,?,?,?,?,?,?,?)",
                                (datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), str(s["date"].date()), flag,
                                 json.dumps(reasons, ensure_ascii=False), used, s["btc"], s["regime_g"], news.strip()[:2000])); c.commit(); c.close()
            old = pd.read_csv(LOG_P); old["ai_flag"] = old["ai_flag"].astype(object); old.loc[old.date == str(s["date"].date()), "ai_flag"] = flag; old.to_csv(LOG_P, index=False)
            st.success(f"پرچم: {flag} — " + "؛ ".join(reasons))
        except Exception as e:
            st.error(f"ناموفق: {e}")
    try:
        c = db(); lg = pd.read_sql_query("SELECT date,flag,model,btc,reasons FROM ai_risk_log ORDER BY id DESC LIMIT 200", c); c.close()
    except Exception:
        lg = pd.DataFrame()
    if len(lg):
        lg["date_ts"] = pd.to_datetime(lg["date"])
        fwd = (btc.shift(-7) / btc - 1)
        lg["بازده ۷ روز بعد BTC"] = [fwd.get(d, np.nan) for d in lg["date_ts"]]
        st.dataframe(lg.drop(columns=["date_ts"]).head(20), hide_index=True, use_container_width=True)
        done = lg.dropna(subset=["بازده ۷ روز بعد BTC"])
        if len(done):
            g = done.groupby("flag")["بازده ۷ روز بعد BTC"].agg(["count", "mean"]).rename(columns={"count": "تعداد", "mean": "میانگین بازده ۷ روز"})
            st.dataframe(g, use_container_width=True)
            st.caption("تا چند ده نمونه جمع نشده، این جدول را مبنای تصمیم نکن.")

with st.expander("لاگ روزانه"):
    if os.path.exists(LOG_P):
        st.dataframe(pd.read_csv(LOG_P).tail(30), hide_index=True, use_container_width=True)
