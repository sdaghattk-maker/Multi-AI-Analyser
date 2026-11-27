# pages/1_Investment.py  - System A card for the Streamlit app (does NOT modify app.py)
import os, json, re, sqlite3, datetime
import numpy as np, pandas as pd, requests, streamlit as st
import trend_regime as tr

st.set_page_config(page_title="سرمایه‌گذاری روندی (آزمایشی)", layout="centered")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
html, body, .stApp, .stMarkdown, label, p, h1, h2, h3, h4, button, input, textarea {font-family: Vazirmatn, Tahoma, sans-serif !important;}
.stMarkdown, .stMarkdown p, .stMarkdown li, h1, h2, h3, label, .stCaption, [data-testid="stCaptionContainer"],
[data-testid="stAlert"], [data-testid="stAlert"] p, [data-testid="stMetric"], [data-testid="stMetric"] *,
[data-testid="stExpander"] summary, [data-testid="stExpander"] p, .stSelectbox label, .stTextInput label, .stTextArea label,
textarea, input {direction: rtl !important; text-align: right !important;}
[data-testid="stMetricValue"], [data-testid="stMetricDelta"] {justify-content: flex-end;}
.block-container {max-width: 900px; padding-top: 1.5rem;}
pre, code, [data-testid="stDataFrame"] {direction: ltr !important; text-align: left !important;}
.stButton button {width: 100%; border-radius: 12px; font-weight: 700;}
</style>""", unsafe_allow_html=True)

DIR = os.getenv("TREND_DATA_DIR", ".")
STATE_P = os.path.join(DIR, "trend_state.json")
LOG_P = os.path.join(DIR, "trend_log.csv")
DB_P = os.getenv("ANALYSES_DB_PATH", "analyses.db")
SYN = os.getenv("TREND_SYNTHETIC") or None
GIST_DESC = "chart-ai-hub-model-health"
AUTO = "خودکار (اولین مدل سالم)"
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


@st.cache_data(ttl=300, show_spinner=False)
def load_health():
    tok = secret(["GITHUB_TOKEN", "GH_TOKEN"])
    if not tok:
        return {"_err": "GITHUB_TOKEN در Secrets پیدا نشد."}
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    try:
        gid = secret(["GIST_ID", "GISTID"])
        g = None
        if gid:
            r = requests.get(f"https://api.github.com/gists/{gid}", headers=h, timeout=20)
            g = r.json() if r.status_code == 200 else None
        for page in (1, 2, 3):
            if g: break
            r = requests.get("https://api.github.com/gists", headers=h, params={"per_page": 100, "page": page}, timeout=20)
            if r.status_code != 200:
                return {"_err": f"خواندن Gist ناموفق بود: {r.status_code} (دسترسی gist توکن را بررسی کن)."}
            for x in r.json():
                if x.get("description") == GIST_DESC:
                    g = requests.get(f"https://api.github.com/gists/{x['id']}", headers=h, timeout=20).json(); break
            if len(r.json()) < 100: break
        if not g or not g.get("files"):
            return {"_err": "Gist مدل‌های سالم پیدا نشد؛ در اپ اصلی یک بار تست سلامت مدل‌ها را اجرا کن."}
        f = list(g["files"].values())[0]
        txt = requests.get(f["raw_url"], timeout=20).text if f.get("truncated") else f.get("content")
        return json.loads(txt or "{}")
    except Exception as e:
        return {"_err": f"خطا در خواندن Gist: {e}"}


def healthy_models(prov):
    text = {}
    d = load_health()
    text.update((d.get("results") or {}).get("text") or {})
    text.update(st.session_state.get("healtht") or {})
    rk = d.get("ranking") or {}
    order = list(rk.get("judge") or []) + list(rk.get("data") or []) + list(text)
    out = []
    for fid in order:
        p, _, m = fid.partition("/")
        if p == prov and m and (text.get(fid) or {}).get("ok") and m not in out:
            out.append(m)
    if not out:
        try:
            c = db()
            c.execute("CREATE TABLE IF NOT EXISTS health(day TEXT, fid TEXT, kind TEXT, ok INTEGER, sec REAL, reason TEXT, soft INTEGER, PRIMARY KEY(day,fid,kind))")
            rows = c.execute("SELECT fid FROM health WHERE kind='text' AND ok=1 ORDER BY day DESC, sec").fetchall(); c.close()
            out = [r[0].partition("/")[2] for r in rows if r[0].startswith(prov + "/")]
        except Exception:
            pass
    return out


def ai_call(prov, model, prompt):
    base, names = PROV[prov]; key = secret(names)
    if not key: raise RuntimeError("کلید API برای این ارائه‌دهنده تنظیم نشده است.")
    hdr = {"Authorization": f"Bearer {key}"}
    cands = [model] if model else healthy_models(prov)[:3]
    if not cands: raise RuntimeError("مدل سالمی برای این ارائه‌دهنده ثبت نشده؛ در اپ اصلی تست سلامت مدل‌ها را اجرا کن.")
    last = ""
    for m in cands:
        body = {"model": m, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 1200}
        if prov in ("gemini", "groq"): body["response_format"] = {"type": "json_object"}
        if prov == "gemini" or (prov == "groq" and "gpt-oss" in m): body["reasoning_effort"] = "low"
        try:
            r = requests.post(f"{base}/chat/completions", headers=hdr, json=body, timeout=90)
            if r.status_code == 400:
                for k in ("response_format", "reasoning_effort"): body.pop(k, None)
                r = requests.post(f"{base}/chat/completions", headers=hdr, json=body, timeout=90)
            if r.status_code != 200:
                last = f"{m}: {r.status_code} {r.text[:150]}"; continue
            txt = r.json()["choices"][0]["message"]["content"] or ""
            mm = re.search(r"\{.*\}", txt, re.S)
            j = json.loads(mm.group(0)) if mm else {}
            flag = str(j.get("risk_flag", "")).lower()
            if flag not in ("low", "medium", "high"):
                last = f"{m}: خروجی نامعتبر"; continue
            return flag, [str(x)[:200] for x in (j.get("reasons") or [])][:3], m
        except Exception as e:
            last = f"{m}: {e}"
    raise RuntimeError(last or "همه مدل‌ها ناموفق بودند")


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
    st.caption("مدل به اینترنت دسترسی ندارد؛ فقط با اعداد بالا و خبرهایی که خودت پیست می‌کنی قضاوت می‌کند. فهرست مدل‌ها فقط مدل‌های سالمِ ثبت‌شده در اپ اصلی است.")
    prov = st.selectbox("ارائه‌دهنده", [p for p in PROV if secret(PROV[p][1])] or list(PROV))
    hm = healthy_models(prov)
    sel = st.selectbox("مدل (فقط مدل‌های سالم)", [AUTO] + hm)
    model = "" if sel == AUTO else sel
    if not hm:
        st.warning(load_health().get("_err") or "برای این ارائه‌دهنده مدل سالمی ثبت نشده؛ در اپ اصلی تست سلامت مدل‌ها را اجرا کن.")
    if st.button("به‌روزرسانی فهرست مدل‌ها"):
        load_health.clear(); st.rerun()
    news = st.text_area("تیترها و خبرهای امروز (اختیاری)", height=100)
    if st.button("ثبت پرچم ریسک امروز"):
        prompt = ("You are a risk reviewer for a long-only crypto daily trend system. You cannot browse the web. "
                  "Use ONLY the data and headlines below. Do NOT predict prices. Answer in JSON: "
                  '{"risk_flag":"low|medium|high","reasons":["max 3 short Persian strings"]}. '
                  "Use high only if data or headlines show exchange, stablecoin, regulatory or systemic stress.\n"
                  f"date={s['date'].date()} btc={s['btc']:.0f} btc_vs_ma150={s['btc_vs_ma150']} btc_vs_ma200={s['btc_vs_ma200']} "
                  f"regime_gate={s['regime_g']} cash_target={cash_t:.2f}\nheadlines:\n{news.strip() or '(none provided)'}")
        try:
            flag, reasons, used = ai_call(prov, model, prompt)
            c = db(); c.execute("INSERT INTO ai_risk_log(ts,date,flag,reasons,model,btc,gate,headlines) VALUES(?,?,?,?,?,?,?,?)",
                                (datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), str(s["date"].date()), flag,
                                 json.dumps(reasons, ensure_ascii=False), used, s["btc"], s["regime_g"], news.strip()[:2000])); c.commit(); c.close()
            old = pd.read_csv(LOG_P); old["ai_flag"] = old["ai_flag"].astype(object); old.loc[old.date == str(s["date"].date()), "ai_flag"] = flag; old.to_csv(LOG_P, index=False)
            st.success(f"پرچم: {flag} (مدل: {used}) — " + "؛ ".join(reasons))
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
