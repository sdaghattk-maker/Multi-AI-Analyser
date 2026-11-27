# pages/1_Investment.py  - System A card for the Streamlit app (does NOT modify app.py)
import os, json, re, html, sqlite3, datetime
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
.cards {display: grid; grid-template-columns: 1fr 1fr; gap: 12px; direction: rtl; margin: 6px 0 12px;}
@media (max-width: 640px) {.cards {grid-template-columns: 1fr;}}
.card {border: 2px solid var(--c); border-radius: 16px; padding: 14px; background: #fff; text-align: right; direction: rtl;}
.card .t {font-size: .78rem; color: #6b7280;}
.card .h {font-size: 1.25rem; font-weight: 700; color: var(--c); margin: 2px 0;}
.card .b {font-size: .85rem; color: #374151; margin: 4px 0;}
.card .f {font-size: .72rem; color: #6b7280; margin-top: 6px;}
.card ul {margin: 4px 0; padding-right: 18px; font-size: .82rem; color: #374151;}
</style>""", unsafe_allow_html=True)

DIR = os.getenv("TREND_DATA_DIR", ".")
STATE_P = os.path.join(DIR, "trend_state.json")
LOG_P = os.path.join(DIR, "trend_log.csv")
DB_P = os.getenv("ANALYSES_DB_PATH", "analyses.db")
SYN = os.getenv("TREND_SYNTHETIC") or None
GIST_DESC = "chart-ai-hub-model-health"
RISK_FILE = "risk_flags.json"
AUTO = "خودکار (اولین مدل سالم)"
LABELS = {1.0: ("ریسک‌پذیر", "بیت‌کوین بالای هر دو میانگین ۱۵۰ و ۲۰۰ روزه است.", "#16a34a"),
          0.5: ("نیمه‌محتاط", "بیت‌کوین فقط بالای یکی از دو میانگین است.", "#d97706"),
          0.0: ("نقد", "بیت‌کوین زیر هر دو میانگین است؛ خرید جدید نداریم.", "#dc2626")}
FLAGS = {"low": ("ریسک کم", "#16a34a"), "medium": ("ریسک متوسط", "#d97706"), "high": ("ریسک زیاد", "#dc2626")}
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


def split_fid(fid):
    fid = str(fid)
    for p in PROV:
        if fid.startswith(p) and len(fid) > len(p) and not fid[len(p)].isalnum():
            m = re.sub(r"^[^A-Za-z0-9]+", "", fid[len(p):])
            if m: return p, m
    return "", ""


def gh_headers():
    return {"Authorization": f"Bearer {secret(['GITHUB_TOKEN', 'GH_TOKEN'])}", "Accept": "application/vnd.github+json"}


def find_gist():
    if not secret(["GITHUB_TOKEN", "GH_TOKEN"]):
        raise RuntimeError("GITHUB_TOKEN در Secrets پیدا نشد.")
    h = gh_headers()
    gid = secret(["GIST_ID", "GISTID"]) or st.session_state.get("_gist_id")
    if gid:
        r = requests.get(f"https://api.github.com/gists/{gid}", headers=h, timeout=20)
        if r.status_code == 200: return r.json()
    for page in (1, 2, 3):
        r = requests.get("https://api.github.com/gists", headers=h, params={"per_page": 100, "page": page}, timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"خواندن Gist ناموفق بود: {r.status_code} (دسترسی gist توکن را بررسی کن).")
        for x in r.json():
            if x.get("description") == GIST_DESC:
                st.session_state["_gist_id"] = x["id"]
                return requests.get(f"https://api.github.com/gists/{x['id']}", headers=h, timeout=20).json()
        if len(r.json()) < 100: break
    raise RuntimeError("Gist مدل‌های سالم پیدا نشد؛ در اپ اصلی یک بار تست سلامت مدل‌ها را اجرا کن.")


def gist_file_text(f):
    return requests.get(f["raw_url"], timeout=20).text if f.get("truncated") else (f.get("content") or "")


@st.cache_data(ttl=300, show_spinner=False)
def load_health():
    try:
        g = find_gist()
        files = g.get("files") or {}
        names = [n for n in files if n != RISK_FILE]
        if not names: raise RuntimeError("فایل سلامت مدل‌ها در Gist نیست.")
        name = next((n for n in names if "health" in n.lower()), names[0])
        return json.loads(gist_file_text(files[name]) or "{}")
    except Exception as e:
        return {"_err": str(e)}


def healthy_models(prov):
    def ok_ids(d):
        out = []
        for fid, r in (d or {}).items():
            p, m = split_fid(fid)
            if p == prov and isinstance(r, dict) and r.get("ok") and m not in out:
                out.append(m)
        return out
    h = load_health()
    res = h.get("results") or {}
    gist = {}
    for k in ("text", "vision"):
        gist.update(res.get(k) or {})
    sess = {}
    for k in ("healtht", "health"):
        sess.update(st.session_state.get(k) or {})
    dbm = []
    try:
        c = db()
        c.execute("CREATE TABLE IF NOT EXISTS health(day TEXT, fid TEXT, kind TEXT, ok INTEGER, sec REAL, reason TEXT, soft INTEGER, PRIMARY KEY(day,fid,kind))")
        rows = c.execute("SELECT fid FROM health WHERE ok=1 ORDER BY day DESC, sec").fetchall(); c.close()
        for (fid,) in rows:
            p, m = split_fid(fid)
            if p == prov and m not in dbm: dbm.append(m)
    except Exception:
        pass
    rk = h.get("ranking") or {}
    rank = []
    for f in list(rk.get("judge") or []) + list(rk.get("data") or []):
        p, m = split_fid(f)
        if p == prov: rank.append(m)
    allm = ok_ids(gist) + ok_ids(sess) + dbm
    out = [m for m in rank if m in allm]
    out += [m for m in allm if m not in out]
    out = list(dict.fromkeys(out))
    st.session_state["_hsrc"] = f"Gist: {len(ok_ids(gist))} | جلسه: {len(ok_ids(sess))} | دیتابیس: {len(dbm)}"
    return out


def flags_from_db():
    try:
        c = db()
        rows = c.execute("SELECT ts,date,flag,reasons,model,btc,gate FROM ai_risk_log ORDER BY id").fetchall(); c.close()
        return [dict(ts=r[0], date=r[1], flag=r[2], reasons=json.loads(r[3] or "[]"), model=r[4], btc=r[5], gate=r[6]) for r in rows]
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def read_flags():
    """returns (records, source, error)"""
    try:
        g = find_gist()
        f = (g.get("files") or {}).get(RISK_FILE)
        recs = json.loads(gist_file_text(f) or "[]") if f else []
        return recs, "Gist", ""
    except Exception as e:
        return flags_from_db(), "محلی (با Reboot پاک می‌شود)", str(e)


def save_flag(rec):
    c = db()
    c.execute("INSERT INTO ai_risk_log(ts,date,flag,reasons,model,btc,gate,headlines) VALUES(?,?,?,?,?,?,?,?)",
              (rec["ts"], rec["date"], rec["flag"], json.dumps(rec["reasons"], ensure_ascii=False), rec["model"], rec["btc"], rec["gate"], rec.get("headlines", "")))
    c.commit(); c.close()
    try:
        g = find_gist()
        f = (g.get("files") or {}).get(RISK_FILE)
        recs = json.loads(gist_file_text(f) or "[]") if f else []
        recs.append({k: v for k, v in rec.items() if k != "headlines"})
        r = requests.patch(f"https://api.github.com/gists/{g['id']}", headers=gh_headers(), timeout=20,
                           json={"files": {RISK_FILE: {"content": json.dumps(recs[-400:], ensure_ascii=False, indent=1)}}})
        read_flags.clear()
        if r.status_code != 200: return False, f"ذخیره در Gist ناموفق بود: {r.status_code}"
        return True, ""
    except Exception as e:
        read_flags.clear()
        return False, f"فقط محلی ذخیره شد: {e}"


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
            if flag not in FLAGS:
                last = f"{m}: خروجی نامعتبر"; continue
            return flag, [str(x)[:200] for x in (j.get("reasons") or [])][:3], m
        except Exception as e:
            last = f"{m}: {e}"
    raise RuntimeError(last or "همه مدل‌ها ناموفق بودند")


def cards_html(s, cash_t, last, n_flags):
    title, text, color = LABELS[s["regime_g"]]
    m150 = f"{s['btc_vs_ma150'] * 100:+.1f}%" if s["btc_vs_ma150"] is not None else "-"
    m200 = f"{s['btc_vs_ma200'] * 100:+.1f}%" if s["btc_vs_ma200"] is not None else "-"
    inv = (f'<div class="card" style="--c:{color}"><div class="t">وضعیت سرمایه‌گذاری</div><div class="h">{title}</div>'
           f'<div class="b">نقد هدف: {cash_t * 100:.0f}٪<br>BTC از MA150: {m150} | از MA200: {m200}</div>'
           f'<div class="b">{text}</div>'
           f'<div class="f">قیمت BTC: {s["btc"]:,.0f} | کندل: {s["date"].date()} | بازتنظیم بعدی: {s["next_rebalance"].date()}</div></div>')
    if not last:
        ai = ('<div class="card" style="--c:#9ca3af"><div class="t">پرچم ریسک هوش مصنوعی</div><div class="h">هنوز ثبت نشده</div>'
              '<div class="b">در بخش «پرچم ریسک» پایین صفحه یک پرچم ثبت کن.</div>'
              '<div class="f">فقط اطلاع؛ وزن‌ها را تغییر نمی‌دهد</div></div>')
    else:
        label, fc = FLAGS.get(last["flag"], ("نامشخص", "#9ca3af"))
        fresh = last.get("date") == str(s["date"].date())
        if not fresh: fc = "#9ca3af"
        stale = "" if fresh else f" <span style='font-size:.75rem;color:#6b7280'>(قدیمی: {html.escape(str(last.get('date')))})</span>"
        rs = "".join(f"<li>{html.escape(str(x))}</li>" for x in (last.get("reasons") or [])[:3])
        ai = (f'<div class="card" style="--c:{fc}"><div class="t">پرچم ریسک هوش مصنوعی</div><div class="h">{label}{stale}</div>'
              f'<ul>{rs}</ul>'
              f'<div class="f">مدل: {html.escape(str(last.get("model")))} | ثبت: {html.escape(str(last.get("ts", ""))[:16])} UTC | تعداد پرچم‌ها: {n_flags}<br>فقط اطلاع؛ وزن‌ها را تغییر نمی‌دهد</div></div>')
    return f'<div class="cards">{inv}{ai}</div>'


st.title("سرمایه‌گذاری روندی (آزمایشی)")
st.caption("خرید و نگهداری و فروش روزانه، فقط لانگ و اسپات. پارامترها قفل‌شده‌اند. این ابزار پژوهشی است و توصیه مالی نیست.")
try:
    s, paper, btc = load()
except Exception as e:
    st.error(f"دریافت داده ناموفق بود: {e}")
    st.stop()

cash_t, cash_e = log_today(s, paper)
flags, flags_src, flags_err = read_flags()
last = flags[-1] if flags else None
st.markdown(cards_html(s, cash_t, last, len(flags)), unsafe_allow_html=True)
if last and last.get("date") == str(s["date"].date()) and last.get("flag") == "high" and s["regime_g"] > 0:
    st.warning("تضاد: سیستم روندی ریسک‌پذیر یا نیمه‌محتاط است ولی پرچم هوش مصنوعی «زیاد» است. تصمیم با خودت؛ وزن‌ها تغییر نکرده‌اند.")
st.caption(f"محل ذخیره پرچم‌ها: {flags_src}" + (f" — {flags_err}" if flags_err and flags_src != "Gist" else ""))

rows = [dict(نماد=k.replace("USDT", ""), سیگنال=s["signal"][k], نوسان=f"{s['vol'][k] * 100:.0f}%",
             هدف=f"{s['target_today'][k] * 100:.1f}%", اجرایی=f"{s['effective'][k] * 100:.1f}%")
        for k in s["signal"] if s["target_today"][k] > 0 or s["effective"][k] > 0]
if rows:
    st.subheader("وزن‌های پیشنهادی")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("«هدف» وزن امروز است و «اجرایی» وزن آخرین روز بازتنظیم؛ فقط در روز بازتنظیم تغییرات را اجرا کن.")
else:
    st.info("هیچ دارایی‌ای انتخاب نشده؛ ۱۰۰٪ نقد. " + ("امروز روز بازتنظیم است." if s["is_rebalance_day"] else "امروز فقط نگاه کن."))

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

with st.expander("ثبت پرچم ریسک هوش مصنوعی و تاریخچه"):
    st.caption("مدل به اینترنت دسترسی ندارد؛ فقط با اعداد صفحه و خبرهایی که خودت پیست می‌کنی قضاوت می‌کند. فهرست مدل‌ها فقط مدل‌های سالمِ ثبت‌شده در اپ اصلی است.")
    prov = st.selectbox("ارائه‌دهنده", [p for p in PROV if secret(PROV[p][1])] or list(PROV))
    hm = healthy_models(prov)
    st.caption("تعداد مدل سالم از هر منبع → " + st.session_state.get("_hsrc", ""))
    sel = st.selectbox("مدل (فقط مدل‌های سالم)", [AUTO] + hm)
    model = "" if sel == AUTO else sel
    if not hm:
        st.warning(load_health().get("_err") or "برای این ارائه‌دهنده مدل سالمی ثبت نشده؛ در اپ اصلی تست سلامت مدل‌ها را اجرا کن.")
    if st.button("به‌روزرسانی فهرست مدل‌ها"):
        load_health.clear(); st.rerun()
    news = st.text_area("تیترها و خبرهای امروز (اختیاری)", height=100)
    if st.session_state.get("_msg"):
        st.success(st.session_state.pop("_msg"))
    saved = False
    if st.button("ثبت پرچم ریسک امروز"):
        prompt = ("You are a risk reviewer for a long-only crypto daily trend system. You cannot browse the web. "
                  "Use ONLY the data and headlines below. Do NOT predict prices. Answer in JSON: "
                  '{"risk_flag":"low|medium|high","reasons":["max 3 short Persian strings"]}. '
                  "Use high only if data or headlines show exchange, stablecoin, regulatory or systemic stress.\n"
                  f"date={s['date'].date()} btc={s['btc']:.0f} btc_vs_ma150={s['btc_vs_ma150']} btc_vs_ma200={s['btc_vs_ma200']} "
                  f"regime_gate={s['regime_g']} cash_target={cash_t:.2f}\nheadlines:\n{news.strip() or '(none provided)'}")
        try:
            flag, reasons, used = ai_call(prov, model, prompt)
            rec = dict(ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), date=str(s["date"].date()),
                       flag=flag, reasons=reasons, model=used, btc=float(s["btc"]), gate=float(s["regime_g"]), headlines=news.strip()[:2000])
            ok, msg = save_flag(rec)
            if os.path.exists(LOG_P):
                old = pd.read_csv(LOG_P); old["ai_flag"] = old["ai_flag"].astype(object)
                old.loc[old.date == rec["date"], "ai_flag"] = flag; old.to_csv(LOG_P, index=False)
            st.session_state["_msg"] = f"پرچم: {flag} (مدل: {used}) — " + "؛ ".join(reasons) + ("" if ok else f"  |  هشدار: {msg}")
            saved = True
        except Exception as e:
            st.error(f"ناموفق: {e}")
    if saved:
        st.rerun()
    if flags:
        lg = pd.DataFrame(flags).iloc[::-1].reset_index(drop=True)
        lg["date_ts"] = pd.to_datetime(lg["date"])
        fwd = (btc.shift(-7) / btc - 1)
        lg["بازده ۷ روز بعد BTC"] = [fwd.get(d, np.nan) for d in lg["date_ts"]]
        lg["reasons"] = lg["reasons"].apply(lambda x: "؛ ".join(x) if isinstance(x, list) else x)
        st.dataframe(lg[["date", "flag", "model", "btc", "reasons", "بازده ۷ روز بعد BTC"]].head(20), hide_index=True, use_container_width=True)
        done = lg.dropna(subset=["بازده ۷ روز بعد BTC"])
        if len(done):
            g = done.groupby("flag")["بازده ۷ روز بعد BTC"].agg(["count", "mean"]).rename(columns={"count": "تعداد", "mean": "میانگین بازده ۷ روز"})
            st.dataframe(g, use_container_width=True)
            st.caption("تا چند ده نمونه جمع نشده، این جدول را مبنای تصمیم نکن.")

with st.expander("لاگ روزانه"):
    if os.path.exists(LOG_P):
        st.dataframe(pd.read_csv(LOG_P).tail(30), hide_index=True, use_container_width=True)
