import warnings
warnings.filterwarnings('ignore')

import streamlit as st

@st.cache_resource
def get_predictor():
    from models.predict import predict_title_desc
    return predict_title_desc

predict_title_desc = get_predictor()

st.set_page_config(
    page_title="Gold Sentiment Engine",
    page_icon="⚜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, .stApp {
        background-color: #090d18 !important;
        color: #d0daea !important;
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 1.5rem 2rem 2rem !important; max-width: 720px !important; }

    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        background: #0e1525;
        border: 1px solid rgba(212,175,55,0.22);
        border-radius: 16px; padding: 0.9rem 1.4rem; margin-bottom: 1.25rem;
    }
    .topbar-left { display: flex; align-items: center; gap: 12px; }
    .topbar-icon {
        width: 38px; height: 38px;
        background: linear-gradient(135deg, #c9a227, #f0e080);
        border-radius: 10px; display: flex; align-items: center;
        justify-content: center; font-size: 19px; line-height: 1;
    }
    .topbar-title { font-size: 1rem; font-weight: 600; color: #eed98a; letter-spacing: 0.3px; line-height: 1.25; }
    .topbar-sub   { font-size: 0.73rem; color: #8a9bb5; letter-spacing: 0.2px; margin-top: 1px; }
    .topbar-pill  {
        font-size: 0.68rem; font-weight: 600;
        background: rgba(212,175,55,0.12); color: #d4af37;
        border: 1px solid rgba(212,175,55,0.35);
        padding: 4px 13px; border-radius: 20px; letter-spacing: 0.8px; text-transform: uppercase;
    }

    .stat-card {
        background: #0e1525;
        border: 1px solid rgba(212,175,55,0.16);
        border-radius: 14px; padding: 1rem 1.1rem;
    }
    .stat-label {
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1.2px; color: #8a9bb5; margin-bottom: 5px;
    }
    .stat-value { font-size: 1.55rem; font-weight: 500; color: #eed98a; }
    .stat-desc { font-size: 0.73rem; color: #7a8ba0; margin-top: 3px; }

    /* section header pengganti panel-card */
    .section-header {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.8px; color: #d4af37; margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(212,175,55,0.14);
    }

    /* wrapper untuk input & hasil — pakai container streamlit */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background: #0e1525;
        border: 1px solid rgba(212,175,55,0.16);
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #080c16 !important;
        border: 1px solid #253348 !important;
        border-radius: 10px !important;
        color: #c8d8ea !important;
        font-size: 0.92rem !important;
        padding: 0.75rem 1rem !important;
        caret-color: #d4af37 !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(212,175,55,0.55) !important;
        box-shadow: 0 0 0 3px rgba(212,175,55,0.1) !important;
    }
    .stTextArea > div > div > textarea { min-height: 130px !important; line-height: 1.65 !important; }
    input::placeholder, textarea::placeholder { color: #4a5e78 !important; }
    label, .stTextInput label, .stTextArea label {
        color: #9aaec5 !important;
        font-size: 0.73rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #c9a227, #e8c85a) !important;
        color: #07090f !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 1rem !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.85 !important; }
    .stButton > button[kind="secondary"] {
        background: rgba(212,175,55,0.07) !important;
        color: #b09a5a !important;
        border: 1px solid rgba(212,175,55,0.25) !important;
        border-radius: 8px !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0.5rem !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(212,175,55,0.14) !important;
        color: #d4af37 !important;
        border-color: rgba(212,175,55,0.45) !important;
    }

    .sentiment-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 12px 48px; border-radius: 50px;
        font-size: 1.2rem; font-weight: 700; letter-spacing: 1.5px;
    }
    .badge-positive { background: rgba(76,175,125,0.12); border: 2px solid rgba(76,175,125,0.6); color: #52c48a; }
    .badge-neutral  { background: rgba(212,175,55,0.12);  border: 2px solid rgba(212,175,55,0.6);  color: #d4af37; }
    .badge-negative { background: rgba(224,82,82,0.12);   border: 2px solid rgba(224,82,82,0.6);   color: #e85c5c; }

    .hint-text { font-size: 0.73rem; color: #8a9bb5; }
    .stAlert {
        background: rgba(224,82,82,0.09) !important;
        border-left: 3px solid #e85c5c !important;
        border-radius: 10px !important;
        color: #d0daea !important;
    }
    .stSpinner > div { border-top-color: #d4af37 !important; }

    .page-footer {
        text-align: center; margin-top: 2rem; padding-top: 1rem;
        border-top: 1px solid rgba(212,175,55,0.08);
        font-size: 0.73rem; color: #6a7e98;
    }
    .page-footer strong { color: #a08840; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# SAMPLE DATA
# ============================================================
EXAMPLE_TITLES = {
    "📈 Bullish": (
        "Gold hits new all-time high",
        "Gold price rallies to a new record high as investors seek safe haven amid rising geopolitical tensions and uncertainty in global markets",
    ),
    "📉 Bearish": (
        "Gold futures tumble sharply",
        "Gold futures drop sharply today as stronger dollar and rising bond yields continue to put pressure on precious metal prices",
    ),
    "➡️ Neutral": (
        "Gold holds steady with no clear direction",
        "Gold holds steady with no clear direction as market waits for further data on inflation and economic outlook",
    ),
    "💥 Crash": (
        "Gold crash deepens on panic",
        "Gold crash intensifies as panic selling grips the market with prices plunging to multi-month lows amid margin calls",
    ),
    "🚀 Rally": (
        "Gold rally shows strong momentum",
        "Gold rally continues with strong bullish momentum as inflation concerns and weak dollar drive safe-haven buying across the board",
    ),
}

EXAMPLE_LIST = list(EXAMPLE_TITLES.items())

def set_example(idx):
    t, d = EXAMPLE_LIST[idx][1]
    st.session_state["title_input"] = t
    st.session_state["desc_input"] = d

# ============================================================
# TOPBAR
# ============================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-icon">⚜</div>
        <div>
            <div class="topbar-title">Gold Sentiment Engine</div>
            <div class="topbar-sub">Two-Stage SVM &nbsp;·&nbsp; TF-IDF ngram 1–2 &nbsp;·&nbsp; 628 sampel berita emas</div>
        </div>
    </div>
    <div class="topbar-pill">Live Model</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STAT ROW
# ============================================================
s1, s2, s3 = st.columns(3, gap="small")
with s1:
    st.markdown("""<div class="stat-card">
        <div class="stat-label">Total Sampel</div>
        <div class="stat-value">628</div>
        <div class="stat-desc">Berita emas 2021–2026</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.markdown("""<div class="stat-card">
        <div class="stat-label">Kelas Sentimen</div>
        <div class="stat-value">3</div>
        <div class="stat-desc">Positive · Neutral · Negative</div>
    </div>""", unsafe_allow_html=True)
with s3:
    st.markdown("""<div class="stat-card">
        <div class="stat-label">Max Features</div>
        <div class="stat-value">3,000</div>
        <div class="stat-desc">TF-IDF vocab size</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ============================================================
# INPUT CARD — pakai st.container dengan border
# ============================================================
with st.container(border=True):
    st.markdown('<div class="section-header">📰 &nbsp;Input Berita</div>', unsafe_allow_html=True)

    title = st.text_input(
        "Judul Berita",
        placeholder="Contoh: Gold price surges to record high...",
        key="title_input",
    )
    description = st.text_area(
        "Isi Berita / Snippet",
        placeholder="Tempel isi berita atau deskripsi singkat di sini...",
        height=130,
        key="desc_input",
    )
    predict_btn = st.button(
        "🔍  Predict Sentiment",
        type="primary",
        use_container_width=True,
    )

    st.markdown('<div class="hint-text" style="margin-top:0.5rem;">Coba contoh cepat:</div>', unsafe_allow_html=True)
    ex_cols = st.columns(5, gap="small")
    for i, (lbl, _) in enumerate(EXAMPLE_LIST):
        with ex_cols[i]:
            st.button(lbl, on_click=set_example, args=(i,), key=f"ex_{i}", type="secondary")

# ============================================================
# HASIL ANALISIS CARD
# ============================================================
if predict_btn:
    with st.container(border=True):
        st.markdown('<div class="section-header">📊 &nbsp;Hasil Analisis</div>', unsafe_allow_html=True)

        if not title and not description:
            st.error("⚠️ Masukkan minimal judul atau isi berita.")
        else:
            with st.spinner("Menganalisis sentimen..."):
                try:
                    label, confidence, cleaned, distribution = predict_title_desc(
                        title, description
                    )
                    label_display = {
                        "positive": "✅  POSITIVE",
                        "neutral":  "⚠️  NEUTRAL",
                        "negative": "❌  NEGATIVE",
                    }[label]
                    badge_class = f"badge-{label}"

                    st.markdown(f"""
                    <div style="text-align:center; padding: 1.5rem 0 1rem;">
                        <div class="sentiment-badge {badge_class}">{label_display}</div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi error: {e}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="page-footer">
    <strong>Two-Stage SVM + Balanced</strong>
    &nbsp;·&nbsp; TF-IDF (ngram 1–2, max 3000)
    &nbsp;·&nbsp; Dataset: 628 sampel berita emas 2021–2026
</div>
""", unsafe_allow_html=True)