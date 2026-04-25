import streamlit as st
import numpy as np
import pickle
import time
import random
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle

def pad_sequences(sequences, maxlen, padding="pre"):
    padded = []
    for seq in sequences:
        if len(seq) >= maxlen:
            padded.append(seq[-maxlen:])
        else:
            pad = [0] * (maxlen - len(seq))
            padded.append((pad + seq) if padding == "pre" else (seq + pad))
    return np.array(padded)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quotes Next Word Predictor",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Inspirational Quote Seeds ─────────────────────────────────────────────────
QUOTE_SEEDS = [
    "The only way to do great work is to",
    "In the middle of every difficulty lies",
    "Life is what happens when you are busy",
    "The future belongs to those who",
    "It always seems impossible until",
    "You miss 100% of the shots you",
    "The greatest glory in living lies not in",
    "In the end it is not the years in your life",
    "Darkness cannot drive out darkness only",
    "The best time to plant a tree was",
]

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@300;400;500;600&family=Space+Mono&display=swap');

:root {
    --bg:        #080b14;
    --surface:   #0e1220;
    --surface2:  #141826;
    --border:    #1f2640;
    --accent:    #f59e0b;
    --accent2:   #ec4899;
    --accent3:   #10b981;
    --blue:      #3b82f6;
    --purple:    #8b5cf6;
    --text:      #f1f5f9;
    --muted:     #64748b;
    --muted2:    #94a3b8;
    --glow-amber: rgba(245,158,11,0.25);
    --glow-pink:  rgba(236,72,153,0.2);
    --glow-blue:  rgba(59,130,246,0.2);
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    max-width: 820px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* ═══════════════ HERO BANNER ═══════════════ */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 3rem 1rem 2rem;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 20% 30%, rgba(245,158,11,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 80% 60%, rgba(236,72,153,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 50% 10%, rgba(139,92,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(236,72,153,0.15));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 99px;
    padding: 0.3rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: clamp(2.2rem, 6vw, 3.6rem);
    line-height: 1.1;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #f59e0b 0%, #f97316 35%, #ec4899 65%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.9rem;
    color: var(--muted2);
    letter-spacing: 0.03em;
    margin-top: 0.3rem;
}
.hero-divider {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    margin: 1.2rem auto 0;
    border-radius: 2px;
}

/* ═══════════════ STATUS CHIPS ═══════════════ */
.chip-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.8rem;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: 0.3rem 0.85rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted2);
}
.chip.green { border-color: rgba(16,185,129,0.4); color: #34d399; background: rgba(16,185,129,0.07); }
.chip.amber { border-color: rgba(245,158,11,0.4); color: #fbbf24; background: rgba(245,158,11,0.07); }
.chip.blue  { border-color: rgba(59,130,246,0.4);  color: #93c5fd; background: rgba(59,130,246,0.07); }

/* ═══════════════ SECTION LABELS ═══════════════ */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

# /* ═══════════════ INPUT AREA ═══════════════ */
# .stTextArea > div > div > textarea {
#     # background: var(--surface) !important;
#     border: 1px solid rgba(245,158,11,0.2);
#     border-radius: 18px;
#     background: linear-gradient(145deg, rgba(245,158,11,0.05), rgba(236,72,153,0.04));
#     # border: 1.5px solid var(--border) !important;
#     # border-radius: 14px !important;
#     # color: var(--text) !important;
#     font-family: 'Playfair Display', serif !important;
#     font-size: 1.05rem !important;
#     font-style: italic !important;
#     line-height: 1.8 !important;
#     padding: 1rem 1.2rem !important;
#     resize: vertical !important;
#     # box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(245,158,11,0.1);
#     transition: all 0.3s ease !important;
# }
# .stTextArea > div > div > textarea:focus {
#     border-color: var(--accent) !important;
#     box-shadow: 0 0 0 3px var(--glow-amber), 0 4px 24px rgba(0,0,0,0.4) !important;
# }
# .stTextArea > div > div > textarea::placeholder {
#     color: #334155 !important;
#     font-style: italic !important;
# }


/* ═══════════════ INPUT AREA ═══════════════ */
.stTextArea > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.stTextArea > div > div > textarea {
    background: rgba(245,158,11,0.2) !important;
    border: 1.5px solid rgba(245,158,11,0.55) !important;
    border-radius: 18px !important;
    color: #000000 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1.05rem !important;
    font-style: italic !important;
    line-height: 1.8 !important;
    padding: 1rem 1.2rem !important;
    resize: vertical !important;
    box-shadow: none !important;
    outline: none !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.3) !important;
    outline: none !important;
}
.stTextArea > div > div > textarea::placeholder {
    color: rgba(0,0,0,0.4) !important;
    font-style: italic !important;
}

/* ═══════════════ SLIDERS ═══════════════ */
.stSlider > label {
    color: var(--muted2) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ═══════════════ BUTTONS ═══════════════ */
.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #f97316, #ec4899) !important;
    color: #0a0b0e !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 16px rgba(245,158,11,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(245,158,11,0.45), 0 2px 8px rgba(236,72,153,0.3) !important;
    filter: brightness(1.08) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─ secondary random button ─ */
div[data-testid="column"]:last-child .stButton > button {
    background: var(--surface2) !important;
    color: var(--muted2) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-size: 0.88rem !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 12px var(--glow-amber) !important;
    filter: none !important;
}

/* ═══════════════ RESULTS ═══════════════ */
.result-header {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-style: italic;
    color: var(--muted2);
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}

/* ── Quote Output Box ── */
.quote-out-wrap {
    background: linear-gradient(145deg, rgba(245,158,11,0.05), rgba(236,72,153,0.04));
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    position: relative;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(245,158,11,0.1);
}
.quote-mark {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    line-height: 0.6;
    display: block;
    margin-bottom: 0.5rem;
}
.quote-seed-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-style: italic;
    # color: rgba(241,245,249,0.55);
     background: linear-gradient(90deg, #f59e0b, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.75;
}
.quote-gen-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-style: italic;
    font-weight: 700;
    background: linear-gradient(90deg, #f59e0b, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.75;
}
.quote-close {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    color: rgba(236,72,153,0.2);
    line-height: 0;
    display: block;
    text-align: right;
    margin-top: 0.3rem;
}
.copy-hint {
    font-size: 0.72rem;
    color: var(--muted);
    text-align: right;
    margin-top: 0.6rem;
    font-family: 'Space Mono', monospace;
}

/* ── Top-K Probability Card ── */
.topk-card {
    background: linear-gradient(145deg, var(--surface), var(--surface2));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.topk-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--purple));
}
.topk-card::after {
    content: '"';
    position: absolute;
    right: 1.2rem;
    top: 0.5rem;
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    color: rgba(245,158,11,0.06);
    line-height: 1;
    pointer-events: none;
}
.topk-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
    animation: slideIn 0.4s ease both;
}
.prob-row:nth-child(2) { animation-delay: 0.05s; }
.prob-row:nth-child(3) { animation-delay: 0.10s; }
.prob-row:nth-child(4) { animation-delay: 0.15s; }
.prob-row:nth-child(5) { animation-delay: 0.20s; }
.prob-row:nth-child(6) { animation-delay: 0.25s; }
.prob-row:nth-child(7) { animation-delay: 0.30s; }
.prob-row:nth-child(8) { animation-delay: 0.35s; }
.prob-row:nth-child(9) { animation-delay: 0.40s; }
.prob-row:nth-child(10){ animation-delay: 0.45s; }

@keyframes slideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}

.prob-word {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--text);
    min-width: 130px;
}
.prob-bar-wrap {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 99px;
}
.prob-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted2);
    min-width: 46px;
    text-align: right;
}

.bar-0 { background: linear-gradient(90deg, #f59e0b, #f97316); }
.bar-1 { background: linear-gradient(90deg, #ec4899, #f43f5e); }
.bar-2 { background: linear-gradient(90deg, #8b5cf6, #3b82f6); }
.bar-3 { background: linear-gradient(90deg, #10b981, #06b6d4); }
.bar-4 { background: linear-gradient(90deg, #f97316, #ec4899); }
.bar-5 { background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
.bar-6 { background: linear-gradient(90deg, #06b6d4, #10b981); }
.bar-7 { background: linear-gradient(90deg, #f43f5e, #f59e0b); }
.bar-8 { background: linear-gradient(90deg, #8b5cf6, #ec4899); }
.bar-9 { background: linear-gradient(90deg, #10b981, #3b82f6); }

/* ═══════════════ EXPANDER ═══════════════ */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted2) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.streamlit-expanderContent {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ═══════════════ ERROR ═══════════════ */
.err-box {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.25);
    color: #fca5a5;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
}

/* ═══════════════ MISC ═══════════════ */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
.spacer { margin-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Artifacts ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len


# ─── Prediction Logic ─────────────────────────────────────────────────────────
def predict_next_words(model, tokenizer, max_len, seed_text, num_words=5, top_k=5):
    index_word = {v: k for k, v in tokenizer.word_index.items()}

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
    predictions = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_words = [(index_word.get(i, "?"), float(predictions[i])) for i in top_indices]

    current_text = seed_text
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding="pre")
        pred = model.predict(token_list, verbose=0)[0]
        next_index = np.argmax(pred)
        next_word = index_word.get(next_index, "")
        if not next_word:
            break
        current_text += " " + next_word

    generated_words = current_text[len(seed_text):].strip()
    return top_words, current_text, generated_words


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✦ LSTM · Deep Learning · NLP ✦</div>
    <h1 class="hero-title">Quotes Next Word<br>Predictor</h1>
    <p class="hero-sub">Let the model complete your thoughts — one word at a time</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Waking up the model…"):
    try:
        model, tokenizer, max_len = load_artifacts()
        vocab_size = len(tokenizer.word_index) + 1
        st.markdown(
            f'<div class="chip-row">'
            f'<span class="chip green">● Model Ready</span>'
            f'<span class="chip amber">⬡ Vocab · {vocab_size:,} words</span>'
            f'<span class="chip blue">⧖ Max Len · {max_len}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        model_loaded = True
    except Exception as e:
        st.markdown(
            f'<div class="err-box">⚠ Could not load model files.<br><br>'
            f'<code>{e}</code><br><br>'
            f'Make sure <b>lstm_model.h5</b>, <b>tokenizer.pkl</b>, and <b>max_len.pkl</b> '
            f'are in the same folder as this script.</div>',
            unsafe_allow_html=True,
        )
        model_loaded = False

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════
if model_loaded:

    # ── Seed Input ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">✦ Your Quote Seed</div>', unsafe_allow_html=True)

    if "seed_val" not in st.session_state:
        st.session_state.seed_val = ""

    col_in, col_btn = st.columns([3, 1])
    with col_btn:
        st.markdown("<div style='padding-top:1.85rem;'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random Seed", key="rand"):
            st.session_state.seed_val = random.choice(QUOTE_SEEDS)
            st.rerun()

    with col_in:
        seed_text = st.text_area(
            label="seed",
            label_visibility="collapsed",
            value=st.session_state.seed_val,
            placeholder='✦ Begin a quote…  e.g. "The only way to do great work is to"',
            height=120,
            key="seed_input",
        )

    # ── Controls ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1rem;">⚙ Generation Controls</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        num_words = st.slider("📝 Words to generate", min_value=1, max_value=25, value=7,
                              help="How many words the model appends after your seed")
    with col2:
        top_k = st.slider("🎯 Top-K candidates", min_value=1, max_value=10, value=5,
                          help="Number of next-word alternatives to display")

    st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
    predict_btn = st.button("✨  Generate Prediction", key="predict")

    # ═══════════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    if predict_btn:
        clean = seed_text.strip()
        if not clean:
            st.markdown(
                '<div class="err-box">Please enter or pick a seed quote before predicting.</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("✦ Completing your quote…"):
                time.sleep(0.25)
                try:
                    top_words, full_sentence, new_words = predict_next_words(
                        model, tokenizer, max_len, clean, num_words, top_k
                    )

                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown('<div class="result-header">✦ Prediction Results</div>', unsafe_allow_html=True)

                    # ── Generated Quote ──────────────────────────────────────
                    st.markdown('<div class="section-label">Generated Quote</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="quote-out-wrap">'
                        f'  <span class="quote-mark">&ldquo;</span>'
                        f'  <span class="quote-seed-text">{clean} </span>'
                        f'  <span class="quote-gen-text">{new_words}</span>'
                        f'  <span class="quote-close">&rdquo;</span>'
                        f'  <div class="copy-hint">LSTM · {num_words} words generated</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Top-K Probability Bars ───────────────────────────────
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Next-Word Candidates</div>', unsafe_allow_html=True)
                    max_prob = top_words[0][1] if top_words else 1.0

                    rows_html = (
                        '<div class="topk-card">'
                        '<div class="topk-label">Top candidates · probability distribution</div>'
                    )
                    for i, (word, prob) in enumerate(top_words):
                        bar_pct = (prob / max_prob) * 100 if max_prob > 0 else 0
                        rows_html += (
                            f'<div class="prob-row">'
                            f'  <span class="prob-word">{word}</span>'
                            f'  <div class="prob-bar-wrap">'
                            f'    <div class="prob-bar bar-{i}" style="width:{bar_pct:.1f}%"></div>'
                            f'  </div>'
                            f'  <span class="prob-pct">{prob:.1%}</span>'
                            f'</div>'
                        )
                    rows_html += '</div>'
                    st.markdown(rows_html, unsafe_allow_html=True)

                    # ── Full Sentence Copyable ────────────────────────────────
                    with st.expander("📋 Copy full generated sentence"):
                        st.code(full_sentence, language=None)

                except Exception as e:
                    st.markdown(
                        f'<div class="err-box">⚠ Prediction error: {e}</div>',
                        unsafe_allow_html=True,
                    )

    # ═══════════════════════════════════════════════════════════════════════════
    #  HOW IT WORKS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️  How it works"):
        st.markdown("""
**Quotes Next Word Predictor** uses an LSTM (Long Short-Term Memory) neural network trained on quote data.

| Feature | Detail |
|---|---|
| **Top-K candidates** | Softmax probabilities for the most likely next tokens |
| **Sentence generation** | Greedy decoding — best token appended repeatedly |
| **🎲 Random seed** | Picks an inspirational quote fragment to get you started |

**Required files (same folder as `app.py`):**
```
lstm_model.h5    ← trained LSTM model
tokenizer.pkl    ← fitted Keras tokenizer
max_len.pkl      ← sequence max length integer
```
        """)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center; margin-top:2.5rem; padding-top:1rem; '
        'border-top:1px solid #1f2640; font-family:Space Mono,monospace; '
        'font-size:0.68rem; color:#334155; letter-spacing:0.1em;">'
        'QUOTES NEXT WORD PREDICTOR &nbsp;·&nbsp; LSTM &nbsp;·&nbsp; BUILT WITH STREAMLIT'
        '</div>',
        unsafe_allow_html=True,
    )
