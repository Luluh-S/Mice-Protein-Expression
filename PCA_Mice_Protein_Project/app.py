import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Anyone Can Cluster", page_icon="🐀", layout="wide",
                   initial_sidebar_state="collapsed")

HERE = Path(__file__).parent
BASE_CSS = (HERE / "components" / "base.css").read_text(encoding="utf-8")

# the vegetables of a ratatouille, tuned to read on the dark kitchen ground (data only)
TOMATO, BASIL, SAFFRON, BLUE = "#e8573f", "#7fb05a", "#f2bd4b", "#6a9fd0"
VEG = [TOMATO, BASIL, SAFFRON, BLUE, "#f39255", "#d9667f", "#4fae95", "#d08a5a", "#b9b55a", "#9aa6b0"]
NOISE = "#4a4540"

COURSES = ["Welcome", "Prep work", "Before PCA", "Remy's nose", "After PCA", "The verdict", "Test kitchen"]
NEXT_LABELS = {"Welcome": "the welcome page", "Prep work": "the prep work", "Before PCA": "clustering before PCA",
               "Remy's nose": "PCA at work", "After PCA": "clustering after PCA", "The verdict": "the verdict",
               "Test kitchen": "the bonus experiments"}


def show(name, data, height):
    """Render one of our animated components with its data."""
    page = (HERE / "components" / f"{name}.html").read_text(encoding="utf-8")
    page = page.replace("/*BASE*/", BASE_CSS).replace("/*DATA*/null", json.dumps(data))
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        components.html(page, height=height)


def script(code, key):
    """A tiny hidden frame that runs a script on the page."""
    with st.container(key=key):
        page = f"<script>{code}</script>"
        if hasattr(st, "iframe"):
            st.iframe(page, height=1)
        else:
            components.html(page, height=1)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


# ---------------- Page style (taste-skill deal: warm-black-gold, fraunces + figtree, soft radius) ----------------

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT,WONK@0,9..144,300..700,0..100,0..1;1,9..144,300..700,0..100,0..1&family=Figtree:wght@400..600&display=swap');
:root {
  --bg: #0c0a08; --alt: #12100c; --surface: #17140f; --surface-2: #211c16; --inset: #0f0d0a;
  --ink: #ffffff; --ink-2: #e6dfd4; --muted: #948d83; --faint: #5f5951;
  --line: rgba(255,255,255,.08); --line-2: rgba(255,255,255,.16);
  --accent: #c9a45a; --accent-hover: #d8b877; --accent-ink: #0c0a08; --accent-soft: rgba(201,164,90,.14);
  --good: #8cc063;
  --font-display: "Fraunces", Georgia, serif;
  --font-text: "Figtree", system-ui, -apple-system, "Segoe UI", sans-serif;
  --display-axes: 'SOFT' 30, 'WONK' 0;
  --r-lg: 20px; --r-md: 12px; --r-sm: 8px;
  --ease: cubic-bezier(.2,.8,.2,1);
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, label, p, li, textarea { font-family: var(--font-text); }
.stApp { background: var(--bg); color: var(--ink); }
.stApp::before { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background: radial-gradient(1200px 600px at 75% -10%, rgba(201,164,90,.07), transparent 60%); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], #MainMenu, footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1280px; padding: 0 48px 0; }
iframe { border: 0 !important; display: block; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stMarkdownContainer"] p { font-size: 16px; line-height: 1.65; }
.st-key-scrolltop, .st-key-localefix { height: 0; overflow: hidden; margin: 0 !important; position: absolute; }

/* ---- header: one opaque bar, 1px hairline, the same at every scroll depth ---- */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar), .st-key-topbar { position: sticky; top: 0; z-index: 990; }
.st-key-topbar { background: rgba(12,10,8,1); border-bottom: 1px solid var(--line); padding: 16px 0 10px; gap: 10px !important; }
.pc-brand { display: flex; align-items: center; gap: 12px; font: 400 22px var(--font-display);
  font-variation-settings: var(--display-axes); letter-spacing: -.015em; color: var(--ink); white-space: nowrap; }
.pc-brand small { font: 500 13px var(--font-text); color: var(--muted); letter-spacing: 0; padding-left: 12px; border-left: 1px solid var(--line-2); }
.pc-live { display: flex; gap: 22px; justify-content: flex-end; align-items: baseline; flex-wrap: wrap; }
.pc-live span { font: 500 13px var(--font-text); color: var(--muted); white-space: nowrap; }
.pc-live b { font: 400 19px var(--font-display); font-variation-settings: var(--display-axes); color: var(--ink); margin-left: 6px; font-variant-numeric: tabular-nums; }
.pc-live b.up { color: var(--accent); }
.st-key-topbar [data-testid="stWidgetLabel"] p { font: 500 12px var(--font-text) !important; color: var(--muted); }

/* the course menu: plain words, the current one carries a 2px gold bar */
.st-key-nav [data-testid="stButtonGroup"] > div { background: transparent !important; border: 0 !important; padding: 0 !important; gap: 4px !important; }
.st-key-nav [data-testid="stButtonGroup"] button { background: transparent !important; box-shadow: none !important; border-radius: 0 !important;
  padding: 10px 10px 12px !important; min-height: 40px; color: var(--muted) !important; }
.st-key-nav [data-testid="stButtonGroup"] button p { font: 500 15px var(--font-text) !important; }
.st-key-nav [data-testid="stButtonGroup"] button:hover { color: var(--ink) !important; background: transparent !important; box-shadow: inset 0 -1px 0 var(--line-2) !important; }
.st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"], .st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"]:hover {
  color: var(--ink) !important; background: transparent !important; box-shadow: inset 0 -2px 0 var(--accent) !important; }

/* other segmented controls: one inset track, the picked option raised */
[data-testid="stButtonGroup"] > div { background: var(--inset); border: 1px solid var(--line); border-radius: 12px; padding: 4px; gap: 2px !important; flex-wrap: wrap; }
[data-testid="stButtonGroup"] button { border: 0 !important; background: transparent !important; color: var(--muted) !important;
  border-radius: 9px !important; min-height: 34px; padding: 5px 14px !important; box-shadow: none !important;
  transition: color 140ms var(--ease), background-color 200ms var(--ease), box-shadow 200ms var(--ease) !important; }
[data-testid="stButtonGroup"] button p { font: 500 14px var(--font-text) !important; }
[data-testid="stButtonGroup"] button:hover { color: var(--ink) !important; background: rgba(255,255,255,.05) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--surface-2) !important; color: var(--ink) !important; box-shadow: 0 0 0 1px var(--line-2) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"]:hover { background: #2a241d !important; box-shadow: 0 0 0 1px rgba(255,255,255,.32) !important; }
[data-testid="stSliderTickBar"] { opacity: 1 !important; }
[data-testid="stSliderTickBar"] p { color: var(--faint) !important; font-size: 12px !important; }

/* ---- course opening: a big outlined course number beside the title ---- */
.pc-sec { display: grid; grid-template-columns: 170px 1fr; gap: 32px; align-items: start; margin: 72px 0 40px; animation: pcIn 480ms var(--ease) both; }
.pc-sec .num { font: 300 150px/0.8 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.04em;
  color: transparent; -webkit-text-stroke: 1px rgba(201,164,90,.55); }
.pc-sec .eyebrow { font: 500 13px var(--font-text); color: var(--accent); }
.pc-sec h2 { font: 400 64px/1 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.03em; margin: 10px 0 16px; padding: 0; color: var(--ink); }
.pc-sec h2 em { font-style: italic; font-weight: 300; color: var(--muted); }
.pc-sec p { max-width: 60ch; color: var(--muted) !important; font-size: 18px !important; line-height: 1.6 !important; margin: 0; }
@keyframes pcIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }

.pc-h3 { display: flex; align-items: baseline; gap: 14px; font: 400 34px var(--font-display); font-variation-settings: var(--display-axes);
  letter-spacing: -.02em; margin: 8px 0 8px; color: var(--ink); }
.pc-h3 small { font: 500 13px var(--font-text); color: var(--accent); letter-spacing: 0; }
[data-testid="stMarkdownContainer"] p.pc-lead, .pc-lead { color: var(--muted); margin: 0 0 24px; font-size: 17px !important; line-height: 1.6 !important; max-width: 62ch; }

/* full-bleed bands on a second ground break the long page into rooms */
[class*="st-key-band"] { background: var(--alt); box-shadow: 0 0 0 100vmax var(--alt); clip-path: inset(0 -100vmax);
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 64px 0 64px; margin: 64px 0 0; }

/* ---- open figures: big numbers on hairlines, no boxes ---- */
.pc-figs { display: grid; border-top: 1px solid var(--line-2); margin: 8px 0 32px; }
.pc-figs > div { padding: 18px 24px 4px 0; }
.pc-figs > div + div { padding-left: 24px; border-left: 1px solid var(--line); }
.pc-figs .lab { font: 500 13px var(--font-text); color: var(--muted); }
.pc-figs .val { font: 300 60px/1.05 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.035em; color: var(--ink);
  margin-top: 10px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pc-figs .val small { font: 600 14px var(--font-text); color: var(--good); letter-spacing: 0; margin-left: 10px; vertical-align: 18px; }

/* ---- chef's note: a serif aside with a gold label, no box ---- */
.pc-note { border-left: 1px solid var(--accent); padding: 4px 0 4px 24px; margin: 8px 0; }
.pc-note .lab { font: 500 13px var(--font-text); color: var(--accent); margin-bottom: 8px; }
.pc-note .txt { font: 400 21px/1.5 var(--font-display); font-variation-settings: var(--display-axes); color: var(--ink-2); letter-spacing: -.005em; }
.pc-note .txt b { color: var(--ink); font-weight: 500; }

/* ---- cards: used only for the four questions ---- */
.pc-grid { display: grid; gap: 16px; }
.pc-g2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.pc-qa { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 28px 28px 30px; }
.pc-qa .n { font: 300 44px/1 var(--font-display); color: var(--accent); }
.pc-qa h4 { font: 400 24px/1.25 var(--font-display); font-variation-settings: var(--display-axes); margin: 18px 0 10px; padding: 0; color: var(--ink); letter-spacing: -.01em; }
.pc-qa p { margin: 0; color: var(--ink-2); font-size: 16px; line-height: 1.65; }
.pc-qa b { color: var(--ink); font-weight: 600; }

/* ---- the tasting menu card ---- */
.pc-menu { max-width: 940px; margin: 0 auto; padding: 56px 64px 48px; border: 1px solid var(--line-2); border-radius: 4px;
  outline: 1px solid var(--line); outline-offset: 8px; background: linear-gradient(180deg, rgba(201,164,90,.04), transparent 40%); }
.pc-menu .top { text-align: center; margin-bottom: 36px; }
.pc-menu .top .lab { font: 500 13px var(--font-text); color: var(--accent); }
.pc-menu .top h3 { font: 400 52px/1.05 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.025em; margin: 10px 0 8px; padding: 0; color: var(--ink); }
.pc-menu .top h3 em { font-style: italic; font-weight: 300; }
.pc-menu .top p { color: var(--muted); margin: 0; }
.pc-menu .row { display: grid; grid-template-columns: 56px auto 1fr auto; align-items: baseline; gap: 14px; padding: 16px 0; }
.pc-menu .row + .row { border-top: 1px solid var(--line); }
.pc-menu .no { font: 400 15px var(--font-display); color: var(--accent); font-style: italic; }
.pc-menu .name { font: 400 30px var(--font-display); font-variation-settings: var(--display-axes); font-style: italic; color: var(--ink); letter-spacing: -.01em; white-space: nowrap; }
.pc-menu .dots { border-bottom: 1px dotted var(--faint); transform: translateY(-6px); min-width: 24px; }
.pc-menu .desc { font: 400 15px var(--font-text); color: var(--muted); text-align: right; }
.pc-menu .row.bonus .name { color: var(--accent); }

/* ---- editorial two-column spread ---- */
.pc-spread { display: grid; grid-template-columns: 1.05fr 1fr; gap: 64px; align-items: start; }
[data-testid="stMarkdownContainer"] p.pc-big, .pc-big { font: 300 46px/1.12 var(--font-display) !important; font-variation-settings: var(--display-axes); letter-spacing: -.025em; color: var(--ink); margin: 0; }
.pc-big em { font-style: italic; color: var(--accent); font-weight: 300; }
.pc-rows { margin: 0; padding: 0 !important; list-style: none; }
.pc-rows li { display: flex; justify-content: space-between; gap: 24px; padding: 15px 0; border-bottom: 1px solid var(--line); margin: 0 !important; font-size: 16px; }
.pc-rows li:first-child { border-top: 1px solid var(--line-2); }
.pc-rows li span { color: var(--muted); }
.pc-rows li b { font-weight: 500; text-align: right; color: var(--ink); }
.pc-rows-title { font: 500 13px var(--font-text); color: var(--accent); margin: 0 0 14px; }

/* ---- final summary ---- */
.pc-list { margin: 0; padding: 0 !important; list-style: none; counter-reset: k; columns: 2; column-gap: 56px; }
.pc-list li { counter-increment: k; break-inside: avoid; display: grid; grid-template-columns: 44px 1fr; padding: 18px 0; border-top: 1px solid var(--line); margin: 0 !important;
  font-size: 17px; line-height: 1.6; color: var(--ink-2); }
.pc-list li::before { content: counter(k, decimal-leading-zero); font: 400 15px var(--font-display); color: var(--accent); padding-top: 2px; }

[data-testid="stMarkdownContainer"] p.pc-small, .pc-small { color: var(--faint); font-size: 13px !important; margin: -20px 0 24px; }

/* ---- closing ---- */
.pc-close { text-align: center; padding: 120px 0 40px; }
.pc-close .big { font: 300 clamp(48px, 7vw, 96px)/1 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.04em; color: var(--ink); }
.pc-close .big em { font-style: italic; color: var(--accent); }
.pc-close p { color: var(--muted); margin-top: 18px; }
.pc-foot { padding: 22px 0 28px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px;
  font: 500 13px var(--font-text); color: var(--faint); flex-wrap: wrap; }

/* ---- buttons: one filled gold action, one outline; hover steps 1px, press sinks ---- */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  min-height: 52px; border-radius: 999px !important; padding: 0 28px !important;
  transition: transform 140ms var(--ease), background-color 140ms var(--ease), border-color 140ms var(--ease) !important; }
[data-testid="stBaseButton-primary"] { background: var(--accent) !important; border: 1px solid var(--accent) !important; color: var(--accent-ink) !important; }
[data-testid="stBaseButton-primary"]:hover { background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; transform: translateY(-1px); }
[data-testid="stBaseButton-secondary"] { background: transparent !important; border: 1px solid var(--line-2) !important; color: var(--ink) !important; }
[data-testid="stBaseButton-secondary"]:hover { background: rgba(255,255,255,.04) !important; border-color: rgba(255,255,255,.42) !important; transform: translateY(-1px); }
[data-testid="stBaseButton-primary"]:active, [data-testid="stBaseButton-secondary"]:active { transform: translateY(1px) scale(.985); transition-duration: 90ms !important; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p { font: 500 16px var(--font-text) !important; }
button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
.st-key-pager { margin-top: 80px; }

[data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: var(--r-md) !important; background: var(--surface); }
[data-testid="stExpander"] summary p { font: 500 15px var(--font-text) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: var(--surface) !important; border-radius: 10px; border: 1px solid var(--line-2) !important; }
[data-testid="stWidgetLabel"] p { font: 500 13px var(--font-text) !important; color: var(--muted); }

@media (max-width: 1100px) {
  .pc-sec { grid-template-columns: 1fr; gap: 0; }
  .pc-sec .num { font-size: 96px; margin-bottom: 8px; }
}
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 0 18px; }
  [data-testid="stLayoutWrapper"]:has(> .st-key-topbar), .st-key-topbar { position: static; }
  .pc-live { justify-content: flex-start; gap: 14px; }
  .pc-sec h2 { font-size: 42px; }
  .pc-figs { grid-template-columns: 1fr 1fr !important; }
  .pc-figs > div:nth-child(3) { border-left: 0; padding-left: 0; }
  .pc-figs .val { font-size: 42px; }
  [data-testid="stMarkdownContainer"] p.pc-big, .pc-big { font-size: 32px !important; }
  .pc-g2, .pc-spread { grid-template-columns: 1fr; gap: 32px; }
  .pc-menu { padding: 36px 22px 28px; }
  .pc-menu .row { grid-template-columns: 36px 1fr; }
  .pc-menu .dots { display: none; }
  .pc-menu .desc { grid-column: 2; text-align: left; }
  .pc-menu .name { font-size: 24px; }
  .pc-list { columns: 1; }
  .pc-brand small { display: none; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""")


def section(num, eyebrow, title, subtitle, text):
    html(f'<div class="pc-sec"><div class="num">{num}</div><div><div class="eyebrow">{eyebrow}</div>'
         f'<h2>{title} <em>{subtitle}</em></h2><p>{text}</p></div></div>')


def figures(items):
    cells = ""
    for label, value, delta in items:
        d = f"<small>{delta}</small>" if delta else ""
        cells += f'<div><div class="lab">{label}</div><div class="val">{value}{d}</div></div>'
    html(f'<div class="pc-figs" style="grid-template-columns: repeat({len(items)}, minmax(0, 1fr))">{cells}</div>')


def note(label, text):
    html(f'<div class="pc-note"><div class="lab">{label}</div><div class="txt">{text}</div></div>')


def subhead(text, label=""):
    small = f"<small>{label}</small>" if label else ""
    html(f'<div class="pc-h3">{text}{small}</div>')


def lead(text):
    html(f'<p class="pc-lead">{text}</p>')


def gap(px=24):
    html(f'<div style="height:{px}px"></div>')


# ---------------- Data and computations ----------------

@st.cache_data
def load_data():
    return pd.read_excel(HERE / "Data_Cortex_Nuclear.xls")


df = load_data()
label_cols = ["MouseID", "Genotype", "Treatment", "Behavior", "class"]
X = df.drop(columns=label_cols)


@st.cache_data
def preprocess(strategy):
    imputer = SimpleImputer(strategy=strategy)
    X_filled = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    return X_filled, X_scaled


@st.cache_data
def run_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, silhouette_score(data, clusters)


@st.cache_data
def run_pca(X_scaled):
    pca_all = PCA().fit(X_scaled)
    pc3 = pca_all.transform(X_scaled)[:, 2]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return pca_all.explained_variance_ratio_, pca.explained_variance_ratio_, X_pca, pc3, pca.components_


@st.cache_data
def elbow(X_scaled):
    inertia = []
    scores = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        inertia.append(float(model.inertia_))
        scores.append(float(silhouette_score(X_scaled, labels)) if k > 1 else None)
    return inertia, scores


@st.cache_data
def components_test(X_scaled, k):
    rows = []
    for n in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 77]:
        pca_n = PCA(n_components=n)
        X_n = pca_n.fit_transform(X_scaled)
        km = KMeans(n_clusters=k, random_state=42)
        labels_n = km.fit_predict(X_n)
        rows.append([n, float(silhouette_score(X_n, labels_n)), float(pca_n.explained_variance_ratio_.sum())])
    return rows


@st.cache_data
def run_dbscan(X_pca, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1
    score = float(silhouette_score(X_pca[mask], labels[mask])) if n_clusters >= 2 else None
    return labels, n_clusters, int((labels == -1).sum()), score


def r3(a):
    return [round(float(v), 3) for v in a]


# ---------------- Header: brand, live results, course menu and the two controls ----------------

if "course" not in st.session_state:
    st.session_state.course = COURSES[0]

with st.container(key="topbar"):
    b1, b2 = st.columns([1, 1], vertical_alignment="center")
    b1.markdown("""
<div class="pc-brand">
  <svg width="30" height="30" viewBox="0 0 28 28" aria-hidden="true">
    <path d="M5 13h18v5a6 6 0 0 1-6 6h-6a6 6 0 0 1-6-6z" fill="#e8573f"/>
    <rect x="3" y="11" width="22" height="3" rx="1.5" fill="#f3ede4"/>
    <path d="M10 8c0-2 2-2 2-4M15 8c0-2 2-2 2-4" stroke="#c9a45a" stroke-width="1.8" fill="none" stroke-linecap="round"/>
  </svg>
  Anyone Can Cluster <small>K-Means before and after PCA</small>
</div>""", unsafe_allow_html=True)
    live = b2.empty()
    n1, n2, n3 = st.columns([2.9, 0.75, 1.05], vertical_alignment="bottom", gap="medium")
    with n1:
        with st.container(key="nav"):
            course = st.segmented_control("Course", COURSES, key="course", required=True,
                                          label_visibility="collapsed")
    strategy = n2.segmented_control("Fill missing values", ["median", "mean"], default="median",
                                    required=True, key="strategy")
    k = n3.slider("Clusters (k)", 2, 10, 8, key="k")

# Streamlit's sliders read the browser language: on an Arabic (right-to-left) browser the
# handle is drawn mirrored against the fill. The app is in English, so we force left-to-right.
script("const w = window.parent; try {"
       "if (!w.__pcLocale) { w.__pcLocale = true;"
       "Object.defineProperty(w.navigator, 'language', {get: () => 'en-US', configurable: true});"
       "Object.defineProperty(w.navigator, 'languages', {get: () => ['en-US', 'en'], configurable: true}); }"
       "w.document.documentElement.setAttribute('lang', 'en'); w.document.documentElement.setAttribute('dir', 'ltr');"
       "w.dispatchEvent(new Event('languagechange')); } catch (e) {}", "localefix")

# a new course opens at the top of the page
if st.session_state.get("last_course") != course:
    st.session_state.last_course = course
    script("/* " + course + " */ const d = window.parent.document; window.parent.scrollTo(0, 0);"
           "d.querySelectorAll('[data-testid=\"stMain\"], [data-testid=\"stAppViewContainer\"], section.main')"
           ".forEach(e => e.scrollTo(0, 0));", "scrolltop")

X_filled, X_scaled = preprocess(strategy)
var_all, var_2, X_pca, pc3, loadings = run_pca(X_scaled)
clusters_before, score_before = run_kmeans(X_scaled, k)
clusters_after, score_after = run_kmeans(X_pca, k)
kept = float(var_2.sum())

live.markdown(f"""
<div class="pc-live">
  <span>Before PCA<b>{score_before:.3f}</b></span>
  <span>After PCA<b class="up">{score_after:.3f}</b></span>
  <span>Info kept<b>{kept * 100:.1f}%</b></span>
</div>""", unsafe_allow_html=True)


def go_to(name):
    st.session_state.course = name


def next_prev():
    i = COURSES.index(course)
    with st.container(key="pager"):
        a, _, b = st.columns([1, 1.2, 1])
        if i > 0:
            a.button(f"← Review {NEXT_LABELS[COURSES[i - 1]]}", on_click=go_to, args=(COURSES[i - 1],), width="stretch")
        if i < len(COURSES) - 1:
            b.button(f"See {NEXT_LABELS[COURSES[i + 1]]} →", on_click=go_to, args=(COURSES[i + 1],), type="primary",
                     width="stretch")


# ---------------- Welcome ----------------

if course == "Welcome":
    gap(28)
    spread = np.percentile(np.abs(X_pca), 99, axis=0) * 1.08
    hero_xy = np.clip(X_pca / spread, -1.06, 1.06)
    show("hero", {
        "x": r3(hero_xy[:, 0]), "y": r3(hero_xy[:, 1]),
        "c": [int(c) for c in clusters_after], "palette": VEG,
        "stats": [
            {"value": len(df), "label": "samples"},
            {"value": X.shape[1], "label": "proteins"},
            {"value": int(df["MouseID"].str.split("_").str[0].nunique()), "label": "mice"},
            {"value": round(kept * 100, 1), "dec": 1, "suffix": "%", "label": "kept by PCA"},
        ],
    }, 640)

    with st.container(key="band-menu"):
        menu = [
            ("I", "Prep work", "Duplicates, missing values and scaling"),
            ("II", "Before PCA", "K-Means on all 77 proteins"),
            ("III", "Remy's nose", "PCA keeps the two strongest directions"),
            ("IV", "After PCA", "K-Means again, on two components"),
            ("V", "The verdict", "Silhouette scores and the four questions"),
            ("VI", "Test kitchen", "More components, and DBSCAN"),
        ]
        rows = "".join(
            f'<div class="row {"bonus" if no == "VI" else ""}"><span class="no">{no}</span><span class="name">{name}</span>'
            f'<span class="dots"></span><span class="desc">{desc}</span></div>' for no, name, desc in menu)
        html(f"""<div class="pc-menu"><div class="top"><div class="lab">Tonight's tasting menu</div>
<h3>Five courses <em>and a bonus</em></h3><p>One step of the project in every course. Walk them in order with the
menu above, or with the button at the foot of each page.</p></div>{rows}</div>""")

    gap(96)
    html("""
<div class="pc-spread">
  <div>
    <div class="pc-rows-title">Tonight's ingredients</div>
    <p class="pc-big">72 mice, tasted 15 times each, and <em>77 proteins</em> from every brain. Can K-Means
    tell the groups apart without being told?</p>
  </div>
  <div>
    <div class="pc-rows-title">Three labels make the 8 groups</div>
    <ul class="pc-rows">
      <li><span>Genotype</span><b>Control or trisomy (a Down syndrome model)</b></li>
      <li><span>Treatment</span><b>Memantine drug or saline</b></li>
      <li><span>Behavior</span><b>Stimulated to learn (C/S) or not (S/C)</b></li>
      <li><span>Used for clustering?</span><b>Never, only to judge the results</b></li>
    </ul>
  </div>
</div>""")

    gap(72)
    html("""
<div class="pc-spread">
  <div>
    <div class="pc-rows-title">The kitchen, translated</div>
    <ul class="pc-rows">
      <li><span>The ingredients</span><b>77 proteins</b></li>
      <li><span>Prep work</span><b>Missing values and scaling</b></li>
      <li><span>Remy's nose</span><b>PCA, 77 → 2</b></li>
      <li><span>The dish</span><b>K-Means clusters</b></li>
      <li><span>Anton Ego, the critic</span><b>Silhouette score</b></li>
    </ul>
  </div>
  <div class="pc-note" style="margin-top:34px"><div class="lab">Chef's note</div><div class="txt">A great cook
  doesn't use every ingredient in the pantry. PCA is our nose: it keeps the few flavours that matter most,
  and we taste whether the dish gets better.</div></div>
</div>""")

# ---------------- Prep work ----------------

elif course == "Prep work":
    section("01", "Course 1 · Preprocessing", "Prep work", "preprocessing",
            "Before cooking, every good chef checks the ingredients: are there duplicates, what is "
            "missing, and what do we do about it?")

    duplicates = int(df.duplicated().sum())
    missing_total = int(X.isnull().sum().sum())
    missing_pct = missing_total / X.size * 100
    missing_percent = (X.isnull().sum() / len(X) * 100)
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    figures([("Duplicate rows", duplicates, ""), ("Missing values", f"{missing_total:,}", ""),
             ("Missing overall", f"{missing_pct:.2f}%", ""), ("Proteins with gaps", f"{len(missing_percent)}/77", "")])

    top = missing_percent.head(10)
    show("jars", {
        "title": "The pantry: missing values per protein",
        "subtitle": "The 10 proteins with the most gaps. Each jar fills to its missing share.",
        "items": [{"name": n, "pct": float(p), "count": int(X[n].isnull().sum())} for n, p in top.items()],
        "threshold": 50, "total": len(X),
    }, 400)

    gap(40)
    a, b = st.columns(2, gap="large")
    with a:
        note("Duplicates", "There are no duplicate rows, so nothing to remove." if duplicates == 0
             else f"Found {duplicates} duplicate rows. They should be removed.")
    with b:
        note("Our decision",
             f"Only <b>{missing_pct:.2f}%</b> is missing and the worst protein, <b>{top.index[0]}</b>, is at "
             f"<b>{top.iloc[0]:.1f}%</b>. Nothing is above 50%, so we keep all 77 and fill the gaps with the "
             f"<b>{strategy}</b>.")

    with st.container(key="band-scaling"):
        subhead("Scaling with StandardScaler", "Step 2")
        lead("Proteins live on different ranges, so we scale each one to mean 0 and standard deviation 1. "
             "Pick a protein to compare.")
        protein = st.selectbox("Protein", list(X.columns), index=list(X.columns).index("NR1_N"))
        col = list(X.columns).index(protein)
        gap(8)
        a, b = st.columns(2, gap="medium")
        for box, values, color, name in [(a, X_filled[:, col], SAFFRON, "Before scaling"),
                                          (b, X_scaled[:, col], TOMATO, "After scaling")]:
            counts, edges = np.histogram(values, bins=24)
            centers = (edges[:-1] + edges[1:]) / 2
            with box:
                show("chart", {
                    "type": "line", "title": name,
                    "subtitle": f"Mean {abs(values.mean()):.3f} · Std {values.std():.3f}",
                    "x": [f"{c:.2f}" for c in centers], "xName": protein, "xLabel": protein,
                    "bars": {"values": [int(c) for c in counts], "color": color, "name": "Samples", "fmt": "int"},
                    "yFmt": "int", "tickDec": 0,
                }, 340)
        gap(16)
        with st.expander("Taste the raw data (first 10 rows)"):
            st.dataframe(df.head(10), hide_index=True)

# ---------------- Before PCA ----------------

elif course == "Before PCA":
    section("02", "Course 2 · K-Means before PCA", "The raw dish", "clustering before PCA",
            "K-Means on the original 77 proteins, after scaling. First we look for a good number of clusters.")
    inertia, scores = elbow(X_scaled)
    a, b = st.columns(2, gap="medium")
    with a:
        show("chart", {
            "type": "line", "title": "Elbow method", "subtitle": "Inertia for k = 1 to 10",
            "x": list(range(1, 11)), "xName": "k =", "xLabel": "Number of clusters",
            "series": [{"name": "Inertia", "values": inertia, "color": TOMATO, "fmt": "int"}],
            "yFmt": "k", "hi": k - 1, "hiLabel": f"k = {k}", "area": True,
        }, 360)
    with b:
        show("chart", {
            "type": "line", "title": "Silhouette score for each k", "subtitle": "Higher means better separated",
            "x": list(range(1, 11)), "xName": "k =", "xLabel": "Number of clusters",
            "series": [{"name": "Silhouette", "values": scores, "color": BASIL}],
            "hi": k - 1, "hiLabel": f"k = {k}", "tickDec": 2, "area": True,
        }, 360)
    gap(40)
    note("Choosing k", "There is no clear elbow: the inertia keeps falling slowly. The dataset has 8 mouse groups, "
         "so we cook with <b>k = 8</b>. Change k in the header and every course follows.")

    with st.container(key="band-result"):
        a, b = st.columns([1, 1.7], gap="large")
        with a:
            subhead("The result")
            figures([("Silhouette score before PCA", f"{score_before:.3f}", "")])
            note("What it means", "The score runs from −1 to 1. Close to 0 means the clusters overlap a lot: "
                 "with 77 proteins, the groups are not clearly separated.")
        with b:
            sizes = np.bincount(clusters_before, minlength=k)
            show("chart", {
                "type": "line", "title": "Cluster sizes", "subtitle": "Samples in each K-Means cluster",
                "x": [f"C{i + 1}" for i in range(k)], "xName": "Cluster",
                "bars": {"values": [int(s) for s in sizes], "colors": VEG[:k], "name": "Samples", "fmt": "int", "labels": True},
                "yFmt": "int", "tickDec": 0,
            }, 400)

# ---------------- Remy's nose ----------------

elif course == "Remy's nose":
    section("03", "Course 3 · Applying PCA", "Remy's nose", "applying PCA",
            "Remy can smell the few ingredients that matter in a dish. PCA does the same: it turns 77 "
            "proteins into a few principal components that keep as much of the variance as possible.")
    figures([("PC1 variance", f"{var_2[0] * 100:.1f}%", ""), ("PC2 variance", f"{var_2[1] * 100:.1f}%", ""),
             ("Kept with 2 components", f"{kept * 100:.1f}%", ""), ("Lost", f"{(1 - kept) * 100:.1f}%", "")])
    cum = np.cumsum(var_all)
    show("chart", {
        "type": "line", "title": "Explained variance", "subtitle": "Each component, and the running total",
        "x": list(range(1, 31)), "xName": "Component", "xLabel": "Number of components",
        "bars": {"values": r3(var_all[:30] * 100), "color": SAFFRON, "name": "This component", "fmt": "pct100", "dec": 1},
        "series": [{"name": "Running total", "values": r3(cum[:30] * 100), "color": TOMATO, "fmt": "pct100", "dec": 1}],
        "yFmt": "pct100", "yMin": 0, "yMax": 100, "hi": 1, "hiBelow": True, "hiLabel": f"2 components · {kept * 100:.1f}%",
        "legend": [{"name": "This component", "color": SAFFRON}, {"name": "Running total", "color": TOMATO}],
    }, 420)

    gap(40)
    a, b = st.columns([1, 1], gap="large", vertical_alignment="center")
    with a:
        target = st.slider("How much information do you want to keep? (%)", 40, 99, 80, key="target")
    needed = int(np.argmax(cum >= target / 100) + 1)
    with b:
        note("Remy says", f"To keep <b>{target}%</b> you need <b>{needed} components</b>. "
             f"We use only 2, so we keep <b>{kept * 100:.1f}%</b>.")

    with st.container(key="band-loadings"):
        subhead("Remy's favorite ingredients", "Loadings")
        lead("Which proteins build each component the most.")
        pc = st.segmented_control("Component", ["PC1", "PC2"], default="PC1", required=True, key="pc",
                                  label_visibility="collapsed")
        idx = 0 if pc == "PC1" else 1
        top = pd.Series(np.abs(loadings[idx]), index=X.columns).sort_values(ascending=False).head(10)
        gap(8)
        show("chart", {
            "type": "hbar", "title": f"Top 10 proteins in {pc}",
            "subtitle": "How strongly each protein builds this component (|loading|)",
            "labels": list(top.index), "values": r3(top.values), "xMax": float(top.values.max()) * 1.18,
            "color": TOMATO if idx == 0 else BASIL,
        }, 440)

# ---------------- After PCA ----------------

elif course == "After PCA":
    section("04", "Course 4 · K-Means after PCA", "The plated dish", "clustering after PCA",
            "K-Means again, this time on just the two PCA components. Switch the colors and the view: "
            "every point is one sample, and it glides to its new place.")
    figures([("Silhouette before PCA", f"{score_before:.3f}", ""),
             ("Silhouette after PCA", f"{score_after:.3f}", f"{score_after - score_before:+.3f}"),
             ("Features used", "77 → 2", "")])
    names_k = [f"Cluster {i + 1}" for i in range(k)]
    classes = sorted(df["class"].unique())
    modes = [
        {"id": "after", "label": "After PCA", "values": [int(c) for c in clusters_after], "names": names_k, "colors": VEG[:k]},
        {"id": "before", "label": "Before PCA", "values": [int(c) for c in clusters_before], "names": names_k, "colors": VEG[:k]},
        {"id": "class", "label": "Class", "values": [classes.index(c) for c in df["class"]], "names": classes, "colors": VEG[:8]},
        {"id": "genotype", "label": "Genotype", "values": [0 if g == "Control" else 1 for g in df["Genotype"]],
         "names": ["Control", "Trisomy"], "colors": [BLUE, TOMATO]},
        {"id": "treatment", "label": "Treatment", "values": [0 if t == "Memantine" else 1 for t in df["Treatment"]],
         "names": ["Memantine", "Saline"], "colors": [BASIL, SAFFRON]},
        {"id": "behavior", "label": "Behavior", "values": [0 if b == "C/S" else 1 for b in df["Behavior"]],
         "names": ["C/S (learned)", "S/C (not stimulated)"], "colors": [TOMATO, BLUE]},
    ]
    notes = {
        "after|p12": "K-Means drew these borders on this exact 2D map, so the clusters sit side by side like slices of a tart.",
        "after|p23": "On PC2 × PC3 these clusters look mixed: they were cut on PC1 and PC2, not on this view.",
        "before|p12": "These clusters were cooked in 77 dimensions. Squashed onto two, they overlap, because their borders live in dimensions we can't see.",
        "class": "The eight real mouse groups are mixed together. That's why no clustering finds them cleanly.",
        "genotype": "Control and trisomy mice overlap almost completely on these components.",
        "treatment": "Memantine and saline mice overlap too.",
        "behavior|p12": "See the top and bottom halves? Now switch the view to PC2 × PC3.",
        "behavior|p23": "There it is: the two behavior groups split along PC2. PCA found the learning signal without ever seeing the labels.",
    }
    show("scatter", {
        "title": "The dish, plated", "subtitle": "1,080 samples on the PCA map. Hover a point or a group.",
        "ids": list(df["MouseID"]), "cls": list(df["class"]),
        "x": {"pc1": r3(X_pca[:, 0]), "pc2": r3(X_pca[:, 1]), "pc3": r3(pc3)},
        "modes": modes, "mode": "after",
        "views": [{"id": "p12", "label": "PC1 × PC2", "x": "pc1", "y": "pc2", "xl": "PC1", "yl": "PC2"},
                  {"id": "p23", "label": "PC2 × PC3", "x": "pc2", "y": "pc3", "xl": "PC2", "yl": "PC3"}],
        "view": "p12", "notes": notes,
    }, 740)
    gap(48)
    a, b = st.columns(2, gap="large")
    with a:
        note("Before PCA", "The clusters were made with all 77 proteins, so on a 2D plot they look mixed together.")
    with b:
        note("After PCA", "The clusters were made on the same 2D space we plot, so they form clean groups.")

# ---------------- The verdict ----------------

elif course == "The verdict":
    section("05", "Course 5 · Comparison and analysis", "The verdict", "comparison and analysis",
            "Anton Ego has tasted both dishes. Here is the verdict, and our answers to the four questions.")
    show("verdict", {"before": float(score_before), "after": float(score_after), "kept": kept}, 340)

    gap(72)
    subhead("The four questions", "Analysis")
    gap(12)
    change = "went up" if score_after > score_before else "went down"
    qa = [
        ("How did PCA affect clustering results?",
         f"The silhouette score {change} from <b>{score_before:.2f}</b> to <b>{score_after:.2f}</b>, "
         "so the clusters are more separated after PCA."),
        ("Did visualization become clearer?",
         "Yes. After PCA the clusters are clean groups on the plot. Before PCA the clusters are mixed "
         "together when we plot them in 2D."),
        ("Did you lose important information?",
         f"Yes. The 2 components keep only <b>{kept * 100:.1f}%</b> of the variance, so "
         f"<b>{(1 - kept) * 100:.1f}%</b> of the information is lost."),
        ("Which approach is better and why?",
         f"PCA is better for visualization and gives a higher silhouette score. But it keeps only "
         f"{kept * 100:.0f}% of the data, so the original data has more information. The higher score "
         "after PCA is partly because 2D data is easier to separate."),
    ]
    html('<div class="pc-grid pc-g2">' + "".join(
        f'<div class="pc-qa"><div class="n">{i + 1}</div><h4>{q}</h4><p>{a}</p></div>'
        for i, (q, a) in enumerate(qa)) + "</div>")

    with st.container(key="band-summary"):
        subhead("Final summary", "Key insights")
        gap(16)
        missing_pct = X.isnull().sum().sum() / X.size * 100
        insights = [
            f"No duplicates. Only {missing_pct:.2f}% of the data was missing, so we filled it with the {strategy} and scaled the data.",
            f"K-Means before PCA gave a low silhouette score ({score_before:.2f}).",
            f"PCA with 2 components kept {kept * 100:.0f}% of the variance.",
            f"K-Means after PCA gave a higher silhouette score ({score_after:.2f}) and a clearer plot.",
            "PCA makes clustering faster and easier to see, but we lose some information.",
            "DBSCAN did not work well on this data (see the test kitchen).",
        ]
        html('<ol class="pc-list">' + "".join(f"<li>{t}</li>" for t in insights) + "</ol>")

# ---------------- Test kitchen ----------------

elif course == "Test kitchen":
    section("06", "Course 6 · Bonus", "The test kitchen", "bonus experiments",
            "Two experiments from the back of the kitchen: more PCA components, and a second chef, DBSCAN.")

    subhead("Different number of PCA components", "Experiment 1")
    gap(8)
    rows = components_test(X_scaled, k)
    show("chart", {
        "type": "line", "title": f"Silhouette score vs components (k = {k})",
        "subtitle": "More components keep more information, but the clusters get less separated",
        "x": [r[0] for r in rows], "xName": "Components:", "xLabel": "Number of PCA components",
        "series": [{"name": "Silhouette score", "values": [r[1] for r in rows], "color": TOMATO},
                   {"name": "Information kept", "values": [r[2] for r in rows], "color": BASIL, "fmt": "pct", "dec": 1}],
        "yMin": 0, "yMax": 1, "tickDec": 2, "area": True,
        "legend": [{"name": "Silhouette score", "color": TOMATO}, {"name": "Information kept", "color": BASIL}],
    }, 400)
    gap(40)
    note("Result", "More components → more information but a lower silhouette score. 2 components give the "
         "cleanest clusters, but they keep the least information.")

    with st.container(key="band-dbscan"):
        subhead("Another algorithm: DBSCAN", "Experiment 2")
        lead("DBSCAN finds dense groups and marks lonely points as noise. Try different values on the PCA data.")
        a, b = st.columns(2, gap="large")
        eps = a.slider("eps (neighborhood size)", 0.3, 2.0, 1.0, 0.1, key="eps")
        min_samples = b.slider("min_samples", 3, 30, 10, key="min_samples")
        db_labels, n_clusters, n_noise, db_score = run_dbscan(X_pca, round(eps, 1), min_samples)
        db_text = f"{db_score:.3f}" if db_score is not None else "n/a"
        gap(16)
        figures([("DBSCAN clusters", n_clusters, ""), ("Noise points", n_noise, ""),
                 ("DBSCAN silhouette*", db_text, ""), ("K-Means silhouette", f"{score_after:.3f}", "")])
        html('<p class="pc-small">*Calculated without the noise points. It needs at least 2 clusters.</p>')
        names = [f"Cluster {i + 1}" for i in range(n_clusters)] + ["Noise"]
        colors = [VEG[i % len(VEG)] for i in range(n_clusters)] + [NOISE]
        show("scatter", {
            "title": f"DBSCAN (eps = {eps:.1f}, min_samples = {min_samples})", "subtitle": "Dark grey points are noise.",
            "ids": list(df["MouseID"]), "cls": list(df["class"]),
            "x": {"pc1": r3(X_pca[:, 0]), "pc2": r3(X_pca[:, 1])},
            "modes": [{"id": "db", "label": "DBSCAN", "values": [n_clusters if g == -1 else int(g) for g in db_labels],
                       "names": names, "colors": colors}],
            "views": [{"id": "p12", "label": "PC1 × PC2", "x": "pc1", "y": "pc2", "xl": "PC1", "yl": "PC2"}],
            "notes": {},
        }, 580)
        gap(40)
        note("Result",
             f"With these settings DBSCAN found <b>{n_clusters}</b> clusters and <b>{n_noise}</b> noise points "
             f"(silhouette <b>{db_text}</b> vs <b>{score_after:.3f}</b> for K-Means). The data is one big cloud with no "
             "clear dense groups, so <b>K-Means is the better choice</b> here.")

next_prev()
html("""<div class="pc-close"><div class="big">Anyone can <em>cluster.</em></div>
<p>A mini project on K-Means before and after PCA, served on the Mice Protein Expression dataset.</p></div>
<div class="pc-foot"><span>Anyone Can Cluster</span>
<span>Data: Mice Protein Expression, UCI Machine Learning Repository (Higuera, Gardiner &amp; Cios, 2015)</span></div>""")
