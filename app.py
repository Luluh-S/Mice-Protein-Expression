import base64
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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
st.set_page_config(page_title="Pondering the Clusters",
                   page_icon=str(HERE / "brain.png") if (HERE / "brain.png").exists() else None,
                   layout="wide", initial_sidebar_state="collapsed")

BASE_CSS = (HERE / "components" / "base.css").read_text(encoding="utf-8")
MOTION_JS = (HERE / "components" / "motion.js").read_text(encoding="utf-8")
DATASET_URL = "https://archive.ics.uci.edu/dataset/342/mice+protein+expression"

# chart colours, picked to read on the light page: Pinky pink, navy, green, gold and friends (data only)
PINK, GREEN, GOLD, NAVY = "#d9557a", "#2f8f6f", "#d99a1e", "#4b5bb3"
PALETTE = [PINK, GREEN, GOLD, NAVY, "#8e5bd1", "#e0773a", "#2a9fb5", "#c2449b", "#8a9a2e", "#7a8199"]
NOISE = "#c9ccd8"

COURSES = ["Intro", "The plan", "The experiment", "Lab prep", "Before PCA", "Brain's shortcut", "After PCA",
           "The verdict", "Lab tests", "The game"]
NEXT_LABELS = {"Intro": "the intro", "The experiment": "the experiment", "The plan": "the plan", "Lab prep": "the lab prep",
               "Before PCA": "clustering before PCA", "Brain's shortcut": "PCA at work",
               "After PCA": "clustering after PCA", "The verdict": "the verdict", "Lab tests": "the bonus experiments",
               "The game": "the game"}


SHOW_COUNT = [0]


def show(name, data, height):
    """Render one of our animated components with its data."""
    page = (HERE / "components" / f"{name}.html").read_text(encoding="utf-8")
    page = page.replace("/*BASE*/", BASE_CSS).replace("/*DATA*/null", json.dumps(data))
    # each component sits in its own container, so the phone CSS can give it more height
    SHOW_COUNT[0] = SHOW_COUNT[0] + 1
    with st.container(key=f"cmp_{name}_{SHOW_COUNT[0]}"):
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
    """Render raw HTML. Keep HTML unindented and free of blank lines, or Markdown will mangle it."""
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data
def _img_src(filename):
    path = HERE / filename
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def mouse_img(filename, css_class="", style=""):
    """An <img> of Pinky or the Brain (pinky.png / brain.png next to this file)."""
    src = _img_src(filename)
    if src:
        return f'<img class="{css_class}" src="{src}" style="{style}" alt="">'
    return ""


# ---------------- Page style (light lab: white, navy ink, Pinky pink, fraunces + figtree, soft radius) ----------------

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT,WONK@0,9..144,300..700,0..100,0..1;1,9..144,300..700,0..100,0..1&family=Figtree:wght@400..600&display=swap');
:root {
  --bg: #fbf9f8; --alt: #f5f0f2; --surface: #ffffff; --surface-2: #f3f0f4; --inset: #f1edf0;
  --ink: #1d2340; --ink-2: #3a4060; --muted: #6b7088; --faint: #a3a7bb;
  --line: rgba(29,35,64,.09); --line-2: rgba(29,35,64,.18);
  --accent: #d9557a; --accent-hover: #c4466b; --accent-ink: #ffffff; --accent-soft: #fde9ef;
  --good: #2f8f6f;
  --font-display: "Fraunces", Georgia, serif;
  --font-text: "Figtree", system-ui, -apple-system, "Segoe UI", sans-serif;
  --display-axes: 'SOFT' 30, 'WONK' 0;
  --r-lg: 20px; --r-md: 12px; --r-sm: 8px;
  --ease: cubic-bezier(.2,.8,.2,1);
}
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, label, p, li, textarea { font-family: var(--font-text); }
.stApp { background: var(--bg); color: var(--ink); color-scheme: light; }
.stApp::before { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background: radial-gradient(1200px 600px at 75% -10%, rgba(217,85,122,.07), transparent 60%); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], #MainMenu, footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container { max-width: 1280px; padding: 0 48px 0; }
iframe { border: 0 !important; display: block; }
[data-testid="stHeaderActionElements"] { display: none !important; }
[data-testid="stMarkdownContainer"] p { font-size: 16px; line-height: 1.65; color: var(--ink); }
.st-key-scrolltop, .st-key-localefix, .st-key-motion { height: 0; overflow: hidden; margin: 0 !important; position: absolute; }

/* ---- header: one opaque bar, 1px hairline, the same at every scroll depth ---- */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar), .st-key-topbar { position: sticky; top: 0; z-index: 990; }
.st-key-topbar { background: var(--bg); border-bottom: 1px solid var(--line); padding: 16px 0 10px; gap: 10px !important; }
.pc-brand { display: flex; align-items: center; gap: 12px; font: 400 22px var(--font-display);
  font-variation-settings: var(--display-axes); letter-spacing: -.015em; color: var(--ink); white-space: nowrap; }
.pc-brand small { font: 500 13px var(--font-text); color: var(--muted); letter-spacing: 0; padding-left: 12px; border-left: 1px solid var(--line-2); }
.pc-live { display: flex; gap: 22px; justify-content: flex-end; align-items: baseline; flex-wrap: wrap; }
.pc-live span { font: 500 13px var(--font-text); color: var(--muted); white-space: nowrap; }
.pc-live b { font: 400 19px var(--font-display); font-variation-settings: var(--display-axes); color: var(--ink); margin-left: 6px; font-variant-numeric: tabular-nums; }
.pc-live b.up { color: var(--accent); }
.st-key-topbar [data-testid="stWidgetLabel"] p { font: 500 12px var(--font-text) !important; color: var(--muted); }

/* the page menu: plain words, the current one carries a 2px pink bar */
.st-key-nav [data-testid="stButtonGroup"] > div { background: transparent !important; border: 0 !important; padding: 0 !important; gap: 2px !important; }
.st-key-nav [data-testid="stButtonGroup"] button { background: transparent !important; box-shadow: none !important; border-radius: 0 !important;
  padding: 10px 5px 12px !important; min-height: 40px; color: var(--muted) !important; }
.st-key-nav [data-testid="stButtonGroup"] button p { font: 500 14px var(--font-text) !important; white-space: nowrap; }
.st-key-nav [data-testid="stButtonGroup"] button:hover { color: var(--ink) !important; background: transparent !important; box-shadow: inset 0 -1px 0 var(--line-2) !important; }
.st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"], .st-key-nav [data-testid="stButtonGroup"] button[aria-checked="true"]:hover {
  color: var(--ink) !important; background: transparent !important; box-shadow: inset 0 -2px 0 var(--accent) !important; }

/* other segmented controls: one inset track, the picked option raised */
[data-testid="stButtonGroup"] > div { background: var(--inset); border: 1px solid var(--line); border-radius: 12px; padding: 4px; gap: 2px !important; flex-wrap: wrap; }
[data-testid="stButtonGroup"] button { border: 0 !important; background: transparent !important; color: var(--muted) !important;
  border-radius: 9px !important; min-height: 34px; padding: 5px 14px !important; box-shadow: none !important;
  transition: color 140ms var(--ease), background-color 200ms var(--ease), box-shadow 200ms var(--ease) !important; }
[data-testid="stButtonGroup"] button p { font: 500 14px var(--font-text) !important; }
[data-testid="stButtonGroup"] button:hover { color: var(--ink) !important; background: rgba(29,35,64,.05) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--surface) !important; color: var(--ink) !important; box-shadow: 0 1px 3px rgba(29,35,64,.12), 0 0 0 1px var(--line) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"]:hover { background: var(--surface) !important; box-shadow: 0 1px 3px rgba(29,35,64,.12), 0 0 0 1px var(--line-2) !important; }
[data-testid="stSliderTickBar"] { opacity: 1 !important; }
[data-testid="stSliderTickBar"] p { color: var(--faint) !important; font-size: 12px !important; }

/* ---- page opening: a big outlined step number beside the title ---- */
.pc-sec { display: grid; grid-template-columns: 170px 1fr; gap: 32px; align-items: start; margin: 72px 0 40px; animation: pcIn 480ms var(--ease) both; }
.pc-sec .num { font: 300 150px/0.8 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.04em;
  color: transparent; -webkit-text-stroke: 1px rgba(217,85,122,.6); }
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

/* ---- notes: a serif aside with a pink label; Brain's notes carry his picture ---- */
.pc-note { border-left: 1px solid var(--accent); padding: 4px 0 4px 24px; margin: 8px 0; }
.pc-note.has-mouse { display: flex; gap: 20px; align-items: flex-start; }
.pc-note-mouse { height: 72px; flex-shrink: 0; }
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

/* ---- the plan card ---- */
.pc-menu { max-width: 940px; margin: 0 auto; padding: 56px 64px 48px; border: 1px solid var(--line-2); border-radius: 4px;
  outline: 1px solid var(--line); outline-offset: 8px; background: var(--surface); }
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

/* ---- closing: Pinky and the Brain sign off ---- */
.pc-close { text-align: center; padding: 120px 0 40px; }
.pc-close .mice { display: flex; justify-content: center; align-items: flex-end; gap: 6px; margin-bottom: 28px; }
.pc-close .big { font: 300 clamp(44px, 6.4vw, 88px)/1.02 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.04em; color: var(--ink); }
.pc-close .big em { font-style: italic; color: var(--accent); }
.pc-close p { color: var(--muted) !important; margin-top: 18px; }
.pc-foot { padding: 22px 0 28px; border-top: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px;
  font: 500 13px var(--font-text); color: var(--faint); flex-wrap: wrap; }

/* ---- buttons: one filled pink action, one outline; hover steps 1px, press sinks ---- */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  min-height: 52px; border-radius: 999px !important; padding: 0 28px !important;
  transition: transform 140ms var(--ease), background-color 140ms var(--ease), border-color 140ms var(--ease) !important; }
[data-testid="stBaseButton-primary"] { background: var(--accent) !important; border: 1px solid var(--accent) !important; color: var(--accent-ink) !important; }
[data-testid="stBaseButton-primary"]:hover { background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; transform: translateY(-1px); }
[data-testid="stBaseButton-secondary"] { background: var(--surface) !important; border: 1px solid var(--line-2) !important; color: var(--ink) !important; }
[data-testid="stBaseButton-secondary"]:hover { background: var(--surface) !important; border-color: rgba(29,35,64,.4) !important; transform: translateY(-1px); }
[data-testid="stBaseButton-primary"]:active, [data-testid="stBaseButton-secondary"]:active { transform: translateY(1px) scale(.985); transition-duration: 90ms !important; }
[data-testid="stBaseButton-primary"] p { font: 500 16px var(--font-text) !important; color: var(--accent-ink) !important; }
[data-testid="stBaseButton-secondary"] p { font: 500 16px var(--font-text) !important; color: var(--ink) !important; }
button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
.st-key-pager { margin-top: 80px; }

/* ---- keep Streamlit's own widgets light, whatever the viewer's system theme is ---- */
[data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: var(--r-md) !important; background: var(--surface); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { font: 500 15px var(--font-text) !important; color: var(--ink) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: var(--surface) !important; border-radius: 10px; border: 1px solid var(--line-2) !important; color: var(--ink) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--ink) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: var(--surface) !important; color: var(--ink) !important; }
[data-testid="stWidgetLabel"] p { font: 500 13px var(--font-text) !important; color: var(--muted) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; }
[data-testid="stSliderThumbValue"] { color: var(--accent) !important; }

/* ================= The experiment (intro tab): classes start with pb- ================= */
.st-key-intro {
  --second: #4b5bb3; --second-soft: #e9ecfa; --good-soft: #e3f3ec;
  --bad: #c94848; --bad-soft: #fbe8e8; --gray-soft: #f1f1f4;
}
@keyframes pbRise {from {opacity: 0; transform: translateY(24px);} to {opacity: 1; transform: none;}}
.pb-rise {animation: pbRise .7s var(--ease) both;}
@supports (animation-timeline: view()) {
  .pb-scroll {animation: pbRise linear both; animation-timeline: view(); animation-range: entry 0% cover 25%;}
}
@keyframes pbFloat {0%,100% {transform: translateY(0);} 50% {transform: translateY(-10px);}}
.pb-float {animation: pbFloat 3.5s ease-in-out infinite;}
.pb-float.late {animation-delay: -1.7s;}
@keyframes pbPop {0% {transform: scale(0);} 70% {transform: scale(1.15);} 100% {transform: scale(1);}}
@keyframes pbPulse {0%,100% {box-shadow: 0 0 0 0 rgba(217,85,122,.45);} 50% {box-shadow: 0 0 0 12px rgba(217,85,122,0);}}
.pb-card {background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 24px;}
.pb-card h3 {font: 400 22px var(--font-display); font-variation-settings: var(--display-axes); color: var(--ink); margin: 0 0 8px; padding: 0;}
[data-testid="stMarkdownContainer"] .pb-card p {color: var(--muted); margin: 4px 0; font-size: 15px;}
.pb-card b {color: var(--ink);}
/* hero */
.pb-hero {position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; gap: 32px; flex-wrap: wrap;
  margin-top: 28px; padding: 56px 56px 48px; border-radius: 28px; border: 1px solid var(--line); background: var(--surface);}
.pb-hero::before {content: ""; position: absolute; width: 400px; height: 400px; border-radius: 50%; background: var(--accent-soft); right: -60px; top: -80px;}
.pb-hero .txt {position: relative; z-index: 1;}
.pb-tag {display: inline-block; font: 600 13px var(--font-text); color: var(--accent); background: var(--accent-soft); padding: 5px 12px; border-radius: 99px;}
.pb-hero h1 {font: 400 64px/1.02 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.03em; color: var(--ink); margin: 16px 0; padding: 0;}
.pb-hero h1 em {font-style: italic; font-weight: 300; color: var(--accent);}
[data-testid="stMarkdownContainer"] .pb-hero p {color: var(--muted); font-size: 18px; max-width: 46ch; margin: 0;}
.pb-hero a {display: inline-block; margin-top: 18px; color: var(--accent) !important; font-weight: 500; text-decoration: none; border-bottom: 1px solid var(--accent);}
.pb-hero .mice {display: flex; align-items: flex-end; gap: 4px; position: relative; z-index: 1; flex-shrink: 0; margin: 0 auto;}
/* section titles */
.pb-section {margin: 88px 0 24px;}
.pb-eyebrow {font: 500 13px var(--font-text); color: var(--accent); letter-spacing: .02em;}
.pb-section h2 {font: 400 44px/1.05 var(--font-display); font-variation-settings: var(--display-axes); letter-spacing: -.025em; color: var(--ink); margin: 10px 0 12px; padding: 0;}
[data-testid="stMarkdownContainer"] .pb-section p {color: var(--muted); font-size: 17px; max-width: 62ch; margin: 0;}
.pb-section b {color: var(--ink);}
.pb-row {display: grid; gap: 16px;}
.pb-row-2 {grid-template-columns: 1fr 1fr;}
.pb-row-4 {grid-template-columns: repeat(4, 1fr);}
/* chromosomes */
.pb-chromos {display: flex; align-items: flex-end; gap: 14px; height: 120px; margin: 16px 0 12px 4px;}
.pb-chromo {width: 22px; border-radius: 11px; background: var(--second-soft); border: 2px solid var(--second); position: relative;}
.pb-chromo::after {content: ""; position: absolute; left: -2px; right: -2px; top: 38%; height: 5px; background: var(--surface);}
.pb-chromo.extra {background: var(--accent-soft); border-color: var(--accent); animation: pbPop .6s .3s ease both, pbPulse 2s 1s ease-in-out 3;}
.pb-chromo.chunk {height: 46px;}
.pb-chromo.chunk::after {display: none;}
.pb-small {font-size: 14px; color: var(--muted);}
.pb-small b {color: var(--accent);}
.pb-chips {display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px;}
.pb-chip {background: var(--accent-soft); color: var(--accent); font: 600 12px var(--font-text); padding: 3px 10px; border-radius: 99px;}
.pb-chip.blue {background: var(--second-soft); color: var(--second);}
/* memantine knobs */
.pb-knob-box {text-align: center; margin-top: 16px;}
.pb-knob {width: 86px; height: 86px; border-radius: 50%; background: var(--surface); border: 2px solid var(--line-2); margin: 0 auto 10px; position: relative;}
@keyframes pbTurn {from {transform: rotate(-130deg);} to {transform: rotate(var(--angle));}}
.pb-knob::after {content: ""; position: absolute; width: 5px; height: 32px; border-radius: 3px; left: 50%; top: 8px; margin-left: -2.5px;
  transform-origin: 50% 35px; transform: rotate(var(--angle)); background: var(--c); animation: pbTurn 1.4s ease both;}
.pb-meter {display: flex; gap: 3px; justify-content: center; align-items: flex-end; height: 40px; margin-bottom: 8px;}
.pb-meter i {width: 9px; border-radius: 3px; background: var(--c);}
.pb-knob-box b {display: block; color: var(--ink);}
.pb-knob-box span {color: var(--muted); font-size: 14px;}
/* the learning test */
.pb-story {display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 12px; align-items: stretch;}
.pb-scene {background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 18px; text-align: center;}
.pb-scene h3 {font: 600 16px var(--font-text); color: var(--ink); margin: 14px 0 4px; padding: 0;}
[data-testid="stMarkdownContainer"] .pb-scene p {color: var(--muted); font-size: 14px; line-height: 1.5; margin: 0;}
.pb-scene b {color: var(--ink);}
.pb-cage {height: 110px; border-radius: 14px; border: 1px solid var(--line-2); position: relative; overflow: hidden;
  background: repeating-linear-gradient(90deg, transparent 0 22px, rgba(29,35,64,.07) 22px 24px), var(--bg);}
.pb-cage .m {position: absolute; bottom: 8px; left: 10%;}
.pb-cage .m img {height: 64px; display: block;}
.pb-shock {background: #fff4c2 !important; color: #8a6a00 !important; animation: pbZap 2.4s ease infinite !important;}
@keyframes pbExplore {0%,100% {left: 8%; transform: scaleX(-1);} 45% {left: 68%; transform: scaleX(-1);} 50% {left: 68%; transform: scaleX(1);} 95% {left: 8%; transform: scaleX(1);}}
.pb-cage .m.explore {animation: pbExplore 5s ease-in-out infinite;}
@keyframes pbZap {0%,100% {opacity: 0;} 8%,16% {opacity: 1;} 12% {opacity: .2;}}
.pb-zap {position: absolute; inset: 0; background: rgba(255,214,0,.35); animation: pbZap 2.4s ease infinite;}
.pb-bolt {position: absolute; top: 8px; right: 12px; font-size: 1.6rem; animation: pbZap 2.4s ease infinite;}
@keyframes pbShiver {0%,100% {transform: translateX(0);} 25% {transform: translateX(-1px);} 75% {transform: translateX(1px);}}
.pb-cage .m.freeze {left: 40%; animation: pbShiver .25s linear infinite;}
.pb-ice {position: absolute; top: 10px; left: 50%; transform: translateX(-50%); background: var(--second-soft); color: var(--second);
  font: 700 11px var(--font-text); letter-spacing: .08em; padding: 3px 10px; border-radius: 99px; animation: pbPop .5s ease both;}
.pb-arrow {display: flex; align-items: center; justify-content: center; color: var(--accent); font-size: 22px;}
.pb-verdict {margin-top: 16px; padding: 16px 20px; border-radius: 14px; color: var(--ink-2); font-size: 16px; border-left: 2px solid var(--accent);}
.pb-verdict.cs {background: var(--accent-soft);}
.pb-verdict.sc {background: var(--gray-soft); border-left-color: var(--muted);}
.pb-verdict b {color: var(--ink);}
/* label decoder */
.pb-label {display: flex; justify-content: center; align-items: center; gap: 6px; margin: 28px 0 20px;
  font: 400 56px var(--font-display); font-variation-settings: var(--display-axes);}
.pb-seg {padding: 0 18px 4px; border-radius: 16px; animation: pbPop .4s ease both;}
.pb-dash {color: var(--faint);}
.pb-s1 {background: var(--second-soft); color: var(--second);}
.pb-s2 {background: var(--accent-soft); color: var(--accent);}
.pb-s3 {background: var(--good-soft); color: var(--good);}
.pb-explain {display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 28px;}
.pb-explain div {border-radius: 14px; padding: 14px 16px; font-size: 15px; color: var(--ink-2);}
.pb-explain b {display: block; font: 600 12px var(--font-text); letter-spacing: .08em; text-transform: uppercase; margin-bottom: 4px;}
.pb-grid {display: grid; grid-template-columns: 90px repeat(4, 1fr); gap: 10px;}
.pb-colhead {text-align: center; font: 500 13px var(--font-text); color: var(--muted);}
.pb-rowhead {display: flex; flex-direction: column; align-items: center; justify-content: center; font: 600 16px var(--font-text); color: var(--ink);}
.pb-rowhead span {font-size: 11px; color: var(--muted); font-weight: 500;}
.pb-group {background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 14px 6px; text-align: center; transition: all .35s var(--ease);}
.pb-group code {font-size: 15px; background: none; color: var(--ink); font-weight: 600;}
.pb-group.dim {opacity: .4;}
.pb-group.pick {border-color: var(--ink); transform: scale(1.05); box-shadow: 0 10px 25px rgba(29,35,64,.12); opacity: 1;}
.pb-badge {display: inline-block; margin-top: 8px; padding: 2px 10px; border-radius: 99px; font: 600 12px var(--font-text); animation: pbPop .5s ease both;}
.pb-b-learned {background: var(--good-soft); color: var(--good);}
.pb-b-rescued {background: var(--accent-soft); color: var(--accent);}
.pb-b-failed {background: var(--bad-soft); color: var(--bad);}
.pb-b-baseline {background: var(--gray-soft); color: var(--muted);}
.pb-group.learned {border-color: var(--good);}
.pb-group.rescued {border-color: var(--accent); background: #fff7f9;}
.pb-group.failed {border-color: var(--bad);}
.pb-result {margin: 20px 0 8px; border-left: 2px solid var(--accent); padding: 4px 0 4px 24px;
  font: 400 21px/1.5 var(--font-display); font-variation-settings: var(--display-axes); color: var(--ink-2);}
.pb-result b {color: var(--ink); font-weight: 500;}
/* where they looked */
.pb-zoom {display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 12px; align-items: stretch;}
.pb-zoom .pb-card {text-align: center;}
.pb-cell {width: 90px; height: 90px; border-radius: 50%; margin: 4px auto 12px; background: var(--second-soft); border: 2px solid var(--second);
  display: flex; align-items: center; justify-content: center;}
.pb-nucleus {width: 38px; height: 38px; border-radius: 50%; background: var(--accent); animation: pbPulse 2s ease-in-out infinite;}
.pb-cortex {width: 90px; height: 90px; border-radius: 50%; margin: 4px auto 12px; border: 12px solid var(--accent);
  background: var(--accent-soft); display: flex; align-items: center; justify-content: center; animation: pbPulse 2s ease-in-out infinite;}
.pb-cortex div {width: 40px; height: 40px; border-radius: 50%; background: var(--second-soft); border: 2px solid var(--second);}
/* protein names and families */
.pb-name {display: flex; justify-content: center; gap: 4px; font: 400 44px var(--font-display); margin: 8px 0 20px;}
.pb-name-parts {display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;}
.pb-name-parts div {border-radius: 14px; padding: 14px 16px; font-size: 15px; color: var(--ink-2);}
.pb-name-parts b:first-child {display: block; font: 400 24px var(--font-display); margin-bottom: 2px;}
.pb-flip {perspective: 1000px; height: 240px; cursor: pointer;}
.pb-flip-in {position: relative; height: 100%; transition: transform .7s var(--ease); transform-style: preserve-3d;}
.pb-flip:hover .pb-flip-in {transform: rotateY(180deg);}
.pb-face {position: absolute; inset: 0; backface-visibility: hidden; border-radius: var(--r-lg); padding: 22px;
  border: 1px solid var(--line); display: flex; flex-direction: column; justify-content: center;}
.pb-front {background: var(--surface); align-items: center; text-align: center;}
.pb-front .icon {font-size: 2.4rem;}
.pb-front h3 {font: 400 22px var(--font-display); color: var(--ink); margin: 8px 0 4px; padding: 0;}
.pb-front small {color: var(--faint); margin-top: 10px;}
.pb-back {background: var(--ink); border-color: var(--ink); transform: rotateY(180deg); color: #ffffff; font-size: 16px; line-height: 1.55;}
.pb-back b {color: #ffb3c7;}
/* Brain in the corner */
.pb-corner {position: fixed; right: 22px; bottom: 18px; z-index: 999; text-align: center; pointer-events: none;}
.pb-corner img {height: 115px;}
.pb-bubble {background: var(--surface); border: 1px solid var(--line-2); border-radius: 50%; width: 46px; height: 46px;
  display: flex; align-items: center; justify-content: center; font-size: 1.5rem; margin: 0 0 -6px auto;
  box-shadow: 0 4px 12px rgba(29,35,64,.08); animation: pbPop .4s ease both;}
@keyframes pbBounce {0%,100% {transform: translateY(0);} 30% {transform: translateY(-16px);} 60% {transform: translateY(-4px);}}
@keyframes pbDroop {0% {transform: none;} 100% {transform: translateY(8px) rotate(-6deg);}}
@keyframes pbTilt {0%,100% {transform: rotate(0);} 50% {transform: rotate(8deg);}}
@keyframes pbWobble {0%,100% {transform: rotate(0);} 25% {transform: rotate(-4deg);} 75% {transform: rotate(4deg);}}
.pb-r-learned, .pb-r-rescued {animation: pbBounce 1s ease 2;}
.pb-r-failed {animation: pbDroop .8s ease forwards;}
.pb-r-baseline {animation: pbTilt 1.6s ease 1;}
.pb-r-waiting {animation: pbWobble 2.5s ease-in-out infinite;}
/* Pinky peeking in on the reveal */
@keyframes pbPeek {0% {transform: translateX(-110%);} 20%, 75% {transform: translateX(0) rotate(8deg);} 100% {transform: translateX(-110%);}}
.pb-peek {position: fixed; left: 0; bottom: 25%; z-index: 999; animation: pbPeek 3.8s ease forwards; pointer-events: none;}
.pb-peek img {height: 190px;}

/* ================= Intro tab: title and team (classes start with tt-) ================= */
.tt-hero { position: relative; overflow: hidden; text-align: center; margin-top: 28px; padding: 84px 32px 170px;
  border-radius: 28px; border: 1px solid var(--line);
  background: radial-gradient(640px 320px at 50% 0%, var(--accent-soft), transparent 70%), var(--surface); }
.tt-pill { display: inline-block; font: 500 13px var(--font-text); color: var(--accent); background: var(--accent-soft); padding: 6px 14px; border-radius: 99px; }
.tt-hero h1 { font: 300 clamp(56px, 8vw, 112px)/.95 var(--font-display); font-variation-settings: var(--display-axes);
  letter-spacing: -.04em; color: var(--ink); margin: 22px 0 18px; padding: 0; }
.tt-hero h1 em { font-style: italic; color: var(--accent); }
[data-testid="stMarkdownContainer"] .tt-hero p { color: var(--muted); font-size: 19px; margin: 0 auto; max-width: 52ch; }
.tt-hero .tt-pinky { position: absolute; left: 7%; bottom: -10px; height: 180px; }
.tt-hero .tt-brain { position: absolute; right: 7%; bottom: -6px; height: 112px; }
.tt-team-label { text-align: center; font: 500 13px var(--font-text); color: var(--accent); margin: 72px 0 26px; letter-spacing: .06em; text-transform: uppercase; }
.tt-team { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; max-width: 1000px; margin: 0 auto; }
.tt-member { text-align: center; background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 28px 16px 24px;
  animation: pbRise .7s var(--ease) both; transition: transform .25s var(--ease), box-shadow .25s var(--ease); }
.tt-member:hover { transform: translateY(-6px); box-shadow: 0 14px 30px rgba(29,35,64,.1); }
.tt-member svg { width: 132px; height: 132px; display: block; margin: 0 auto 16px; }
.tt-member b { display: block; font: 400 28px var(--font-display); font-variation-settings: var(--display-axes); color: var(--ink); letter-spacing: -.01em; }

@media (max-width: 1100px) {
  .pc-sec { grid-template-columns: 1fr; gap: 0; }
  .pc-sec .num { font-size: 96px; margin-bottom: 8px; }
}
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 0 18px; }
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
  .pc-note-mouse { height: 52px; }
  .pb-hero { padding: 32px 24px; } .pb-hero h1 { font-size: 42px; }
  .pb-row-2, .pb-row-4, .pb-story, .pb-zoom, .pb-explain, .pb-name-parts { grid-template-columns: 1fr; }
  .pb-arrow { transform: rotate(90deg); } .pb-label { font-size: 36px; } .pb-section h2 { font-size: 34px; }
  .pb-corner img { height: 80px; }
  .tt-team { grid-template-columns: 1fr 1fr; }
  .tt-hero { padding: 56px 20px 130px; }
  .tt-hero .tt-pinky { height: 120px; left: 4%; } .tt-hero .tt-brain { height: 76px; right: 4%; }
}
/* ---- phones: everything stacks, text and spacing get smaller ---- */
@media (max-width: 760px) {
  [class*="st-key-cmp_hero"], [class*="st-key-cmp_hero"] div, [class*="st-key-cmp_hero"] iframe { height: 980px !important; max-height: none !important; }
  [class*="st-key-cmp_scatter"], [class*="st-key-cmp_scatter"] div, [class*="st-key-cmp_scatter"] iframe { height: 820px !important; max-height: none !important; }
  [class*="st-key-cmp_verdict"], [class*="st-key-cmp_verdict"] div, [class*="st-key-cmp_verdict"] iframe { height: 900px !important; max-height: none !important; }
  [class*="st-key-cmp_game"], [class*="st-key-cmp_game"] div, [class*="st-key-cmp_game"] iframe { height: 1080px !important; max-height: none !important; }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"], .block-container { padding: 0 14px; }
  [data-testid="stMarkdownContainer"] p { font-size: 15px; }
  .pc-brand { font-size: 18px; white-space: normal; }
  /* the header stays on top while scrolling, so it is kept small on phones:
     brand, live scores and the page menu stay; the two controls are left for the laptop */
  .st-key-topbar { padding: 8px 0 2px; gap: 4px !important; }
  .st-key-topbar [data-testid="stColumn"]:has(.st-key-strategy),
  .st-key-topbar [data-testid="stColumn"]:has(.st-key-k) { display: none !important; }
  .pc-live { gap: 10px; justify-content: flex-start; }
  .pc-live span { font-size: 11px; }
  .pc-live { gap: 12px; }
  .pc-live b { font-size: 16px; }
  .st-key-nav [data-testid="stButtonGroup"] > div { flex-wrap: nowrap !important; overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
  .st-key-nav [data-testid="stButtonGroup"] > div::-webkit-scrollbar { display: none; }
  .st-key-nav [data-testid="stButtonGroup"] button { flex: none; padding: 8px 6px 10px !important; }
  .st-key-nav [data-testid="stButtonGroup"] button p { font-size: 13px !important; }
  .pc-sec { margin: 44px 0 24px; }
  .pc-sec .num { font-size: 60px; }
  .pc-sec h2 { font-size: 30px; }
  .pc-sec p, [data-testid="stMarkdownContainer"] p.pc-lead, .pc-lead { font-size: 16px !important; }
  .pc-h3 { font-size: 24px; }
  .pc-figs .val { font-size: 30px; }
  [data-testid="stMarkdownContainer"] p.pc-big, .pc-big { font-size: 24px !important; }
  .pc-menu { padding: 26px 14px 20px; }
  .pc-menu .name { font-size: 20px; }
  .pc-rows li { font-size: 14px; gap: 12px; }
  .pc-close .big { font-size: 36px; }
  .pc-note.has-mouse { gap: 12px; }
  .pb-hero { padding: 24px 16px; } .pb-hero h1 { font-size: 32px; }
  [data-testid="stMarkdownContainer"] .pb-hero p, [data-testid="stMarkdownContainer"] .pb-section p { font-size: 15px; }
  .pb-section h2 { font-size: 28px; } .pb-label { font-size: 28px; } .pb-name { font-size: 30px; }
  .pb-grid { grid-template-columns: 54px repeat(4, minmax(0, 1fr)); gap: 6px; }
  .pb-rowhead { font-size: 13px; }
  .pb-colhead { font-size: 11px; }
  .pb-group { padding: 10px 2px; }
  .pb-group code { font-size: 12px; }
  .pb-badge { font-size: 10px; padding: 2px 6px; }
  .pb-group.pick { transform: scale(1.02); }
  .pb-peek img { height: 130px; }
  .pb-result { font-size: 18px; padding-left: 16px; }
  .tt-hero { padding: 40px 16px 110px; border-radius: 20px; }
  .tt-hero h1 { font-size: 46px; }
  [data-testid="stMarkdownContainer"] .tt-hero p { font-size: 16px; }
  .tt-hero .tt-pinky { height: 90px; } .tt-hero .tt-brain { height: 58px; }
  .tt-team-label { margin: 44px 0 18px; }
  .tt-team { gap: 12px; }
  .tt-member { padding: 18px 8px 16px; }
  .tt-member svg { width: 92px; height: 92px; margin-bottom: 10px; }
  .tt-member b { font-size: 22px; }
}

/* ================= motion layer ================= */
/* header: the fill-missing control stays on one line */
.st-key-strategy [data-testid="stButtonGroup"] > div { flex-wrap: nowrap !important; }
.st-key-strategy [data-testid="stButtonGroup"] button { padding: 5px 10px !important; }
/* reading progress along the bottom of the header */
.pc-progress { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background: var(--accent); transform-origin: 0 50%;
  transform: scaleX(0); pointer-events: none; transition: transform 90ms linear; border-radius: 2px; }
/* a live score flashes when it changes */
.pc-live b { border-radius: 6px; padding: 0 2px; }
.pc-live b.pc-flash { animation: pcFlash 900ms var(--ease); }
@keyframes pcFlash { 0% { background: rgba(217,85,122,.2); box-shadow: 0 0 0 4px rgba(217,85,122,.2); } 100% { background: transparent; box-shadow: 0 0 0 4px rgba(217,85,122,0); } }
/* leaving a page: the old one fades out before the new one comes in */
html.pc-leaving [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) ~ * { opacity: 0; transition: opacity 160ms var(--ease); }
/* page titles: the outlined number fills up, the words rise one by one */
.pc-sec .num { background: linear-gradient(to top, rgba(217,85,122,.16) 50%, transparent 50%); background-size: 100% 200%;
  background-position: 0 0; -webkit-background-clip: text; background-clip: text; animation: pcFill 1200ms var(--ease) 250ms both;
  transition: -webkit-text-stroke-color 200ms var(--ease); }
.pc-sec:hover .num { -webkit-text-stroke-color: var(--accent); }
@keyframes pcFill { from { background-position: 0 0; } to { background-position: 0 100%; } }
.pc-sec h2 .pc-w { display: inline-block; animation: pcWord 620ms var(--ease) both; animation-delay: calc(var(--i) * 70ms + 120ms); }
@keyframes pcWord { from { opacity: 0; transform: translateY(.35em); filter: blur(4px); } to { opacity: 1; transform: none; filter: none; } }
/* cards and rows answer the pointer (translate, so it adds to their entrance animation) */
.pc-qa { animation: pcIn 520ms var(--ease) both; transition: translate 200ms var(--ease), box-shadow 200ms var(--ease), border-color 200ms var(--ease); }
.pc-qa:nth-child(2) { animation-delay: 80ms; } .pc-qa:nth-child(3) { animation-delay: 160ms; } .pc-qa:nth-child(4) { animation-delay: 240ms; }
.pc-qa:hover { translate: 0 -4px; box-shadow: 0 16px 34px rgba(29,35,64,.08); border-color: rgba(217,85,122,.45); }
.pc-qa .n { display: inline-block; transition: transform 200ms var(--ease); }
.pc-qa:hover .n { transform: translateX(4px); }
.pc-rows li { transition: padding 180ms var(--ease); }
.pc-rows li:hover { padding-left: 8px; padding-right: 8px; }
.pc-rows li:hover span { color: var(--accent); }
.pc-list li { transition: color 180ms var(--ease); }
.pc-list li:hover { color: var(--ink); }
.pc-figs > div { animation: pcIn 480ms var(--ease) both; }
.pc-figs > div:nth-child(2) { animation-delay: 70ms; } .pc-figs > div:nth-child(3) { animation-delay: 140ms; } .pc-figs > div:nth-child(4) { animation-delay: 210ms; }
.pc-note { transition: border-color 200ms var(--ease), padding 200ms var(--ease); }
.pc-note:hover { padding-left: 30px; }
.pc-note-mouse { transition: transform 220ms var(--ease); }
.pc-note:hover .pc-note-mouse { transform: rotate(-6deg) translateY(-3px); }
.pb-card, .pb-scene { transition: translate 200ms var(--ease), box-shadow 200ms var(--ease); }
.pb-card:hover, .pb-scene:hover { translate: 0 -3px; box-shadow: 0 12px 28px rgba(29,35,64,.07); }
.pb-zoom .pb-card:hover, .pb-flip .pb-face { translate: none; }
/* the plan menu: every row opens its page */
.pc-menu .row { position: relative; cursor: pointer; animation: pcIn 480ms var(--ease) both; animation-delay: calc(var(--i, 0) * 70ms + 150ms); outline: none; }
.pc-menu .row::before { content: ""; position: absolute; inset: 0 -20px; background: var(--accent-soft); border-radius: 10px; opacity: 0;
  transform: scaleX(.97); transition: opacity 180ms var(--ease), transform 180ms var(--ease); }
.pc-menu .row > * { position: relative; }
.pc-menu .row:hover::before, .pc-menu .row:focus-visible::before { opacity: .6; transform: none; }
.pc-menu .row .name { transition: transform 200ms var(--ease), color 180ms var(--ease); }
.pc-menu .row:hover .name, .pc-menu .row:focus-visible .name { transform: translateX(8px); color: var(--accent); }
.pc-menu .row .dots { transition: border-color 180ms var(--ease); }
.pc-menu .row:hover .dots { border-bottom-color: var(--accent); }
.pc-menu .row::after { content: "→"; position: absolute; right: -36px; top: 50%; font: 400 20px var(--font-text); color: var(--accent);
  opacity: 0; transform: translate(-8px, -50%); transition: opacity 180ms var(--ease), transform 180ms var(--ease); }
.pc-menu .row:hover::after, .pc-menu .row:focus-visible::after { opacity: 1; transform: translate(0, -50%); }
.pc-menu .row:active .name { transform: translateX(10px) scale(.99); }
.st-key-plango { position: absolute !important; width: 1px; height: 1px; overflow: hidden; opacity: 0; pointer-events: none; }
/* the heroes: Pinky and the Brain follow the pointer a little, at different depths */
.tt-hero { background: radial-gradient(640px 320px at calc(50% + var(--mx, 0) * 40%) calc(var(--my, 0) * 30%), var(--accent-soft), transparent 70%), var(--surface); }
.tt-hero img, .pb-hero .mice img { translate: calc(var(--mx, 0) * var(--depth, 16px)) calc(var(--my, 0) * var(--depth, 16px) * .6); transition: translate 260ms var(--ease); }
.tt-hero .tt-pinky, .pb-hero .mice img:first-child { --depth: -28px; }
.tt-hero .tt-brain, .pb-hero .mice img:last-child { --depth: 18px; }
/* the team: portraits assemble pixel by pixel, then blink and tilt on hover */
.tt-member .px { transform-box: fill-box; transform-origin: 50% 50%; animation: pxIn 320ms var(--ease) both; }
@keyframes pxIn { from { opacity: 0; transform: scale(.2); } to { opacity: 1; transform: none; } }
.tt-member svg { transition: transform 220ms var(--ease); }
.tt-member:hover svg { transform: rotate(-4deg) scale(1.04); }
.tt-member:hover .eye { animation: pxBlink 1800ms var(--ease) infinite; }
@keyframes pxBlink { 0%, 40%, 54%, 100% { transform: none; } 47% { transform: scaleY(.1); } }
.tt-member b { transition: color 180ms var(--ease); }
.tt-member:hover b { color: var(--accent); }
@media (max-width: 900px) {
  .pc-menu .row::before { inset: 0 -8px; }
  .pc-menu .row::after { display: none; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""")


def _words(text, start=0):
    """Wrap each word in a span, numbered from start, for the word-by-word title entrance."""
    return " ".join(f'<span class="pc-w" style="--i:{start + i}">{w}</span>' for i, w in enumerate(text.split()))


def section(num, eyebrow, title, subtitle, text):
    n = len(title.split())
    html(f'<div class="pc-sec"><div class="num">{num}</div><div><div class="eyebrow">{eyebrow}</div>'
         f'<h2>{_words(title)} <em>{_words(subtitle, n)}</em></h2><p>{text}</p></div></div>')


def figures(items):
    cells = ""
    for label, value, delta in items:
        d = f"<small>{delta}</small>" if delta else ""
        cells += f'<div><div class="lab">{label}</div><div class="val">{value}{d}</div></div>'
    html(f'<div class="pc-figs" style="grid-template-columns: repeat({len(items)}, minmax(0, 1fr))">{cells}</div>')


def note(label, text, who=None):
    """A side note. who="brain" or "pinky" puts that character next to it."""
    if who:
        html(f'<div class="pc-note has-mouse">{mouse_img(f"{who}.png", "pc-note-mouse")}'
             f'<div><div class="lab">{label}</div><div class="txt">{text}</div></div></div>')
    else:
        html(f'<div class="pc-note"><div class="lab">{label}</div><div class="txt">{text}</div></div>')


def subhead(text, label=""):
    small = f"<small>{label}</small>" if label else ""
    html(f'<div class="pc-h3">{text}{small}</div>')


def lead(text):
    html(f'<p class="pc-lead">{text}</p>')


def gap(px=24):
    html(f'<div style="height:{px}px"></div>')


# ---------------- The experiment (intro tab) ----------------

BADGE = {"learned": "Learned", "rescued": "Rescued", "failed": "Didn't learn", "baseline": "Baseline"}


def outcome(code):
    """What happened to a group in the learning test."""
    g, b, t = code.split("-")
    if b == "SC":
        return "baseline"
    if g == "t" and t == "s":
        return "failed"
    if g == "t" and t == "m":
        return "rescued"
    return "learned"


def intro_section(num, eyebrow, title, text=""):
    html(f'<div class="pb-section pb-scroll"><div class="pb-eyebrow">{num} · {eyebrow}</div>'
         f'<h2>{title}</h2>{f"<p>{text}</p>" if text else ""}</div>')


def render_intro():
    if "pb_revealed" not in st.session_state:
        st.session_state.pb_revealed = False

    with st.container(key="intro"):
        # ---------- hero ----------
        html(f"""
<div class="pb-hero pb-rise">
<div class="txt">
<span class="pb-tag">The experiment</span>
<h1>Can a drug help a brain<br><em>learn again?</em></h1>
<p>Scientists tested a drug called memantine on mice that model Down syndrome, then looked inside their brains to see what changed.</p>
<a href="{DATASET_URL}" target="_blank">Dataset: Mice Protein Expression (UCI) ↗</a>
</div>
<div class="mice">{mouse_img("pinky.png", "pb-float", "height:230px")}{mouse_img("brain.png", "pb-float late", "height:142px")}</div>
</div>
""")

        # ---------- 01 why these mice ----------
        intro_section("01", "The mice", "Why these mice?",
                      "Down syndrome happens when a person has an extra copy of chromosome 21. "
                      "Mice can't have that exact chromosome, so scientists use a mouse that carries an extra copy "
                      "of the matching piece of its own DNA.")
        html("""
<div class="pb-row pb-row-2 pb-scroll">
<div class="pb-card"><h3>In people</h3>
<div class="pb-chromos"><div class="pb-chromo" style="height:100px"></div><div class="pb-chromo" style="height:100px"></div><div class="pb-chromo extra" style="height:100px"></div></div>
<div class="pb-small">Chromosome 21: <b>three copies</b> instead of two</div></div>
<div class="pb-card"><h3>In Ts65Dn mice (“trisomic”)</h3>
<div class="pb-chromos"><div class="pb-chromo" style="height:110px"></div><div class="pb-chromo" style="height:110px"></div><div class="pb-chromo extra chunk"></div></div>
<div class="pb-small">An <b>extra chunk</b> of DNA carrying many of the same genes</div>
<div class="pb-chips"><span class="pb-chip">DYRK1A</span><span class="pb-chip">SOD1</span><span class="pb-chip">APP</span><span class="pb-chip">ITSN1</span><span class="pb-chip">…and more</span></div></div>
</div>
<div class="pb-card pb-scroll" style="margin-top:16px"><p style="margin:0">Extra copies of genes can mean <b>extra protein</b>, and that can change how the brain works.
Mice without the extra copy are the <b>control</b> group.</p></div>
""")

        # ---------- 02 why memantine ----------
        intro_section("02", "The drug", "Why memantine?",
                      "Learning relies on NMDA receptors, little “doors” on brain cells that let signals in. "
                      "In these mice, that signal is thought to be too loud.")
        html("""
<div class="pb-row pb-row-2 pb-scroll">
<div class="pb-card"><h3>Saline (placebo)</h3><p>Just salt water, so nothing changes.</p>
<div class="pb-knob-box"><div class="pb-knob" style="--angle:120deg; --c:var(--bad)"></div>
<div class="pb-meter" style="--c:var(--bad)"><i style="height:30%"></i><i style="height:50%"></i><i style="height:70%"></i><i style="height:85%"></i><i style="height:100%"></i></div>
<b>Signal too loud</b><span>hard to learn</span></div></div>
<div class="pb-card"><h3>Memantine</h3><p>Turns the NMDA signal down a little. It's already used to treat Alzheimer's.</p>
<div class="pb-knob-box"><div class="pb-knob" style="--angle:0deg; --c:var(--good)"></div>
<div class="pb-meter" style="--c:var(--good)"><i style="height:30%"></i><i style="height:50%"></i><i style="height:70%"></i><i style="height:22%;opacity:.25"></i><i style="height:15%;opacity:.25"></i></div>
<b>Signal turned down</b><span>learning might work again</span></div></div>
</div>
""")

        # ---------- 03 the learning test ----------
        intro_section("03", "The test", "How do you test if a mouse learned?",
                      "With fear conditioning. A mouse that remembers danger <b>freezes</b> in place. "
                      "Switch between the two versions of the test:")
        test = st.segmented_control("Learning test", ["CS · learning", "SC · baseline"], default="CS · learning",
                                    required=True, key="pb_test", label_visibility="collapsed")
        mouse = mouse_img("pinky.png")
        shock = '<div class="pb-zap"></div><div class="pb-ice pb-shock">SHOCK</div>'
        if test.startswith("CS"):
            s1 = (f'<div class="pb-cage"><div class="m explore">{mouse}</div></div><h3>1 · Explores a new cage</h3>'
                  '<p>It has a few minutes to look around and get to know the place.</p>')
            s2 = (f'<div class="pb-cage">{shock}<div class="m" style="left:40%">{mouse}</div></div>'
                  '<h3>2 · Then a mild shock</h3><p>Now the cage is linked to something bad.</p>')
            s3 = (f'<div class="pb-cage"><div class="pb-ice">FROZEN</div><div class="m freeze">{mouse}</div></div>'
                  '<h3>3 · Back in the cage later</h3><p>It <b>freezes</b>: it remembers this place means danger.</p>')
            verdict = ('<div class="pb-verdict cs"><b>CS = context, then shock.</b> The mouse had time to learn the cage, '
                       'so freezing later means it <b>learned</b>.</div>')
        else:
            s1 = (f'<div class="pb-cage">{shock}<div class="m" style="left:40%">{mouse}</div></div>'
                  '<h3>1 · Shocked right away</h3><p>Before it has any time to look around.</p>')
            s2 = (f'<div class="pb-cage"><div class="m explore">{mouse}</div></div><h3>2 · Then explores</h3>'
                  '<p>The shock already happened, so there\'s nothing to link it to.</p>')
            s3 = (f'<div class="pb-cage"><div class="m explore">{mouse}</div></div>'
                  '<h3>3 · Back in the cage later</h3><p>It keeps exploring. <b>No memory</b> of danger here.</p>')
            verdict = ('<div class="pb-verdict sc"><b>SC = shock, then context.</b> No chance to learn, '
                       'so this group is the <b>baseline</b> to compare against.</div>')
        html(f'<div class="pb-story"><div class="pb-scene">{s1}</div><div class="pb-arrow">→</div>'
             f'<div class="pb-scene">{s2}</div><div class="pb-arrow">→</div><div class="pb-scene">{s3}</div></div>{verdict}')

        # ---------- 04 label decoder ----------
        intro_section("04", "The classes", "Reading a mouse's label",
                      "Every mouse gets a short label built from three choices. Build one yourself:")
        c1, c2, c3 = st.columns(3)
        geno = c1.segmented_control("Genotype", ["Control", "Trisomic"], default="Trisomic", required=True, key="pb_geno")
        behav = c2.segmented_control("Learning test", ["CS", "SC"], default="CS", required=True, key="pb_behav")
        treat = c3.segmented_control("Treatment", ["Memantine", "Saline"], default="Memantine", required=True, key="pb_treat")

        g = "c" if geno == "Control" else "t"
        t = "m" if treat == "Memantine" else "s"
        picked = f"{g}-{behav}-{t}"
        revealed = st.session_state.pb_revealed

        g_text = "typical mouse" if g == "c" else "mouse with the extra DNA chunk (Ts65Dn)"
        b_text = "had the chance to learn (context, then shock)" if behav == "CS" else "baseline: shocked first, nothing to learn"
        t_text = "got the drug" if t == "m" else "got salt water (placebo)"
        html(f"""
<div class="pb-label"><span class="pb-seg pb-s1">{g}</span><span class="pb-dash">-</span><span class="pb-seg pb-s2">{behav}</span><span class="pb-dash">-</span><span class="pb-seg pb-s3">{t}</span></div>
<div class="pb-explain">
<div class="pb-s1"><b>{g} · Genotype</b>{geno}: {g_text}</div>
<div class="pb-s2"><b>{behav} · Learning test</b>{b_text}</div>
<div class="pb-s3"><b>{t} · Treatment</b>{treat}: {t_text}</div>
</div>
""")
        columns = [("c", "m", "Control<br>+ memantine"), ("c", "s", "Control<br>+ saline"),
                   ("t", "m", "Trisomic<br>+ memantine"), ("t", "s", "Trisomic<br>+ saline")]
        cells = ['<div></div>'] + [f'<div class="pb-colhead">{label}</div>' for _, _, label in columns]
        for b, sub in [("CS", "learning"), ("SC", "baseline")]:
            cells.append(f'<div class="pb-rowhead">{b}<span>{sub}</span></div>')
            for gg, tt, _ in columns:
                code = f"{gg}-{b}-{tt}"
                classes = ["pb-group", "pick" if code == picked else "dim"]
                badge = ""
                if revealed:
                    res = outcome(code)
                    classes.append(res)
                    badge = f'<div><span class="pb-badge pb-b-{res}">{BADGE[res]}</span></div>'
                cells.append(f'<div class="{" ".join(classes)}"><code>{code}</code>{badge}</div>')
        html(f'<div class="pb-grid">{"".join(cells)}</div>')

        # ---------- reveal ----------
        st.write("")
        if not revealed:
            if st.button("Reveal who learned", type="primary", key="pb_reveal"):
                st.session_state.pb_revealed = True
                st.session_state.pb_just_revealed = True
                st.rerun()
        else:
            html('<div class="pb-result pb-rise"><b>The result:</b> trisomic mice on saline <b>didn\'t learn</b>. '
                 'Trisomic mice on <b>memantine did</b>: the drug rescued their learning. '
                 'So what changed inside their brains? That\'s where the proteins come in.</div>')
            if st.session_state.pop("pb_just_revealed", False):
                html(f'<div class="pb-peek">{mouse_img("pinky.png")}</div>')
            if st.button("Hide results", key="pb_hide"):
                st.session_state.pb_revealed = False
                st.rerun()

        # ---------- 05 where they looked ----------
        intro_section("05", "The measurement", "Where did they look?",
                      "After the learning test, scientists measured protein levels in each mouse's brain, zooming all the way in:")
        html("""
<div class="pb-zoom pb-scroll">
<div class="pb-card"><div class="pb-cortex"><div></div></div><h3>The cortex</h3><p>The brain's outer layer, where thinking and memory happen.</p></div>
<div class="pb-arrow">→</div>
<div class="pb-card"><div class="pb-cell"><div class="pb-nucleus"></div></div><h3>Inside a cell</h3><p>Brain cells from the cortex were split into parts.</p></div>
<div class="pb-arrow">→</div>
<div class="pb-card"><div class="pb-cell" style="border-color:var(--accent); background:var(--accent-soft)"><div class="pb-nucleus" style="width:62px;height:62px"></div></div>
<h3>The nucleus</h3><p>The cell's control center. That's the <b>_N</b> at the end of every protein name.</p></div>
</div>
""")

        # ---------- 06 the proteins ----------
        intro_section("06", "The proteins", "What are the proteins?",
                      "Proteins are the tiny machines that do the work inside cells. "
                      "How much of each one there is tells us what the brain is busy doing. 77 were measured.")
        html("""
<div class="pb-card pb-scroll">
<h3>How to read a protein name</h3>
<div class="pb-name"><span class="pb-seg pb-s2">p</span><span class="pb-seg pb-s1">ERK</span><span class="pb-seg pb-s3">_N</span></div>
<div class="pb-name-parts">
<div class="pb-s2"><b>p</b>“phosphorylated”: the <b>switched-on</b> version. Not every name has it.</div>
<div class="pb-s1"><b>ERK</b>The protein's name.</div>
<div class="pb-s3"><b>_N</b>Measured in the <b>nucleus</b> of cortex cells.</div>
</div></div>
<p class="pb-small pb-scroll" style="margin:28px 0 14px">The proteins fall into a few families. <b>Hover a card</b> to flip it.</p>
<div class="pb-row pb-row-4 pb-scroll">
<div class="pb-flip"><div class="pb-flip-in">
<div class="pb-face pb-front"><h3>Learning receptors</h3>
<div class="pb-chips" style="justify-content:center"><span class="pb-chip blue">NR1</span><span class="pb-chip blue">NR2A</span><span class="pb-chip blue">NR2B</span><span class="pb-chip blue">GluR3</span><span class="pb-chip blue">GluR4</span></div><small>hover me</small></div>
<div class="pb-face pb-back"><span>The <b>doors</b> on brain cells that let signals in. Learning depends on them, and the NR ones are exactly what <b>memantine</b> acts on.</span></div>
</div></div>
<div class="pb-flip"><div class="pb-flip-in">
<div class="pb-face pb-front"><h3>Signal switches</h3>
<div class="pb-chips" style="justify-content:center"><span class="pb-chip blue">ERK</span><span class="pb-chip blue">AKT</span><span class="pb-chip blue">mTOR</span><span class="pb-chip blue">P70S6</span></div><small>hover me</small></div>
<div class="pb-face pb-back"><span>A <b>chain of switches</b> that passes the message from the door to the inside of the cell. Their “p” forms show which switches are on.</span></div>
</div></div>
<div class="pb-flip"><div class="pb-flip-in">
<div class="pb-face pb-front"><h3>Extra-copy genes</h3>
<div class="pb-chips" style="justify-content:center"><span class="pb-chip">DYRK1A</span><span class="pb-chip">SOD1</span><span class="pb-chip">APP</span><span class="pb-chip">ITSN1</span></div><small>hover me</small></div>
<div class="pb-face pb-back"><span>Made from genes on the <b>extra DNA chunk</b>. Trisomic mice may simply make more of them, which makes them suspects for the learning problems.</span></div>
</div></div>
<div class="pb-flip"><div class="pb-flip-in">
<div class="pb-face pb-front"><h3>Brain support</h3>
<div class="pb-chips" style="justify-content:center"><span class="pb-chip blue">BDNF</span><span class="pb-chip blue">GFAP</span></div><small>hover me</small></div>
<div class="pb-face pb-back"><span><b>BDNF</b> helps brain cells grow and form new connections. <b>GFAP</b> marks the helper cells that keep neurons healthy.</span></div>
</div></div>
</div>
""")

        # ---------- Brain in the corner, reacting to the picked group ----------
        mood = outcome(picked) if revealed else "waiting"
        html(f'<div class="pb-corner"><div class="pb-r-{mood}">{mouse_img("brain.png")}</div></div>')


# ---------------- Intro (title and team) ----------------

TEAM = [
    # name, hijab colour (None = short hair and beard), skin, circle colour, extra detail
    ("Luluh", "#1e1e26", "#e9c19c", "#c9d0f4", "goggles"),
    ("Aleen", "#1e1e26", "#d6a47c", "#f6c5d3", None),
    ("Reema", "#1e1e26", "#f0cfae", "#bfe4d3", "glasses"),
    ("Khalid", None, "#cf9a72", "#f3dca9", None),
]

# 20 x 20 pixel portraits in white lab coats. One letter per pixel; "." is empty.
PX_HIJAB = [
    "....................",
    ".......HHHHHH.......",
    ".....HHHHHHHHHH.....",
    "....HHHHHHHHHHHH....",
    "...HHHHHHHHHHHHHH...",
    "...HHHHSSSSSSHHHH...",
    "...HHHSSSSSSSSHHH...",
    "...HHHSSESSESSHHH...",
    "...HHHSCSSSSCSHHH...",
    "...HHHSSSMMSSSHHH...",
    "...HHHHSSSSSSHHHH...",
    "...HHHHHHHHHHAAHH...",
    "..WhHHHHHHHHHHHHhW..",
    ".WWWhHHHHHHHHHHhWWW.",
    "WWWWWLhHHHHHHhLWWWWW",
    "WWWWWWLhHHHHhLWWWWWW",
    "WWWWWWWLhHHhLWWWPWWW",
    "WWWWWWWWLhhLWWWOOOWW",
    "WWWWWWWWWLLWWWWOOOWW",
    "wWWWWWWWWLLWWWWWWWWw",
]
PX_MAN = [
    "....................",
    "......RRRRRRR.......",
    ".....RRRRRRRRRR.....",
    "....RRRRRRRRRRRR....",
    "....RRRSSSSSSRRR....",
    "....RRSSSSSSSSRR....",
    ".....sSSSSSSSSs.....",
    ".....sSESSSSESs.....",
    ".....bSCSssSCSb.....",
    "......SSbbbbSS......",
    "......bSSMMSSb......",
    ".......bbbbbb.......",
    "........ssss........",
    "..WWWWWwBBBBwWWWWW..",
    ".WWWWWWLBTTBLWWWWWW.",
    ".WWWWWWWLTTLWWWWWWW.",
    "WWWWWWWWLTTLWWWWPWWW",
    "WWWWWWWWWLLWWWWOOOWW",
    "WWWWWWWWWLLWWWWOOOWW",
    "wWWWWWWWWLLWWWWWWWWw",
]
PX_EXTRA = {
    # safety goggles pushed up on the forehead
    "goggles": {3: "....HHGGGGGGGGHH....", 4: "...HHHGggGGggGHHH..."},
    # round-ish glasses: frame (F) around tinted lenses (g)
    "glasses": {6: "...HHHSFFSSFFSHHH...",
                7: "...HHHFgEFFEgFHHH...",
                8: "...HHHCFFSSFFCHHH..."},
}


def _shade(hex_colour, f=0.72):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))


def _mix(a, b, t):
    """Blend colour a towards colour b by t (0 = all a, 1 = all b)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def avatar(i, hijab, skin, bg, extra=None, hair="#2b2220", delay=0.0):
    """A pixel-art portrait in a white lab coat, drawn as crisp SVG squares."""
    rows = list(PX_HIJAB if hijab else PX_MAN)
    for y, row in PX_EXTRA.get(extra, {}).items():
        rows[y] = row
    pin = _shade(bg, .92)      # small hijab pin in each person's circle colour
    frame = _shade(bg, .42)    # glasses frame: a deep version of the circle colour
    colours = {
        # black hijab: folds are a soft lighter grey so the shape still reads
        "H": hijab, "h": _mix(hijab, "#ffffff", .2) if hijab else None,
        "S": skin, "s": _shade(skin, .85),
        "E": "#1d2340", "M": "#b8475a", "C": "#e79a9c", "R": hair,
        "b": _mix(hair, skin, .4),  # light beard / stubble
        "A": pin, "F": frame,
        "W": "#ffffff", "w": "#dfe2ec", "L": "#9aa2bb", "O": "#c7ccdb", "P": "#d9557a",
        "B": "#cdd7f6", "T": "#4b5bb3", "G": "#f2b84b", "g": "#bfe3f2", "Q": "#f7f7f7",
    }
    rects = ""
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            ch, run = row[x], 1
            while x + run < len(row) and row[x + run] == ch:
                run += 1
            if ch != ".":
                # pixels appear top to bottom, left to right, like the portrait is being printed
                d = delay + y * 0.028 + x * 0.006
                cls = "px eye" if ch == "E" else "px"
                rects += (f'<rect class="{cls}" style="animation-delay:{d:.3f}s" x="{x}" y="{y}" width="{run}" '
                          f'height="1.04" fill="{colours[ch]}"/>')
            x += run
    return (f'<svg viewBox="0 0 120 120" aria-hidden="true" shape-rendering="crispEdges">'
            f'<defs><clipPath id="av{i}"><circle cx="60" cy="60" r="55"/></clipPath></defs>'
            f'<circle cx="60" cy="60" r="55" fill="{bg}"/>'
            f'<g clip-path="url(#av{i})"><g transform="translate(5 9) scale(5.5)">{rects}</g></g>'
            f'<circle cx="60" cy="60" r="56.5" fill="none" stroke="{_shade(bg, .78)}" stroke-width="5" shape-rendering="geometricPrecision"/>'
            '</svg>')


def render_title():
    html(f"""
<div class="tt-hero pb-rise">
<span class="tt-pill">Mini project · Unsupervised learning</span>
<h1>Pondering the <em>clusters</em></h1>
<p>K-Means before and after PCA, on the Mice Protein Expression dataset.</p>
{mouse_img("pinky.png", "tt-pinky pb-float")}{mouse_img("brain.png", "tt-brain pb-float late")}
</div>
<div class="tt-team-label">The team</div>
""")
    cards = "".join(
        f'<div class="tt-member" style="animation-delay:{0.15 + i * 0.12:.2f}s">{avatar(i, h, sk, bg, ex, delay=0.3 + i * 0.12)}<b>{name}</b></div>'
        for i, (name, h, sk, bg, ex) in enumerate(TEAM))
    html(f'<div class="tt-team">{cards}</div>')


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
def run_hierarchical(X_pca, k, linkage):
    labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X_pca)
    return labels, float(silhouette_score(X_pca, labels))


names_k_all = [f"Cluster {i + 1}" for i in range(10)]


def r3(a):
    return [round(float(v), 3) for v in a]


# ---------------- Header: title, live results, page menu and the two controls ----------------

if "course" not in st.session_state:
    st.session_state.course = COURSES[0]

with st.container(key="topbar"):
    b1, b2 = st.columns([1, 1], vertical_alignment="center")
    b1.markdown('<div class="pc-brand">Pondering the Clusters <small>K-Means before and after PCA</small></div>',
                unsafe_allow_html=True)
    live = b2.empty()
    n1, n2, n3 = st.columns([3.55, 0.68, 0.85], vertical_alignment="bottom", gap="medium")
    with n1:
        with st.container(key="nav"):
            course = st.segmented_control("Page", COURSES, key="course", required=True,
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

# the motion layer: installed once into the page itself (see components/motion.js)
script("const w = window.parent; if (!w.__pcFx) { const s = w.document.createElement('script'); s.textContent = "
       + json.dumps(MOTION_JS) + "; w.document.head.appendChild(s); }", "motion")

# a new page opens at the top, and its blocks come in one after another
if st.session_state.get("last_course") != course:
    st.session_state.last_course = course
    script("/* " + course + " */ const d = window.parent.document; window.parent.scrollTo(0, 0);"
           "d.querySelectorAll('[data-testid=\"stMain\"], [data-testid=\"stAppViewContainer\"], section.main')"
           ".forEach(e => e.scrollTo(0, 0)); try { window.parent.__pcEnter && window.parent.__pcEnter(); } catch (e) {}",
           "scrolltop")

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


# ---------------- Intro ----------------

if course == "Intro":
    render_title()

# ---------------- The experiment ----------------

elif course == "The experiment":
    render_intro()

# ---------------- The plan ----------------

elif course == "The plan":
    gap(28)
    spread = np.percentile(np.abs(X_pca), 99, axis=0) * 1.08
    hero_xy = np.clip(X_pca / spread, -1.06, 1.06)
    show("hero", {
        "x": r3(hero_xy[:, 0]), "y": r3(hero_xy[:, 1]),
        "c": [int(c) for c in clusters_after], "palette": PALETTE,
        "stats": [
            {"value": len(df), "label": "samples"},
            {"value": X.shape[1], "label": "proteins"},
            {"value": int(df["MouseID"].str.split("_").str[0].nunique()), "label": "mice"},
            {"value": round(kept * 100, 1), "dec": 1, "suffix": "%", "label": "kept by PCA"},
        ],
    }, 640)

    with st.container(key="band-menu"):
        menu = [
            ("I", "The experiment", "The mice, the drug, the learning test"),
            ("II", "Lab prep", "Duplicates, missing values and scaling"),
            ("III", "Before PCA", "K-Means on all 77 proteins"),
            ("IV", "Brain's shortcut", "PCA keeps the two strongest directions"),
            ("V", "After PCA", "K-Means again, on two components"),
            ("VI", "The verdict", "Silhouette scores and the four questions"),
            ("VII", "Lab tests", "More components, and hierarchical clustering"),
        ]
        rows = "".join(
            f'<div class="row {"bonus" if no == "VII" else ""}" style="--i:{i}"><span class="no">{no}</span><span class="name">{name}</span>'
            f'<span class="dots"></span><span class="desc">{desc}</span></div>' for i, (no, name, desc) in enumerate(menu))
        html(f"""<div class="pc-menu"><div class="top"><div class="lab">Tonight's plan to take over the world</div>
<h3>Six steps <em>and a bonus</em></h3><p>One step of the project on every page. Walk them in order with the
menu above, or with the button at the foot of each page.</p></div>{rows}</div>""")
        # one hidden button per row: clicking a row presses its button (see the motion script)
        with st.container(key="plango"):
            for i, (no, name, desc) in enumerate(menu):
                st.button(name, key=f"plango_{i}", on_click=go_to, args=(name,))

    gap(96)
    html("""
<div class="pc-spread">
  <div>
    <div class="pc-rows-title">Tonight's test subjects</div>
    <p class="pc-big">72 mice, measured 15 times each, and <em>77 proteins</em> from every brain. Can K-Means
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
    a, b = st.columns([1.05, 1], gap="large")
    with a:
        html("""
<div class="pc-rows-title">The scheme, translated</div>
<ul class="pc-rows">
  <li><span>The test subjects</span><b>77 proteins</b></li>
  <li><span>Lab prep</span><b>Missing values and scaling</b></li>
  <li><span>Brain's shortcut</span><b>PCA, 77 → 2</b></li>
  <li><span>The master plan</span><b>K-Means clusters</b></li>
  <li><span>Did the plan work?</span><b>Silhouette score</b></li>
</ul>""")
    with b:
        gap(34)
        note("Brain's note", "A genius doesn't read every book in the library, Pinky. PCA is my shortcut: it keeps "
             "the few directions that matter most, and we check whether the plan gets better.", who="brain")

# ---------------- Lab prep ----------------

elif course == "Lab prep":
    section("02", "Step 2 · Preprocessing", "Lab prep", "preprocessing",
            "Before any scheme, Brain checks the lab supplies: are there duplicates, what is "
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
        "title": "The specimen shelf: missing values per protein",
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
        for box, values, color, name in [(a, X_filled[:, col], GOLD, "Before scaling"),
                                          (b, X_scaled[:, col], PINK, "After scaling")]:
            counts, edges = np.histogram(values, bins=24)
            centers = (edges[:-1] + edges[1:]) / 2
            with box:
                show("chart", {
                    "key": f"hist_{name}", "type": "line", "title": name,
                    "subtitle": f"Mean {abs(values.mean()):.3f} · Std {values.std():.3f}",
                    "x": [f"{c:.2f}" for c in centers], "xName": protein, "xLabel": protein,
                    "bars": {"values": [int(c) for c in counts], "color": color, "name": "Samples", "fmt": "int"},
                    "yFmt": "int", "tickDec": 0,
                }, 340)
        gap(16)
        with st.expander("Peek at the raw data (first 10 rows)"):
            st.dataframe(df.head(10), hide_index=True)

# ---------------- Before PCA ----------------

elif course == "Before PCA":
    section("03", "Step 3 · K-Means before PCA", "The first attempt", "clustering before PCA",
            "K-Means on the original 77 proteins, after scaling. First we look for a good number of clusters.")
    inertia, scores = elbow(X_scaled)
    a, b = st.columns(2, gap="medium")
    with a:
        show("chart", {
            "key": "elbow", "type": "line", "title": "Elbow method", "subtitle": "Inertia for k = 1 to 10",
            "x": list(range(1, 11)), "xName": "k =", "xLabel": "Number of clusters",
            "series": [{"name": "Inertia", "values": inertia, "color": PINK, "fmt": "int"}],
            "yFmt": "k", "hi": k - 1, "hiLabel": f"k = {k}", "area": True,
        }, 360)
    with b:
        show("chart", {
            "key": "silk", "type": "line", "title": "Silhouette score for each k", "subtitle": "Higher means better separated",
            "x": list(range(1, 11)), "xName": "k =", "xLabel": "Number of clusters",
            "series": [{"name": "Silhouette", "values": scores, "color": GREEN}],
            "hi": k - 1, "hiLabel": f"k = {k}", "tickDec": 2, "area": True,
        }, 360)
    gap(40)
    note("Choosing k", "There is no clear elbow: the inertia keeps falling slowly. The dataset has 8 mouse groups, "
         "so we go with <b>k = 8</b>. Change k in the header and every page follows.")

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
                "key": "sizes", "type": "line", "title": "Cluster sizes", "subtitle": "Samples in each K-Means cluster",
                "x": [f"C{i + 1}" for i in range(k)], "xName": "Cluster",
                "bars": {"values": [int(s) for s in sizes], "colors": PALETTE[:k], "name": "Samples", "fmt": "int", "labels": True},
                "yFmt": "int", "tickDec": 0,
            }, 400)

# ---------------- Brain's shortcut ----------------

elif course == "Brain's shortcut":
    section("04", "Step 4 · Applying PCA", "Brain's shortcut", "applying PCA",
            "Brain doesn't need all 77 proteins to see the pattern. PCA does the same: it turns 77 "
            "proteins into a few principal components that keep as much of the variance as possible.")
    figures([("PC1 variance", f"{var_2[0] * 100:.1f}%", ""), ("PC2 variance", f"{var_2[1] * 100:.1f}%", ""),
             ("Kept with 2 components", f"{kept * 100:.1f}%", ""), ("Lost", f"{(1 - kept) * 100:.1f}%", "")])
    cum = np.cumsum(var_all)
    show("chart", {
        "key": "variance", "type": "line", "title": "Explained variance", "subtitle": "Each component, and the running total",
        "x": list(range(1, 31)), "xName": "Component", "xLabel": "Number of components",
        "bars": {"values": r3(var_all[:30] * 100), "color": GOLD, "name": "This component", "fmt": "pct100", "dec": 1},
        "series": [{"name": "Running total", "values": r3(cum[:30] * 100), "color": PINK, "fmt": "pct100", "dec": 1}],
        "yFmt": "pct100", "yMin": 0, "yMax": 100, "hi": 1, "hiBelow": True, "hiLabel": f"2 components · {kept * 100:.1f}%",
        "legend": [{"name": "This component", "color": GOLD}, {"name": "Running total", "color": PINK}],
    }, 420)

    gap(40)
    a, b = st.columns([1, 1], gap="large", vertical_alignment="center")
    with a:
        target = st.slider("How much information do you want to keep? (%)", 40, 99, 80, key="target")
    needed = int(np.argmax(cum >= target / 100) + 1)
    with b:
        note("Brain says", f"To keep <b>{target}%</b> you need <b>{needed} components</b>. "
             f"We use only 2, so we keep <b>{kept * 100:.1f}%</b>.", who="brain")

    with st.container(key="band-loadings"):
        subhead("Brain's favorite proteins", "Loadings")
        lead("Which proteins build each component the most.")
        pc = st.segmented_control("Component", ["PC1", "PC2"], default="PC1", required=True, key="pc",
                                  label_visibility="collapsed")
        idx = 0 if pc == "PC1" else 1
        top = pd.Series(np.abs(loadings[idx]), index=X.columns).sort_values(ascending=False).head(10)
        gap(8)
        show("chart", {
            "key": "loadings", "type": "hbar", "title": f"Top 10 proteins in {pc}",
            "subtitle": "How strongly each protein builds this component (|loading|)",
            "labels": list(top.index), "values": r3(top.values), "xMax": float(top.values.max()) * 1.18,
            "color": PINK if idx == 0 else GREEN,
        }, 440)

# ---------------- After PCA ----------------

elif course == "After PCA":
    section("05", "Step 5 · K-Means after PCA", "The second attempt", "clustering after PCA",
            "K-Means again, this time on just the two PCA components. Switch the colors and the view: "
            "every point is one sample, and it glides to its new place.")
    figures([("Silhouette before PCA", f"{score_before:.3f}", ""),
             ("Silhouette after PCA", f"{score_after:.3f}", f"{score_after - score_before:+.3f}"),
             ("Features used", "77 → 2", "")])
    names_k = [f"Cluster {i + 1}" for i in range(k)]
    classes = sorted(df["class"].unique())
    modes = [
        {"id": "after", "label": "After PCA", "values": [int(c) for c in clusters_after], "names": names_k, "colors": PALETTE[:k]},
        {"id": "before", "label": "Before PCA", "values": [int(c) for c in clusters_before], "names": names_k, "colors": PALETTE[:k]},
        {"id": "class", "label": "Class", "values": [classes.index(c) for c in df["class"]], "names": classes, "colors": PALETTE[:8]},
        {"id": "genotype", "label": "Genotype", "values": [0 if g == "Control" else 1 for g in df["Genotype"]],
         "names": ["Control", "Trisomy"], "colors": [NAVY, PINK]},
        {"id": "treatment", "label": "Treatment", "values": [0 if t == "Memantine" else 1 for t in df["Treatment"]],
         "names": ["Memantine", "Saline"], "colors": [GREEN, GOLD]},
        {"id": "behavior", "label": "Behavior", "values": [0 if b == "C/S" else 1 for b in df["Behavior"]],
         "names": ["C/S (learned)", "S/C (not stimulated)"], "colors": [PINK, NAVY]},
    ]
    notes = {
        "after|p12": "K-Means drew these borders on this exact 2D map, so the clusters sit side by side like countries on a world map.",
        "after|p23": "On PC2 × PC3 these clusters look mixed: they were cut on PC1 and PC2, not on this view.",
        "before|p12": "These clusters were made in 77 dimensions. Squashed onto two, they overlap, because their borders live in dimensions we can't see.",
        "class": "The eight real mouse groups are mixed together. That's why no clustering finds them cleanly.",
        "genotype": "Control and trisomy mice overlap almost completely on these components.",
        "treatment": "Memantine and saline mice overlap too.",
        "behavior|p12": "See the top and bottom halves? Now switch the view to PC2 × PC3.",
        "behavior|p23": "There it is: the two behavior groups split along PC2. PCA found the learning signal without ever seeing the labels.",
    }
    show("scatter", {
        "key": "world", "title": "The world map", "subtitle": "1,080 samples on the PCA map. Hover a point or a group.",
        "ids": list(df["MouseID"]), "cls": list(df["class"]),
        "x": {"pc1": r3(X_pca[:, 0]), "pc2": r3(X_pca[:, 1]), "pc3": r3(pc3)},
        "modes": modes, "mode": "after",
        "views": [{"id": "p12", "label": "PC1 × PC2", "x": "pc1", "y": "pc2", "xl": "PC1", "yl": "PC2"},
                  {"id": "p23", "label": "PC2 × PC3", "x": "pc2", "y": "pc3", "xl": "PC2", "yl": "PC3"}],
        "view": "p12", "notes": notes,
        "tour": [{"mode": "after", "view": "p12"}, {"mode": "before", "view": "p12"},
                 {"mode": "behavior", "view": "p12"}, {"mode": "behavior", "view": "p23"}],
    }, 740)
    gap(48)
    a, b = st.columns(2, gap="large")
    with a:
        note("Before PCA", "The clusters were made with all 77 proteins, so on a 2D plot they look mixed together.")
    with b:
        note("After PCA", "The clusters were made on the same 2D space we plot, so they form clean groups.")

# ---------------- The verdict ----------------

elif course == "The verdict":
    section("06", "Step 6 · Comparison and analysis", "The verdict", "comparison and analysis",
            "Both attempts are in. Did the plan work? Here is the verdict, and our answers to the four questions.")
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
        hc_ward = run_hierarchical(X_pca, k, "ward")[1]
        insights = [
            f"No duplicates. Only {missing_pct:.2f}% of the data was missing, so we filled it with the {strategy} and scaled the data.",
            f"K-Means before PCA gave a low silhouette score ({score_before:.2f}).",
            f"PCA with 2 components kept {kept * 100:.0f}% of the variance.",
            f"K-Means after PCA gave a higher silhouette score ({score_after:.2f}) and a clearer plot.",
            "PCA makes clustering faster and easier to see, but we lose some information.",
            f"Hierarchical clustering scored {hc_ward:.2f}, still below K-Means after PCA (see the lab tests).",
        ]
        html('<ol class="pc-list">' + "".join(f"<li>{t}</li>" for t in insights) + "</ol>")

# ---------------- Lab tests ----------------

elif course == "Lab tests":
    section("07", "Bonus", "Lab tests", "bonus experiments",
            "Two side experiments from the back of the lab: more PCA components, and a second algorithm, hierarchical clustering.")

    subhead("Different number of PCA components", "Experiment 1")
    gap(8)
    rows = components_test(X_scaled, k)
    show("chart", {
        "key": "components", "type": "line", "title": f"Silhouette score vs components (k = {k})",
        "subtitle": "More components keep more information, but the clusters get less separated",
        "x": [r[0] for r in rows], "xName": "Components:", "xLabel": "Number of PCA components",
        "series": [{"name": "Silhouette score", "values": [r[1] for r in rows], "color": PINK},
                   {"name": "Information kept", "values": [r[2] for r in rows], "color": GREEN, "fmt": "pct", "dec": 1}],
        "yMin": 0, "yMax": 1, "tickDec": 2, "area": True,
        "legend": [{"name": "Silhouette score", "color": PINK}, {"name": "Information kept", "color": GREEN}],
    }, 400)
    gap(40)
    note("Result", "More components → more information but a lower silhouette score. 2 components give the "
         "cleanest clusters, but they keep the least information.")

    with st.container(key="band-hier"):
        subhead("Another algorithm: hierarchical clustering", "Experiment 2")
        lead("Agglomerative clustering works bottom-up: every sample starts as its own cluster, and the two "
             "closest clusters are merged again and again until only k are left. We run it on the PCA data, "
             "with the same k as K-Means.")
        linkage = st.segmented_control("Linkage: how “closest” is measured", ["ward", "average", "complete"],
                                       default="ward", required=True, key="linkage")
        html('<p class="pc-small" style="margin:8px 0 24px">ward merges the pair that adds the least spread (the default) · '
             'average uses the mean distance between two clusters · complete uses their two farthest points</p>')
        hc_labels, hc_score = run_hierarchical(X_pca, k, linkage)
        figures([("Clusters", k, ""), ("Hierarchical silhouette", f"{hc_score:.3f}", ""),
                 ("K-Means silhouette", f"{score_after:.3f}", "")])
        show("scatter", {
            "key": "hier", "title": f"Hierarchical clustering ({linkage} linkage, k = {k})",
            "subtitle": "Switch to K-Means to compare the two on the same PCA map.",
            "ids": list(df["MouseID"]), "cls": list(df["class"]),
            "x": {"pc1": r3(X_pca[:, 0]), "pc2": r3(X_pca[:, 1])},
            "modes": [{"id": "hc", "label": "Hierarchical", "values": [int(c) for c in hc_labels],
                       "names": names_k_all[:k], "colors": PALETTE[:k]},
                      {"id": "km", "label": "K-Means", "values": [int(c) for c in clusters_after],
                       "names": names_k_all[:k], "colors": PALETTE[:k]}],
            "mode": "hc",
            "views": [{"id": "p12", "label": "PC1 × PC2", "x": "pc1", "y": "pc2", "xl": "PC1", "yl": "PC2"}],
            "notes": {},
        }, 600)
        gap(40)
        better = hc_score > score_after
        note("Result",
             f"With <b>{linkage}</b> linkage, hierarchical clustering scores <b>{hc_score:.3f}</b> against "
             f"<b>{score_after:.3f}</b> for K-Means. "
             + ("Here it does better than K-Means." if better else
                "It finds sensible groups, but <b>K-Means is still the better choice</b> for this data."))

# ---------------- The game ----------------

elif course == "The game":
    section("08", "Bonus game", "Take over the world", "a puzzle game",
            "Four quick puzzles and one big red button, all built from this project: the lab door, the protein "
            "sequence, the power grid and a silhouette score. Help the Brain finish tonight's plan. Five minutes, tops.")
    show("game", {
        "brain": _img_src("brain.png"), "pinky": _img_src("pinky.png"),
        # the door code comes from the project itself: mouse groups, PCA components, team members
        "code": [int(df["class"].nunique()), 2, len(TEAM)],
        "clues": ["How many groups of mice are in the data?", "How many components does PCA keep?",
                  "How many people made this project?"],
        "beakers": [{"name": "NR1", "color": PINK}, {"name": "ERK", "color": GREEN},
                    {"name": "BDNF", "color": GOLD}, {"name": "SOD1", "color": NAVY}],
        "colors": [PINK, NAVY],
        "kept": f"{kept * 100:.1f}", "lost": f"{(1 - kept) * 100:.1f}",
    }, 720)
    gap(40)
    note("Pinky's note", "Brain says nothing can go wrong this time. He says that every night. Narf!", who="pinky")

next_prev()