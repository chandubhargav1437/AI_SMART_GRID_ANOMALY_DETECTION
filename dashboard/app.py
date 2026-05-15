"""
Phase 5 — Real-time Streamlit Dashboard
Premium UI — Deep Navy theme with high-contrast text and glowing accents.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
import sys
import numpy as np
from pathlib import Path

sys.path.append('.')
from engine.response import get_response, get_severity, FAULT_LABELS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Grid Monitor — GE Vernova",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #060d1f !important;
    color: #e8edf5 !important;
}

/* ── Main area ── */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    background-color: #060d1f !important;
}

/* ── Headings ── */
h1 { color: #ffffff !important; font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { color: #e8edf5 !important; font-weight: 700 !important; }
h3 { color: #c9d4e8 !important; font-weight: 600 !important; }
p, span, label, div { color: #c9d4e8 !important; }

/* ── Caption / subtitle ── */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #7b9cbf !important;
    font-size: 0.85rem !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1e3d 0%, #0a1628 100%) !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 0 20px rgba(0, 180, 255, 0.08) !important;
}
[data-testid="stMetricLabel"] p {
    color: #7bb3d9 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f22 0%, #060c1a 100%) !important;
    border-right: 1px solid #1a2d50 !important;
}
[data-testid="stSidebar"] * { color: #d0ddf0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    border-bottom: 1px solid #1a3560 !important;
    padding-bottom: 8px !important;
}
[data-testid="stSidebarContent"] [data-testid="stMetricLabel"] p { color: #88aacc !important; }
[data-testid="stSidebarContent"] [data-testid="stMetricValue"]   { color: #00d4ff !important; font-size: 1.2rem !important; }

/* ── Divider ── */
hr { border-color: #1a2d50 !important; margin: 1.2rem 0 !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { color: #a0b8d8 !important; font-weight: 600 !important; }
[data-testid="stSelectbox"] > div > div {
    background: #0d1e3d !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0066ff 0%, #00aaff 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(0, 160, 255, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 25px rgba(0, 160, 255, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Info box ── */
[data-testid="stInfo"] {
    background: #0a1e3a !important;
    border: 1px solid #1a3560 !important;
    border-radius: 8px !important;
    color: #a0c4e8 !important;
}
[data-testid="stInfo"] p { color: #a0c4e8 !important; }

/* ── Alert / Error / Success boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-weight: 500 !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: rgba(255, 50, 50, 0.12) !important;
    border: 1px solid rgba(255, 80, 80, 0.4) !important;
    color: #ff8080 !important;
}
[data-testid="stAlert"][data-type="error"] p { color: #ff8080 !important; }
[data-testid="stAlert"][data-type="success"] {
    background: rgba(0, 220, 120, 0.10) !important;
    border: 1px solid rgba(0, 220, 120, 0.35) !important;
}
[data-testid="stAlert"][data-type="success"] p { color: #00e87a !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0a1628 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #1a2d50 !important;
}
[data-testid="stTabs"] [role="tab"] {
    color: #7b9cbf !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 7px !important;
    padding: 8px 18px !important;
    border: none !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #0066ff 0%, #0099dd 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(0, 120, 255, 0.4) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    color: #d0e4f5 !important;
    background: #0d1e3d !important;
}

/* ── Subheader ── */
[data-testid="stSubheader"] h2,
[data-testid="stSubheader"] h3 { color: #ffffff !important; }

/* ── Markdown text ── */
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: #c9d4e8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  <div style="background:linear-gradient(135deg,#0066ff,#00ccff);
              border-radius:12px;padding:10px 14px;
              box-shadow:0 0 24px rgba(0,160,255,0.5);">
    <span style="font-size:1.8rem;">⚡</span>
  </div>
  <div>
    <h1 style="margin:0;color:#ffffff;font-size:1.9rem;font-weight:800;">
      Smart Grid Anomaly Detection
      <span style="color:#00aaff;">— Live Monitor</span>
    </h1>
    <p style="margin:2px 0 0;color:#5a8ab0;font-size:0.85rem;font-weight:500;">
      GE Vernova Grid Automation &nbsp;·&nbsp; IEC 61850 &nbsp;·&nbsp; Two-Stage ML Pipeline
    </p>
  </div>
</div>
<hr style="border:1px solid #0f2040;margin:18px 0 10px;">
""", unsafe_allow_html=True)

# ── Load data & models ─────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent

@st.cache_data
def load_data():
    return pd.read_csv(BASE / "data" / "grid_data_detected.csv", parse_dates=["timestamp"])

@st.cache_resource
def load_models():
    iso = joblib.load(BASE / "models" / "saved" / "isolation_forest.pkl")
    xgb = joblib.load(BASE / "models" / "saved" / "xgboost_classifier.pkl")
    return iso, xgb

df = load_data()
iso_model, xgb_model = load_models()

FEATURE_EXCLUDE = ['timestamp', 'fault_type', 'is_anomaly', 'anomaly_score', 'confidence']
features = [c for c in df.columns if c not in FEATURE_EXCLUDE]

if 'confidence' not in df.columns:
    anomaly_mask = df['is_anomaly'] == -1
    confidence_col = np.full(len(df), 0.0)
    if anomaly_mask.any():
        X_anom = df.loc[anomaly_mask, features].values
        proba = xgb_model.predict_proba(X_anom)
        confidence_col[anomaly_mask.values] = proba.max(axis=1)
    df['confidence'] = confidence_col

# ── Sidebar ───────────────────────────────────────────────────────────────────
anomaly_count = int((df['is_anomaly'] == -1).sum())
fault_rate     = anomaly_count / len(df)

st.sidebar.markdown("""
<h2 style="font-size:1.1rem;font-weight:700;letter-spacing:0.3px;margin-bottom:14px;">
  📊 System Metrics
</h2>""", unsafe_allow_html=True)

col_a, col_b = st.sidebar.columns(2)
col_a.metric("Rows Monitored",   f"{len(df):,}")
col_b.metric("Anomalies Found",  f"{anomaly_count:,}")
col_c, col_d = st.sidebar.columns(2)
col_c.metric("Fault Rate",       f"{fault_rate:.1%}")
col_d.metric("Contamination",    "10.0%")

st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,#082040,#041630);
            border:1px solid #1a3560;border-radius:10px;
            padding:12px 14px;margin:12px 0;">
  <p style="color:#00d4ff !important;font-weight:700;font-size:0.8rem;
             text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
    Stage 2 — XGBoost
  </p>
  <p style="color:#ffffff !important;font-size:1.3rem;font-weight:800;margin:0;">
    99.86% F1
  </p>
  <p style="color:#5a8ab0 !important;font-size:0.75rem;margin:2px 0 0;">
    Precision · Recall · F1 all ≥ 99%
  </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("""
<p style="color:#7b9cbf !important;font-size:0.78rem;font-weight:700;
          text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">
  🔌 Fault Severity Map
</p>""", unsafe_allow_html=True)

SEV_COLORS = {"CRITICAL": ("#ff4060", "🔴"), "HIGH": ("#ff8c00", "🟠"), "MEDIUM": ("#ffd700", "🟡")}
for ft, label in FAULT_LABELS.items():
    if ft == 0:
        continue
    sev = get_severity(ft)
    clr, emoji = SEV_COLORS.get(sev, ("#aaffaa", "🟢"))
    st.sidebar.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 10px;margin:3px 0;background:#0a1628;border-radius:7px;'
        f'border:1px solid #1a2d50;">'
        f'<span style="color:#d0e4f5 !important;font-weight:600;">{emoji} {label}</span>'
        f'<span style="color:{clr} !important;font-size:0.75rem;font-weight:700;">{sev}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Stream controls ───────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
with c1:
    stream_speed = st.selectbox("⚙️ Stream Speed",
        ["Fast (0.01s)", "Normal (0.03s)", "Slow (0.1s)"], index=1)
with c2:
    start_btn = st.button("▶ Start Live Stream", type="primary", use_container_width=True)
with c3:
    st.info("📡 Click **Start Live Stream** to begin real-time grid monitoring.")

speed_map  = {"Fast (0.01s)": 0.01, "Normal (0.03s)": 0.03, "Slow (0.1s)": 0.1}
sleep_time = speed_map[stream_speed]

PLOTLY_LAYOUT = dict(
    paper_bgcolor='#060d1f',
    plot_bgcolor='#060d1f',
    font=dict(color='#c9d4e8', family='Inter'),
    xaxis=dict(gridcolor='#0f2040', linecolor='#1a2d50', tickfont=dict(color='#7b9cbf')),
    yaxis=dict(gridcolor='#0f2040', linecolor='#1a2d50', tickfont=dict(color='#7b9cbf')),
    title_font=dict(color='#ffffff', size=15),
    legend=dict(bgcolor='#0a1628', bordercolor='#1a2d50', borderwidth=1,
                font=dict(color='#c9d4e8')),
)

# ── Live Stream area ──────────────────────────────────────────────────────────
placeholder       = st.empty()
alert_placeholder = st.empty()

if start_btn:
    for i in range(10, len(df), 10):
        chunk = df.iloc[:i].copy()

        fig = go.Figure()
        normal_mask  = chunk['is_anomaly'] == 1
        anomaly_mask = chunk['is_anomaly'] == -1

        fig.add_trace(go.Scattergl(
            x=chunk.loc[normal_mask, 'timestamp'],
            y=chunk.loc[normal_mask, 'MMXU1.PhV.phsA'],
            mode='markers', name='Normal',
            marker=dict(color='#00c8ff', size=3, opacity=0.7),
        ))
        fig.add_trace(go.Scattergl(
            x=chunk.loc[anomaly_mask, 'timestamp'],
            y=chunk.loc[anomaly_mask, 'MMXU1.PhV.phsA'],
            mode='markers', name='Anomaly',
            marker=dict(color='#ff3860', size=5, opacity=0.9,
                        symbol='x', line=dict(width=1, color='#ff3860')),
        ))
        fig.update_layout(
            title='⚡ Live Voltage Reading — Phase A',
            xaxis_title='Timestamp',
            yaxis_title='Voltage (V)',
            height=370,
            margin=dict(l=50, r=20, t=50, b=45),
            **PLOTLY_LAYOUT
        )
        placeholder.plotly_chart(fig, use_container_width=True)

        latest = df.iloc[i - 1]
        if latest['is_anomaly'] == -1:
            fault    = int(latest['fault_type']) if latest['fault_type'] != 0 else 1
            conf     = float(latest.get('confidence', 0.85))
            action   = get_response(fault, conf)
            severity = get_severity(fault)
            sev_clr, sev_emoji = SEV_COLORS.get(severity, ("#aaffaa", "🟢"))
            alert_placeholder.error(
                f"{sev_emoji} **ANOMALY DETECTED** &nbsp;|&nbsp; "
                f"Severity: **{severity}** &nbsp;|&nbsp; Confidence: {conf:.0%}\n\n"
                f"📋 {action}"
            )
        else:
            alert_placeholder.success("✅ System Normal — All parameters within safe operating range")

        time.sleep(sleep_time)

    st.balloons()
    st.success("✅ Live stream complete. Full analysis below ↓")

# ── Static Analysis ───────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color:#ffffff;font-size:1.25rem;font-weight:700;margin:24px 0 12px;">
  📈 Full Dataset Analysis
</h3>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  📊  Fault Distribution  ",
                             "  ⚡  Voltage Profile  ",
                             "  🔁  Frequency Profile  "])

with tab1:
    fault_df = (df[df['is_anomaly'] == -1]['fault_type']
                .map(FAULT_LABELS).value_counts().reset_index())
    fault_df.columns = ['Fault Type', 'Count']
    fig2 = px.pie(
        fault_df, names='Fault Type', values='Count',
        title='Detected Fault Type Distribution',
        color_discrete_sequence=['#0077ff','#00ccff','#ff3860','#ffd700','#ff8c00','#00e87a'],
        hole=0.45,
    )
    fig2.update_traces(
        textposition='outside',
        textinfo='percent+label',
        textfont=dict(color='#ffffff', size=13),
        pull=[0.04]*len(fault_df),
    )
    fig2.update_layout(
        height=480,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(font=dict(color='#c9d4e8', size=13), bgcolor='#0a1628',
                    bordercolor='#1a2d50', borderwidth=1),
        title_font=dict(color='#ffffff', size=16),
        paper_bgcolor='#060d1f',
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    sample = df.sample(min(5000, len(df)), random_state=42).sort_values('timestamp')
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=sample['timestamp'], y=sample['MMXU1.PhV.phsA'],
        name='Phase A', line=dict(color='#00aaff', width=1.5), opacity=0.85))
    fig3.add_trace(go.Scatter(
        x=sample['timestamp'], y=sample['MMXU1.PhV.phsB'],
        name='Phase B', line=dict(color='#ffd700', width=1.5), opacity=0.85))
    fig3.update_layout(
        title='Voltage Phase A & B — Sample View',
        xaxis_title='Timestamp', yaxis_title='Voltage (V)',
        height=420, margin=dict(l=50, r=20, t=55, b=45),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=df.loc[df['is_anomaly'] == 1, 'MMXU1.Hz'],
        name='Normal', marker_color='#00aaff', opacity=0.8, nbinsx=80))
    fig4.add_trace(go.Histogram(
        x=df.loc[df['is_anomaly'] == -1, 'MMXU1.Hz'],
        name='Anomaly', marker_color='#ff3860', opacity=0.85, nbinsx=80))
    fig4.update_layout(
        barmode='overlay',
        title='Frequency Distribution — Normal vs Anomaly',
        xaxis_title='Frequency (Hz)', yaxis_title='Count',
        height=420, margin=dict(l=50, r=20, t=55, b=45),
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border:1px solid #0f2040;margin:28px 0 16px;'>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;color:#3a6080 !important;font-size:12px;letter-spacing:0.3px;">
  ⚡ Smart Grid Anomaly Detection &nbsp;·&nbsp; IEC 61850 &nbsp;·&nbsp;
  Isolation Forest + XGBoost &nbsp;·&nbsp; GE Vernova Grid Automation R&D
</p>""", unsafe_allow_html=True)
