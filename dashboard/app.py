import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

METRICS_FILE = Path("/app/metrics/server_metrics.json")
MODEL_FILE = Path("/app/model/final_model.pt")
NUM_ROUNDS = 20
REFRESH_INTERVAL = 5


def load_server_metrics():
    if not METRICS_FILE.exists():
        return []
    with open(METRICS_FILE) as f:
        return json.load(f)


def label(text):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:600;color:#a7afb7;letter-spacing:0.06em;"
        f"text-transform:uppercase;margin:0.6rem 0 0.2rem 0'>{text}</p>",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="FL Admin", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    :root {
        --bg: #111315;
        --panel: #181b1f;
        --border: #2b3138;
        --text: #f3f4f6;
        --muted: #a7afb7;
    }

    .stApp { background-color: var(--bg); color: var(--text); }

    .block-container {
        max-width: 1300px;
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    p, div, span, label { color: var(--text); }
    h1, h2, h3, h4, h5, h6 { display: none; }

    [data-testid="stCaptionContainer"] { color: var(--muted); font-size: 0.75rem !important; }

    [data-testid="metric-container"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.5rem 0.8rem !important;
    }

    [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: var(--muted) !important; }

    hr { margin: 0.5rem 0 !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

CHART_STYLE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f3f4f6", size=11),
    margin=dict(l=10, r=10, t=24, b=10),
    xaxis=dict(gridcolor="#2b3138", linecolor="#2b3138"),
    yaxis=dict(gridcolor="#2b3138", linecolor="#2b3138"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    height=190,
)

with st.sidebar:
    auto_refresh = st.toggle("Auto-refresh (5s)", value=True)

data = load_server_metrics()
srv = pd.DataFrame(data) if data else pd.DataFrame()
training_done = MODEL_FILE.exists()

if training_done:
    badge = "<span style='background:#14532d;color:#4ade80;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:600'>Done</span>"
elif not srv.empty:
    r = int(srv["round"].max())
    badge = f"<span style='background:#78350f;color:#fbbf24;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:600'>Round {r} / {NUM_ROUNDS}</span>"
else:
    badge = "<span style='background:#1e3a5f;color:#93c5fd;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:600'>Waiting</span>"

st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem'>
    <div>
        <span style='font-size:1rem;font-weight:700;color:#f3f4f6'>Admin Dashboard</span>
        <span style='font-size:0.75rem;color:#a7afb7;margin-left:0.75rem'>Global model metrics and client contribution scores. No client data is accessible here.</span>
    </div>
    {badge}
</div>
""", unsafe_allow_html=True)

st.divider()

label("Global model")

if srv.empty:
    st.caption("No data yet.")
else:
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Accuracy (latest)", f"{srv['global_eval_acc'].iloc[-1]:.2%}")
    with k2:
        st.metric("Eval loss (latest)", f"{srv['global_eval_loss'].iloc[-1]:.4f}")
    with k3:
        st.metric("Rounds completed", len(srv))

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=srv["round"], y=srv["global_eval_acc"],
            mode="lines+markers", name="accuracy",
            line=dict(color="#818cf8", width=2), marker=dict(size=4),
        ))
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Round", **CHART_STYLE)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=srv["round"], y=srv["global_eval_loss"],
            mode="lines+markers", name="eval loss",
            line=dict(color="#f472b6", width=2), marker=dict(size=4),
        ))
        fig.update_layout(xaxis_title="Round", **CHART_STYLE)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

label("Client contributions")
st.caption("Cosine similarity and L2 norm of each client's update. Low cosine sim or near-zero norm may indicate a free-rider.")

if not srv.empty:
    contrib_rounds = [
        (row["round"], row["client_contributions"])
        for _, row in srv.iterrows()
        if row["client_contributions"]
    ]

    if not contrib_rounds:
        st.caption("Will appear after the first training round.")
    else:
        latest_round, latest_contrib = contrib_rounds[-1]
        st.caption(f"Round {latest_round} (latest)")

        nodes = list(latest_contrib.keys())
        labels_x = [f"node {i+1}" for i in range(len(nodes))]
        cos_vals = [latest_contrib[n]["cosine_sim"] for n in nodes]
        norm_vals = [latest_contrib[n]["l2_norm"] for n in nodes]

        b1, b2, b3 = st.columns(3)
        with b1:
            fig = go.Figure(go.Bar(x=labels_x, y=cos_vals, marker_color="#4ade80"))
            fig.update_layout(title_text="Cosine similarity", yaxis_range=[-1, 1], **CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

        with b2:
            fig = go.Figure(go.Bar(x=labels_x, y=norm_vals, marker_color="#fbbf24"))
            fig.update_layout(title_text="L2 norm", **CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

        with b3:
            if len(contrib_rounds) > 1:
                fig = go.Figure()
                for i, node_id in enumerate(nodes):
                    rounds = [r for r, c in contrib_rounds if node_id in c]
                    sims = [c[node_id]["cosine_sim"] for r, c in contrib_rounds if node_id in c]
                    fig.add_trace(go.Scatter(
                        x=rounds, y=sims,
                        mode="lines+markers", name=f"node {i+1}",
                        line=dict(width=2), marker=dict(size=3),
                    ))
                fig.update_layout(title_text="Cosine sim history", xaxis_title="Round", yaxis_range=[-1, 1], **CHART_STYLE)
                st.plotly_chart(fig, use_container_width=True)

if auto_refresh and not training_done:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
