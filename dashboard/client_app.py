import json
import os
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

CLIENT_ID = os.environ.get("CLIENT_ID", "0")
METRICS_DIR = Path("/app/metrics")
MODEL_FILE = Path("/app/model/final_model.pt")
NUM_ROUNDS = 20
REFRESH_INTERVAL = 5


def load_json(path):
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def label(text):
    st.markdown(
        f"<p style='font-size:0.7rem;font-weight:600;color:#a7afb7;letter-spacing:0.06em;"
        f"text-transform:uppercase;margin:0.6rem 0 0.2rem 0'>{text}</p>",
        unsafe_allow_html=True,
    )


client_data = load_json(METRICS_DIR / f"client_{CLIENT_ID}_metrics.json")
server_data = load_json(METRICS_DIR / "server_metrics.json")
training_done = MODEL_FILE.exists()

evals = [r for r in client_data if r["type"] == "eval"]
trains = [r for r in client_data if r["type"] == "train"]

st.set_page_config(
    page_title=f"Client {CLIENT_ID}",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

if training_done:
    badge = "<span style='background:#14532d;color:#4ade80;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:600'>Done</span>"
elif server_data:
    r = server_data[-1]["round"]
    badge = f"<span style='background:#78350f;color:#fbbf24;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:600'>Round {r} / {NUM_ROUNDS}</span>"
else:
    badge = "<span style='background:#1e3a5f;color:#93c5fd;padding:2px 10px;border-radius:6px;font-size:0.75rem;font-weight:600'>Waiting</span>"

st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem'>
    <div>
        <span style='font-size:1rem;font-weight:700;color:#f3f4f6'>Client {CLIENT_ID}</span>
        <span style='font-size:0.75rem;color:#a7afb7;margin-left:0.75rem'>Your local training metrics and your contribution to the federation.</span>
    </div>
    {badge}
</div>
""", unsafe_allow_html=True)

st.divider()

if not client_data:
    st.caption("No data yet.")
else:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        if evals:
            st.metric("Eval accuracy", f"{evals[-1]['eval_acc']:.2%}")
    with k2:
        if evals:
            st.metric("Eval loss", f"{evals[-1]['eval_loss']:.4f}")
    with k3:
        if trains:
            st.metric("Train loss", f"{trains[-1]['train_loss']:.4f}")
    with k4:
        if evals:
            st.metric("Local samples", f"{evals[-1]['num_examples']:,}")

    c1, c2 = st.columns(2)

    with c1:
        if evals:
            local_acc = [r["eval_acc"] for r in evals]
            rounds = list(range(1, len(local_acc) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=local_acc,
                mode="lines+markers", name="your accuracy",
                line=dict(color="#4ade80", width=2), marker=dict(size=4),
            ))
            if server_data:
                fig.add_trace(go.Scatter(
                    x=[r["round"] for r in server_data],
                    y=[r["global_eval_acc"] for r in server_data],
                    mode="lines+markers", name="global",
                    line=dict(color="#818cf8", width=2, dash="dash"), marker=dict(size=3),
                ))
            fig.update_layout(title_text="Local vs global accuracy", yaxis_tickformat=".0%", xaxis_title="Round", **CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if trains:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(trains) + 1)),
                y=[r["train_loss"] for r in trains],
                mode="lines+markers", name="train loss",
                line=dict(color="#f472b6", width=2), marker=dict(size=4),
            ))
            fig.update_layout(title_text="Local train loss", xaxis_title="Round", **CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    label("Your contribution score")
    st.caption("Cosine similarity: how aligned your update was with the global direction. L2 norm: how much your model changed.")

    my_cos = []
    my_norm = []
    contrib_rounds = []

    for row in server_data:
        contribs = row.get("client_contributions", {})
        if not contribs:
            continue
        nodes = list(contribs.keys())
        if len(nodes) > int(CLIENT_ID):
            node_id = nodes[int(CLIENT_ID)]
            my_cos.append(contribs[node_id]["cosine_sim"])
            my_norm.append(contribs[node_id]["l2_norm"])
            contrib_rounds.append(row["round"])

    if not contrib_rounds:
        st.caption("Will appear after the first training round.")
    else:
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Cosine similarity (latest)", f"{my_cos[-1]:.4f}")
        with s2:
            st.metric("L2 norm (latest)", f"{my_norm[-1]:.4f}")

        if len(contrib_rounds) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=contrib_rounds, y=my_cos,
                mode="lines+markers", name="cosine similarity",
                line=dict(color="#4ade80", width=2), marker=dict(size=4),
            ))
            fig.update_layout(title_text="Cosine similarity over rounds", xaxis_title="Round", yaxis_range=[-1, 1], **CHART_STYLE)
            st.plotly_chart(fig, use_container_width=True)

if auto_refresh and not training_done:
    time.sleep(REFRESH_INTERVAL)
    st.rerun()
