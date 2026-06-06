import torch
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from models.cnn import Net


CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CANVAS_SIZE = 340


st.set_page_config(
    page_title="Handwriting Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(f"""
<style>
    :root {{
        --bg: #111315;
        --panel: #181b1f;
        --panel-2: #1d2126;
        --border: #2b3138;
        --text: #f3f4f6;
        --muted: #a7afb7;
        --canvas-bg: #0b0d10;
        --canvas-stroke: #f5f5f5;
        --canvas-size: {CANVAS_SIZE}px;
    }}

    .stApp {{
        background-color: var(--bg);
        color: var(--text);
    }}

    .block-container {{
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        color: var(--text);
    }}

    [data-testid="stCaptionContainer"] {{
        color: var(--muted);
    }}

    .drawing-wrap {{
        width: var(--canvas-size);
    }}

    .drawing-panel {{
        width: var(--canvas-size);
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem;
    }}

    .panel-soft {{
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
    }}

    .card-title {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.55rem;
    }}

    .card-text {{
        font-size: 0.95rem;
        line-height: 1.55;
        color: var(--muted);
    }}

    .section-label {{
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.6rem;
    }}

    .result-box {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem;
        min-height: 110px;
    }}

    .result-label {{
        font-size: 0.88rem;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }}

    .result-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
        line-height: 1.2;
    }}

    .result-note {{
        font-size: 0.92rem;
        color: var(--muted);
        margin-top: 0.35rem;
    }}

    .space {{
        height: 1rem;
    }}

    [data-testid="stImage"] img {{
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #ffffff;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
    model.eval()
    return model


def preprocess(canvas_image):
    img = Image.fromarray(canvas_image.astype("uint8")).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    return tensor, img


model = load_model()

st.markdown("## Federated Handwriting Recognition")
st.caption("Draw a character and inspect the model prediction.")

left, right = st.columns([1.2, 0.9], gap="large")

with left:
    st.markdown('<div class="drawing-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Drawing area</div>', unsafe_allow_html=True)
    st.markdown('<div class="drawing-panel">', unsafe_allow_html=True)

    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#f5f5f5",
        background_color="#0b0d10",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-soft">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Model architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-text">A compact convolutional neural network processes a 28×28 grayscale image and predicts one of 62 possible character classes.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="space"></div>', unsafe_allow_html=True)

    if canvas.image_data is not None:
        x, preview = preprocess(canvas.image_data)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        st.markdown('<div class="section-label">Model input</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-soft">', unsafe_allow_html=True)
        st.image(preview.resize((170, 170)))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="space"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Prediction</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">{CLASSES[pred_idx]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">{confidence:.2%}</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-note">Predicted class probability.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel-soft">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-text">No input yet. Draw a character on the left to run inference.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)