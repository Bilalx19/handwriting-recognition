import torch
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from models.cnn import Net

# Page config
st.set_page_config(
    page_title="Federated Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .confidence-bar {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 30px;
        border-radius: 15px;
        transition: width 0.5s ease;
    }
    .metric-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("global_model/final_model.pt", map_location="cpu"))
    model.eval()
    return model


def preprocess(canvas_image):
    # RGBA -> grayscale
    img = Image.fromarray(canvas_image.astype("uint8")).convert("L")
    
    # invert: canvas black on white -> model expects white-ish digit on black-ish
    img = ImageOps.invert(img)
    
    # resize to FEMNIST size
    img = img.resize((28, 28))
    
    # normalize to tensor [1, 1, 28, 28]
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    
    return tensor, img


def get_top_k_predictions(probs, k=5):
    top_probs, top_indices = torch.topk(probs, k)
    top_classes = [CLASSES[idx] for idx in top_indices[0].tolist()]
    return top_classes, top_probs[0].tolist()


# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">✍️ Federated Handwriting Recognition</h1>
    <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">Draw any character and let AI recognize it!</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

# Create two columns for main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🎨 Drawing Canvas")
    st.markdown("*Draw your character in the canvas below*")
    
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=400,
        height=400,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )
    
    # Add clear canvas button with custom styling
    if st.button("🗑️ Clear Canvas", use_container_width=True):
        st.rerun()

with col2:
    st.markdown("### 📊 Recognition Results")
    
    if canvas.image_data is not None:
        x, preview = preprocess(canvas.image_data)
        
        # Display preprocessed image
        col2a, col2b = st.columns(2)
        with col2a:
            st.markdown("**Processed Input**")
            st.image(preview.resize((112, 112)), caption="28x28 pixels", use_container_width=True)
        
        with col2b:
            st.markdown("**Model Architecture**")
            with st.expander("ℹ️ CNN Details", expanded=False):
                st.markdown("""
                - 2 Convolutional layers
                - 64 → 128 filters
                - 2 Fully connected layers
                - Dropout for regularization
                - 62 output classes
                """)
        
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            
            # Get top 5 predictions
            top_classes, top_probs = get_top_k_predictions(probs, k=5)
        
        # Display main prediction with animation
        st.markdown(f"""
        <div class="prediction-box fade-in">
            <h2 style="color: white; margin: 0;">Top Prediction</h2>
            <div style="font-size: 5rem; font-weight: bold; color: white; margin: 0.5rem;">
                {CLASSES[pred_idx]}
            </div>
            <div style="font-size: 1.2rem; color: rgba(255,255,255,0.9);">
                Confidence: {confidence:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge using Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            title = {'text': "Confidence Score", 'font': {'size': 16}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 50], 'color': "#ff6b6b"},
                    {'range': [50, 80], 'color': "#ffd93d"},
                    {'range': [80, 100], 'color': "#6bcf7f"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Top 5 predictions bar chart
        df_top = pd.DataFrame({
            'Character': top_classes,
            'Confidence': [p * 100 for p in top_probs]
        })
        
        fig_bar = px.bar(df_top, x='Character', y='Confidence', 
                         title="Top 5 Predictions",
                         color='Confidence',
                         color_continuous_scale=['#ff6b6b', '#ffd93d', '#6bcf7f'],
                         labels={'Confidence': 'Confidence (%)'})
        fig_bar.update_layout(showlegend=False, height=300, 
                              plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)')
        fig_bar.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("👈 Draw a character on the canvas to see recognition results!", icon="ℹ️")
        
        # Show example predictions
        with st.expander("📚 How it works", expanded=True):
            st.markdown("""
            **The model recognizes:**
            - Digits (0-9)
            - Uppercase letters (A-Z)
            - Lowercase letters (a-z)
            
            **Tips for best results:**
            - Draw clearly in the center
            - Use thick strokes
            - Avoid overlapping lines
            - Clear canvas between characters
            """)
            
            st.markdown("**Example Characters:**")
            example_cols = st.columns(5)
            examples = ['A', 'b', '9', 'Z', 'm']
            for col, ex in zip(example_cols, examples):
                with col:
                    st.markdown(f"<div style='text-align: center; font-size: 2rem;'>{ex}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
colf1, colf2, colf3 = st.columns(3)
with colf1:
    st.markdown("### 🧠 Model Info")
    st.caption("Federated Learning • CNN Architecture • 62 Classes")
with colf2:
    st.markdown("### 📊 Dataset")
    st.caption("FEMNIST • 800k+ samples • Balanced classes")
with colf3:
    st.markdown("### ⚡ Performance")
    st.caption("Real-time inference • High accuracy • Efficient")

# Add some metrics in sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Stats")
    st.markdown("---")
    
    st.markdown("**Supported Characters:**")
    st.markdown(f"<div style='font-size: 2rem; line-height: 1.2;'>🔢 0-9<br>🔤 A-Z<br>📝 a-z</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Model Architecture:**")
    st.code("""
    Conv2d(1, 64, 5)
    ReLU + MaxPool
    Conv2d(64, 128, 5)
    ReLU + MaxPool  
    FC(2048, 512)
    ReLU + Dropout
    FC(512, 62)
    """, language="python")
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.info("""
    1. **Use thick strokes** for better recognition
    2. **Center your drawing** in the canvas
    3. **Keep characters simple** and clear
    4. **Clear between characters** for best results
    """)