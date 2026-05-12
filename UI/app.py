import torch
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from models.cnn import Net


CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
    model.eval()
    return model


def preprocess(canvas_image):
    # RGBA -> grayscale
    img = Image.fromarray(canvas_image.astype("uint8")).convert("L")

    # invert: canvas black on white -> model expects white-ish digit on black-ish? Test both if needed
    img = ImageOps.invert(img)

    # resize to FEMNIST size
    img = img.resize((28, 28))

    # normalize to tensor [1, 1, 28, 28]
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    return tensor, img


st.title("Federated Handwriting Recognition")

model = load_model()

canvas = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    x, preview = preprocess(canvas.image_data)

    st.image(preview.resize((140, 140)), caption="Model input 28x28")

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    st.subheader(f"Prediction: `{CLASSES[pred_idx]}`")
    st.write(f"Confidence: `{confidence:.2%}`")