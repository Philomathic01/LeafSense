from pathlib import Path
import json
import urllib.request

import numpy as np
import cv2
from PIL import Image
import streamlit as st

from src.predict import load_bundle, predict_bgr

APP_TITLE = "LeafSense: Potato Disease Recognition (SVM + Image Features)"

MODEL_PATH = Path("artifacts/model_pipeline.joblib")

# ✅ After you create a GitHub Release, paste the direct asset URL here:
# Example:
# https://github.com/Philomathic01/LeafSense/releases/download/v1.0/model_pipeline.joblib
MODEL_URL = ""  # <-- paste release URL later

ADVICE_PATH = Path("disease_advice.json")

@st.cache_resource
def get_bundle():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        if not MODEL_URL:
            raise RuntimeError("MODEL_URL is empty. Paste your GitHub Release asset URL in app/app.py")
        with st.spinner("Downloading model from GitHub Release..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_bundle(str(MODEL_PATH))

def pil_to_bgr(pil_img: Image.Image):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def load_advice():
    if ADVICE_PATH.exists():
        return json.loads(ADVICE_PATH.read_text(encoding="utf-8"))
    return {}

def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.caption("Classical ML model: **SVM (RBF)** trained on handcrafted features (HSV hist + LBP + HOG).")

    advice = load_advice()

    with st.sidebar:
        mode = st.radio("Input", ["Camera (snapshot)", "Upload image"])
        topk = st.slider("Top-K", 1, 5, 3)
        conf_th = st.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.01)

    try:
        bundle = get_bundle()
    except Exception as e:
        st.error(str(e))
        st.stop()

    img_bgr = None

    if mode == "Camera (snapshot)":
        cam = st.camera_input("Take a photo")
        if cam is not None:
            pil_img = Image.open(cam)
            st.image(pil_img, caption="Captured", use_container_width=True)
            img_bgr = pil_to_bgr(pil_img)
    else:
        up = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            pil_img = Image.open(up)
            st.image(pil_img, caption="Uploaded", use_container_width=True)
            img_bgr = pil_to_bgr(pil_img)

    if img_bgr is None:
        st.info("Provide an image using camera or upload.")
        return

    if st.button("Predict", type="primary"):
        pred, top = predict_bgr(bundle, img_bgr, topk=topk)

        best_conf = top[0][1]
        if best_conf < conf_th:
            st.warning(f"Low confidence ({best_conf:.2f}). Retake photo / use clearer leaf image.")
        else:
            st.success(f"Prediction: **{pred}** (conf: {best_conf:.2f})")

        st.write("### Top predictions")
        for cls, p in top:
            st.write(f"- **{cls}** — {p:.3f}")

        st.write("### Advisory")
        info = advice.get(pred)
        if not info:
            st.info("No advisory text for this class (edit disease_advice.json).")
        else:
            st.markdown(f"**Symptoms:** {info.get('symptoms','-')}")
            st.markdown("**Actions:**")
            for a in info.get("actions", []):
                st.write(f"• {a}")
            st.markdown("**Prevention:**")
            for p in info.get("prevention", []):
                st.write(f"• {p}")

if __name__ == "__main__":
    main()
