import numpy as np
import joblib
from src.features import extract_features

def load_bundle(model_path: str):
    return joblib.load(model_path)

def predict_bgr(bundle, img_bgr, topk=3):
    pipe = bundle["pipeline"]
    le = bundle["label_encoder"]

    x = extract_features(img_bgr).reshape(1, -1)

    pred_idx = int(pipe.predict(x)[0])
    pred_label = le.inverse_transform([pred_idx])[0]

    top = [(pred_label, 1.0)]
    if hasattr(pipe[-1], "predict_proba"):
        proba = pipe.predict_proba(x)[0]
        order = np.argsort(-proba)[:topk]
        top = [(le.inverse_transform([i])[0], float(proba[i])) for i in order]

    return pred_label, top
