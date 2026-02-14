import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern

def extract_features(img_bgr, size=(224, 224),
                     use_color=True, use_lbp=True, use_hog=True):
    img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    feats = []

    if use_color:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, (8, 8, 8),
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        feats.append(hist)

    if use_lbp:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        n_bins = int(lbp.max() + 1)
        h, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        feats.append(h.astype(np.float32))

    if use_hog:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True
        ).astype(np.float32)
        feats.append(h)

    return np.concatenate(feats).astype(np.float32)
