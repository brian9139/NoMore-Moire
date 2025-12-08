# src/adapt/criteria.py
import numpy as np
import cv2


def compute_residual_score(input_img, demoire_img, mask=None) -> float:
    inp = input_img.astype(np.float32)
    out = demoire_img.astype(np.float32)
    diff = inp - out

    if mask is not None:
        m = mask.astype(np.float32)
        if m.ndim == 2:
            m = m[..., None]
        diff = diff * m

        denom = np.maximum(np.mean(m), 1e-6)
        energy = np.mean(diff ** 2) / denom
    else:
        energy = np.mean(diff ** 2)

    return float(energy)


def _sobel_mag(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag


def compute_edge_score(input_img, demoire_img, bins: int = 32) -> float:
    mag_in = _sobel_mag(input_img)
    mag_out = _sobel_mag(demoire_img)

    max_v = float(max(mag_in.max(), mag_out.max(), 1.0))
    hist_in, _ = np.histogram(mag_in, bins=bins, range=(0.0, max_v), density=True)
    hist_out, _ = np.histogram(mag_out, bins=bins, range=(0.0, max_v), density=True)

    diff = np.mean(np.abs(hist_in - hist_out))  # 0 ~ about 2
    score = 1.0 - diff
    score = float(np.clip(score, 0.0, 1.0))
    return score
