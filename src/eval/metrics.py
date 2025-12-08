# src/eval/metrics.py
import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_fn

try:
    import torch
    import pyiqa
    _HAS_PYIQA = True
except Exception:
    torch = None
    pyiqa = None
    _HAS_PYIQA = False


# PSNR
def psnr(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 999.0  # identical images
    return 20 * np.log10(255.0 / np.sqrt(mse))


# SSIM
def ssim(pred, gt):
    pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_fn(pred_g, gt_g, full=True, data_range=255)
    return float(score)


def _to_pyiqa_tensor(img_bgr):
    # pyiqa expects RGB, NCHW, 0~1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x


def _create_metric(name: str):
    if not _HAS_PYIQA:
        return None
    try:
        return pyiqa.create_metric(name).eval()
    except Exception:
        return None


# NIQE / BRISQUE 
niqe_model = _create_metric("niqe")
brisque_model = _create_metric("brisque")


def niqe(img):
    if niqe_model is None:
        return None
    x = _to_pyiqa_tensor(img)
    score = niqe_model(x)
    return float(score)


def brisque(img):
    if brisque_model is None:
        return None
    x = _to_pyiqa_tensor(img)
    score = brisque_model(x)
    return float(score)


# TIME WRAP
def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    end = time.perf_counter()
    return out, end - start
