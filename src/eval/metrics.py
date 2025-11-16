import time
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim_fn

# PSNR
def psnr(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 999.0                                 # identical images
    return 20 * np.log10(255.0 / np.sqrt(mse))


# SSIM
def ssim(pred, gt):
    pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gt_g = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    score, _ = ssim_fn(pred_g, gt_g, full=True, data_range=255)
    return float(score)


# NIQE / BRISQUE
import torch
import pyiqa
import cv2


niqe_model = pyiqa.create_metric('niqe').eval()
brisque_model = pyiqa.create_metric('brisque').eval()


def niqe(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.tensor(img_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
    score = niqe_model(x)
    return float(score)


def brisque(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.tensor(img_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
    score = brisque_model(x)
    return float(score)



# TIME WRAP
def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    out = func(*args, **kwargs)
    end = time.perf_counter()
    return out, end - start
