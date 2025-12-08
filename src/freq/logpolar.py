import numpy as np
import cv2
from scipy import ndimage
from src.freq.fft2d import load_specpack_npz, save_specpack_npz, fft2d, load_image
import os

def cart2logpolar(logmag, return_maps=False):
    """
    將 FFT log-magnitude 影像從 Cartesian 轉到 log-polar 表示。

    參數:
        logmag : 2D numpy array
        return_maps : 回傳對應座標映射 (debug 用)

    回傳:
        logpolar_img
        (x_map, y_map)  若 return_maps=True
    """
    H, W = logmag.shape
    cx, cy = W / 2, H / 2

    max_radius = np.sqrt(cx * cx + cy * cy)
    M = H / np.log(max_radius)

    logpolar_img = cv2.logPolar(
        logmag,
        (cx, cy),
        M,
        flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
    )

    if not return_maps:
        return logpolar_img

    # 回傳座標映射：log-polar → Cartesian
    rp = np.linspace(0, H - 1, H)
    tp = np.linspace(0, W - 1, W)
    rp_grid, tp_grid = np.meshgrid(rp, tp, indexing='ij')

    r = np.exp(rp_grid / M)
    theta = (tp_grid / W) * 2 * np.pi

    x_map = cx + r * np.cos(theta)
    y_map = cy + r * np.sin(theta)

    return logpolar_img, (x_map, y_map)

def logpolar2cart(logpolar, out_shape, center=None):
    """
    將 log-polar 影像反向映射到 Cartesian。

    參數:
        logpolar : 2D numpy array (log-polar 表示)
        out_shape : (H, W) 輸出 Cartesian 影像大小
        center : 頻譜中心 (預設影像中心)

    回傳:
        cart_img : 回到 Cartesian 的影像
    """
    Hp, Wp = logpolar.shape
    H, W = out_shape

    if center is None:
        cx, cy = W / 2, H / 2
    else:
        cx, cy = center

    # 依 cart2logpolar 計算相同 scale
    max_radius = np.sqrt(cx * cx + cy * cy)
    M = Hp / np.log(max_radius)

    # 使用 OpenCV 反向 log-polar
    cart_img = cv2.logPolar(
        logpolar,
        (cx, cy),
        M,
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )

    return cart_img


def visualize_mappings(logmag, logpolar, maps):
    """
    回傳可視化用的多張圖：
    - Cartesian 原圖
    - Log-Polar 圖
    - Log-Polar 網格（映射回 Cartesian）
    """
    x_map, y_map = maps

    H, W = logmag.shape

    # ----------- 產生 log-polar 網格 ----------
    grid_img = np.zeros((H, W, 3), dtype=np.uint8)
    step_r = H // 20
    step_t = W // 20

    # 畫 radial grid
    for r in range(0, H, step_r):
        xx = x_map[r, :]
        yy = y_map[r, :]
        pts = np.vstack([xx, yy]).T
        pts = pts.astype(np.int32)
        valid = (pts[:,0] >= 0) & (pts[:,0] < W) & (pts[:,1] >= 0) & (pts[:,1] < H)
        pts = pts[valid]
        for x, y in pts:
            grid_img[y, x] = (0, 255, 0)  # green line

    # 畫 angle grid
    for t in range(0, W, step_t):
        xx = x_map[:, t]
        yy = y_map[:, t]
        pts = np.vstack([xx, yy]).T
        pts = pts.astype(np.int32)
        valid = (pts[:,0] >= 0) & (pts[:,0] < W) & (pts[:,1] >= 0) & (pts[:,1] < H)
        pts = pts[valid]
        for x, y in pts:
            grid_img[y, x] = (0, 0, 255)  # red line

    return {
        "cartesian": logmag,
        "logpolar": logpolar,
        "grid": grid_img
    }
