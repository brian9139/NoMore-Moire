# src/notch/ms_mask.py
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import cv2


@dataclass
class MSNotchLevelParam:
    scale: float         # e.g. 1.0, 0.5, 0.25
    delta_r: float       # radial half-width in px at this scale
    delta_theta_deg: float  # angular half-width in degrees
    alpha: float         # attenuation strength (0~1)


@dataclass
class MSNotchConfig:
    levels: List[MSNotchLevelParam]


def _make_coordinate_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    # 回傳以中心為原點的座標網格 (X, Y)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    x = np.arange(w) - cx  # [W]
    y = np.arange(h) - cy  # [H]
    X, Y = np.meshgrid(x, y)
    return X, Y


def _build_single_peak_mask(
    h: int,
    w: int,
    peak_r: float,
    peak_theta_deg: float,
    delta_r: float,
    delta_theta_deg: float,
    alpha: float,) -> np.ndarray:
    """
    在頻譜空間 (已 fftshift) 上，為單一峰對 (r, theta) 建一個 cosine-tapered band-stop mask。

    回傳 shape [H, W]，值在 [1 - alpha, 1]，乘在 magnitude 上使用。
    """
    X, Y = _make_coordinate_grid(h, w)

    # 每個 pixel 在頻域的極座標 (r, theta)
    R = np.sqrt(X**2 + Y**2)
    theta = np.rad2deg(np.arctan2(Y, X))  # [-180, 180]

    # 徑向距離差與角度差
    dr = np.abs(R - peak_r)

    dtheta = np.abs(theta - peak_theta_deg)
    # 考慮 360 wrap-around
    dtheta = np.minimum(dtheta, 360.0 - dtheta)

    # radial taper: 0 outside, cos inside
    radial_mask = np.zeros_like(R, dtype=np.float32)
    inside_r = dr <= delta_r
    radial_mask[inside_r] = 0.5 * (1.0 + np.cos(np.pi * dr[inside_r] / delta_r))

    # angular taper
    angular_mask = np.zeros_like(R, dtype=np.float32)
    inside_th = dtheta <= delta_theta_deg
    angular_mask[inside_th] = 0.5 * (1.0 + np.cos(np.pi * dtheta[inside_th] / delta_theta_deg))

    # 結合徑向 + 角向
    weight = radial_mask * angular_mask  # [0, 1]

    # notch：1 -> no change, 1-alpha -> strongest suppression
    mask = 1.0 - alpha * weight
    return mask.astype(np.float32)


def build_ms_mask_from_arrays(
    mag: np.ndarray,
    peaks: List[Dict],
    config: MSNotchConfig,
) -> Dict[str, np.ndarray]:
    """
    核心函式：給一張 magnitude、峰列表、多尺度設定，回傳:
      - mask_ms: [L, H, W] 各尺度合成遮罩 (上 sample 回原尺度後)
      - each_ms: [L, K, H, W] 各尺度、各峰的遮罩 (optional, 可視化用)
      - final_mask: [H, W] 多尺度融合後最終遮罩
    這個函式不做 IFFT，只負責「遮罩」本身。
    """
    h, w = mag.shape[:2]
    K = len(peaks)
    L = len(config.levels)

    mask_ms = np.ones((L, h, w), dtype=np.float32)
    each_ms = np.ones((L, K, h, w), dtype=np.float32)

    for lvl_idx, lvl in enumerate(config.levels):
        # 以 scale 重採樣 mag 尺寸（這裡只為了計算 mask，可以直接用原尺寸算，也可以真的 downsample）
        # 為簡化，你可以直接在原尺寸上算，只是 delta_r 以 scale 調整：
        delta_r = lvl.delta_r * lvl.scale
        delta_theta = lvl.delta_theta_deg
        alpha = lvl.alpha

        level_mask = np.ones((h, w), dtype=np.float32)

        for k, pk in enumerate(peaks):
            r0 = pk["r"] * lvl.scale
            theta0 = pk["theta_deg"]

            peak_mask = _build_single_peak_mask(
                h=h,
                w=w,
                peak_r=r0,
                peak_theta_deg=theta0,
                delta_r=delta_r,
                delta_theta_deg=delta_theta,
                alpha=alpha,
            )
            level_mask *= peak_mask
            each_ms[lvl_idx, k] = peak_mask

        mask_ms[lvl_idx] = level_mask

    # 多尺度融合：這裡先採用「逐像素最小值」（最強抑制），也可以改成加權平均
    final_mask = np.min(mask_ms, axis=0)

    return {
        "mask_ms": mask_ms,      # [L, H, W]
        "each_ms": each_ms,      # [L, K, H, W]
        "final_mask": final_mask,  # [H, W]
    }


def build_ms_mask_from_files(
    specpack_path: str,
    peaks_json_path: str,
    config: MSNotchConfig,
):
    """
    方便 A / D 直接從檔案呼叫的版本：
      - 讀 specpack.npz → mag
      - 讀 peaks.json → peaks list
      - 回傳 dict，可直接存成 maskpack_ms.npz
    """
    specpack = np.load(specpack_path)
    mag = specpack["mag"]

    with open(peaks_json_path, "r") as f:
        peaks = json.load(f)

    masks = build_ms_mask_from_arrays(mag, peaks, config)
    return masks
