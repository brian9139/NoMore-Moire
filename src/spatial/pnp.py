# src/spatial/pnp.py

from dataclasses import dataclass
from typing import Literal, Dict, Optional

import numpy as np

try:
    import cv2  # bilateral 用
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .guided import guided_filter  # 根據你實際的 guided.py 介面調整


PnPMode = Literal["none", "guided", "bilateral", "bm3d"]


@dataclass
class PnPConfig:
    mode: PnPMode = "guided"
    num_iters: int = 1        # PnP 疊代次數
    lam: float = 0.2          # data term vs prior 的混合權重

    # guided 相關超參
    guided_radius: int = 6
    guided_eps: float = 1e-3

    # bilateral 相關超參（OpenCV 實作）
    bilateral_diameter: int = 7          # 鄰域直徑（odd，例如 5,7,9）
    bilateral_sigma_color: float = 0.1   # 顏色空間 sigma（以 0~1 為尺度）
    bilateral_sigma_space: float = 3.0   # 空間距離 sigma（pixel）

    # BM3D 相關超參
    bm3d_sigma: float = 0.05             # 噪音標準差（假設影像在 0~1 範圍）
    bm3d_profile: str = "np"             # "np" / "high" 等，依 bm3d 套件實作選擇


def _ifft_from_mag_phase(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    給 magnitude + phase（已 fftshift），重建空間域影像（實值 float32）。
    """
    F = mag * np.exp(1j * phase)
    F_ishift = np.fft.ifftshift(F)
    img_complex = np.fft.ifft2(F_ishift)
    img = np.real(img_complex)
    return img.astype(np.float32)


def _apply_mask_to_spectrum(
    mag: np.ndarray,
    phase: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    單次頻域抑制：mag * mask → IFFT。
    """
    mag_supp = mag * mask
    return _ifft_from_mag_phase(mag_supp, phase)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ====== 各種 prior（guided / bilateral / bm3d）的一步 ======

def _pnp_guided_step(
    x: np.ndarray,
    y: np.ndarray,
    radius: int,
    eps: float,
    lam: float,
) -> np.ndarray:
    """
    Guided PnP 一步：
        x_{k+1} = (1-lam) * x_k + lam * G(x_k | y)
    其中 G 為 guided_filter，以 y 為 guidance。
    """
    x_denoised = guided_filter(
        guidance=y,
        src=x,
        radius=radius,
        eps=eps,
    )
    x_next = (1.0 - lam) * x + lam * x_denoised
    return _clamp01(x_next)


def _pnp_bilateral_step(
    x: np.ndarray,
    y: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
    lam: float,
) -> np.ndarray:
    """
    Bilateral PnP 一步：
        x_{k+1} = (1-lam) * x_k + lam * B(x_k | y)
    這裡採用 OpenCV bilateral filter，實作上用 y 當 guidance 较難，
    先簡化成對 x 做 bilateral；如要更貼 spec，可將 x 替換成 y 或
    對 x,y 做某種融合後再 bilateral。
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (cv2) 未安裝，無法使用 bilateral PnP 模式。")

    # OpenCV bilateralFilter 的 sigmaColor 取決於輸入範圍；這裡假設 x,y 在 [0,1]，
    # 為了更穩定，轉到 [0,255] 域再套用。
    x_8u = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    # sigma_color 以 0~1 為尺度，轉成 0~255
    sigma_color_cv = sigma_color * 255.0

    x_denoised_8u = cv2.bilateralFilter(
        src=x_8u,
        d=diameter,
        sigmaColor=sigma_color_cv,
        sigmaSpace=sigma_space,
    )
    x_denoised = x_denoised_8u.astype(np.float32) / 255.0

    x_next = (1.0 - lam) * x + lam * x_denoised
    return _clamp01(x_next)


def _pnp_bm3d_step(
    x: np.ndarray,
    sigma: float,
    profile: str,
    lam: float,) -> np.ndarray:
    """
    BM3D PnP 一步：
        x_{k+1} = (1-lam) * x_k + lam * BM3D(x_k)
    需要第三方套件 `bm3d`（pip install bm3d）。
    """
    try:
        # 常見套件名為 bm3d，若你們用的是其他（例如 pybm3d），在這裡改 import 即可。
        from bm3d import bm3d, BM3DProfile  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "BM3D 套件未安裝或 import 失敗，無法使用 bm3d 模式。"
            " 請先安裝 pip install bm3d"
        ) from e

    # BM3D 通常假設輸入在 [0,1]
    x = _clamp01(x)

    if profile == "high":
        prof = BM3DProfile.HIGH
    elif profile == "np":
        prof = BM3DProfile.NP
    else:
        prof = BM3DProfile.NP

    x_denoised = bm3d(x, sigma_psd=sigma, profile=prof)
    x_next = (1.0 - lam) * x + lam * x_denoised.astype(np.float32)
    return _clamp01(x_next)


# ====== 對外主 API ======

def pnp_restore(
    img_input: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    mask: np.ndarray,
    pnp_cfg: Optional[PnPConfig] = None,) -> Dict[str, np.ndarray]:
    """
    PnP 還原主函式：
      1. 頻域抑制：mag * mask → IFFT 得到 x_freq
      2. 若 mode != "none"，以 img_input 作為 guidance（或先驗），做 num_iters 步 PnP

    參數：
      img_input : 原始輸入 Y/灰階影像，float32，建議已經在 [0,1]
      mag, phase: 來自 specpack.npz 的 magnitude / phase（已 fftshift）
      mask      : notch mask，shape 與 mag 相同，值在 [0,1]
      pnp_cfg   : PnPConfig，包含 mode / num_iters / lam / 各模式超參

    回傳：
      {
        "x_freq":  freq-suppression 後直接 IFFT 的結果（未 PnP），float32, [0,1]（裁切）
        "x_final": PnP 之後的最終影像，float32, [0,1]
      }
    """
    if pnp_cfg is None:
        pnp_cfg = PnPConfig()

    # 保證輸入在合理範圍
    img_input = _clamp01(img_input)

    # Step 1: 頻域抑制 + IFFT
    x_freq = _apply_mask_to_spectrum(mag, phase, mask)
    x_freq = _clamp01(x_freq)

    # 若不做 PnP，直接回傳
    if pnp_cfg.mode == "none" or pnp_cfg.num_iters <= 0:
        return {
            "x_freq": x_freq,
            "x_final": x_freq,
        }

    # Step 2: PnP 疊代
    x = x_freq.copy()

    if pnp_cfg.mode == "guided":
        for _ in range(pnp_cfg.num_iters):
            x = _pnp_guided_step(
                x=x,
                y=img_input,
                radius=pnp_cfg.guided_radius,
                eps=pnp_cfg.guided_eps,
                lam=pnp_cfg.lam,
            )

    elif pnp_cfg.mode == "bilateral":
        for _ in range(pnp_cfg.num_iters):
            x = _pnp_bilateral_step(
                x=x,
                y=img_input,
                diameter=pnp_cfg.bilateral_diameter,
                sigma_color=pnp_cfg.bilateral_sigma_color,
                sigma_space=pnp_cfg.bilateral_sigma_space,
                lam=pnp_cfg.lam,
            )

    elif pnp_cfg.mode == "bm3d":
        for _ in range(pnp_cfg.num_iters):
            x = _pnp_bm3d_step(
                x=x,
                sigma=pnp_cfg.bm3d_sigma,
                profile=pnp_cfg.bm3d_profile,
                lam=pnp_cfg.lam,
            )

    else:
        # 未知模式，直接回傳 freq 結果避免 silent wrong behavior
        raise ValueError(f"Unknown PnP mode: {pnp_cfg.mode}")

    return {
        "x_freq": x_freq,
        "x_final": x,
    }
