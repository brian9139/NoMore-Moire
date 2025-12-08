# src/spatial/pnp.py
from dataclasses import dataclass
from typing import Literal, Dict, Optional

import numpy as np

from .guided import guided_filter  


PnPMode = Literal["none", "guided"]  # 之後要加 bilateral / bm3d 再擴充


@dataclass
class PnPConfig:
    mode: PnPMode = "guided"
    num_iters: int = 1
    lam: float = 0.2  # data vs prior 的權重


def _ifft_from_mag_phase(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """
    給 magnitude + phase，在頻域（已 fftshift）重建空間域影像（實值）。
    """
    # 還原複數頻譜
    F = mag * np.exp(1j * phase)

    # 把中心 shift 回去再做 ifft
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


def _pnp_guided_step(
    x: np.ndarray,
    y: np.ndarray,
    radius: int,
    eps: float,
    lam: float,
) -> np.ndarray:
    """
    單步 PnP 更新：
      x_{k+1} = (1-lam) * x_k + lam * G(x_k | y)
    其中 G 是 guided filter，以 y 當 guidance。
    """
    x_denoised = guided_filter(
        guidance=y,
        src=x,
        radius=radius,
        eps=eps,
    )
    return (1.0 - lam) * x + lam * x_denoised


def pnp_restore(
    img_input: np.ndarray,
    mag: np.ndarray,
    phase: np.ndarray,
    mask: np.ndarray,
    pnp_cfg: Optional[PnPConfig] = None,
    guided_radius: int = 6,
    guided_eps: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """
    對外 API：
      1. 先做一次「頻域抑制 + IFFT」得到 x0。
      2. 如果 mode != 'none'，用 PnP 疊代幾步，輸出 x_final。
    回傳：
      - x_freq: freq-suppression 後直接 IFFT 的結果
      - x_final: PnP 後的最終影像
    """
    if pnp_cfg is None:
        pnp_cfg = PnPConfig()

    # Step 1: baseline 頻域抑制 → 空域
    x_freq = _apply_mask_to_spectrum(mag, phase, mask)

    if pnp_cfg.mode == "none":
        return {
            "x_freq": x_freq,
            "x_final": x_freq,
        }

    x = x_freq.copy()

    if pnp_cfg.mode == "guided":
        for _ in range(pnp_cfg.num_iters):
            x = _pnp_guided_step(
                x=x,
                y=img_input,
                radius=guided_radius,
                eps=guided_eps,
                lam=pnp_cfg.lam,
            )
    else:
        # 預留：之後可以新增 bilateral / bm3d 模式
        pass

    return {
        "x_freq": x_freq,
        "x_final": x,
    }
