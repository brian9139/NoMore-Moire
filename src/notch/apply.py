
import json
import os
from typing import Iterable, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from notch.mask import build_notch_masks_from_polar
from spatial.guided import guided_filter


def _iter_image_dirs(
    out_root: str,
    categories: Iterable[str] = ("real", "synth"),
):
    """
    Yield (category, name, img_dir) for each image result directory under out_root.
    out_root/category/name/
    """
    for category in categories:
        cat_dir = os.path.join(out_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for name in sorted(os.listdir(cat_dir)):
            img_dir = os.path.join(cat_dir, name)
            if not os.path.isdir(img_dir):
                continue
            yield category, name, img_dir


def _load_logmag_from_spec(spec_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mag / phase / log-magnitude from specpack.npz, being robust to key name.
    """
    data = np.load(spec_path)
    mag = data["mag"]
    phase = data["phase"]
    if "logmag" in data:
        logmag = data["logmag"]
    elif "log_magnitude" in data:
        logmag = data["log_magnitude"]
    else:
        # Fallback: compute
        logmag = np.log(np.abs(mag) + 1e-8)
    return mag, phase, logmag


def _save_notch_heatmap(logmag: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    """
    Save notch heatmap: log-magnitude as background (gray),
    (1 - mask) as colored overlay.
    """
    logmag = logmag.astype(np.float32)
    mask = mask.astype(np.float32)

    log_norm = logmag - logmag.min()
    if log_norm.max() > 0:
        log_norm /= log_norm.max()

    notch = 1.0 - mask  # higher -> stronger attenuation
    if notch.max() > 0:
        notch_norm = notch / notch.max()
    else:
        notch_norm = notch

    plt.figure(figsize=(4, 4))
    plt.imshow(log_norm, cmap="gray", interpolation="nearest")
    if notch_norm.max() > 0:
        plt.imshow(
            notch_norm,
            cmap="jet",
            alpha=0.6,
            interpolation="nearest",
        )
    plt.axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(pad=0.0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def run_mask_stage(
    out_root: str = "./out",
    categories: Iterable[str] = ("real", "synth"),
    delta_r: float = 4.0,
    delta_theta_deg: float = 6.0,
    alpha: float = 0.8,
) -> None:
    """
    Stage: 'mask'
    - Input : specpack.npz, peaks.json
    - Output: maskpack.npz, notch_heatmap.png
    """
    for category, name, img_dir in _iter_image_dirs(out_root, categories):
        spec_path = os.path.join(img_dir, "specpack.npz")
        peaks_path = os.path.join(img_dir, "peaks.json")

        if not (os.path.exists(spec_path) and os.path.exists(peaks_path)):
            continue

        mag, phase, logmag = _load_logmag_from_spec(spec_path)

        with open(peaks_path, "r") as f:
            raw_peaks = json.load(f)

        peaks_for_notch = []
        for p in raw_peaks:
            if isinstance(p, dict) and ("radius" in p or "r" in p):
                # pair-style
                peaks_for_notch.append(
                    {
                        "r": p.get("r", p.get("radius", 0.0)),
                        "theta_deg": p.get("theta_deg", p.get("angle", 0.0)),
                        "strength": p.get("strength", p.get("avg_strength", 0.0)),
                    }
                )

        mask, each = build_notch_masks_from_polar(
            shape=mag.shape,
            peaks=peaks_for_notch,
            delta_r=delta_r,
            delta_theta_deg=delta_theta_deg,
            alpha=alpha,
        )

        params = {
            "delta_r": float(delta_r),
            "delta_theta_deg": float(delta_theta_deg),
            "alpha": float(alpha),
            "num_peaks": int(len(peaks_for_notch)),
        }

        maskpack_path = os.path.join(img_dir, "maskpack.npz")
        np.savez(
            maskpack_path,
            mask=mask.astype(np.float32),
            each=each.astype(np.float32),
            params=params,
        )

        heatmap_path = os.path.join(img_dir, "notch_heatmap.png")
        _save_notch_heatmap(logmag, mask, heatmap_path)

        print(f"[mask] {category}/{name} -> maskpack.npz, notch_heatmap.png")


def run_restore_stage(
    out_root: str = "./out",
    categories: Iterable[str] = ("real", "synth"),
    guided_r: int = 6,
    guided_eps: float = 1e-3,
) -> None:
    """
    Stage: 'restore'
    - Input : specpack.npz, maskpack.npz
    - Output: demoire.png
    """
    for category, name, img_dir in _iter_image_dirs(out_root, categories):
        spec_path = os.path.join(img_dir, "specpack.npz")
        maskpack_path = os.path.join(img_dir, "maskpack.npz")

        if not (os.path.exists(spec_path) and os.path.exists(maskpack_path)):
            continue

        spec = np.load(spec_path)
        mag = spec["mag"].astype(np.float32)
        phase = spec["phase"].astype(np.float32)

        maskpack = np.load(maskpack_path, allow_pickle=True)
        mask = maskpack["mask"].astype(np.float32)

        if mask.shape != mag.shape:
            raise ValueError(
                f"Mask shape {mask.shape} != mag shape {mag.shape} "
                f"for {category}/{name}"
            )

        # reconstruct complex spectrum with attenuation
        fshift = mag * mask * np.exp(1j * phase)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_real = np.real(img_back).astype(np.float32)

        # normalize to [0, 1] for guided filter
        vmin = float(img_real.min())
        vmax = float(img_real.max())
        if vmax > vmin:
            img_norm = (img_real - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img_real, dtype=np.float32)

        # self-guided filter
        refined = guided_filter(img_norm, img_norm, r=guided_r, eps=guided_eps)

        out_8bit = np.clip(refined * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out_path = os.path.join(img_dir, "demoire.png")

        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(out_path, out_8bit)

        print(f"[restore] {category}/{name} -> demoire.png")


if __name__ == "__main__":
    # 簡單單獨跑的入口（開發測試用）
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["mask", "restore", "mask+restore"],
        default="mask+restore",
    )
    parser.add_argument("--delta_r", type=float, default=4.0)
    parser.add_argument("--delta_theta", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--guided_r", type=int, default=6)
    parser.add_argument("--guided_eps", type=float, default=1e-3)

    args = parser.parse_args()

    if args.stage in ("mask", "mask+restore"):
        run_mask_stage(
            out_root=args.out,
            delta_r=args.delta_r,
            delta_theta_deg=args.delta_theta,
            alpha=args.alpha,
        )
    if args.stage in ("restore", "mask+restore"):
        run_restore_stage(
            out_root=args.out,
            guided_r=args.guided_r,
            guided_eps=args.guided_eps,
        )
