import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from src.freq.detect_adv import detect_peaks_adv, specpack_adv
from src.freq.peak_detect import detect_peaks
from src.freq.fft2d import fft2d
from src.io.artifact import load_maskpack, load_maskpack_ms
from src.notch.apply import run_mask_stage, run_restore_stage
from src.eval.summarize import evaluate_pair_folder
from src.io.loader import DataLoader
from src.notch.ms_mask import MSNotchLevelParam, build_ms_mask_from_files, MSNotchConfig
from src.spatial.guided import guided_filter


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def infer_split(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for token in ("synth", "real"):
        if token in parts:
            return token
    return path.name

def stage_spec(input_dir: Path) -> None:
    print("[spec] Calling fft2d() to generate specpack.npz ...")
    loader = DataLoader(input_dir, batch_size=4, color_mode='bgr')
    fft2d(loader=loader)
    print("[spec] Specpack generation completed.")

def iter_result_dirs(out_root: Path, pack: str="specpack.npz") -> Iterable[Path]:
    cat_dir = out_root
    if not cat_dir.exists():
        return []
    for p in sorted(cat_dir.rglob("*")):
        if p.is_dir() and (p / pack).exists():
            yield p

def stage_detect(out_root: Path, r_min: float, max_pairs: int) -> None:
    detect_peaks(
        r_min=r_min,
        max_pairs=max_pairs,
        path=str(out_root)
    )

def stage_mask(out_root: Path, delta_r: float, delta_theta: float, alpha: float, method: str) -> None:
    img_dirs = list(iter_result_dirs(out_root))

    run_mask_stage(
        out_root=str(out_root),
        categories=('real', 'synth'),
        delta_r=delta_r,
        delta_theta_deg=delta_theta,
        alpha=alpha,
    )

    if method == "final":
        for img_dir in img_dirs:
            src = img_dir / "maskpack.npz"
            dst = img_dir / "maskpack_ms.npz"
            if dst.exists() or not src.exists():
                continue
            data = np.load(src, allow_pickle=True)
            params = data["params"]
            if isinstance(params, np.ndarray):
                params = params.item()
            np.savez_compressed(
                dst,
                mask_ms=data["mask"],
                each_ms=data["each"],
                params_ms=params,
            )

def stage_restore(out_root: Path, guided_r: int, guided_eps: float) -> None:
    run_restore_stage(
        out_root=str(out_root),
        categories=('real', 'synth'),
        guided_r=guided_r,
        guided_eps=guided_eps,
    )

def stage_eval(data_root: Path, out_root: Path, out_final: Path, results_path: Path) -> None:
    categories = ('real', 'synth')
    rows = []
    for category in categories:
        row = evaluate_pair_folder(
            str(data_root) + f'/{category}',
            str(out_root) + f'/{category}',
            str(out_final) + f'/{category}',
            category,
        )
        rows += row
    if not rows:
        print("[eval] No rows to save.")
        return
    df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"[eval] Saved {results_path}")

def final_spec_stage(out_root: Path) -> None:
    print("[spec_adv] Calling specpack_adv() to generate advanced specpack.npz ...")
    specpack_adv(out_root=out_root)
    print("[spec_adv] Advanced specpack generation completed.")

def final_supperss_stage(out_root: Path, r_min: float=5.0, max_pairs: int=10) -> None:
    print("[detect_adv] Calling detect_peaks_adv() to generate advanced peaks ...")
    print(str(out_root))
    detect_peaks_adv(r_min=r_min, max_pairs=max_pairs, path=str(out_root))
    print("[detect_adv] Advanced peak detection completed.")

def final_mask_stage(out_root: Path, delta_r: float, delta_theta: float, alpha: float) -> None:
    config = MSNotchConfig(
        levels=[
            MSNotchLevelParam(scale=1.0, delta_r=delta_r, delta_theta_deg=delta_theta, alpha=alpha),
            MSNotchLevelParam(scale=0.5, delta_r=delta_r, delta_theta_deg=delta_theta, alpha=alpha),
            MSNotchLevelParam(scale=0.25, delta_r=delta_r, delta_theta_deg=delta_theta, alpha=alpha),
        ]
    )
    print(f"[mask_adv] Using multi-scale config: {config}")
    img_dirs = list(iter_result_dirs(out_root, pack="specpack_adv.npz"))
    # print(img_dirs)
    for img_dir in img_dirs:
        # print(img_dir)
        specpack_path = img_dir / "specpack_adv.npz"
        peaks_json_path = img_dir / "peaks.json"
        mask = build_ms_mask_from_files(
            specpack_path=specpack_path,
            peaks_json_path=peaks_json_path,
            config=config,
        )
        maskpack_path = img_dir / "maskpack_ms.npz"
        np.savez_compressed(
            maskpack_path,
            mask_ms=mask["mask_ms"].astype(np.float32),
            each_ms=mask["each_ms"].astype(np.float32),
            final_mask=mask["final_mask"].astype(np.float32),
            params_ms={
                "levels": [
                    {
                        "scale": lvl.scale,
                        "delta_r": lvl.delta_r,
                        "delta_theta_deg": lvl.delta_theta_deg,
                        "alpha": lvl.alpha,
                    }
                    for lvl in config.levels
                ],
                'num_peaks': int(len(mask["each_ms"][0])),
            },
        )
    print("[mask_adv] Multi-scale mask building completed.")

def final_restore_stage(out_root: Path, guided_r: int, guided_eps: float) -> None:
    out_root = Path(out_root)
    img_dirs = list(iter_result_dirs(out_root, pack="specpack_adv.npz"))
    for img_dir in img_dirs:
        # prefer specpack_adv if available
        spec_adv = img_dir / "specpack_adv.npz"
        spec_base = img_dir / "specpack.npz"
        spec_path = spec_adv if spec_adv.exists() else spec_base
        if not spec_path.exists():
            continue

        # prefer multi-scale mask
        mask = None
        mask_ms_path = img_dir / "maskpack_ms.npz"
        if mask_ms_path.exists():
            # print(mask_ms_path.parent)
            mdata = load_maskpack_ms(mask_ms_path.parent)
            if mdata.mask_ms is not None:
                mm = mdata.mask_ms
                mask = mm.min(axis=0) if mm.ndim == 3 else mm

        spec = np.load(spec_path)
        mag = spec["mag"].astype(np.float32)
        phase = spec["phase"].astype(np.float32)
        mask = mask.astype(np.float32)

        if mask.shape != mag.shape:
            mask = cv2.resize(mask, (mag.shape[1], mag.shape[0]), interpolation=cv2.INTER_LINEAR)

        fshift = mag * mask * np.exp(1j * phase)
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
        img_real = np.real(img_back).astype(np.float32)

        vmin, vmax = float(img_real.min()), float(img_real.max())
        if vmax > vmin:
            img_norm = (img_real - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img_real, dtype=np.float32)

        refined = guided_filter(img_norm, img_norm, r=guided_r, eps=guided_eps)

        out_y = np.clip(refined * 255.0 + 0.5, 0, 255).astype(np.uint8)
        yuv_path = img_dir / "yuv.npz"
        if yuv_path.exists():
            yuv = np.load(yuv_path)
            img = np.zeros((out_y.shape[0], out_y.shape[1], 3), dtype=np.uint8)
            img[:, :, 0] = out_y
            img[:, :, 1] = yuv["u"]
            img[:, :, 2] = yuv["v"]
            out_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        else:
            out_bgr = cv2.cvtColor(out_y, cv2.COLOR_GRAY2BGR)

        out_path = img_dir / "demoire_final.png"
        cv2.imwrite(str(out_path), out_bgr)
        print(f"[final-restore] {out_path} -> demoire_final.png")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='data', help="Input folder (e.g., data/synth)")
    parser.add_argument("--out", type=str, default='out', help="Output root or split folder (e.g., out or out/synth)")
    parser.add_argument("--out_final", type=str, default='out_final', help="Output root for final method")
    parser.add_argument("--config", type=str, default="configs/final.yaml")
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "final"],
        default="final",
        help="Pipeline variant to run.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["spec", "detect", "suppress", "restore", "adapt_eval", "full"],
        default="full",
    )
    parser.add_argument("--results", type=str, default="results_final.csv", help="CSV path for eval stage.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input)
    split = infer_split(input_dir)
    out_root = Path(args.out)
    out_final = Path(args.out_final)

    # print(f"[Info] Input dir: {input_dir}")
    # print(f"[Info] Output root: {out_root}")
    # print(f"[Info] Split: {split}")

    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    params = cfg.get("params", {})
    guided_cfg = cfg.get("guided", {})

    r_min = float(params.get("r_min", 10))
    max_pairs = int(params.get("K", 4))
    delta_r = float(params.get("delta_r", 4.0))
    delta_theta = float(params.get("delta_theta", 6.0))
    alpha = float(params.get("alpha", 0.8))
    guided_r = int(guided_cfg.get("r", 6))
    guided_eps = float(guided_cfg.get("eps", 1e-3))

    stage = args.stage
    method = args.method

    if stage in ("spec", "full"):
        if method == "baseline":
            stage_spec(input_dir)
        else:
            final_spec_stage(out_final)
    if stage in ("detect", "full"):
        if method == "baseline":
            stage_detect(out_root, r_min=r_min, max_pairs=max_pairs)
        else:
            final_supperss_stage(out_final, r_min=r_min, max_pairs=max_pairs)
    if stage in ("suppress", "full"):
        if method == "baseline":
            stage_mask(out_root, delta_r=delta_r, delta_theta=delta_theta, alpha=alpha, method=method)
        else:
            final_mask_stage(out_final, delta_r=delta_r, delta_theta=delta_theta, alpha=alpha)
    if stage in ("restore", "full"):
        if method == "baseline":
            stage_restore(out_root, guided_r=guided_r, guided_eps=guided_eps)
        else:
            final_restore_stage(out_final, guided_r=guided_r, guided_eps=guided_eps)
    if stage in ("adapt_eval", "full"):
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = Path.cwd() / results_path
        data_root = input_dir if input_dir.name in ("real", "synth") else input_dir.parent
        stage_eval(data_root, out_root, out_final, results_path)


if __name__ == "__main__":
    main()
