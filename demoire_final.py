import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from src.freq.peak_detect import _detect_peaks, save_peaks_json, detect_peaks
from src.freq.fft2d import fft2d
from src.notch.apply import run_mask_stage, run_restore_stage
from src.eval.summarize import evaluate_pair_folder
from src.io.artifact import load_specpack, save_specpack , load_specpack_adv, load_peaks
from src.io.loader import DataLoader


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

def infer_split(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for token in ("synth", "real"):
        if token in parts:
            return token
    return path.name

def stage_spec(input_dir: Path, out_root: Path, split: str) -> None:
    print("[spec] Calling fft2d() to generate specpack.npz ...")
    loader = DataLoader(input_dir, batch_size=4, color_mode='bgr')
    fft2d(loader=loader)

    # sanity: check again
    after = list(iter_result_dirs(out_root, split))
    if not after:
        print(f"[spec] Warning: no specpack found under {out_root}/{split} after fft2d(). "
              f"Ensure input/out paths match fft2d defaults.")

def iter_result_dirs(out_root: Path, split: str) -> Iterable[Path]:
    cat_dir = out_root / split
    if not cat_dir.exists():
        return []
    for p in sorted(cat_dir.rglob("*")):
        if p.is_dir() and (p / "specpack.npz").exists():
            yield p

def stage_detect(out_root: Path, r_min: float, max_pairs: int) -> None:
    detect_peaks(
        r_min=r_min,
        max_pairs=max_pairs,
        path=str(out_root)
    )

def stage_mask(out_root: Path, split: str, delta_r: float, delta_theta: float, alpha: float, method: str) -> None:
    img_dirs = list(iter_result_dirs(out_root, split))

    run_mask_stage(
        out_root=str(out_root),
        categories=(split,),
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

def stage_restore(out_root: Path, split: str, guided_r: int, guided_eps: float, method: str) -> None:
    run_restore_stage(
        out_root=str(out_root),
        categories=(split,),
        guided_r=guided_r,
        guided_eps=guided_eps,
    )

def stage_eval(data_root: Path, out_root: Path, split: str, results_path: Path, method: str) -> None:
    rows = evaluate_pair_folder(
        str(data_root),
        str(out_root),
        str(out_root),
        split,
    )
    if not rows:
        print("[eval] No rows to save.")
        return
    df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"[eval] Saved {results_path}")

def final_supperss_stage(out_root: Path, r_min: float, max_pairs: int) -> None:
    pass

def final_mask_stage(out_root: Path, split: str, delta_r: float, delta_theta: float, alpha: float, method: str) -> None:
    pass

def final_restore_stage(out_root: Path, split: str, guided_r: int, guided_eps: float, method: str) -> None:
    pass

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='data/synth', help="Input folder (e.g., data/synth)")
    parser.add_argument("--out", type=str, default='out/synth', help="Output root or split folder (e.g., out or out/synth)")
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
    out_root = args.out

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
        stage_spec('data', out_root, split)
    if stage in ("detect", "full"):
        if method == "baseline":
            stage_detect(out_root, r_min=r_min, max_pairs=max_pairs)
        else:
            final_supperss_stage(out_root, r_min=r_min, max_pairs=max_pairs)
    if stage in ("suppress", "full"):
        if method == "baseline":
            stage_mask(out_root, split, delta_r=delta_r, delta_theta=delta_theta, alpha=alpha, method=method)
        else:
            final_mask_stage(out_root, split, delta_r=delta_r, delta_theta=delta_theta, alpha=alpha, method=method)
    if stage in ("restore", "full"):
        if method == "baseline":
            stage_restore(out_root, split, guided_r=guided_r, guided_eps=guided_eps, method=method)
        else:
            final_restore_stage(out_root, split, guided_r=guided_r, guided_eps=guided_eps, method=method)
    if stage in ("adapt_eval", "full"):
        results_path = Path(args.results)
        if not results_path.is_absolute():
            results_path = Path.cwd() / results_path
        data_root = input_dir if input_dir.name in ("real", "synth") else input_dir.parent
        stage_eval(data_root, out_root, split, results_path, method=args.method)


if __name__ == "__main__":
    main()
