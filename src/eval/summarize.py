# src/eval/summarize.py
import os
import argparse
import cv2
import pandas as pd
# from src.eval.metrics import psnr, ssim, niqe, brisque, measure_time
from .metrics import psnr, ssim, niqe, brisque, measure_time

IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]


def find_gt(data_root, rel_path):
    parent = os.path.dirname(rel_path)
    img_base = parent
    for ext in IMG_EXT:
        gt_path = os.path.join(data_root, img_base + "_gt" + ext)
        if os.path.exists(gt_path):
            return gt_path
    return None


def find_demoire(out_dir, is_final=False):
    for ext in IMG_EXT:
        p = None
        if is_final == True:
            p = os.path.join(out_dir, "demoire_final" + ext)
        else:
            p = os.path.join(out_dir, "demoire" + ext)
        if os.path.exists(p):
            return p, ext
    return None, None


def _load_demoire(out_root, subdir, is_final=False):
    out_dir = os.path.join(out_root, subdir)
    if not os.path.isdir(out_dir):
        return None, None, None
    demo_path, ext = find_demoire(out_dir, is_final=is_final)
    if demo_path is None:
        return None, None, None
    img = cv2.imread(demo_path)
    if img is None:
        return None, None, None
    if is_final == True:
        rel = os.path.join(subdir, "demoire_final" + ext)
    else:
        rel = os.path.join(subdir, "demoire" + ext)
    # print(img, rel, ext)
    return img, rel, ext


def evaluate_pair_folder(data_root, out_base_root, out_final_root, split):
    rows = []

    base_subs = set(os.listdir(out_base_root)) if os.path.exists(out_base_root) else set()
    final_subs = set(os.listdir(out_final_root)) if os.path.exists(out_final_root) else set()
    all_subs = sorted([s for s in (base_subs | final_subs)])
    print(f"Evaluating {split} data: {len(all_subs)} subdirectories found.")

    for subdir in all_subs:
        img_b, rel_b, _ = _load_demoire(out_base_root, subdir)
        img_f, rel_f, _ = _load_demoire(out_final_root, subdir, is_final=True)
        print(f"Evaluating subdir: {subdir}, baseline image: {rel_b}, final image: {rel_f}")
        # choose a rel_path for reporting (prefer baseline)
        rel = rel_b or rel_f
        if rel is None:
            continue

        row = {
            "split": split,
            "rel_path": rel,
            "psnr_base": None,
            "psnr_final": None,
            "ssim_base": None,
            "ssim_final": None,
            "niqe_base": None,
            "niqe_final": None,
            "brisque_base": None,
            "brisque_final": None,
            "time_base": None,
            "time_final": None,
            "time_overhead": None,
            "psnr_gain": None,
            "ssim_gain": None,
            "niqe_gain": None,
            "brisque_gain": None,
        }

        # Synth: with GT
        if split == "synth":
            gt_path = find_gt(data_root, rel)
            if gt_path is not None:
                gt = cv2.imread(gt_path)

                t_base = 0.0
                t_final = 0.0

                if img_b is not None:
                    psnr_val, t1 = measure_time(psnr, img_b, gt)
                    ssim_val, t2 = measure_time(ssim, img_b, gt)
                    row["psnr_base"] = psnr_val
                    row["ssim_base"] = ssim_val
                    t_base = t1 + t2

                if img_f is not None:
                    psnr_val, t1 = measure_time(psnr, img_f, gt)
                    ssim_val, t2 = measure_time(ssim, img_f, gt)
                    row["psnr_final"] = psnr_val
                    row["ssim_final"] = ssim_val
                    t_final = t1 + t2

                row["time_base"] = t_base if img_b is not None else None
                row["time_final"] = t_final if img_f is not None else None
                if row["time_base"] is not None and row["time_final"] is not None:
                    row["time_overhead"] = row["time_final"] - row["time_base"]

                if row["psnr_base"] is not None and row["psnr_final"] is not None:
                    row["psnr_gain"] = row["psnr_final"] - row["psnr_base"]
                if row["ssim_base"] is not None and row["ssim_final"] is not None:
                    row["ssim_gain"] = row["ssim_final"] - row["ssim_base"]

                rows.append(row)
                continue

        # Real: no GT, use NIQE/BRISQUE if available
        t_base = 0.0
        t_final = 0.0

        if img_b is not None:
            niqe_val, t1 = measure_time(niqe, img_b)
            brisque_val, t2 = measure_time(brisque, img_b)
            row["niqe_base"] = niqe_val
            row["brisque_base"] = brisque_val
            t_base = t1 + t2

        if img_f is not None:
            niqe_val, t1 = measure_time(niqe, img_f)
            brisque_val, t2 = measure_time(brisque, img_f)
            row["niqe_final"] = niqe_val
            row["brisque_final"] = brisque_val
            t_final = t1 + t2

        row["time_base"] = t_base if img_b is not None else None
        row["time_final"] = t_final if img_f is not None else None
        if row["time_base"] is not None and row["time_final"] is not None:
            row["time_overhead"] = row["time_final"] - row["time_base"]

        # gains (positive means better)
        if row["niqe_base"] is not None and row["niqe_final"] is not None:
            row["niqe_gain"] = row["niqe_base"] - row["niqe_final"]
        if row["brisque_base"] is not None and row["brisque_final"] is not None:
            row["brisque_gain"] = row["brisque_base"] - row["brisque_final"]

        rows.append(row)

    return rows


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(project_root, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_base", default=os.path.join(project_root, "out"))
    parser.add_argument("--out_final", default=os.path.join(project_root, "out_final"))
    parser.add_argument("--output", default=os.path.join(project_root, "results_final.csv"))
    args = parser.parse_args()

    synth_data = os.path.join(data_dir, "synth")
    real_data  = os.path.join(data_dir, "real")

    synth_out_base = os.path.join(args.out_base, "synth")
    real_out_base  = os.path.join(args.out_base, "real")
    synth_out_final = os.path.join(args.out_final, "synth")
    real_out_final  = os.path.join(args.out_final, "real")

    rows = []
    if os.path.exists(synth_out_base) or os.path.exists(synth_out_final):
        rows += evaluate_pair_folder(synth_data, synth_out_base, synth_out_final, "synth")
    if os.path.exists(real_out_base) or os.path.exists(real_out_final):
        rows += evaluate_pair_folder(real_data, real_out_base, real_out_final, "real")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"[OK] Saved → {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()
