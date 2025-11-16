import os
import cv2
import pandas as pd
from .metrics import psnr, ssim, niqe, brisque, measure_time


IMG_EXT = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]

# def is_image(fname):
#     return os.path.splitext(fname.lower())[1] in IMG_EXT

# Find GT 
def find_gt(data_root, rel_path):
    
    parent = os.path.dirname(rel_path)       
    img_base = parent                        

    for ext in IMG_EXT:
        gt_path = os.path.join(data_root, img_base + "_gt" + ext)
        if os.path.exists(gt_path):
            return gt_path
    return None

def find_demoire(out_dir):
    for ext in IMG_EXT:
        p = os.path.join(out_dir, "demoire" + ext)
        if os.path.exists(p):
            return p, ext
    return None, None

def evaluate_folder(data_root, out_root, split):
    """
    split: 'synth' or 'real'
    data_root: data/synth or data/real
    out_root: out/synth or out/real
    """
    rows = []

    for subdir in os.listdir(out_root):
        out_dir = os.path.join(out_root, subdir)
        if not os.path.isdir(out_dir):
            continue

        demo_path, ext = find_demoire(out_dir)
        if demo_path is None:
            continue

        img = cv2.imread(demo_path)
        if img is None:
            continue

        rel = os.path.join(subdir, "demoire" + ext)


        if split == "synth":
            gt_path = find_gt(data_root, rel)
            if gt_path is not None:
                gt = cv2.imread(gt_path)

                psnr_val, t1 = measure_time(psnr, img, gt)
                ssim_val, t2 = measure_time(ssim, img, gt)

                rows.append({
                    "split": split,
                    "path": rel,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "niqe": None,
                    "brisque": None,
                    "time_s": t1 + t2
                })
                continue

        # Real (Without GT)
        niqe_val, t1 = measure_time(niqe, img)
        brisque_val, t2 = measure_time(brisque, img)

        rows.append({
            "split": split,
            "path": rel,
            "psnr": None,
            "ssim": None,
            "niqe": niqe_val,
            "brisque": brisque_val,
            "time_s": t1 + t2
        })

    return rows

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    data_dir = os.path.join(project_root, "data")
    out_dir = os.path.join(project_root, "out")

    synth_data = os.path.join(data_dir, "synth")
    real_data  = os.path.join(data_dir, "real")

    synth_out = os.path.join(out_dir, "synth")
    real_out  = os.path.join(out_dir, "real")

    rows = []
    if os.path.exists(synth_out):
        rows += evaluate_folder(synth_data, synth_out, "synth")
    if os.path.exists(real_out):
        rows += evaluate_folder(real_data, real_out, "real")

    df = pd.DataFrame(rows)

    out_csv = os.path.join(project_root, "results.csv")
    df.to_csv(out_csv, index=False)

    print(f"[OK] Saved → {out_csv}")
    print(df.head())
