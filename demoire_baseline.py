import argparse, os, yaml
import pandas as pd

from src.freq.fft2d import fft2d
from src.io.loader import DataLoader
from src.freq.peak_detect import detect_peaks
from src.notch.apply import run_mask_stage, run_restore_stage
# from src.eval.summarize import evaluate_folder

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/synth')
    parser.add_argument('--out', type=str, default='out/synth')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--stage', type=str, choices=['spec', 'peaks', 'mask', 'restore', 'eval'])

    args = parser.parse_args()

    stage = args.stage
    loader = DataLoader('data', batch_size=4, color_mode='bgr')

    # read config/baseline.yaml if needed
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # print(type(config['guided']['eps']))

    if stage == 'spec':
        print('Generating specpacks...')
        fft2d(loader=loader)
        print('Specpacks generation completed.')
    elif stage == 'peaks':
        print('Generating peaks...')
        detect_peaks(r_min=config['params']['r_min'], max_pairs=config['params']['K'])
        print('Peaks generation completed.')
    elif stage == 'mask':
        print('Generating maskpacks...')
        run_mask_stage(delta_r=config['params']['delta_r'],
                       delta_theta_deg=config['params']['delta_theta'],
                       alpha=config['params']['alpha'])
        print('Maskpacks generation completed.')
    elif stage == 'restore':
        print('Restoring images...')
        run_restore_stage(guided_r=config['guided']['r'],
                          guided_eps=config['guided']['eps'])
        print('Image restoration completed.')
    elif stage == 'eval':
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))

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

        out_csv = os.path.join('./', "results.csv")
        df.to_csv(out_csv, index=False)

        print(f"[OK] Saved → {out_csv}")


if __name__ == '__main__':
    main()

# test viz
# import cv2

# from src.io.artifact import load_maskpack, load_specpack
# from src.viz.spectrum import show_spectrum_and_notch
# from src.viz.panels import show_before_after_panels
# from pathlib import Path

# if __name__ == "__main__":

#     dir_path = Path("out/synth/0001")
#     mag, phase, logmag = load_specpack(dir_path)
#     maskpack = load_maskpack(dir_path)

#     show_spectrum_and_notch(logmag, mag, maskpack.mask, dir_path)

#     before_img = cv2.imread('data/synth/0001.jpg')
#     after_img = cv2.imread('out/synth/0001/demoire.png')
#     show_before_after_panels(before_img, after_img)
