import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _norm8(x):
    x = x.astype(np.float64)
    x -= x.min()
    if x.max() > 0:
        x /= (x.max() + 1e-12)
    return (x * 255).astype(np.uint8)

def show_spectrum_and_notch(logmag_before, mag, mask, out_dir):
    """
    - 三張圖合併顯示：before / after / notch_heatmap
    - notch_heatmap 從 out_dir 讀取，而不是重新計算
    """
    out_dir = Path(out_dir)

    # -----------------------------
    # 產生 before 圖
    # -----------------------------
    before_vis = cv2.applyColorMap(_norm8(logmag_before),
                                   cv2.COLORMAP_INFERNO)
    before_vis = cv2.cvtColor(before_vis, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 產生 after 圖
    # -----------------------------
    mag_after = mag * np.clip(mask, 1e-6, 1.0)
    logmag_after = np.log1p(mag_after)

    after_vis = cv2.applyColorMap(_norm8(logmag_after),
                                  cv2.COLORMAP_INFERNO)
    after_vis = cv2.cvtColor(after_vis, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # notch_heatmap.png 從資料夾讀取
    # -----------------------------
    heatmap_path = out_dir / "notch_heatmap.png"
    if not heatmap_path.exists():
        raise FileNotFoundError(f"{heatmap_path} 不存在")

    heatmap_vis = cv2.cvtColor(cv2.imread(str(heatmap_path)),
                               cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 合併顯示三張圖
    # -----------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(before_vis)
    plt.title("Spectrum Before")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(after_vis)
    plt.title("Spectrum After")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(heatmap_vis)
    plt.title("Notch Heatmap")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

