import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def show_before_after_panels(before, after, out_png=None,
                             crop=None, zoom=2):
    """
    - 改成用 plt.imshow 顯示
    - 不再寫入 cv2.imwrite（除非 out_png 給出）
    - before / after / zoom 合併顯示在一張圖
    """
    b = before.copy()
    a = after.copy()

    # 轉成 BGR → RGB for imshow
    if b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

    b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    H, W = b.shape[:2]

    # 自動中心 crop
    if crop is None:
        h = min(128, H // 2)
        w = min(128, W // 2)
        y = (H - h) // 2
        x = (W - w) // 2
        crop = (y, x, h, w)

    y, x, h, w = crop

    # 畫框
    b_box = b_rgb.copy()
    a_box = a_rgb.copy()
    cv2.rectangle(b_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(a_box, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Zoom in
    cb = b_rgb[y:y + h, x:x + w]
    ca = a_rgb[y:y + h, x:x + w]
    cbz = cv2.resize(cb, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    caz = cv2.resize(ca, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

    # ---------------------------
    # 使用 matplotlib 合併顯示
    # ---------------------------
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(b_box)
    plt.title("Before")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(a_box)
    plt.title("After")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cbz)
    plt.title("Before (Zoom)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(caz)
    plt.title("After (Zoom)")
    plt.axis("off")

    plt.tight_layout()

    # 若 out_png 有給 → 儲存整張面板
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_png), dpi=200)

    plt.show()


def show_baseline_final_panels(before, baseline, final, out_png=None,
                               crop=None, zoom=2):
    """
    Render input / baseline / final with a shared crop and zoom; saves to out_png if provided.
    """
    b = before.copy()
    bl = baseline.copy()
    f = final.copy()

    # ensure 3-channel for plotting
    if b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    if bl.ndim == 2:
        bl = cv2.cvtColor(bl, cv2.COLOR_GRAY2BGR)
    if f.ndim == 2:
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

    b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    bl_rgb = cv2.cvtColor(bl, cv2.COLOR_BGR2RGB)
    f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    H, W = b.shape[:2]
    if crop is None:
        h = min(128, H // 2)
        w = min(128, W // 2)
        y = (H - h) // 2
        x = (W - w) // 2
        crop = (y, x, h, w)
    y, x, h, w = crop

    def draw_box(img_rgb):
        boxed = img_rgb.copy()
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return boxed

    b_box = draw_box(b_rgb)
    bl_box = draw_box(bl_rgb)
    f_box = draw_box(f_rgb)

    cb = b_rgb[y:y + h, x:x + w]
    cbl = bl_rgb[y:y + h, x:x + w]
    cf = f_rgb[y:y + h, x:x + w]

    cbz = cv2.resize(cb, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    cblz = cv2.resize(cbl, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    cfz = cv2.resize(cf, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); plt.imshow(b_box); plt.title("Input"); plt.axis("off")
    plt.subplot(2, 3, 2); plt.imshow(bl_box); plt.title("Baseline"); plt.axis("off")
    plt.subplot(2, 3, 3); plt.imshow(f_box); plt.title("Final"); plt.axis("off")
    plt.subplot(2, 3, 4); plt.imshow(cbz); plt.title("Input (Zoom)"); plt.axis("off")
    plt.subplot(2, 3, 5); plt.imshow(cblz); plt.title("Baseline (Zoom)"); plt.axis("off")
    plt.subplot(2, 3, 6); plt.imshow(cfz); plt.title("Final (Zoom)"); plt.axis("off")

    plt.tight_layout()

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_png), dpi=200)

    plt.show()
