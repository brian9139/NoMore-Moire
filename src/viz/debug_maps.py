import matplotlib.pyplot as plt
import numpy as np


def _show_map(arr: np.ndarray, title: str = "", cmap: str = "magma"):
    arr = np.asarray(arr)
    plt.imshow(arr, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")


def show_debug_maps(logpolar=None, orient=None, mask=None, mask_ms=None, residual=None):
    """
    Quick visualization of B/C intermediates:
      - logpolar: log-polar magnitude
      - orient: orientation/energy map
      - mask: single-scale notch mask or heatmap
      - mask_ms: multi-scale mask heatmap
      - residual: PnP residual / influence map
    Only the provided arrays will be shown.
    """
    entries = [
        ("logpolar", logpolar, "magma"),
        ("orient", orient, "magma"),
        ("mask", mask, "inferno"),
        ("mask_ms", mask_ms, "inferno"),
        ("residual", residual, "viridis"),
    ]
    to_show = [(t, a, c) for t, a, c in entries if a is not None]
    if not to_show:
        return

    n = len(to_show)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (title, arr, cmap) in enumerate(to_show, 1):
        plt.subplot(rows, cols, i)
        _show_map(arr, title=title, cmap=cmap)
    plt.tight_layout()
    plt.show()
