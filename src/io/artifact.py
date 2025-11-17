# artifacts.py — load/save helpers for specpack.npz, peaks.json, maskpack.npz
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ---------------------------
# Dataclasses
# ---------------------------
@dataclass
class Peak:
    r: float
    theta_deg: float
    strength: float

@dataclass
class MaskPack:
    mask: np.ndarray      # (H, W)
    each: np.ndarray      # (K, H, W) or empty
    params: dict          # all notch parameters

# ---------------------------
# specpack.npz
# ---------------------------
def load_specpack(path: Path):
    """
    Load specpack.npz from directory.
    Returns:
        mag   : float32[H,W]
        phase : float32[H,W]
        log_magnitude: float32[H,W]
    """
    f = Path(path) / "specpack.npz"
    d = np.load(f)
    return d['mag'], d['phase'], d['log_magnitude']

# ---------------------------
# peaks.json
# ---------------------------
def load_peaks(path: Path):
    """
    Load peaks.json → list[Peak]
    """
    f = Path(path) / "peaks.json"
    with open(f, 'r', encoding='utf-8') as fp:
        arr = json.load(fp)
    return [Peak(**p) for p in arr]

# ---------------------------
# maskpack.npz
# ---------------------------
def load_maskpack(path: Path) -> MaskPack:
    """
    Load maskpack.npz
    Returns MaskPack(mask, each, params)
    """
    f = Path(path) / "maskpack.npz"
    d = np.load(f, allow_pickle=True)
    mask = d['mask']
    each = d['each']
    params = d['params'].item() if isinstance(d['params'], np.ndarray) else d['params']
    return MaskPack(mask=mask, each=each, params=params)


# ---------------------------
# SAVE functions
# ---------------------------

def save_specpack(path: Path, mag: np.ndarray, phase: np.ndarray, logmag: np.ndarray):
    """
    Save specpack.npz
    {"mag": HxW, "phase": HxW, "logmag": HxW}
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "specpack.npz",
                        mag=mag.astype(np.float32),
                        phase=phase.astype(np.float32),
                        logmag=logmag.astype(np.float32))


def save_peaks(path: Path, peaks: list[Peak]):
    """
    Save peaks.json
    Format: [{"r": ..., "theta_deg": ..., "strength": ...}, ...]
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    arr = [
        {"r": p.r, "theta_deg": p.theta_deg, "strength": p.strength}
        for p in peaks
    ]
    with open(path / "peaks.json", 'w', encoding='utf-8') as fp:
        json.dump(arr, fp, indent=2, ensure_ascii=False)


def save_maskpack(path: Path, mask: np.ndarray, each: np.ndarray, params: dict):
    """
    Save maskpack.npz
    mask: (H,W)
    each: (K,H,W)
    params: dict
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # ensure correct dtype
    mask = mask.astype(np.float32)
    each = each.astype(np.float32) if isinstance(each, np.ndarray) else each

    np.savez_compressed(path / "maskpack.npz",
                        mask=mask,
                        each=each,
                        params=params)