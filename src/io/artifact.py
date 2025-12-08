# artifacts.py — load/save helpers for specpack.npz, peaks.json, maskpack.npz
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ---------------------------
# Dataclasses
# ---------------------------

@dataclass
class SpecPack:
    mag: np.ndarray          # (H, W)
    phase: np.ndarray        # (H, W)
    log_magnitude: np.ndarray  # (H, W)

@dataclass
class SpecPackAdv:
    mag: np.ndarray          # (H, W)
    logmag: np.ndarray       # (H, W)
    phase: np.ndarray        # (H, W)
    logpolar: np.ndarray     # (H', W')
    orient_energy: np.ndarray  # (H, W, C)

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

@dataclass
class MaskPackMs:
    mask_ms: np.ndarray   # (H, W)
    each_ms: np.ndarray   # (K, H, W)
    params_ms: dict     # all ms notch parameters
# ---------------------------
# specpack.npz
# ---------------------------
def load_specpack(path: Path) -> SpecPack:
    """
    Load specpack.npz from directory.
    Returns SpecPack(mag, phase, log_magnitude)
    """
    f = Path(path) / "specpack.npz"
    d = np.load(f)
    return SpecPack(mag=d['mag'], phase=d['phase'], log_magnitude=d['logmag'])

def load_specpack_adv(path: Path) -> SpecPackAdv:
    """
    Load advanced specpack.npz from directory.
    Returns SpecPackAdv(mag, logmag, phase, logpolar, orient_energy)
    """
    f = Path(path) / "specpack.npz"
    d = np.load(f)
    return SpecPackAdv(mag=d['mag'],
                       logmag=d['logmag'],
                       phase=d['phase'],
                       logpolar=d['logpolar'],
                       orient_energy=d['orient_energy'])

# ---------------------------
# peaks.json
# ---------------------------
def load_peaks(path: Path) -> list[Peak]:
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

def load_maskpack_(path: Path) -> MaskPackMs:
    """
    Load maskpack_ms.npz
    Returns MaskPackMs(mask_ms, each_ms, params_ms)
    """
    f = Path(path) / "maskpack_ms.npz"
    d = np.load(f, allow_pickle=True)
    mask_ms = d['mask_ms']
    each_ms = d['each_ms']
    params_ms = d['params_ms'].item() if isinstance(d['params_ms'], np.ndarray) else d['params_ms']
    return MaskPackMs(mask_ms=mask_ms, each_ms=each_ms, params_ms=params_ms)
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