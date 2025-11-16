
from typing import Iterable, Tuple, List, Dict, Any

import numpy as np


def _angular_distance_deg(theta: np.ndarray, center_deg: float) -> np.ndarray:
    """
    Compute smallest angular distance between each element in theta
    and center_deg, in degrees, modulo 360, folded to [-180, 180].
    """
    diff = np.abs(theta - center_deg)
    # wrap to [-180, 180]
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.abs(diff)


def build_notch_masks_from_polar(
    shape: Tuple[int, int],
    peaks: Iterable[Dict[str, Any]],
    delta_r: float = 4.0,
    delta_theta_deg: float = 6.0,
    alpha: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build cosine-tapered band-stop notch masks from a list of peaks.

    shape: (H, W) of spectrum magnitude
    peaks: list of dicts, each having either:
           - {"r": float, "theta_deg": float, "strength": ...}
           - or {"radius": float, "angle": float, "avg_strength": ...}
    delta_r: radial half width in pixels
    delta_theta_deg: angular half width in degrees
    alpha: attenuation strength (1=no attenuation, 1-alpha=center attenuation)
    """
    H, W = shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0

    # coordinate grid
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]
    dy = y - cy
    dx = x - cx
    R = np.sqrt(dx * dx + dy * dy)  # radius
    Theta = np.degrees(np.arctan2(dy, dx))  # [-180, 180]

    base_mask = np.ones_like(R, dtype=np.float32)
    each_masks: List[np.ndarray] = []

    for peak in peaks:
        # unify key names
        if "r" in peak:
            r0 = float(peak["r"])
        else:
            r0 = float(peak.get("radius", 0.0))

        if "theta_deg" in peak:
            theta0 = float(peak["theta_deg"])
        else:
            theta0 = float(peak.get("angle", 0.0))

        # radial & angular distance
        dr = np.abs(R - r0)
        dtheta = _angular_distance_deg(Theta, theta0)

        radial_region = dr <= delta_r
        angular_region = dtheta <= delta_theta_deg
        region = radial_region & angular_region

        if not np.any(region):
            # nothing to do for this peak
            continue

        # cosine taper in radial dimension
        t_r = np.clip(dr / delta_r, 0.0, 1.0)
        w_r = 0.5 * (1.0 + np.cos(np.pi * t_r))  # 1 at center, 0 at boundary

        # cosine taper in angular dimension
        t_theta = np.clip(dtheta / delta_theta_deg, 0.0, 1.0)
        w_theta = 0.5 * (1.0 + np.cos(np.pi * t_theta))

        w = w_r * w_theta  # combined smooth window

        # attenuation: 1 outside; 1 - alpha at center
        atten = 1.0 - alpha * w

        mask_i = np.ones_like(R, dtype=np.float32)
        mask_i[region] = atten[region]

        each_masks.append(mask_i)
        base_mask *= mask_i

    if each_masks:
        each_arr = np.stack(each_masks, axis=0).astype(np.float32)
    else:
        # no detected peaks -> empty K dimension
        each_arr = np.ones((0, H, W), dtype=np.float32)

    return base_mask.astype(np.float32), each_arr
