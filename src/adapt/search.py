# src/adapt/search.py
import os
import json
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Tuple, Union

import cv2
import yaml

from .criteria import compute_residual_score, compute_edge_score


@dataclass
class SearchSpace:
    delta_r: List[float]
    delta_theta: List[float]
    alpha: List[float]
    residual_w: float = 1.0
    edge_w: float = 1.0
    max_evals: Optional[int] = None
    patience: int = 10
    min_improve: float = 1e-6


def load_search_space(final_yaml_path: str) -> SearchSpace:
    with open(final_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    s = cfg.get("search", cfg)

    # allow flexible key names
    delta_r = s.get("delta_r") or s.get("dr") or s.get("Δr") or []
    delta_theta = s.get("delta_theta") or s.get("dtheta") or s.get("Δθ") or []
    alpha = s.get("alpha") or s.get("α") or []

    residual_w = float(s.get("residual_w", 1.0))
    edge_w = float(s.get("edge_w", 1.0))
    max_evals = s.get("max_evals", None)
    patience = int(s.get("patience", 10))
    min_improve = float(s.get("min_improve", 1e-6))

    if not delta_r or not delta_theta or not alpha:
        raise ValueError("Search space is empty. Please define delta_r/delta_theta/alpha in final.yaml.")

    return SearchSpace(
        delta_r=list(delta_r),
        delta_theta=list(delta_theta),
        alpha=list(alpha),
        residual_w=residual_w,
        edge_w=edge_w,
        max_evals=max_evals,
        patience=patience,
        min_improve=min_improve,
    )


def _normalize_restore_output(out: Union[str, Any]):
    if isinstance(out, str):
        img = cv2.imread(out)
        return img
    return out


def compute_combo_score(input_img, demoire_img, residual_w=1.0, edge_w=1.0, mask=None) -> Tuple[float, float, float]:
    """
    Combine proxies into a single ranking score.
    """
    residual = compute_residual_score(input_img, demoire_img, mask=mask)  # lower better
    edge = compute_edge_score(input_img, demoire_img)                     # higher better
    combo = edge_w * edge - residual_w * residual
    return float(combo), float(residual), float(edge)


def grid_search_image(
    input_path: str,
    space: SearchSpace,
    restore_fn: Callable[[str, Dict[str, Any]], Union[str, Any]],
    mask: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    restore_fn signature:
        out = restore_fn(input_path, params)
    out can be:
        - demoired image ndarray (BGR)
        - or path to demoired image
    """
    input_img = cv2.imread(input_path)
    if input_img is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    best = {
        "params": None,
        "combo": -1e18,
        "residual": None,
        "edge": None,
    }

    no_improve = 0
    eval_count = 0

    for dr, dth, a in itertools.product(space.delta_r, space.delta_theta, space.alpha):
        params = {"delta_r": dr, "delta_theta": dth, "alpha": a}

        out = restore_fn(input_path, params)
        demo_img = _normalize_restore_output(out)
        if demo_img is None:
            continue

        combo, residual, edge = compute_combo_score(
            input_img, demo_img, residual_w=space.residual_w, edge_w=space.edge_w, mask=mask
        )

        eval_count += 1
        improved = combo > best["combo"] + space.min_improve
        if improved:
            best.update({"params": params, "combo": combo, "residual": residual, "edge": edge})
            no_improve = 0
        else:
            no_improve += 1

        if space.max_evals is not None and eval_count >= int(space.max_evals):
            break
        if no_improve >= space.patience:
            break

    return best


def grid_search_dataset(
    input_list: List[str],
    final_yaml_path: str,
    restore_fn: Callable[[str, Dict[str, Any]], Union[str, Any]],
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    space = load_search_space(final_yaml_path)

    results = {}
    for p in input_list:
        best = grid_search_image(p, space, restore_fn)
        results[p] = best

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results
