
import numpy as np
import cv2


def _boxfilter(img: np.ndarray, r: int) -> np.ndarray:
    """
    Fast box filter using OpenCV.
    img: float32 array, HxW
    r: radius -> kernel size = 2*r+1
    """
    ksize = 2 * r + 1
    return cv2.boxFilter(
        img,
        ddepth=-1,
        ksize=(ksize, ksize),
        borderType=cv2.BORDER_REFLECT,
    )


def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 6, eps: float = 1e-3) -> np.ndarray:
    """
    Gray-scale guided filter.

    I: guidance image, float32 in [0, 1] or arbitrary range
    p: filtering input (we use p=I for self-guided)
    r: radius of local window
    eps: regularization parameter
    """
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    mean_I = _boxfilter(I, r)
    mean_p = _boxfilter(p, r)

    mean_Ip = _boxfilter(I * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _boxfilter(I * I, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _boxfilter(a, r)
    mean_b = _boxfilter(b, r)

    q = mean_a * I + mean_b
    return q
