import numpy as np
from skimage.filters import window

def apply_window(image):
    return image * window('hann', image.shape)