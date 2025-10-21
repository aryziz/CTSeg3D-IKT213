from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import skimage
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import ball, binary_erosion, binary_dilation, remove_small_objects
from skimage.measure import label as cc_label

# import helpers
from preprocess import clip_and_scale, gaussian_3d, nlm2d_opencv
@dataclass
class OtsuSeedingConfig:
    # Preprocessing

