from typing import Any, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import ball


# ---------- gradients ----------
def sobel_3d_gradient(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float32, copy=False)
    gx = ndi.sobel(v, axis=0, mode="nearest")
    gy = ndi.sobel(v, axis=1, mode="nearest")
    gz = ndi.sobel(v, axis=2, mode="nearest")
    gxy = np.hypot(gx, gy, dtype=np.float32)
    return np.hypot(gxy, gz, dtype=np.float32)


# ---------- utilities ----------


def multi_otsu_3d(
    volume: np.ndarray,
    classes: int = 3,
    min_bg_size: int = 5000,
    max_bg_size: int = 200000,
) -> Tuple[Any, Any, float | int]:
    """
    3D multi-Otsu segmentation with background exclusion.
    Returns thresholds, labeled volume, and interclass variance.
    """
    v = volume.astype(np.float32, copy=False)

    # Detect background using Otsu threshold
    try:
        otsu_threshold = threshold_otsu(v)
        background_threshold = otsu_threshold * 0.3
    except Exception:
        background_threshold = np.percentile(v, 2)

    # Create background mask from largest low-intensity region
    low_intensity_mask = v < background_threshold

    if np.any(low_intensity_mask):
        labeled_mask, num_features = ndi.label(low_intensity_mask)

        if num_features > 0:
            component_sizes = np.bincount(labeled_mask.ravel())
            if len(component_sizes) > 1:
                largest_component = np.argmax(component_sizes[1:]) + 1
                background_mask = labeled_mask == largest_component

                # Apply size constraints to background region
                region_size = component_sizes[largest_component]
                if min_bg_size <= region_size <= max_bg_size:
                    background_mask = ndi.binary_dilation(
                        background_mask, structure=ball(1)
                    )
                else:
                    background_mask = low_intensity_mask
            else:
                background_mask = low_intensity_mask
        else:
            background_mask = low_intensity_mask
    else:
        background_mask = np.zeros_like(v, dtype=bool)

    # Segment only foreground regions
    foreground_mask = ~background_mask
    foreground_data = v[foreground_mask]

    if foreground_data.size == 0:
        return np.array([]), np.zeros_like(v, dtype=np.uint8), float("nan")

    # Apply multi-Otsu to foreground
    thresholds = threshold_multiotsu(foreground_data, classes=classes)

    # Label the volume (0 = background, 1..classes = foreground classes)
    labeled_volume = np.zeros_like(v, dtype=np.uint8)

    if foreground_data.size == 0:
        # nothing to segment; return empty foreground and NaN variance
        return np.array([]), labeled_volume, float("nan")

    t = thresholds  # len = classes-1

    # First foreground class
    labeled_volume[foreground_mask & (v <= t[0])] = 1

    # Middle classes (if any)
    for i in range(1, len(t)):
        mask = foreground_mask & (v > t[i - 1]) & (v <= t[i])
        labeled_volume[mask] = i + 1

    # Last foreground class
    labeled_volume[foreground_mask & (v > t[-1])] = classes

    # Interclass variance over foreground classes (background=0 is excluded)
    N_fg = int(foreground_mask.sum())
    if N_fg == 0:
        interclass_variance = float("nan")
    else:
        global_mean = float(v[foreground_mask].mean())

        class_means = []
        class_weights = []
        for c in range(1, classes + 1):  # foreground classes only
            cmask = labeled_volume == c
            n = int(cmask.sum())
            if n == 0:
                continue
            class_means.append(float(v[cmask].mean()))
            class_weights.append(n / N_fg)

        interclass_variance = sum(
            w * (m - global_mean) ** 2 for w, m in zip(class_weights, class_means)
        )

    return thresholds, labeled_volume, interclass_variance
