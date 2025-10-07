from typing import Tuple

import numpy as np


def clip_and_scale(
    v: np.ndarray, pcts: Tuple[float, float] = (0.5, 99.5), eps: float = 1e-6
) -> Tuple[np.ndarray, float, float]:
    """Clip and scale a value to the range [0, 1] based on given percentiles.

    Args:
        v: The input value or array to be clipped and scaled.
        pcts: A tuple containing the lower and upper percentiles for clipping.
        eps: A small epsilon value to prevent division by zero.

    Returns:
        A list containing the clipped and scaled array,
        the lower percentile value, and the upper percentile value.
    """
    p1, p2 = np.percentile(v, pcts)

    if not np.isfinite(p1) or not np.isfinite(p2):
        raise ValueError("Percentiles are NaN/Inf")

    if p2 <= p1:
        scaled = np.zeros_like(v, dtype=np.float32)
        return scaled, float(p1), float(p2)
    else:
        clipped: np.ndarray = np.clip(v, p1, p2)
        scaled_v: np.ndarray = (clipped - p1) / (p2 - p1 + eps)
        scaled = scaled_v.astype(np.float32, copy=False)

    scaled[scaled == 0] = 0.0
    np.clip(scaled, 0.0, 1.0, out=scaled)
    return scaled, float(p1), float(p2)


if __name__ == "__main__":
    # Testing purposes
    from utils.tif import read_tif, tif_to_float32

    v_raw = read_tif("data/Litarion.tif")
    v = tif_to_float32(v_raw)
    v_scaled, p1, p2 = clip_and_scale(v, pcts=(0.5, 99.5))

    print(f"Scaled to [0,1] using percentiles: {p1:.2e}-{p2:.2e}")
    print("min, max, unique=", np.min(v), np.max(v), np.unique(v))
