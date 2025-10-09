from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
from lib.plot import show3
from skimage.filters import threshold_otsu
from skimage.restoration import denoise_tv_chambolle


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

    np.clip(scaled, 0.0, 1.0, out=scaled)
    return scaled, float(p1), float(p2)


def n4_bias_correction_sitk(volume: np.ndarray) -> Tuple[np.float32, np.ndarray]:
    image = sitk.GetImageFromArray(volume.astype(np.float32))

    mask_array: np.int8 = volume > threshold_otsu(volume)
    mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])

    corrected_image = corrector.Execute(image, mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image)

    corrected = sitk.GetArrayFromImage(corrected_image)
    bias_field = np.exp(sitk.GetArrayFromImage(log_bias_field))

    return bias_field, corrected


def denoise_tv(
    volume: np.ndarray,
    weight: float = 0.05,
    n_iter_max: int = 200,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    v = volume.astype(np.float32, copy=False)

    if mask is None:
        out = denoise_tv_chambolle(
            v,
            weight=weight,
            max_num_iter=n_iter_max,
            channel_axis=None,
        ).astype(np.float32, copy=False)
    else:
        den = denoise_tv_chambolle(
            v, weight=weight, max_num_iter=n_iter_max, channel_axis=None
        ).astype(np.float32, copy=False)
        out = v.copy()
        out[mask.astype(bool)] = den[mask.astype(bool)]

    np.clip(out, 0.0, 1.0, out=out)
    return out


if __name__ == "__main__":
    # Testing purposes
    from lib.tif import read_tif, tif_to_float32

    v_raw = read_tif("data/Litarion.tif")
    v = tif_to_float32(v_raw)
    v_scaled, p1, p2 = clip_and_scale(v, pcts=(0.5, 99.5))

    print(f"Scaled to [0,1] using percentiles: {p1:.2e}-{p2:.2e}")
    print("min, max, unique=", np.min(v), np.max(v), np.unique(v))

    specimen_mask = v_scaled > v_scaled.mean()
    tv_denoised = denoise_tv(v_scaled, weight=0.05, n_iter_max=200, mask=specimen_mask)

    show3(tv_denoised)
