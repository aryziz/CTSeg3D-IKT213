from typing import Any, Dict, Tuple

import cv2 as cv
import numpy as np
import SimpleITK as sitk
import tifffile as tiff
from scipy.ndimage import gaussian_filter
from skimage import exposure, restoration, util
from skimage.filters import threshold_otsu


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


def total_variation_filter(file_path: str) -> np.ndarray:
    try:
        image_stack = tiff.imread(file_path)
    except FileNotFoundError:
        print("File not found")
        exit()

    if image_stack.ndim != 3:
        print(f"Expected 3D input data, but got {image_stack.ndim} instead.")
        exit()

    image_float = util.img_as_float(image_stack)

    weight_value = 0.1

    print(f"Denoising {image_float.shape} with weight {weight_value}.")

    denoised_stack = restoration.denoise_tv_chambolle(image_float, weight=weight_value)

    print("Denoising complete.")

    return denoised_stack


def nlm2d_opencv(
    volume: np.ndarray,
    h: float = 0.9,
    template_window: int = 5,
    search_window: int = 13,
) -> np.ndarray:
    """Non-local means using openCV per-slice operations

    Args:
        volume (np.ndarray): 3D stack
        h (float, optional): How heavy denoising should be done. Defaults to 0.9.
        template_window (int, optional): Size of patch. Defaults to 5.
        search_window (int, optional): Size of search area. Defaults to 13.

    Returns:
        _type_: _description_
    """
    z, y, x = volume.shape
    out = np.empty_like(volume, dtype=np.float32)
    for i in range(z):
        img8 = np.clip(volume[i] * 255, 0, 255).astype(np.uint8)
        den8 = cv.fastNlMeansDenoising(
            img8,
            None,
            h=float(h * 25),  # OpenCV h is ~0–30 for 8-bit
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )
        out[i] = den8.astype(np.float32) / 255.0
    return out


def gaussian_3d(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a 3D Gaussian filter to the input volume.

    Args:
        volume: A 3D numpy array representing the input volume.
        sigma: The standard deviation for Gaussian kernel.

    Returns:
        A 3D numpy array representing the filtered volume.
    """
    return gaussian_filter(volume, sigma=sigma)


def clahe_2d_opencv(
    volume: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) using OpenCV per-slice operations.

    Args:
        volume (np.ndarray): 3D stack
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (Tuple[int, int], optional): Size of grid for histogram equalization. Defaults to (8, 8).

    Returns:
        np.ndarray: The contrast-enhanced volume.
    """
    z, y, x = volume.shape
    out = np.empty_like(volume, dtype=np.float32)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    for i in range(z):
        img8 = np.clip(volume[i] * 255, 0, 255).astype(np.uint8)
        clahe_img8 = clahe.apply(img8)
        out[i] = clahe_img8.astype(np.float32) / 255.0
    return out


def clahe_3d_skimage(
    volume: np.ndarray,
    clip_limit: float = 0.01,
    nbins: int = 256,
    kernel_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) using skimage per-slice operations.

    Args:
        volume (np.ndarray): 3D stack
        clip_limit (float, optional): Normalized threshold for contrast limiting. Defaults to 0.01.
        nbins (int, optional): Number of histogram bins. Defaults to 256.
        kernel_size (Tuple[int, int], optional): Size of grid for histogram equalization. Defaults to (8, 8).

    Returns:
        np.ndarray: The contrast-enhanced volume.
    """
    z, y, x = volume.shape
    out = np.empty_like(volume, dtype=np.float32)
    for i in range(z):
        out[i] = exposure.equalize_adapthist(
            volume[i],
            clip_limit=clip_limit,
            nbins=nbins,
            kernel_size=kernel_size,
        )
    return out


def preprocess_stack(
    stack: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Preprocess the input stack based on the provided configuration.

    Args:
        stack: Input 3D numpy array representing the stack.
        cfg: Configuration dictionary specifying preprocessing steps.

    Returns:
        Preprocessed 3D numpy array.
    """
    preprocessed_stack = stack.copy()
    print(cfg)

    pcts = tuple(cfg.get("normalize_pcts", (0.5, 99.5)))
    preprocessed_stack, p1, p2 = clip_and_scale(preprocessed_stack, pcts=pcts)
    print(f"Clipped and scaled using percentiles: {p1:.2e}-{p2:.2e}")

    if cfg.get("n4_bias_correction", False):
        bias_field, preprocessed_stack = n4_bias_correction_sitk(preprocessed_stack)
        print("Applied N4 bias field correction.")

    if cfg.get("gaussian_filter", False):
        sigma = cfg.get("gaussian_sigma", 1.0)
        preprocessed_stack = gaussian_3d(preprocessed_stack, sigma=sigma)
        print(f"Applied 3D Gaussian filter with sigma={sigma}.")

    if cfg.get("nlm_denoising", False):
        h = cfg.get("nlm_h", 0.9)
        template_window = cfg.get("nlm_template_window", 5)
        search_window = cfg.get("nlm_search_window", 13)
        preprocessed_stack = nlm2d_opencv(
            preprocessed_stack,
            h=h,
            template_window=template_window,
            search_window=search_window,
        )
        print(f"Applied 2D Non-Local Means denoising with h={h}.")

    if cfg.get("clahe", False):
        clip_limit = cfg.get("clahe_clip_limit", 2.0)
        kernel_size = tuple(cfg.get("clahe_tile_grid_size", (8, 8)))
        preprocessed_stack = clahe_3d_skimage(
            preprocessed_stack,
            clip_limit=clip_limit,
            nbins=256,
            kernel_size=kernel_size,
        )
        print(
            f"Applied CLAHE with clip_limit={clip_limit} and kernel_size={kernel_size}."
        )

    return preprocessed_stack


if __name__ == "__main__":
    # Testing purposes
    from utils.tif import read_tif, save_stack, tif_to_float32

    v_raw = read_tif("data/Litarion.tif")
    v = tif_to_float32(v_raw)
    v_scaled, p1, p2 = clip_and_scale(v, pcts=(0.5, 99.5))

    print(f"Scaled to [0,1] using percentiles: {p1:.2e}-{p2:.2e}")
    print("min, max, unique=", np.min(v), np.max(v), np.unique(v))

    gauss = gaussian_3d(v_scaled, sigma=1.0)
    save_stack(gauss, "data/results/Litarion_gauss.tif")
