from pathlib import Path

import numpy as np
import tifffile as tiff


def read_tif(image_path: str | Path) -> np.ndarray:
    """Read a tif image.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.ndarray: Loaded image.
    """
    image = tiff.imread(image_path)
    return image


def tif_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert a tif image to float32.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Converted image.
    """

    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.float32:
        pass
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

    return image


def save_stack(input_stack: np.ndarray, output_path: str | Path) -> None:
    print(f"Saving image to {output_path}")
    tiff.imwrite(output_path, input_stack)
