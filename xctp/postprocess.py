import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    generate_binary_structure,
)


def postprocess_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Postprocess a binary mask using morphological operations.

    Args:
        mask (np.ndarray): Input binary mask.
        Radius (int, optional): Number of iterations for morphological operations. Defaults to 1.

    Returns:
        np.ndarray: Postprocessed binary mask.
    """
    print("Postprocessing mask...")
    # Ensure mask is binary
    bin_mask = (mask > 0).astype(bool)

    structure = generate_binary_structure(rank=3, connectivity=1)

    closed = binary_closing(bin_mask, structure=structure, iterations=radius)
    opened = binary_opening(closed, structure=structure, iterations=radius)

    return opened.astype(np.uint8)


def fill_internal_pores(mask: np.ndarray) -> np.ndarray:
    """Fill pores using ndimage binary_fill_holes

    Args:
        mask (np.ndarray): Input binary mask

    Returns:
        np.ndarray: Postprocessed binary mask
    """
    bin_mask = (mask > 0).astype(bool)
    filled = binary_fill_holes(bin_mask)
    return filled.astype(np.uint8)
