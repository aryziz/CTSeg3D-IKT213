from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from xctp.preprocess import preprocess_stack
from xctp.seeds import multi_otsu_3d
from xctp.segment import binary_hysteresis_xct
from xctp.utils.tif import read_tif, save_stack, tif_to_float32


def run_pipeline(
    in_path: Path,
    out_mask_path: Path,
    cfg: Dict[str, Any],
    out_preproc_path: Optional[Path] = None,
) -> None:
    """Run the segmentation pipeline on a single input stack.

    Args:
        in_path: Path to input .tif/.tiff file.
        out_mask_path: Path to output segmentation mask .tif file.
        cfg: Configuration dictionary for the pipeline.
        out_preproc_path: Optional path to save preprocessed stack .tif file.
    """
    # Load input stack
    stack = tif_to_float32(read_tif(in_path))

    # Preprocessing
    preproc_cfg = cfg["preprocess"]
    preprocessed_stack = preprocess_stack(stack, preproc_cfg)

    # Save preprocessed stack if requested
    if out_preproc_path is not None:
        save_stack(preprocessed_stack, out_preproc_path)

    # Seeding
    # seeding_cfg = cfg.get("seeding", {})
    thresholds, labels, icv = multi_otsu_3d(preprocessed_stack)
    t1, t2 = thresholds
    foreground_mask = labels > 0

    print(f"Threshold: {thresholds}, Interclass: {icv}")

    # Segmentation
    # segmentation_cfg = cfg.get("segmentation", {})

    bin_mask = binary_hysteresis_xct(
        preprocessed_stack, t1, t2, foreground_mask=foreground_mask, debug=True
    )
    save_stack(bin_mask, out_mask_path)

    # save_stack(fg_mask, out_mask_path)

    print("Finished segmentation.")

    # Save output mask
    # save_tiff_stack(mask, out_mask_path)
