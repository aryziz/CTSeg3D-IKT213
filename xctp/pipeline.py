from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from xctp.preprocess import preprocess_stack
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
    # seeds = generate_seeds(preprocessed_stack, seeding_cfg)

    # Segmentation
    # segmentation_cfg = cfg.get("segmentation", {})
    # mask = segment_stack(preprocessed_stack, seeds, segmentation_cfg)

    # Save output mask
    # save_tiff_stack(mask, out_mask_path)
