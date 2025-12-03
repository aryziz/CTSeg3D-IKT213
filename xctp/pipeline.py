from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from xctp.postprocess import fill_internal_pores, postprocess_mask
from xctp.preprocess import preprocess_stack
from xctp.seeds import multi_otsu_3d
from xctp.segment import binarise
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
    seeding_cfg = cfg.get("seeding", {})
    max_bg_size = seeding_cfg.get("max_bg_size", 200000)
    min_bg_size = seeding_cfg.get("min_bg_size", 5000)
    classes = seeding_cfg.get("class", 3)
    thresholds, labels, icv = multi_otsu_3d(
        preprocessed_stack,
        classes=classes,
        min_bg_size=min_bg_size,
        max_bg_size=max_bg_size,
    )

    t1, t2 = thresholds
    foreground_mask = labels > 0

    print(f"Threshold: {thresholds}, Interclass: {icv}")

    # Segmentation
    bin_mask = binarise(
        preprocessed_stack, t1, t2, foreground_mask=foreground_mask, debug=True
    )

    postprocess_cfg = cfg.get("postprocess", {})
    morphological_post = postprocess_mask(
        bin_mask,
        radius=postprocess_cfg.get("radius", 1),
    )
    postprocessed_mask = fill_internal_pores(morphological_post)

    save_stack(postprocessed_mask, out_mask_path)

    print("Finished segmentation.")
