#!/usr/bin/env python3
"""
Run the XCT pipeline over all .tif/.tiff stacks in an input directory.

Usage:
  python run_pipeline.py --in-dir data/raw --cfg config/test.yaml
                                        --out-dir results/run_001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from xctp.pipeline import run_pipeline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="""Run XCT filtering/segmentation
        pipeline over a directory of stacks."""
    )
    p.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Directory containing input .tif/.tiff stack(s).",
    )
    p.add_argument(
        "--cfg",
        required=True,
        type=Path,
        help="YAML config file with pipeline parameters.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write outputs (created if missing).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for stacks.",
    )
    p.add_argument(
        "--write-preproc",
        action="store_true",
        help="If set, also write preprocessed grayscale volume(s).",
    )
    return p.parse_args()


def _load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a mapping/dict.")
    return cfg


def _find_stacks(in_dir: Path, recursive: bool) -> list[Path]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    stacks: list[Path] = []
    for pat in patterns:
        stacks.extend(in_dir.rglob(pat) if recursive else in_dir.glob(pat))
    stacks = sorted({p.resolve() for p in stacks})
    return stacks


def main() -> int:
    args = _parse_args()

    if not args.in_dir.exists() or not args.in_dir.is_dir():
        print(
            f"[ERROR] --in-dir not found or not a directory: {args.in_dir}",
            file=sys.stderr,
        )
        return 2
    if not args.cfg.exists():
        print(f"[ERROR] --cfg not found: {args.cfg}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(args.cfg)

    stacks = _find_stacks(args.in_dir, args.recursive)
    if not stacks:
        print(
            f"[WARN] No .tif/.tiff files found in {args.in_dir} (recursive={args.recursive})."
        )
        return 0

    # Session log
    session: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "input_dir": str(args.in_dir.resolve()),
        "config_file": str(args.cfg.resolve()),
        "output_dir": str(args.out_dir.resolve()),
        "recursive": bool(args.recursive),
        "write_preproc": bool(args.write_preproc),
        "num_inputs": len(stacks),
        "files": [],
    }

    for i, in_path in enumerate(stacks, start=1):
        stem = in_path.stem
        out_subdir = args.out_dir / stem
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_mask_path = out_subdir / f"{stem}_mask.tif"
        out_preproc_path = (
            (out_subdir / f"{stem}_preproc.tif") if args.write_preproc else None
        )

        print(f"[{i}/{len(stacks)}] Processing: {in_path.name}")
        try:
            cfg_run = dict(cfg)  # shallow copy
            cfg_run.setdefault("io", {})
            cfg_run["io"].update(
                {
                    "input_path": str(in_path),
                    "output_mask_path": str(out_mask_path),
                    "output_preproc_path": (
                        str(out_preproc_path) if out_preproc_path else None
                    ),
                }
            )

            run_pipeline(
                in_path=in_path,
                out_mask_path=out_mask_path,
                cfg=cfg_run,
                out_preproc_path=out_preproc_path,
            )

            status = "ok"
            err_msg = None
        except Exception as e:  # noqa: BLE001 (simple CLI)
            status = "error"
            err_msg = f"{type(e).__name__}: {e}"
            print(f"[ERROR] {in_path.name}: {err_msg}", file=sys.stderr)

        # Append file record
        session["files"].append(
            {
                "input": str(in_path),
                "output_mask": str(out_mask_path),
                "output_preproc": str(out_preproc_path) if out_preproc_path else None,
                "status": status,
                "error": err_msg,
            }
        )

    # Write session log (JSON) and a copy of the config used
    (args.out_dir / "run_log.json").write_text(json.dumps(session, indent=2))
    cfg_copy_path = (
        args.out_dir
        / f"config_used_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.yaml"
    )
    cfg_copy_path.write_text(Path(args.cfg).read_text())

    print(f"[DONE] Processed {len(stacks)} stack(s). Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
