import numpy as np
from scipy import ndimage as ndi


def _estimate_q_tail(
    dark_band: np.ndarray, bright_band: np.ndarray, target_seed_fraction: float = 0.15
) -> float:
    """Adaptively set q_tail to get ~15% of each band as seeds
    Handles cases where bands are narrow or wide

    Args:
        dark_band: Voxels in the dark band.
        bright_band: Voxels in the bright band.
        target_seed_fraction: Target fraction of voxels to be selected as seeds.

    Returns:
        Estimated q_tail value.
    """
    dark_std = np.std(dark_band)
    bright_std = np.std(bright_band)

    avg_std = 0.5 * (dark_std + bright_std)
    q_tail = np.clip(0.3 * (avg_std / 0.1), 0.1, 0.4)
    return q_tail


def binarise(
    volume: np.ndarray,
    t1: float,
    t2: float,
    foreground_mask: np.ndarray | None = None,
    q_tail: float | None = None,
    connectivity: int = 1,
    debug: bool = False,
) -> np.ndarray:
    v = volume.astype(np.float32, copy=False)
    if foreground_mask is None:
        foreground_mask = np.ones(v.shape, bool)

    # ---- bands ----
    dark = (v < t1) & foreground_mask
    mid = (v >= t1) & (v < t2) & foreground_mask
    bright = (v >= t2) & foreground_mask

    if q_tail is None:
        if dark.any() and bright.any():
            q_tail = _estimate_q_tail(v[dark], v[bright])
            if debug:
                print(f"Auto-estimated q_tail: {q_tail:.3f}")
        else:
            q_tail = 0.30  # Fallback default
            if debug:
                print(f"Using default q_tail: {q_tail:.3f} (insufficient data)")

    if debug:
        print(
            {
                "frac_dark": float(dark.mean()),
                "frac_mid": float(mid.mean()),
                "frac_bright": float(bright.mean()),
                "t1": float(t1),
                "t2": float(t2),
            }
        )

    # ---- strong seeds (tails) ----
    if dark.any():
        bg_cut = np.quantile(v[dark], q_tail)
    else:
        bg_cut = t1
    sure_bg = (v <= bg_cut) & foreground_mask

    if bright.any():
        fg_cut = np.quantile(v[bright], 1 - q_tail)
    else:
        fg_cut = t2
    sure_fg = (v >= fg_cut) & foreground_mask

    if debug:
        print(
            {
                "bg_cut": float(bg_cut),
                "fg_cut": float(fg_cut),
                "sure_bg_voxels": int(sure_bg.sum()),
                "sure_fg_voxels": int(sure_fg.sum()),
            }
        )

    # ---- allowed regions for growth ----
    bg_allowed = dark | mid
    fg_allowed = bright | mid

    # ---- propagate ----
    S = ndi.generate_binary_structure(3, min(connectivity, 3))
    bg_grown = ndi.binary_propagation(input=sure_bg, mask=bg_allowed, structure=S)
    fg_grown = ndi.binary_propagation(input=sure_fg, mask=fg_allowed, structure=S)

    if debug:
        print(
            {
                "bg_grown": int(bg_grown.sum()),
                "fg_grown": int(fg_grown.sum()),
                "allowed_bg_voxels": int(bg_allowed.sum()),
                "allowed_fg_voxels": int(fg_allowed.sum()),
            }
        )

    # ---- resolve overlaps ----
    both = bg_grown & fg_grown
    if both.any():
        dist_bg = np.abs(v[both] - t1)
        dist_fg = np.abs(v[both] - t2)
        choose_fg = dist_fg < dist_bg
        idx = np.where(both)
        fg_grown = fg_grown.copy()
        fg_grown[idx[0][choose_fg], idx[1][choose_fg], idx[2][choose_fg]] = True
        bg_grown[idx[0][~choose_fg], idx[1][~choose_fg], idx[2][~choose_fg]] = True

    # ---- build binary ----
    bin_mask = np.zeros_like(v, np.uint8)
    bin_mask[bg_grown] = 0
    bin_mask[fg_grown] = 1

    leftover = foreground_mask & ~(bg_grown | fg_grown)
    if leftover.any():
        bin_mask[leftover & (v >= 0.5 * (t1 + t2))] = 1

    if debug:
        labeled, ncomp = ndi.label(
            bin_mask, structure=ndi.generate_binary_structure(3, 1)
        )
        counts = np.bincount(labeled.ravel())
        largest = int(counts[1:].max()) if counts.size > 1 else 0
        fg_frac = float(bin_mask.mean())
        largest_frac = largest / bin_mask.size
        mid_claim_fg = (bin_mask.astype(bool) & mid).sum() / max(1, mid.sum())

        print(
            {
                "fg_frac": fg_frac,
                "n_components": int(ncomp),
                "largest_frac": largest_frac,
                "mid_claim_fg": float(mid_claim_fg),
            }
        )

    return bin_mask
