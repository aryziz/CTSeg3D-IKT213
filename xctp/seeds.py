import os
import numpy as np
import tifffile as tiff
from scipy import ndimage as ndi



"""
def load_volume(data_dir):
    tif_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])
    slices = []
    for f in tif_files:
        path = os.path.join(data_dir, f)
        imf = tiff.imread(path)
        slices.append(img)
    volume = np.stack(slices, axis=0)
    return volume
    """


def sobel_3d_gradient(volume):
    gx = ndi.sobel(volume, axis=0)
    gy = ndi.sobel(volume, axis=1)
    gz = ndi.sobel(volume, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    return grad_mag

def gradient_aware_filter(volume, percentile=70):
    """
      Keep only voxels below gradient percentile threshold (interior regions).
      If seeds are provided, mask them; otherwise, return the interior mask.
      """
    grad = sobel_3d_gradient(volume)
    cutoff =  np.percentile(grad, percentile)
    mask = grad < cutoff
    return mask

if __name__ == '__main__':
    from utils.tif import read_tif, save_stack, tif_to_float32
    from preprocess import clip_and_scale, gaussian_3d

    v_raw = read_tif("data/Litarion.tif")
    v = tif_to_float32(v_raw)

    v_scaled, p1, p2 = clip_and_scale(v, pcts=(0.5, 99.5))
    print(f"Scaled to [0,1] using percentiles: {p1:.2e}-{p2:.2e}")
    print(f"min, max =", np.min(v_scaled), np.max(v_scaled))

    gauss = gaussian_3d(v_scaled)
    interior_mask = gradient_aware_filter(gauss)

    save_stack(interior_mask.astype(np.uint8) * 255, "data/results/Litarion_gradmask.tif")
    print(" Saved gradient-aware mask to data/results/Litarion_gradmask.tif")
