import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_multiotsu, threshold_otsu
from scipy import ndimage
from skimage.morphology import ball, dilation

def compute_otsu_interclass_variance(histogram, thresholds):
    """
    Compute Otsu's interclass variance for multi-threshold segmentation.
    Higher variance indicates better separation between classes.
    """
    total_pixels = np.sum(histogram)
    if total_pixels == 0:
        return 0

    # Normalize histogram to get probabilities
    probabilities = histogram / total_pixels
    cumulative_sum = np.cumsum(probabilities)

    # Calculate means for each class
    class_means = []
    class_weights = []

    # Class 0: below first threshold
    mask_0 = np.arange(len(histogram)) <= thresholds[0]
    if np.sum(mask_0) > 0:
        weight_0 = np.sum(probabilities[mask_0])
        if weight_0 > 0:
            mean_0 = np.sum(np.arange(len(histogram))[mask_0] * probabilities[mask_0]) / weight_0
            class_means.append(mean_0)
            class_weights.append(weight_0)

    # Middle classes
    for i in range(1, len(thresholds)):
        mask_i = (np.arange(len(histogram)) > thresholds[i-1]) & (np.arange(len(histogram)) <= thresholds[i])
        if np.sum(mask_i) > 0:
            weight_i = np.sum(probabilities[mask_i])
            if weight_i > 0:
                mean_i = np.sum(np.arange(len(histogram))[mask_i] * probabilities[mask_i]) / weight_i
                class_means.append(mean_i)
                class_weights.append(weight_i)

    # Last class: above last threshold
    mask_last = np.arange(len(histogram)) > thresholds[-1]
    if np.sum(mask_last) > 0:
        weight_last = np.sum(probabilities[mask_last])
        if weight_last > 0:
            mean_last = np.sum(np.arange(len(histogram))[mask_last] * probabilities[mask_last]) / weight_last
            class_means.append(mean_last)
            class_weights.append(weight_last)

    # Calculate interclass variance
    global_mean = np.sum(np.arange(len(histogram)) * probabilities)
    interclass_variance = 0

    for weight, mean in zip(class_weights, class_means):
        interclass_variance += weight * (mean - global_mean) ** 2

    return interclass_variance

def adaptive_uncertainty_band(volume, k_sigma=0.6, m_abs=3.0):
    """
    Calculate adaptive uncertainty band based on local standard deviation.
    Args:
        volume: 3D numpy array
        k_sigma: Multiplier for standard deviation-based uncertainty
        m_abs: Minimum absolute uncertainty margin
    Returns:
        uncertainty_band: Array of uncertainty values for each voxel
    """
    # Calculate local standard deviation using a 3x3x3 neighborhood
    local_std = ndimage.generic_filter(
        volume,
        np.std,
        size=3,
        mode='reflect'
    )

    # Calculate adaptive uncertainty band
    uncertainty_band = m_abs + k_sigma * local_std

    return uncertainty_band

def uncertainty_aware_thresholding(foreground_data, uncertainty_band_foreground, classes=3):
    """
    Apply uncertainty-aware multi-Otsu thresholding.
    Adjusts thresholds based on uncertainty band.
    """
    # Get base thresholds using standard multi-Otsu
    base_thresholds = threshold_multiotsu(foreground_data, classes=classes)

    # Calculate uncertainty-weighted adjustments
    adjusted_thresholds = []

    for i, threshold in enumerate(base_thresholds):
        # Find voxels near this threshold
        near_threshold_mask = np.abs(foreground_data - threshold) < np.percentile(uncertainty_band_foreground, 50)

        if np.any(near_threshold_mask):
            # Calculate average uncertainty near threshold
            avg_uncertainty = np.mean(uncertainty_band_foreground[near_threshold_mask])

            # Adjust threshold based on uncertainty
            # Higher uncertainty -> more conservative adjustment
            adjustment = avg_uncertainty * 0.1  # Small adjustment factor
            adjusted_threshold = threshold + adjustment
        else:
            adjusted_threshold = threshold

        adjusted_thresholds.append(adjusted_threshold)

    return np.array(adjusted_thresholds)

def automated_multi_otsu_3d(volume: np.ndarray, classes: int = 3, m_abs: float = 3.0, k_sigma: float = 0.6,
                           min_seed_bg: int = 5000, max_seed_bg: int = 200000,
                           variance_threshold: float = 0.7, max_iterations: int = 3):
    """
    Enhanced automated 3D multi-Otsu thresholding with interclass variance optimization
    and uncertainty band adjustment.

    Args:
        volume: 3D numpy array (Z, Y, X)
        classes: Number of classes to segment
        m_abs: Minimum margin for uncertainty band
        k_sigma: Multiplier for uncertainty margin based on local standard deviation
        min_seed_bg: Minimum size for a valid background region (in voxels)
        max_seed_bg: Maximum size for a valid background region (in voxels)
        variance_threshold: Minimum acceptable interclass variance ratio (0-1)
        max_iterations: Maximum iterations for variance optimization

    Returns:
        thresholds: List of threshold values
        labeled_volume: Volume with classes labeled from 0 to classes-1
        variance_metrics: Dictionary containing variance information
    """
    v = volume.astype(np.float32, copy=False)

    print("ENHANCED MULTI-OTSU WITH UNCERTAINTY BAND")

    # Calculate uncertainty band for the entire volume
    print("Calculating uncertainty band...")
    uncertainty_band = adaptive_uncertainty_band(v, k_sigma=k_sigma, m_abs=m_abs)

    # Automated background detection using Otsu threshold with variance optimization
    best_variance = 0
    best_background_threshold = None
    best_foreground_mask = None

    print("Optimizing background detection...")

    # Try multiple background thresholds to maximize interclass variance
    candidate_thresholds = [
        threshold_otsu(v) * 0.2,
        threshold_otsu(v) * 0.3,
        threshold_otsu(v) * 0.4,
        np.percentile(v, 1),
        np.percentile(v, 2),
        np.percentile(v, 5)
    ]

    for bg_candidate in candidate_thresholds:
        low_intensity_mask = v < bg_candidate

        if np.any(low_intensity_mask):
            labeled_mask, num_features = ndimage.label(low_intensity_mask)
            component_sizes = np.bincount(labeled_mask.ravel())

            valid_background_mask = np.zeros_like(labeled_mask, dtype=bool)
            for i in range(1, num_features + 1):
                region_size = component_sizes[i]
                if min_seed_bg <= region_size <= max_seed_bg:
                    valid_background_mask |= (labeled_mask == i)

            foreground_mask = ~valid_background_mask
            foreground_data = v[foreground_mask]

            if len(foreground_data) > 100:  # Ensure sufficient foreground data
                try:
                    # Calculate histogram and interclass variance
                    hist, bin_edges = np.histogram(foreground_data, bins=256, density=True)
                    temp_thresholds = threshold_multiotsu(foreground_data, classes=classes)
                    variance = compute_otsu_interclass_variance(hist, temp_thresholds)

                    if variance > best_variance:
                        best_variance = variance
                        best_background_threshold = bg_candidate
                        best_foreground_mask = foreground_mask.copy()
                except:
                    continue

    if best_background_threshold is None:
        # Fallback to conservative approach
        best_background_threshold = threshold_otsu(v) * 0.3
        low_intensity_mask = v < best_background_threshold
        if np.any(low_intensity_mask):
            labeled_mask, num_features = ndimage.label(low_intensity_mask)
            component_sizes = np.bincount(labeled_mask.ravel())
            valid_background_mask = np.zeros_like(labeled_mask, dtype=bool)
            for i in range(1, num_features + 1):
                if min_seed_bg <= component_sizes[i] <= max_seed_bg:
                    valid_background_mask |= (labeled_mask == i)
            best_foreground_mask = ~valid_background_mask
        else:
            best_foreground_mask = np.ones_like(v, dtype=bool)

    print(f"Optimized background threshold: {best_background_threshold:.4f}")
    print(f"Interclass variance: {best_variance:.4f}")

    # Expand background mask with dilation
    background_mask = ndimage.binary_dilation(~best_foreground_mask, structure=ball(1))
    foreground_mask = ~background_mask
    foreground_data = v[foreground_mask]
    uncertainty_foreground = uncertainty_band[foreground_mask]

    print(f"Final foreground voxels: {np.sum(foreground_mask)} ({np.sum(foreground_mask)/v.size*100:.2f}% of total)")

    # Apply uncertainty-aware multi-Otsu
    print("Applying uncertainty-aware thresholding...")
    thresholds = uncertainty_aware_thresholding(foreground_data, uncertainty_foreground, classes=classes)

    # Calculate final interclass variance
    hist, bin_edges = np.histogram(foreground_data, bins=256, density=True)
    final_variance = compute_otsu_interclass_variance(hist, thresholds)

    print(f"Final interclass variance: {final_variance:.4f}")

    # Create labeled volume
    labeled_volume = np.zeros_like(v, dtype=np.uint8)

    # Apply thresholds within foreground regions
    labeled_foreground = np.zeros(foreground_data.shape, dtype=np.uint8)

    # Class 0: below first threshold
    labeled_foreground[foreground_data <= thresholds[0]] = 0

    # Middle classes
    for i in range(1, len(thresholds)):
        labeled_foreground[
            (foreground_data > thresholds[i-1]) & (foreground_data <= thresholds[i])
        ] = i

    # Last class: above last threshold
    labeled_foreground[foreground_data > thresholds[-1]] = len(thresholds)

    # Place foreground labels into full volume
    labeled_volume[foreground_mask] = labeled_foreground

    # Prepare variance metrics
    variance_metrics = {
        'interclass_variance': final_variance,
        'background_threshold': best_background_threshold,
        'foreground_ratio': np.sum(foreground_mask) / v.size,
        'uncertainty_mean': np.mean(uncertainty_band),
        'uncertainty_std': np.std(uncertainty_band)
    }

    return thresholds, labeled_volume, variance_metrics

def show_enhanced_segmentation_results(original_volume, segmented_volume, thresholds, uncertainty_band=None,
                                     variance_metrics=None, title="Enhanced 3D Multi-Otsu Segmentation"):
    """
    Enhanced visualization showing uncertainty bands and variance metrics.
    """
    z = original_volume.shape[0] // 2
    original_slice = original_volume[z, :, :]
    segmented_slice = segmented_volume[z, :, :]

    if uncertainty_band is not None:
        uncertainty_slice = uncertainty_band[z, :, :]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = [axes]  # Make it 2D for consistent indexing

    # Original image
    axes[0, 0].imshow(original_slice, cmap='gray')
    axes[0, 0].set_title(f'Original (slice z={z})')
    axes[0, 0].axis('off')

    # Histogram with thresholds
    axes[0, 1].hist(original_volume.ravel(), bins=255, alpha=0.7)
    axes[0, 1].set_title('Histogram with Auto Thresholds')
    axes[0, 1].set_xlabel('Intensity')
    axes[0, 1].set_ylabel('Frequency')

    for i, thresh in enumerate(thresholds):
        axes[0, 1].axvline(thresh, color='red', linestyle='--',
                          label=f'Threshold {i+1}: {thresh:.4f}')
    axes[0, 1].legend()

    # Segmented result
    axes[0, 2].imshow(segmented_slice, cmap='jet')
    axes[0, 2].set_title('Segmented Result')
    axes[0, 2].axis('off')

    if uncertainty_band is not None:
        # Uncertainty band visualization
        im3 = axes[1, 0].imshow(uncertainty_slice, cmap='hot')
        axes[1, 0].set_title('Uncertainty Band')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])

        # Combined view (segmentation overlaid on uncertainty)
        axes[1, 1].imshow(uncertainty_slice, cmap='gray')
        axes[1, 1].imshow(segmented_slice, cmap='jet', alpha=0.6)
        axes[1, 1].set_title('Segmentation + Uncertainty')
        axes[1, 1].axis('off')

        # Variance metrics display
        axes[1, 2].axis('off')
        if variance_metrics:
            metrics_text = f"""VARIANCE METRICS:
Interclass Variance: {variance_metrics['interclass_variance']:.4f}
Background Threshold: {variance_metrics['background_threshold']:.4f}
Foreground Ratio: {variance_metrics['foreground_ratio']:.3f}
Uncertainty Mean: {variance_metrics['uncertainty_mean']:.3f}
Uncertainty Std: {variance_metrics['uncertainty_std']:.3f}"""
            axes[1, 2].text(0.1, 0.5, metrics_text, fontfamily='monospace',
                           fontsize=10, verticalalignment='center')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Main execution for the tif file
print("=== ENHANCED AUTOMATED 3D MULTI-OTSU SEGMENTATION ===")

# Using the output from gaussian filter as testing
image = out_gauss

# Apply enhanced automated multi-Otsu with uncertainty band
thresholds, segmented_volume, variance_metrics = automated_multi_otsu_3d(
    image,
    classes=3,
    m_abs=3.0,
    k_sigma=0.6,
    min_seed_bg=5000,
    max_seed_bg=200000,
    variance_threshold=0.7
)

print(f"Automatically found thresholds: {thresholds}")
print(f"Classes in segmentation: {np.unique(segmented_volume)}")
print(f"Voxel counts per class: {np.bincount(segmented_volume.flatten())}")
print(f"Variance metrics: {variance_metrics}")

# Calculate uncertainty band for visualization
uncertainty_band = adaptive_uncertainty_band(image, k_sigma=0.6, m_abs=3.0)

# Show enhanced results
show_enhanced_segmentation_results(
    image,
    segmented_volume,
    thresholds,
    uncertainty_band,
    variance_metrics,
    "Enhanced 3D Multi-Otsu with Uncertainty Band"
)

# Show 3D views
def show_3d_views(volume, title=""):
    """Show axial, coronal and sagittal views"""
    z, y, x = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(volume[z//2], cmap='jet')
    axes[0].set_title(f'{title} - Axial')
    axes[0].axis('off')

    axes[1].imshow(volume[:, y//2, :], cmap='jet')
    axes[1].set_title(f'{title} - Coronal')
    axes[1].axis('off')

    axes[2].imshow(volume[:, :, x//2], cmap='jet')
    axes[2].set_title(f'{title} - Sagittal')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

show_3d_views(segmented_volume, "Enhanced Segmentation Results")