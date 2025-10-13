import imageio as iio
import numpy as np
import tifffile as tiff
from pint.compat import ndarray
from skimage import io, restoration, util


# 3D Total Variation
def total_variation_filter(file_path: str) -> ndarray:
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


# Normalization (Min-Max Norm)
def normalize(filepath: str, output_path: str) -> ndarray:

    print(f"Loading image from {filepath}")
    img_stack = io.imread(filepath)

    print(f"Normalizing image with shape {img_stack.shape} and type {img_stack.dtype}")

    normalized_stack = np.zeros_like(img_stack, dtype=img_stack.dtype)

    for i in range(img_stack.shape[0]):
        img_slice = img_stack[i]
        norm_slice = normalized_stack[i]
        for row in range(img_slice.shape[0]):
            for pxl in range(img_slice.shape[1]):
                new_pxl_value = (img_slice[row][pxl] - img_slice.min()) / (
                    img_slice.max() - img_slice.min()
                )
                norm_slice[row][pxl] = new_pxl_value

    print(f"Saving image to {output_path}")
    iio.imwrite(output_path, normalized_stack)
    return normalized_stack


def save_stack(input_stack: ndarray, output_path: str) -> None:
    print(f"Saving image to {output_path}")
    iio.imwrite(output_path, input_stack)


if __name__ == "__main__":
    tv_stack = total_variation_filter(file_path="data/Tesla.tif")

    save_stack(tv_stack, "data/Tesla_3dtv.tif")
