import matplotlib.pyplot as plt
import numpy as np


def show3(v: np.ndarray, title: str = "") -> None:
    z, y, x = v.shape
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(v[z // 2], cmap="gray")
    axs[0].set_title("Axial")
    axs[1].imshow(v[:, y // 2, :], cmap="gray")
    axs[1].set_title("Coronal")
    axs[2].imshow(v[:, :, x // 2], cmap="gray")
    axs[2].set_title("Sagittal")
    for a in axs:
        a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
