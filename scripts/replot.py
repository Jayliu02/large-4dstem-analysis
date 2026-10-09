
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("outputs/origin_check")

names = [
    "bf_fixed",
    "bf_corrected",
    "adf_fixed",
    "adf_corrected",
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, name in zip(axes.flat, names):
    image = np.load(out / f"{name}.npy").astype(float)

    # 每张图独立计算显示范围
    vmin, vmax = np.percentile(image, [2, 98])

    if vmax <= vmin:
        vmin = image.min()
        vmax = image.max()
        if vmax <= vmin:
            vmax = vmin + 1

    print(
        f"{name}: "
        f"min={image.min():.1f}, "
        f"median={np.median(image):.1f}, "
        f"max={image.max():.1f}"
    )

    ax.imshow(
        image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(name)
    ax.axis("off")

fig.tight_layout()

fig.savefig(
    out / "bf_adf_autocontrast.png",
    dpi=300,
)

plt.close(fig)
