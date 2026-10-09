from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

origin_dir = Path("outputs/origin_check")
analysis_dir = Path("outputs/mib_analysis")

bf = np.load(origin_dir / "bf_corrected.npy")
adf = np.load(origin_dir / "adf_corrected.npy")
total = np.load(analysis_dir / "total.npy")

bf_norm = np.divide(
    bf, total,
    out=np.zeros_like(bf, dtype=float),
    where=total > 0,
)

adf_norm = np.divide(
    adf, total,
    out=np.zeros_like(adf, dtype=float),
    where=total > 0,
)

images = {
    "BF Corrected": bf,
    "ADF Corrected": adf,
    "BF / Total": bf_norm,
    "ADF / Total": adf_norm,
}

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, (name, img) in zip(axes.flat, images.items()):
    vmin, vmax = np.percentile(img, [2, 98])

    ax.imshow(
        img,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(name)
    ax.axis("off")

    print(
        name,
        "min:", img.min(),
        "median:", np.median(img),
        "max:", img.max(),
    )

plt.tight_layout()
plt.savefig(
    origin_dir / "normalized_bf_adf.png",
    dpi=300,
)
plt.close()