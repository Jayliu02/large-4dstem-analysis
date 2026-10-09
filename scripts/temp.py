import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsciio.quantumdetector import file_reader

mib_path = (
    r"data\3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib"
)

data = file_reader(
    mib_path,
    lazy=True,
    navigation_shape=(256, 256),
)[0]["data"]

# 扫描位置 (y, x)，先作为候选点
positions = {
    "Center": (120, 120),
    "Top-left": (40, 40),
    "Right": (120, 210),
    "Bottom-right": (210, 210),
}

fig, axes = plt.subplots(2, 2, figsize=(9, 9))

for ax, (name, (y, x)) in zip(
    axes.flat, positions.items()
):
    dp = np.asarray(data[y, x].compute(), dtype=np.float64)

    # 检查总强度和直射束中心
    total = dp.sum()

    yy, xx = np.indices(dp.shape)

    if total > 0:
        cy = (yy * dp).sum() / total
        cx = (xx * dp).sum() / total
    else:
        cy, cx = np.nan, np.nan

    print(
        f"{name}: total={total:.0f}, "
        f"max={dp.max():.0f}, "
        f"CoM=({cy:.2f}, {cx:.2f})"
    )

    view = np.log1p(dp)

    ax.imshow(view, cmap="gray")
    ax.set_title(
        f"{name} ({y}, {x})\n"
        f"CoM=({cy:.1f}, {cx:.1f})"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    "outputs/mib_analysis/dp_comparison.png",
    dpi=200,
)
plt.close()


import numpy as np
from scipy.ndimage import gaussian_filter

positions = {
    "Center": (120, 120),
    "Top-left": (40, 40),
    "Right": (120, 210),
    "Bottom-right": (210, 210),
}

for name, (y, x) in positions.items():

    dp = np.asarray(
        data[y, x].compute(),
        dtype=np.float32
    )

    # 适度平滑，抑制孤立噪声
    smooth = gaussian_filter(dp, sigma=1.5)

    # 在探测器中心附近寻找直射束
    search = smooth[64:192, 64:192]

    iy, ix = np.unravel_index(
        np.argmax(search),
        search.shape
    )

    cy = iy + 64
    cx = ix + 64

    print(
        f"{name}: direct beam ≈ "
        f"(y={cy}, x={cx})"
    )