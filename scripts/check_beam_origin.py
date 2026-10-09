from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, shift
from rsciio.quantumdetector import file_reader


mib_path = Path(
    r"data\3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib"
)

out_dir = Path("outputs/origin_check")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. 读取 MIB
data = file_reader(
    str(mib_path),
    lazy=True,
    navigation_shape=(256, 256),
    chunks=(1, 16, 256, 256),
)[0]["data"]

# 2. 抽样扫描坐标
stride = 8

ys = np.arange(0, 256, stride)
xs = np.arange(0, 256, stride)

origin_y = np.zeros((len(ys), len(xs)))
origin_x = np.zeros_like(origin_y)

mean_raw = np.zeros((256, 256), dtype=np.float64)
mean_aligned = np.zeros_like(mean_raw)

reference_center = (127.5, 127.5)

n_frames = 0

# 3. 测量直射束中心
for i, y in enumerate(ys):

    frames = np.asarray(
        data[int(y), ::stride].compute(),
        dtype=np.float32,
    )

    smoothed = gaussian_filter(
        frames,
        sigma=(0, 1.5, 1.5),
    )

    # 已知直射束大致在中心区域
    cropped = smoothed[:, 64:192, 64:192]

    peak_indices = np.argmax(
        cropped.reshape(len(xs), -1),
        axis=1,
    )

    centers_y = 64 + peak_indices // 128
    centers_x = 64 + peak_indices % 128

    origin_y[i] = centers_y
    origin_x[i] = centers_x

    # 4. 比较对齐前后的 DP
    for j, dp in enumerate(frames):

        cy = centers_y[j]
        cx = centers_x[j]

        mean_raw += dp

        aligned = shift(
            dp,
            shift=(
                reference_center[0] - cy,
                reference_center[1] - cx,
            ),
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )

        mean_aligned += aligned
        n_frames += 1

    print(f"Processed {i + 1}/{len(ys)} rows")

mean_raw /= n_frames
mean_aligned /= n_frames

# 5. 保存中心位置数据
np.savez(
    out_dir / "origin_sampled.npz",
    scan_y=ys,
    scan_x=xs,
    origin_y=origin_y,
    origin_x=origin_x,
)

# 6. 绘制直射束中心分布
magnitude = np.hypot(
    origin_y - reference_center[0],
    origin_x - reference_center[1],
)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, image, title in zip(
    axes,
    (origin_x, origin_y, magnitude),
    ("Origin X", "Origin Y", "Offset magnitude"),
):
    im = ax.imshow(image, cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

fig.tight_layout()
fig.savefig(out_dir / "origin_maps.png", dpi=200)
plt.close(fig)

# 7. 比较平均 DP
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

before = np.log1p(mean_raw)
after = np.log1p(mean_aligned)

vmin, vmax = np.percentile(
    np.stack([before, after]), [1, 99.5]
)

axes[0].imshow(
    before, cmap="gray", vmin=vmin, vmax=vmax
)
axes[0].set_title("Before alignment")

axes[1].imshow(
    after, cmap="gray", vmin=vmin, vmax=vmax
)
axes[1].set_title("After alignment")

for ax in axes:
    ax.axis("off")

fig.tight_layout()
fig.savefig(
    out_dir / "mean_dp_comparison.png",
    dpi=200,
)
plt.close(fig)

print("Offset range:", magnitude.min(), magnitude.max())
print("Saved results to:", out_dir.resolve())