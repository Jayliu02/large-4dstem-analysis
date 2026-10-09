
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from rsciio.quantumdetector import file_reader


# ---------- 配置 ----------

mib_path = Path(
    r"data\3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib"
)

out = Path("outputs/origin_check")
out.mkdir(parents=True, exist_ok=True)

batch_size = 16
bf_radius = 15
adf_inner = 18
adf_outer = 50


# ---------- 1. 读取已测量的原点 ----------

sample = np.load(out / "origin_sampled.npz")

sy_sample = sample["scan_y"]
sx_sample = sample["scan_x"]

Y, X = np.meshgrid(
    sy_sample, sx_sample, indexing="ij"
)

# 平面：q0 = c0 + c1 * scan_x + c2 * scan_y
A = np.column_stack([
    np.ones(X.size),
    X.ravel(),
    Y.ravel(),
])


# ---------- 2. 稳健二维平面拟合 ----------

def fit_origin(z, name):
    values = z.ravel()
    valid = np.isfinite(values)

    a = A[valid]
    b = values[valid]

    initial = np.linalg.lstsq(
        a, b, rcond=None
    )[0]

    result = least_squares(
        lambda p: a @ p - b,
        initial,
        loss="soft_l1",
        f_scale=2.0,
    )

    residual = b - a @ result.x

    print(
        name,
        "median abs residual:",
        np.median(np.abs(residual)),
        "px"
    )
    print(
        name,
        "95% abs residual:",
        np.percentile(np.abs(residual), 95),
        "px"
    )

    return result.x


px = fit_origin(sample["origin_x"], "Origin X")
py = fit_origin(sample["origin_y"], "Origin Y")


# ---------- 3. 生成全扫描原点场 ----------

yy, xx = np.indices((256, 256))

origin_x = px[0] + px[1] * xx + px[2] * yy
origin_y = py[0] + py[1] * xx + py[2] * yy

np.savez(
    out / "origin_fitted.npz",
    origin_x=origin_x,
    origin_y=origin_y,
    coefficients_x=px,
    coefficients_y=py,
)


# ---------- 4. 延迟读取原始 MIB ----------

data = file_reader(
    str(mib_path),
    lazy=True,
    navigation_shape=(256, 256),
    chunks=(1, batch_size, 256, 256),
)[0]["data"]

scan_y, scan_x, qy, qx = data.shape

det_y, det_x = np.ogrid[:qy, :qx]

center_y = (qy - 1) / 2
center_x = (qx - 1) / 2

# 固定探测器
r2_fixed = (
    (det_y - center_y) ** 2
    + (det_x - center_x) ** 2
)

bf_mask_fixed = r2_fixed <= bf_radius**2
adf_mask_fixed = (
    (r2_fixed >= adf_inner**2)
    & (r2_fixed < adf_outer**2)
)


# ---------- 5. 分块计算 BF / ADF ----------

bf_fixed = np.zeros((scan_y, scan_x))
bf_corrected = np.zeros_like(bf_fixed)

adf_fixed = np.zeros_like(bf_fixed)
adf_corrected = np.zeros_like(bf_fixed)

for y in range(scan_y):
    for x0 in range(0, scan_x, batch_size):
        x1 = min(x0 + batch_size, scan_x)

        frames = np.asarray(
            data[y, x0:x1].compute(),
            dtype=np.uint16,
        )

        # 原始固定探测器
        bf_fixed[y, x0:x1] = frames[
            :, bf_mask_fixed
        ].sum(axis=1, dtype=np.uint64)

        adf_fixed[y, x0:x1] = frames[
            :, adf_mask_fixed
        ].sum(axis=1, dtype=np.uint64)

        # 每个扫描位置采用拟合的直射束中心
        cy = origin_y[y, x0:x1][:, None, None]
        cx = origin_x[y, x0:x1][:, None, None]

        r2 = (
            (det_y[None, :, :] - cy) ** 2
            + (det_x[None, :, :] - cx) ** 2
        )

        bf_mask = r2 <= bf_radius**2

        adf_mask = (
            (r2 >= adf_inner**2)
            & (r2 < adf_outer**2)
        )

        bf_corrected[y, x0:x1] = (
            (frames * bf_mask).sum(
                axis=(1, 2), dtype=np.uint64
            )
        )

        adf_corrected[y, x0:x1] = (
            (frames * adf_mask).sum(
                axis=(1, 2), dtype=np.uint64
            )
        )

    if (y + 1) % 16 == 0:
        print(f"Processed {y + 1}/{scan_y} rows")


# ---------- 6. 保存数值结果 ----------

for name, image in {
    "bf_fixed": bf_fixed,
    "bf_corrected": bf_corrected,
    "adf_fixed": adf_fixed,
    "adf_corrected": adf_corrected,
}.items():
    np.save(out / f"{name}.npy", image)


# ---------- 7. 绘制对比 ----------

fig, axes = plt.subplots(
    2, 2, figsize=(10, 10)
)

pairs = [
    (bf_fixed, bf_corrected, "BF"),
    (adf_fixed, adf_corrected, "ADF"),
]

for row, (raw, corrected, label) in enumerate(pairs):

    a = np.log1p(raw)
    b = np.log1p(corrected)

    # 同一对图使用一致的显示范围
    vmin, vmax = np.percentile(
        np.concatenate([a.ravel(), b.ravel()]),
        [1, 99],
    )

    axes[row, 0].imshow(
        a, cmap="gray", vmin=vmin, vmax=vmax
    )
    axes[row, 1].imshow(
        b, cmap="gray", vmin=vmin, vmax=vmax
    )

    axes[row, 0].set_title(f"{label} - Fixed")
    axes[row, 1].set_title(f"{label} - Corrected")

for ax in axes.flat:
    ax.axis("off")

fig.tight_layout()
fig.savefig(
    out / "bf_adf_comparison.png",
    dpi=300,
)
plt.close(fig)

print("Completed:", out.resolve())
