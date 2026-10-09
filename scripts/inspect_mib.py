from pathlib import Path
from rsciio.quantumdetector import file_reader

file_path = r"data\3-1-ss5.8nm-c2 50-cl 110-sp7-gunlens2-12bit-256x256-0.62ms-67k-1045.mib"
mib_path = Path(
    file_path
)

# 文件大小
size_gb = mib_path.stat().st_size / 1024**3
print(f"File size: {size_gb:.3f} GiB")

# 读取 MIB，保持延迟加载
result = file_reader(
    file_path,
    lazy=True,
    print_info=True,
)

data = result[0]["data"]

print("Data shape:", data.shape)
print("Data dtype:", data.dtype)
print("Dask chunks:", data.chunks)

print("\nMetadata:")
print(result[0]["metadata"])

print("\nOriginal metadata:")
print(result[0]["original_metadata"])

import numpy as np
import matplotlib.pyplot as plt

# data 来自之前的 file_reader
# 每个扫描位置取探测器中心附近的积分强度

mask_y, mask_x = np.ogrid[:256, :256]

r = np.sqrt(
    (mask_y - 128) ** 2 +
    (mask_x - 128) ** 2
)

mask = r < 15

bf = np.zeros((256, 256))

for y in range(256):
    frames = data[y, :, :, :].compute()
    bf[y] = frames[:, mask].sum(axis=1)

plt.figure(figsize=(7, 7))
plt.imshow(bf, cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()