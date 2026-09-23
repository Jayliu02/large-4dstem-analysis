from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .array_utils import as_numpy_block, iter_navigation_slices
from .dataset import DatasetHandle, save_jsonable_metadata
from .logging import get_logger, log_block_progress

log = get_logger(__name__)


@dataclass(slots=True)
class VirtualImageResult:
    images: dict[str, np.ndarray]
    com_x: np.ndarray
    com_y: np.ndarray
    mean_diffraction: np.ndarray
    max_diffraction: np.ndarray
    output_dir: Path | None = None
    saturation_fraction: np.ndarray | None = None


def compute_virtual_images(
    dataset: DatasetHandle,
    masks: dict[str, np.ndarray],
    *,
    output_dir: str | Path | None = None,
    block_shape: tuple[int, int] = (8, 8),
) -> VirtualImageResult:
    """Compute virtual detector images and diffraction previews in navigation blocks."""
    nav_shape = dataset.navigation_shape
    sig_shape = dataset.signal_shape
    images = {name: np.zeros(nav_shape, dtype=np.float32) for name in masks}
    com_x = np.zeros(nav_shape, dtype=np.float32)
    com_y = np.zeros(nav_shape, dtype=np.float32)
    mean_sum = np.zeros(sig_shape, dtype=np.float64)
    max_diff = np.zeros(sig_shape, dtype=np.float32)
    n_patterns = 0

    yy, xx = np.indices(sig_shape, dtype=np.float32)
    blocks = list(iter_navigation_slices(nav_shape, block_shape))
    n_blocks = len(blocks)
    log.info("Computing %d virtual images across %d navigation blocks (block_shape=%s)", len(masks), n_blocks, block_shape)

    for idx, (ys, xs) in enumerate(blocks, start=1):
        log_block_progress(log, block=idx, total_blocks=n_blocks, stage="virtual")
        block = as_numpy_block(dataset.data[ys, xs, :, :]).astype(np.float32, copy=False)
        for name, mask in masks.items():
            images[name][ys, xs] = block[..., mask].sum(axis=-1)
        total = np.maximum(block.sum(axis=(-2, -1)), 1e-12)
        com_x[ys, xs] = (block * xx).sum(axis=(-2, -1)) / total
        com_y[ys, xs] = (block * yy).sum(axis=(-2, -1)) / total
        mean_sum += block.sum(axis=(0, 1), dtype=np.float64)
        max_diff = np.maximum(max_diff, block.max(axis=(0, 1)))
        n_patterns += block.shape[0] * block.shape[1]

    result = VirtualImageResult(
        images=images,
        com_x=com_x,
        com_y=com_y,
        mean_diffraction=(mean_sum / max(n_patterns, 1)).astype(np.float32),
        max_diffraction=max_diff,
        output_dir=Path(output_dir) if output_dir else None,
    )
    if output_dir:
        _save_virtual_result(result, dataset)
    return result


def _save_virtual_result(result: VirtualImageResult, dataset: DatasetHandle) -> None:
    assert result.output_dir is not None
    result.output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in result.images.items():
        np.save(result.output_dir / f"virtual_{name}.npy", image)
    np.save(result.output_dir / "com_x.npy", result.com_x)
    np.save(result.output_dir / "com_y.npy", result.com_y)
    np.save(result.output_dir / "mean_diffraction.npy", result.mean_diffraction)
    np.save(result.output_dir / "max_diffraction.npy", result.max_diffraction)
    save_jsonable_metadata(result.output_dir / "dataset_summary.json", dataset.describe())
