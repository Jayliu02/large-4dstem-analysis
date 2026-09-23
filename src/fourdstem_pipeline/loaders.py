from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DatasetHandle
from .logging import get_logger
from .synthetic import make_synthetic_4dstem

log = get_logger(__name__)


def _load_h5(
    data_path: Path,
    *,
    lazy: bool = True,
    scan_shape: tuple | None = None,
    detector_shape: tuple | None = None,
    backend: str | None = None,
) -> DatasetHandle:
    """Load a 4D-STEM dataset from an HDF5 file via h5py."""
    import h5py

    f = h5py.File(data_path, "r")

    # Find the largest 4D dataset in the file.
    candidate_paths: list[tuple[str, tuple[int, ...], type]] = []

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            candidate_paths.append((name, obj.shape, obj.dtype))

    f.visititems(_visit)

    # Prefer datasets named "data" under a "datacube" group (common convention).
    data_path_h5: str | None = None
    for name, shape, _dtype in candidate_paths:
        if name.endswith("/data") and "datacube" in name and len(shape) >= 2:
            data_path_h5 = name
            break

    # Fall back to the largest 4D dataset.
    if data_path_h5 is None:
        datasets_4d = [(n, s, d) for n, s, d in candidate_paths if len(s) == 4]
        if datasets_4d:
            datasets_4d.sort(key=lambda x: int(np.prod(x[1])), reverse=True)
            data_path_h5 = datasets_4d[0][0]
        elif candidate_paths:
            # No 4D dataset — take the largest 2D/3D one.
            candidate_paths.sort(key=lambda x: int(np.prod(x[1])), reverse=True)
            data_path_h5 = candidate_paths[0][0]

    if data_path_h5 is None:
        f.close()
        raise ValueError(f"No suitable dataset found in {data_path}. Available: {[n for n, _, _ in candidate_paths]}")

    ds = f[data_path_h5]
    shape = ds.shape
    dtype_np = ds.dtype

    if len(shape) < 2:
        f.close()
        raise ValueError(f"Dataset {data_path_h5} in {data_path} has only {len(shape)} dimensions; need ≥ 2.")

    # For 2D data (single diffraction pattern), add dummy navigation dims.
    if len(shape) == 2:
        data = np.asarray(ds[:], dtype=np.float32)
        data = data[None, None, :, :]  # (1, 1, qy, qx)
    elif len(shape) == 3:
        data = np.asarray(ds[:], dtype=np.float32)
        data = data[:, None, :, :]  # (ny, 1, qy, qx)
    else:
        # 4D: load eagerly for small/medium data; h5py doesn't support
        # memory-mapping, but slicing + numpy is fine for ≤ 1 GB datasets.
        data = np.asarray(ds[:], dtype=np.float32)
        # Handle 5+D if needed.
        if len(shape) > 4:
            data = data.reshape(-1, shape[-2], shape[-1])
            data = data.reshape((1, 1) + data.shape)

    f.close()

    if scan_shape is None:
        scan_shape = tuple(int(v) for v in data.shape[:2])
    if detector_shape is None:
        detector_shape = tuple(int(v) for v in data.shape[-2:])

    return DatasetHandle(
        data=data,
        source="h5",
        metadata={
            "path": str(data_path),
            "h5_dataset": data_path_h5,
            "source_backend": backend or "h5py",
            "scan_shape": scan_shape,
            "detector_shape": detector_shape,
            "original_dtype": str(dtype_np),
        },
    )


def load_dataset(
    path: str | Path,
    *,
    lazy: bool = True,
    cache: str | Path | None = None,
    chunks: dict[str, Any] | tuple[int, ...] | None = None,
    backend: str | None = None,
    scan_shape: tuple[int, int] | list[int] | None = None,
    detector_shape: tuple[int, int] | list[int] | None = None,
    dtype: str | np.dtype | None = None,
    mib_header_bytes: int | None = None,
) -> DatasetHandle:
    """Load a 4D-STEM dataset.

    `.mib` files are passed to HyperSpy with lazy loading when HyperSpy is
    installed. `.npy` and `.npz` are supported for tests and cached workflows.
    Use `synthetic://demo` for a self-contained demonstration dataset.
    """
    source = str(path)
    if source.startswith("synthetic://"):
        data, labels = make_synthetic_4dstem()
        return DatasetHandle(
            data=data,
            source="synthetic",
            metadata={
                "labels": labels,
                "path": source,
                "source_backend": backend or "synthetic",
                "scan_shape": data.shape[:2],
                "detector_shape": data.shape[-2:],
            },
        )

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(data_path, mmap_mode="r" if lazy else None)
        return DatasetHandle(
            data=data,
            source="npy",
            metadata={
                "path": str(data_path),
                "cache": str(cache) if cache else None,
                "source_backend": backend or "numpy",
                "scan_shape": tuple(data.shape[:2]),
                "detector_shape": tuple(data.shape[-2:]),
            },
        )

    if suffix == ".npz":
        if lazy:
            log.info(
                "NPZ lazy loading requested for %s, but NumPy cannot memory-map NPZ members; "
                "loading the selected array eagerly.",
                data_path,
            )
        archive = np.load(data_path)
        key = "data" if "data" in archive.files else archive.files[0]
        data = archive[key]
        return DatasetHandle(
            data=data,
            source="npz",
            metadata={
                "path": str(data_path),
                "npz_key": key,
                "source_backend": backend or "numpy",
                "scan_shape": tuple(data.shape[:2]),
                "detector_shape": tuple(data.shape[-2:]),
            },
        )

    if suffix in (".h5", ".hdf5"):
        return _load_h5(data_path, lazy=lazy, scan_shape=scan_shape, detector_shape=detector_shape, backend=backend)

    if suffix == ".mib":
        if backend == "mib_memmap":
            if scan_shape is None or detector_shape is None or dtype is None or mib_header_bytes is None:
                raise ValueError("Explicit mib_memmap requires scan_shape, detector_shape, dtype (including byte order), and mib_header_bytes.")
            expected_bytes = int(np.prod(scan_shape)) * (int(mib_header_bytes) + int(np.prod(detector_shape)) * np.dtype(dtype).itemsize)
            if data_path.stat().st_size != expected_bytes:
                raise ValueError("MIB file size does not match the explicit fixed-frame layout.")
            return _load_mib_memmap(data_path, scan_shape=scan_shape, detector_shape=detector_shape,
                                    dtype=dtype, header_bytes=mib_header_bytes, cache=cache, backend=backend)
        return _load_mib_with_hyperspy(
            data_path,
            lazy=lazy,
            cache=cache,
            chunks=chunks,
            backend=backend,
            scan_shape=scan_shape,
            detector_shape=detector_shape,
            dtype=dtype,
            mib_header_bytes=mib_header_bytes,
        )

    return _load_with_hyperspy(data_path, lazy=lazy, cache=cache, chunks=chunks, backend=backend)


def _load_mib_with_hyperspy(
    path: Path,
    *,
    lazy: bool,
    cache: str | Path | None,
    chunks: Any,
    backend: str | None,
    scan_shape: tuple[int, int] | list[int] | None,
    detector_shape: tuple[int, int] | list[int] | None,
    dtype: str | np.dtype | None,
    mib_header_bytes: int | None,
) -> DatasetHandle:
    try:
        return _load_with_hyperspy(path, lazy=lazy, cache=cache, chunks=chunks, backend=backend or "hyperspy_pyxem", source="mib")
    except ImportError:
        if scan_shape is None or detector_shape is None:
            raise
        return _load_mib_memmap(
            path,
            scan_shape=scan_shape,
            detector_shape=detector_shape,
            dtype=dtype or np.uint16,
            header_bytes=mib_header_bytes,
            cache=cache,
            backend=backend,
        )


class MIBMemmapArray:
    """NumPy-like view for fixed-frame MIB files with per-frame headers."""

    def __init__(self, path: Path, scan_shape: tuple[int, int], detector_shape: tuple[int, int], dtype: np.dtype, header_bytes: int):
        self.path = path
        self.scan_shape = scan_shape
        self.detector_shape = detector_shape
        self.dtype = dtype
        self.header_bytes = int(header_bytes)
        frame_dtype = np.dtype([("header", f"V{self.header_bytes}"), ("image", self.dtype, self.detector_shape)])
        self._frames = np.memmap(path, mode="r", dtype=frame_dtype, shape=self.scan_shape)
        self.shape = self.scan_shape + self.detector_shape

    def __getitem__(self, key: Any) -> np.ndarray:
        y_key, x_key, qy_key, qx_key = _normalize_mib_key(key)
        return np.asarray(self._frames[y_key, x_key]["image"][..., qy_key, qx_key])


def _load_mib_memmap(
    path: Path,
    *,
    scan_shape: tuple[int, int] | list[int],
    detector_shape: tuple[int, int] | list[int],
    dtype: str | np.dtype,
    header_bytes: int | None,
    cache: str | Path | None,
    backend: str | None,
) -> DatasetHandle:
    scan = tuple(int(v) for v in scan_shape)
    detector = tuple(int(v) for v in detector_shape)
    np_dtype = np.dtype(dtype)
    if header_bytes is None:
        header_bytes = _infer_mib_header_bytes(path, scan, detector, np_dtype)
    data = MIBMemmapArray(path, scan, detector, np_dtype, header_bytes)
    return DatasetHandle(
        data=data,
        source="mib",
        metadata={
            "path": str(path),
            "cache": str(cache) if cache else None,
            "lazy": True,
            "source_backend": backend or "mib_memmap",
            "scan_shape": scan,
            "detector_shape": detector,
            "mib_header_bytes": header_bytes,
            "axes": [],
            "pyxem_available": False,
            "pyxem_signal_type": None,
            "pyxem_error": "Fixed-frame memmap reader used; HyperSpy was not used.",
        },
    )


def _infer_mib_header_bytes(path: Path, scan_shape: tuple[int, int], detector_shape: tuple[int, int], dtype: np.dtype) -> int:
    n_frames = int(np.prod(scan_shape))
    image_bytes = int(np.prod(detector_shape) * dtype.itemsize)
    frame_bytes, remainder = divmod(path.stat().st_size, n_frames)
    if remainder or frame_bytes < image_bytes:
        raise ValueError(
            f"Cannot infer fixed MIB frame layout for {path}: file size is not compatible "
            f"with scan_shape={scan_shape}, detector_shape={detector_shape}, dtype={dtype}."
        )
    return frame_bytes - image_bytes


def _normalize_mib_key(key: Any) -> tuple[Any, Any, Any, Any]:
    if not isinstance(key, tuple):
        key = (key,)
    key = key + (slice(None),) * (4 - len(key))
    if len(key) != 4:
        raise IndexError("MIBMemmapArray expects 4D indexing.")
    return key


def _load_with_hyperspy(
    path: Path,
    *,
    lazy: bool,
    cache: str | Path | None,
    chunks: Any,
    backend: str | None,
    source: str | None = None,
) -> DatasetHandle:
    try:
        import hyperspy.api as hs
    except ImportError as exc:
        raise ImportError(
            "HyperSpy is required for this file type. Install the optional large-data dependencies "
            "or convert the dataset to .npy/.npz for the synthetic/test workflow."
        ) from exc

    signal = hs.load(str(path), lazy=lazy)
    pyxem_info = _try_make_pyxem_compatible(signal) if backend == "hyperspy_pyxem" else {"pyxem_available": False, "pyxem_signal_type": None, "pyxem_error": None}
    data = signal.data
    if chunks and hasattr(data, "rechunk"):
        data = data.rechunk(_normalize_chunks(chunks, data.shape))
        signal.data = data
    metadata = {
        "path": str(path),
        "cache": str(cache) if cache else None,
        "lazy": lazy,
        "source_backend": backend or "hyperspy",
        "axes": _extract_axes_metadata(signal),
        "scan_shape": tuple(int(v) for v in data.shape[:2]),
        "detector_shape": tuple(int(v) for v in data.shape[-2:]),
        **pyxem_info,
    }
    return DatasetHandle(data=data, signal=signal, source=source or path.suffix.lower().lstrip("."), metadata=metadata)


def _normalize_chunks(chunks: Any, shape: tuple[int, ...]) -> Any:
    if isinstance(chunks, dict):
        nav = tuple(chunks.get("navigation", (8, 8)))
        sig = tuple(chunks.get("signal", shape[-2:]))
        return nav + sig
    return chunks


def _try_make_pyxem_compatible(signal: Any) -> dict[str, Any]:
    try:
        import pyxem  # noqa: F401
    except ImportError as exc:
        return {"pyxem_available": False, "pyxem_signal_type": None, "pyxem_error": str(exc)}

    signal_type = None
    error = None
    setter = getattr(signal, "set_signal_type", None)
    if callable(setter):
        for candidate in ("electron_diffraction", "electron_diffraction2d"):
            try:
                setter(candidate)
                signal_type = candidate
                break
            except Exception as exc:  # pyxem/HyperSpy versions differ in registered names.
                error = str(exc)
    if signal_type is None:
        signal_type = _signal_type_name(signal)
    return {"pyxem_available": True, "pyxem_signal_type": signal_type, "pyxem_error": error}


def _signal_type_name(signal: Any) -> str | None:
    metadata = getattr(signal, "metadata", None)
    signal_node = getattr(metadata, "Signal", None)
    signal_type = getattr(signal_node, "signal_type", None)
    return str(signal_type) if signal_type else None


def _extract_axes_metadata(signal: Any) -> list[dict[str, Any]]:
    axes_manager = getattr(signal, "axes_manager", None)
    if axes_manager is None:
        return []
    axes = []
    for axis in axes_manager:
        axes.append(
            {
                "name": str(getattr(axis, "name", "")),
                "size": int(getattr(axis, "size", 0)),
                "scale": _safe_axis_value(getattr(axis, "scale", None)),
                "offset": _safe_axis_value(getattr(axis, "offset", None)),
                "units": str(getattr(axis, "units", "")),
                "navigate": bool(getattr(axis, "navigate", False)),
            }
        )
    return axes


def _safe_axis_value(value: Any) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)
