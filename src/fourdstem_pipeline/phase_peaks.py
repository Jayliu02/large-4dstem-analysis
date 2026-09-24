"""Per-pattern beam localization and sparse peaks, with resumable bounded I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import ndimage

from .loaders import load_dataset
from .basic_analysis import inspect_mib

# This exact predecessor differs only in atomic-write retries and the cache
# migration below. Numerical extraction functions and array layouts are identical.
_IO_ONLY_PREDECESSOR = '26bb235852f1e04a1525c8a2255e339fe115a44688a09504b157b9b8a55e5ef6'


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False), encoding='utf-8')
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows indexing/virus scanners can briefly hold the destination.
            # Keep the old checkpoint intact and retry only the atomic replace.
            if attempt == 7:
                raise
            time.sleep(0.05 * 2**attempt)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def centroid(image, y, x, radius=3):
    y, x = int(y), int(x)
    y0, y1 = max(0, y-radius), min(image.shape[0], y+radius+1)
    x0, x1 = max(0, x-radius), min(image.shape[1], x+radius+1)
    patch = image[y0:y1, x0:x1]
    weights = np.maximum(patch - np.percentile(patch, 20), 0)
    yy, xx = np.indices(patch.shape)
    total = weights.sum()
    if total <= 0:
        return np.array([float(y), float(x)]), 0.0
    return np.array([y0+(yy*weights).sum()/total, x0+(xx*weights).sum()/total]), float(total)


def find_pattern_peaks(pattern: np.ndarray, predicted_yx, cfg: dict):
    """Return raw detector positions, intensities, centre and validity diagnostics."""
    image = np.asarray(pattern, dtype=np.float32).copy()
    invalid = image >= cfg['invalid_count']
    if invalid.any():
        filtered = ndimage.median_filter(image, size=3)
        image[invalid] = filtered[invalid]
    smooth = ndimage.gaussian_filter(image, 1)
    background = ndimage.gaussian_filter(image, 3)
    signal = np.maximum(smooth-background, 0)
    sy, sx = image.shape
    py, px = np.rint(predicted_yx).astype(int)
    radius = int(cfg['center_search_radius'])
    y0, y1 = max(3, py-radius), min(sy-3, py+radius+1)
    x0, x1 = max(3, px-radius), min(sx-3, px+radius+1)
    if y0 >= y1 or x0 >= x1:
        return np.zeros((0, 2)), np.zeros(0), np.array(predicted_yx), False, 0, np.inf
    dy, dx = np.unravel_index(signal[y0:y1, x0:x1].argmax(), (y1-y0, x1-x0))
    center, center_intensity = centroid(image, y0+dy, x0+dx)
    displacement = float(np.linalg.norm(center-predicted_yx))
    local_max = signal == ndimage.maximum_filter(signal, size=2*int(cfg['minimum_spacing'])-1)
    mad = np.median(np.abs(signal-np.median(signal))) * 1.4826
    threshold = max(1.5, cfg['noise_sigma']*mad, cfg['relative_threshold']*float(signal.max()))
    yy, xx = np.indices(image.shape)
    rr = np.hypot(yy-center[0], xx-center[1])
    allowed = (rr >= cfg['exclusion_radius']) & (rr <= cfg['max_radius'])
    allowed[:4] = allowed[-4:] = False
    allowed[:, :4] = allowed[:, -4:] = False
    allowed &= ~ndimage.maximum_filter(invalid, size=7)
    points = np.argwhere(local_max & allowed & (signal > threshold))
    if len(points):
        order = np.argsort(signal[points[:, 0], points[:, 1]])[::-1][:cfg['max_peaks']]
        peaks, intensity = zip(*(centroid(image, *point) for point in points[order]))
        peaks, intensity = np.array(peaks), np.array(intensity)
    else:
        peaks, intensity = np.empty((0, 2)), np.empty(0)
    # Friedel-pair midpoints cross-check the local direct-beam localization.
    mids = []
    for i in range(len(peaks)):
        for j in range(i+1, len(peaks)):
            middle = (peaks[i]+peaks[j])/2
            if np.linalg.norm(middle-center) <= 1.5:
                mids.append(middle)
    if len(mids) >= 2:
        midpoint = np.median(mids, axis=0)
        center = (center+midpoint)/2
    valid = center_intensity > 0 and displacement <= cfg['center_max_residual']
    return peaks, intensity, center, valid, len(mids), displacement


def extract_scan(path: Path, basic: Path, output: Path, scan_shape, cfg: dict, *, resume=True):
    output.mkdir(parents=True, exist_ok=True)
    motion = read_json(basic/'beam_motion.json')
    signature_data = {'path': str(path.resolve()), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                      'shape': scan_shape, 'config': cfg, 'motion': motion,
                      'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature = digest(signature_data)
    checkpoint_path = output/'peaks_checkpoint.json'
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else {}
    legacy_data = {**signature_data, 'implementation': _IO_ONLY_PREDECESSOR}
    if checkpoint and resume and checkpoint.get('signature') == digest(legacy_data):
        checkpoint.setdefault('migrations', []).append({
            'previous_signature': checkpoint['signature'], 'previous_provenance': checkpoint['provenance'],
            'reason': 'atomic JSON write retry only; unchanged numerical peak extraction'})
        checkpoint['signature'], checkpoint['provenance'] = signature, signature_data
        atomic_json(checkpoint_path, checkpoint)
    if checkpoint and (checkpoint.get('signature') != signature or not resume):
        raise ValueError('Peak cache input/configuration changed; use a new output directory.')
    if not checkpoint:
        header = inspect_mib(path, tuple(scan_shape))
        atomic_json(output/'input_metadata.json', header)
        checkpoint = {'signature': signature, 'provenance': signature_data, 'tiles': [], 'complete': False}
    ny, nx = scan_shape
    max_peaks = cfg['max_peaks']
    shapes = {'peak_yx': ((ny,nx,max_peaks,2), 'float32'), 'intensity': ((ny,nx,max_peaks), 'float32'),
              'count': ((ny,nx), 'int16'), 'center_yx': ((ny,nx,2), 'float32'),
              'center_valid': ((ny,nx), 'bool'), 'friedel_pairs': ((ny,nx), 'int16'),
              'center_residual': ((ny,nx), 'float32')}
    arrays = {}
    for name, (shape, dtype) in shapes.items():
        file = output/f'{name}.npy'
        if file.exists():
            arrays[name] = np.load(file, mmap_mode='r+')
        else:
            arrays[name] = np.lib.format.open_memmap(file, mode='w+', dtype=dtype, shape=shape)
            arrays[name][:] = 0
    if checkpoint.get('complete'):
        return arrays
    metadata = read_json(output/'input_metadata.json')
    dataset = load_dataset(path, backend='mib_memmap', scan_shape=scan_shape,
                           detector_shape=metadata['detector_shape'], dtype='>u2', mib_header_bytes=384)
    coeff, intercept = np.array(motion['affine_coefficients_yx']), np.array(motion['affine_intercept_yx'])
    block = cfg['block_size']
    done = set(checkpoint['tiles'])
    started = time.perf_counter()
    for y0 in range(0, ny, block):
        for x0 in range(0, nx, block):
            key = f'{y0},{x0}'
            if key in done:
                continue
            cube = np.asarray(dataset.data[y0:min(y0+block,ny), x0:min(x0+block,nx), :, :])
            for iy in range(cube.shape[0]):
                for ix in range(cube.shape[1]):
                    y, x = y0+iy, x0+ix
                    predicted = coeff @ np.array([y,x]) + intercept
                    peaks, intensity, center, valid, pairs, residual = find_pattern_peaks(cube[iy,ix], predicted, cfg)
                    n = len(peaks)
                    arrays['peak_yx'][y,x] = 0
                    arrays['intensity'][y,x] = 0
                    arrays['peak_yx'][y,x,:n] = peaks
                    arrays['intensity'][y,x,:n] = intensity
                    arrays['count'][y,x] = n
                    arrays['center_yx'][y,x] = center
                    arrays['center_valid'][y,x] = valid
                    arrays['friedel_pairs'][y,x] = pairs
                    arrays['center_residual'][y,x] = residual
            for arr in arrays.values():
                arr.flush()
            done.add(key)
            checkpoint['tiles'] = sorted(done)
            atomic_json(checkpoint_path, checkpoint)
        print(f'  peaks {path.stem[-4:]} rows {min(y0+block,ny)}/{ny}, {time.perf_counter()-started:.0f}s', flush=True)
    gy,gx = np.indices((ny,nx))
    predicted = np.stack([gy,gx],axis=-1) @ coeff.T + intercept
    residual = arrays['center_yx']-predicted
    local = ndimage.median_filter(residual, size=(3,3,1))
    discontinuity = np.linalg.norm(residual-local, axis=-1) > 2.0
    arrays['center_valid'][:] &= ~discontinuity
    arrays['center_valid'].flush()
    checkpoint['complete'] = True
    atomic_json(checkpoint_path, checkpoint)
    atomic_json(output/'peaks_summary.json', {'patterns': ny*nx, 'valid_center_fraction': float(arrays['center_valid'].mean()),
                'median_peaks': float(np.median(arrays['count'])), 'at_least_six_fraction': float(np.mean(arrays['count']>=6)),
                'friedel_pair_fraction': float(np.mean(arrays['friedel_pairs']>=2)), 'neighbor_rejections': int(discontinuity.sum()),
                'input_unchanged': path.stat().st_mtime_ns == signature_data['mtime_ns']})
    return arrays
