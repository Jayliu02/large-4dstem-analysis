"""Phase-independent weak-peak and nonlocal averaging experiments."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .phase_peaks import centroid


def local_signal(image, invalid_count):
    image = np.asarray(image, np.float32)
    invalid = image >= invalid_count
    clean = image.copy()
    clean[invalid] = ndimage.median_filter(clean, size=3)[invalid]
    background = ndimage.gaussian_filter(clean, 3)
    signal = ndimage.gaussian_filter(clean, 1)-background
    # Poisson propagation through the same difference-of-Gaussians filter.
    impulse = np.zeros((25, 25))
    impulse[12, 12] = 1
    kernel = ndimage.gaussian_filter(impulse, 1)-ndimage.gaussian_filter(impulse, 3)
    variance = ndimage.convolve(np.maximum(background, 1), kernel**2)
    return signal, np.sqrt(np.maximum(variance, .01)), invalid


def raw_peak_support(image, point):
    """Background-subtracted integrated raw counts in a 3-px disk vs 4-6-px annulus."""
    y, x = np.rint(point).astype(int)
    if y < 6 or x < 6 or y >= image.shape[0]-6 or x >= image.shape[1]-6:
        return 0.
    patch = np.asarray(image[y-6:y+7, x-6:x+7], float)
    yy, xx = np.indices(patch.shape)
    rr = np.hypot(yy-6, xx-6)
    core, ring = rr <= 3, (rr >= 4) & (rr <= 6)
    background = np.median(patch[ring])
    net = patch[core].sum()-core.sum()*background
    variance = np.maximum(patch[core], 1).sum()+core.sum()**2*np.maximum(patch[ring], 1).mean()/ring.sum()
    return float(net/np.sqrt(variance))


def adaptive_peaks(image, raw, center, peaks_cfg, experiment):
    signal, noise, invalid = local_signal(image, peaks_cfg['invalid_count'])
    yy, xx = np.indices(signal.shape)
    radius = np.hypot(yy-center[0], xx-center[1])
    allowed = ((radius >= peaks_cfg['exclusion_radius']) & (radius <= peaks_cfg['max_radius']))
    allowed[:6] = allowed[-6:] = False
    allowed[:, :6] = allowed[:, -6:] = False
    # Invalid raw pixels remain excluded even after averaging.
    invalid |= np.asarray(raw) >= peaks_cfg['invalid_count']
    allowed &= ~ndimage.maximum_filter(invalid, size=7)
    maxima = signal == ndimage.maximum_filter(signal, size=2*peaks_cfg['minimum_spacing']-1)
    candidates = np.argwhere(allowed & maxima & (signal >= experiment['snr']*noise))
    if len(candidates):
        order = np.argsort(signal[candidates[:, 0], candidates[:, 1]])[::-1]
        candidates = candidates[order]
    positions, intensities, support = [], [], []
    for point in candidates:
        # Always localize and measure on the original image; averaged data only proposes peaks.
        location, intensity = centroid(np.asarray(raw, np.float32), *point)
        snr = raw_peak_support(raw, location)
        if snr < experiment['raw_support_snr']:
            continue
        if positions and np.min(np.linalg.norm(np.asarray(positions)-location, axis=1)) < peaks_cfg['minimum_spacing']:
            continue
        positions.append(location)
        intensities.append(intensity)
        support.append(snr)
        if len(positions) == peaks_cfg['max_peaks']:
            break
    return np.asarray(positions).reshape(-1, 2), np.asarray(intensities), np.asarray(support)


def nonlocal_average(raw, neighbors, center, neighbor_centers, peaks_cfg, experiment):
    """Small-neighborhood Poisson-normalized similarity averaging inspired by NLPAR.

    This is an explicit, conservative ablation, not a reproduction of the paper.
    Transmitted/saturated pixels are excluded from similarity. Neighbors must be
    supplied from the same guarded partition. Values at invalid shifted pixels
    do not contribute to the average.
    """
    raw = np.asarray(raw, np.float32)
    yy, xx = np.indices(raw.shape)
    rr = np.hypot(yy-center[0], xx-center[1])
    mask = (rr >= peaks_cfg['exclusion_radius']) & (rr <= peaks_cfg['max_radius'])
    raw_valid = raw < peaks_cfg['invalid_count']
    mask &= raw_valid
    total, weight = raw*raw_valid, raw_valid.astype(float)
    used = 0
    for neighbor, other_center in zip(neighbors, neighbor_centers):
        delta = np.asarray(center)-other_center
        valid = ndimage.shift((np.asarray(neighbor) < peaks_cfg['invalid_count']).astype(float), delta,
                             order=0, mode='constant', cval=0) > .5
        aligned = ndimage.shift(np.asarray(neighbor, np.float32), delta, order=1, mode='constant', cval=0)
        overlap = mask & valid
        if overlap.sum() < 100:
            continue
        distance = np.mean((raw[overlap]-aligned[overlap])**2/(raw[overlap]+aligned[overlap]+2))
        if distance > experiment['similarity_cutoff']:
            continue
        w = np.exp(-max(float(distance)-1, 0)/experiment['similarity_bandwidth']**2)
        total += w*aligned*valid
        weight += w*valid
        used += 1
    averaged = np.divide(total, weight, out=raw.astype(float).copy(), where=weight > 0)
    averaged[~raw_valid] = peaks_cfg['invalid_count']
    return averaged, used
