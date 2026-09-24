"""Separate permissive peak detection from reproducible indexing selection."""
from __future__ import annotations

import numpy as np


def select_observations(peaks, config):
    """Remove weak local maxima before geometric scoring, preserving detector coordinates.

    Intensities are background-subtracted centroid-patch sums from extraction.
    A per-pattern relative cutoff prevents low contrast maxima near a diffuse
    central halo from overwhelming the geometrical precision/recall score.
    No phase or template information enters this selection.
    """
    positions = np.asarray(peaks['peak_yx'])
    intensity = np.asarray(peaks['intensity'])
    counts = np.asarray(peaks['count'])
    valid = np.arange(positions.shape[-2]) < counts[..., None]
    maximum = np.max(np.where(valid, intensity, 0), axis=-1, keepdims=True)
    threshold = np.maximum(config['minimum_peak_intensity'], maximum*config['minimum_peak_relative_intensity'])
    keep = valid & (intensity >= threshold)
    # Brightness order also determines the anchor set used for pose search.
    order = np.argsort(np.where(keep, -intensity, np.inf), axis=-1, kind='stable')
    selected = dict(peaks)
    selected['peak_yx'] = np.take_along_axis(positions, order[...,None], axis=-2).copy()
    selected['intensity'] = np.take_along_axis(intensity, order, axis=-1).copy()
    selected['count'] = keep.sum(axis=-1).astype(np.int16)
    padding = np.arange(positions.shape[-2]) >= selected['count'][...,None]
    selected['peak_yx'][padding] = 0
    selected['intensity'][padding] = 0
    selected['selected_peak_indices'] = np.where(padding, -1, order).astype(np.int16)
    return selected
