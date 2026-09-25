"""Shared deterministic geometry for the Fe BCC/FCC synthetic benchmark.

Coordinate contract (fixed in benchmarks/fe_fcc_bcc/config.yaml):
    q_x = (col - c_x) * dq
    q_y = s_y * (row - c_y) * dq      with s_y = +1
    dq = 0.020 A^-1/px, center (c_y, c_x) = (127.5, 127.5), detector 256x256.

Crystal->lab convention (verified against the pipeline template libraries,
outputs/phase_identification_fe/templates/*.npz, 2026-09-25):
    R0(zone) is the minimal-twist rotation mapping the crystal zone axis to
    the beam direction (R0 @ zone_unit = z_hat).  A spot with reciprocal-lattice
    coordinates g appears at qxy = (R0 @ g)[:2].  The in-plane angle theta is a
    CCW rotation of the crystal about the beam:
        R(theta) = Rz(theta) @ R0.
    Reference spots computed this way reproduce the pipeline template geometry
    for zones [001], [011], [111] of both phases with residual < 1e-6 A^-1.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_z2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def zone_to_rotation(zone):
    """Minimal-twist rotation with R @ zone_unit = z_hat (crystal -> lab)."""
    v = np.asarray(zone, dtype=float)
    v = v / np.linalg.norm(v)
    ez = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(v, ez)) > 1.0 - 1e-9:
        return np.eye(3)
    a = np.cross(v, ez)
    a = a / np.linalg.norm(a)
    phi = np.arccos(np.clip(np.dot(v, ez), -1.0, 1.0))
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    R = np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)
    if not np.allclose(R @ v, ez, atol=1e-9):
        raise ValueError(f'zone_to_rotation failed for {zone}: R @ zone = {R @ v}')
    return R


def pattern_rotation(zone, in_plane_deg):
    """Crystal -> lab rotation for a zone axis and CCW in-plane crystal rotation."""
    theta = np.deg2rad(float(in_plane_deg))
    return rotation_z(theta) @ zone_to_rotation(zone)


def reference_spots(cif_path, zone, q_min, q_max, extinction_floor=1e-12):
    """Independent kinematic ZOLZ spot list for one (CIF, zone) orientation.

    Uses pymatgen TEMCalculator (Mott-Bethe electron scattering factors plus
    the relativistic wavelength), NOT the py4DSTEM Crystal engine used by the
    pipeline.  q is computed from hkl through the crystallographic reciprocal
    lattice (no 2*pi), and the beam_direction selects the ZOLZ.  Returns spots
    sorted by descending intensity_norm, each {'hkl', 'qxy', 'intensity_norm',
    'd_hkl'} with intensity renormalized so the strongest in-window spot is 1.
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.tem import TEMCalculator

    structure = Structure.from_file(cif_path)
    calculator = TEMCalculator(voltage=300, beam_direction=tuple(int(v) for v in zone),
                               camera_length=160, debye_waller_factors={})
    pattern = calculator.get_pattern(structure)
    rec = structure.lattice.reciprocal_lattice_crystallographic.matrix
    R = zone_to_rotation(zone)
    seen = {}
    for hkl, raw in zip(pattern['(hkl)'], pattern['Intensity (norm)']):
        if float(raw) < extinction_floor:
            continue
        hkl = tuple(int(v) for v in hkl)
        g = np.asarray(hkl, dtype=float) @ rec
        qxy = (R @ g)[:2]
        q = float(np.linalg.norm(qxy))
        if q < q_min or q > q_max:
            continue
        # TEMCalculator may repeat a reflection across its grid; keep the strongest.
        if hkl in seen and seen[hkl][2] >= float(raw):
            continue
        d_hkl = float(np.linalg.norm(g))
        seen[hkl] = (hkl, qxy, float(raw), d_hkl)
    spots = [{'hkl': hkl, 'qxy': qxy, 'intensity_norm': raw, 'd_hkl': d_hkl}
             for hkl, qxy, raw, d_hkl in seen.values()]
    if not spots:
        raise ValueError(f'No reference spots for {Path(cif_path).name} zone {zone}')
    maximum = max(s['intensity_norm'] for s in spots)
    for spot in spots:
        spot['intensity_norm'] /= maximum
    spots.sort(key=lambda s: (-s['intensity_norm'], np.linalg.norm(s['qxy'])))
    return spots


def rasterize(shape, spots_px, sigma_px):
    """Sum of fixed 2D Gaussians; spots_px is [(y, x, amplitude), ...]."""
    ys, xs = np.indices(shape)
    image = np.zeros(shape, dtype=np.float64)
    for y0, x0, amplitude in spots_px:
        image += amplitude * np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2.0 * sigma_px**2))
    return image


def mib_header_template(source_dir: Path):
    """First 384 header bytes of the first real .mib file; only the sequence is rewritten."""
    paths = sorted(Path(source_dir).glob('*.mib'))
    if not paths:
        raise ValueError(f'No .mib header source in {source_dir}')
    header = paths[0].read_bytes()[:384]
    if len(header) != 384 or not header.startswith(b'MQ1,'):
        raise ValueError(f'Unexpected MIB header template: {paths[0].name}')
    return bytearray(header)


def write_mib(path, cube, header_template: bytearray):
    """Frames of 384-byte header + 256x256 big-endian uint16, row-major scan order."""
    header = bytearray(header_template)
    detector = cube.shape[-2:]
    with open(path, 'wb') as stream:
        for sequence, frame in enumerate(cube.reshape(-1, *detector), 1):
            header[4:10] = f'{sequence:06d}'.encode()
            stream.write(header)
            stream.write(np.ascontiguousarray(frame, dtype='>u2').tobytes())


def mib_payload_sha256(path: Path, frames: int, header_bytes=384, detector=(256, 256)):
    """Hash of the concatenated frame payloads (headers excluded)."""
    digest = hashlib.sha256()
    stride = header_bytes + detector[0] * detector[1] * 2
    with open(path, 'rb') as stream:
        for _ in range(frames):
            stream.seek(header_bytes, 1)
            digest.update(stream.read(stride - header_bytes))
    return digest.hexdigest()


def h5_payload_sha256(path: Path):
    import h5py
    with h5py.File(path, 'r') as handle:
        return sha256_bytes(np.ascontiguousarray(handle['/datacube/data']).tobytes())


def cubic_rotations():
    """The 24 proper rotations of the cubic point group 432 (used for misorientation)."""
    operations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            matrix = np.zeros((3, 3))
            for i, p in enumerate(perm):
                matrix[i, p] = signs[i]
            if np.linalg.det(matrix) > 0:
                operations.append(matrix.astype(np.float64))
    return np.array(operations)


def rotation_angle_deg(matrix):
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))))
