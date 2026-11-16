"""Baseline matcher for vessel point templates.

Two photos of the same eye are never lined up exactly: the head moves and the
camera tilts, so the whole point pattern is shifted and slightly rotated.
The matcher first finds the rotation + shift that lines the two point sets up
best, then counts how many points have a partner within `tol` px. The pairing
step is Person B's `match_point_sets` (Features/bifurcations.py); this file
adds the alignment in front of it, which that function does not do.

Alignment uses Hough-style voting (the standard trick from fingerprint
minutiae matching): for each trial rotation, every pair of points (one from
each set) votes for the shift that would put one on top of the other. The
shift with the most votes wins, because points from the same eye all agree
on it while random pairs scatter their votes.
"""
import sys
from pathlib import Path

import numpy as np

_FEATURES = str(Path(__file__).resolve().parent.parent / "Features")
if _FEATURES not in sys.path:
    sys.path.insert(0, _FEATURES)
from bifurcations import match_point_sets    # noqa: E402  Person B's pairing


def _pairs(a, b, tol):
    """Number of one-to-one pairs within tol px, using Person B's
    match_point_sets (greedy closest-first pairing)."""
    return match_point_sets(a, b, tol=tol)[0]




def align(a, b, angles=np.arange(-12, 12.5, 1.0), bin_px=8, max_shift=350):
    """Find the rotation (deg, about a's centroid) and shift that best map
    point set a onto point set b. Returns (angle, shift, a_aligned).
    max_shift is large (350 px of a 1504 px image) because some RIDB photos
    are centred on the optic disc and others on the macula (e.g. person 7)."""
    a = np.asarray(a, float).reshape(-1, 2)
    b = np.asarray(b, float).reshape(-1, 2)
    c = a.mean(axis=0)
    edges = np.arange(-max_shift, max_shift + bin_px, bin_px)
    best = (-1, 0.0, np.zeros(2))
    for ang in angles:
        t = np.deg2rad(ang)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        ar = (a - c) @ R.T + c
        diff = (b[None] - ar[:, None]).reshape(-1, 2)      # every candidate shift
        h, _, _ = np.histogram2d(diff[:, 0], diff[:, 1], bins=[edges, edges])
        # Sum each 2x2 block of bins so a peak on a bin border is not split
        h2 = h[:-1, :-1] + h[1:, :-1] + h[:-1, 1:] + h[1:, 1:]
        i, j = np.unravel_index(np.argmax(h2), h2.shape)  # block = bins i..i+1
        if h2[i, j] > best[0]:
            sel = ((diff[:, 0] >= edges[i]) & (diff[:, 0] < edges[i + 2]) &
                   (diff[:, 1] >= edges[j]) & (diff[:, 1] < edges[j + 2]))
            best = (h2[i, j], ang, np.median(diff[sel], axis=0))
    _, ang, shift = best
    t = np.deg2rad(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return ang, shift, (a - c) @ R.T + c + shift


def point_match(a, b, tol=10):
    """Similarity score in [0, 1] between two point templates.
    score = matched pairs / sqrt(|a| * |b|) after alignment."""
    a = np.asarray(a, float).reshape(-1, 2)
    b = np.asarray(b, float).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    _, _, a_al = align(a, b)
    return _pairs(a_al, b, tol) / np.sqrt(len(a) * len(b))


def no_align_match(a, b, tol=10):
    """Same score without the alignment step (ablation)."""
    a = np.asarray(a, float).reshape(-1, 2)
    b = np.asarray(b, float).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return _pairs(a, b, tol) / np.sqrt(len(a) * len(b))
