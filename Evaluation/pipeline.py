"""Joins the three parts of the system into one feature extractor:
fundus image -> vessel map (Person A) -> bifurcations/crossovers (Person B)
-> point template (N x 2 array of x, y).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for d in ("Features",):
    if str(ROOT / d) not in sys.path:
        sys.path.insert(0, str(ROOT / d))

from segmentation_copy import segment_vessels  # noqa: E402  Person A (copy)
from bifurcations import find_points          # noqa: E402  Person B


def vessel_points(img_bgr, return_all=False):
    """Run segmentation + point detection on one BGR fundus image."""
    vessels, steps = segment_vessels(img_bgr, return_steps=True)
    mask = steps["2 retina mask"]
    r = find_points(vessels, mask)
    pts = np.vstack([r["bifurcations"], r["crossovers"]]).reshape(-1, 2)
    if return_all:
        return pts, vessels, mask, r
    return pts


class CachedExtractor:
    """Segmentation takes ~0.5 s per image, so remember results by image id
    (the same image is used several times during scoring)."""

    def __init__(self, fn=vessel_points):
        self.fn, self.cache = fn, {}

    def __call__(self, img):
        key = id(img)
        if key not in self.cache:
            self.cache[key] = self.fn(img)
        return self.cache[key]
