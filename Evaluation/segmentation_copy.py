"""COPY of Person A's (David Byrne) vessel segmentation.

Copied unchanged from Segmentation/C23308943_RetinaProject.ipynb, because a
notebook cannot be imported. The original notebook is the master version:
if it changes, re-copy its code cell into this file.
"""
import os
import sys

import cv2
import numpy as np


REFERENCE_DIAMETER = 540
BASE_SIGMAS = [1.5, 2.0, 3.0, 4.0]


FRANGI_BETA = 0.5
FRANGI_C_FRAC = 0.5


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not read image: " + path)
    return img


def get_green_channel(img_bgr):
    return img_bgr[:, :, 1]


def make_retina_mask(green):
    blurred = cv2.GaussianBlur(green, (0, 0), 3)

    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = max(5, 0.4 * otsu_value)
    _, rough = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(rough, connectivity=8)
    if count <= 1:
        return np.full(green.shape, 255, np.uint8)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    retina = np.where(labels == biggest, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(retina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(retina)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def retina_diameter(mask):
    area = cv2.countNonZero(mask)
    return 2.0 * np.sqrt(area / np.pi)


def shrink_mask(mask, diameter):
    size = max(3, int(0.03 * diameter) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(mask, kernel)


def fill_outside_retina(green, mask):
    filled = green.copy()
    average = int(cv2.mean(green, mask=mask)[0])
    filled[mask == 0] = average
    return filled


def enhance_contrast(green):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(green)


def reduce_noise(img, scale):
    return cv2.GaussianBlur(img, (0, 0), 1.0 * scale)


def frangi_filter(img, sigmas, beta=FRANGI_BETA, c_frac=FRANGI_C_FRAC):
    img = img.astype(np.float32)
    best = np.zeros(img.shape, np.float32)

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)

        ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3, scale=0.25)
        iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3, scale=0.25)
        ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3, scale=0.25)

        ixx *= sigma ** 2
        iyy *= sigma ** 2
        ixy *= sigma ** 2

        half_sum = (ixx + iyy) / 2.0
        root = np.sqrt(((ixx - iyy) / 2.0) ** 2 + ixy ** 2)
        e_a = half_sum + root
        e_b = half_sum - root

        a_is_smaller = np.abs(e_a) < np.abs(e_b)
        l1 = np.where(a_is_smaller, e_a, e_b)
        l2 = np.where(a_is_smaller, e_b, e_a)

        rb = np.abs(l1) / (np.abs(l2) + 1e-10)
        s = np.sqrt(l1 ** 2 + l2 ** 2)

        c = c_frac * np.percentile(s, 99)
        c = max(c, 1e-10)

        score = np.exp(-(rb ** 2) / (2 * beta ** 2)) * (1.0 - np.exp(-(s ** 2) / (2 * c ** 2)))
        score[l2 <= 0] = 0.0

        best = np.maximum(best, score.astype(np.float32))

    return best


def threshold_otsu(vesselness, mask):

    scaled = np.sqrt(vesselness / (vesselness.max() + 1e-10))
    scaled = (scaled * 255).astype(np.uint8)

    inside = scaled[mask > 0].reshape(-1, 1)
    cutoff, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = np.where(scaled > cutoff, 255, 0).astype(np.uint8)
    binary[mask == 0] = 0
    return binary


def clean_vessel_map(binary, diameter):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    min_area = int(40 * (diameter / REFERENCE_DIAMETER) ** 2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def segment_vessels(img_bgr, return_steps=False):
    green = get_green_channel(img_bgr)

    full_mask = make_retina_mask(green)
    diameter = retina_diameter(full_mask)
    mask = shrink_mask(full_mask, diameter)

    scale = diameter / REFERENCE_DIAMETER
    sigmas = [s * scale for s in BASE_SIGMAS]

    filled = fill_outside_retina(green, full_mask)
    enhanced = enhance_contrast(filled)
    smooth = reduce_noise(enhanced, scale)

    vesselness = frangi_filter(smooth, sigmas)
    vesselness[mask == 0] = 0

    binary = threshold_otsu(vesselness, mask)
    vessel_map = clean_vessel_map(binary, diameter)

    if return_steps:
        steps = {
            "1 green channel": green,
            "2 retina mask": mask,
            "3 CLAHE + blur": smooth,
            "4 Frangi score": (255 * np.sqrt(vesselness / (vesselness.max() + 1e-10))).astype(np.uint8),
            "5 Otsu threshold": binary,
            "6 final vessel map": vessel_map,
        }
        return vessel_map, steps
    return vessel_map


def evaluate(predicted, truth, fov_mask):
    inside = fov_mask > 0
    p = predicted[inside] > 0
    t = truth[inside] > 0
    tp = np.sum(p & t)
    tn = np.sum(~p & ~t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn + 1e-10),
        "specificity": tn / (tn + fp + 1e-10),
    }


def save_step_figure(img_bgr, steps, path, tile_width=400):
    tiles = [("original", img_bgr)] + list(steps.items())
    prepared = []
    for title, tile in tiles:
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        height = int(tile.shape[0] * tile_width / tile.shape[1])
        tile = cv2.resize(tile, (tile_width, height))
        cv2.putText(tile, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        prepared.append(tile)

    while len(prepared) % 4 != 0:
        prepared.append(np.zeros_like(prepared[0]))
    rows = [cv2.hconcat(prepared[i:i + 4]) for i in range(0, len(prepared), 4)]
    cv2.imwrite(path, cv2.vconcat(rows))
