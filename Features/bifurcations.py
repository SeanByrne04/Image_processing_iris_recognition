import cv2
import numpy as np

from skeleton import skeletonise, prune_spurs, neighbour_count


def find_junctions(sk, ring=2):
    """Find junctions on a skeleton and count the branches leaving each one.

    1. Candidates: skeleton pixels with 3+ skeleton neighbours. At a real
       junction these form a small clump of touching pixels (thinning rarely
       leaves a single perfect centre pixel), so each clump = one junction.
    2. Branches: look at the skeleton in a thin ring `ring` px outside the
       clump. Each separate piece of skeleton crossing that ring is one
       branch. This is the crossing-number idea applied to a ring around the
       whole clump instead of the 8 pixels around one pixel, which fails when
       the centre is a clump (an X then scores 2 instead of 4).

    Returns [(x, y, n_branches), ...]; 3 = bifurcation, 4+ = crossover,
    2 or less = just a kink in a vessel (ignored by the caller).
    """
    cand = ((sk > 0) & (neighbour_count(sk) >= 3)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    h, w = sk.shape
    pad = ring + 2
    k_in = np.ones((2 * ring - 1, 2 * ring - 1), np.uint8)
    k_out = np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)
    out = []
    for i in range(1, n):
        x, y, bw, bh = stats[i, :4]
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
        blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        band = cv2.dilate(blob, k_out) - cv2.dilate(blob, k_in)
        hits = (band * sk[y0:y1, x0:x1]).astype(np.uint8)
        n_branches = cv2.connectedComponents(hits, connectivity=8)[0] - 1
        out.append((cents[i][0], cents[i][1], n_branches))
    return out


def _merge_close_bifurcations(bifs, merge_dist):
    """A crossover usually thins into two bifurcations joined by a short
    segment (the vessels cross at an angle, not at one pixel). Pair up the
    closest bifurcations that are under merge_dist apart and replace each
    pair with one crossover at their midpoint."""
    bifs = np.asarray(bifs, dtype=float).reshape(-1, 2)
    if len(bifs) < 2:
        return bifs, np.empty((0, 2))
    d = np.linalg.norm(bifs[:, None] - bifs[None], axis=2)
    np.fill_diagonal(d, np.inf)
    used, crosses = set(), []
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > merge_dist:
            break
        if i in used or j in used:
            continue
        used.update((i, j))
        crosses.append((bifs[i] + bifs[j]) / 2)
    keep = [k for k in range(len(bifs)) if k not in used]
    return bifs[keep], np.array(crosses).reshape(-1, 2)


def _inside(points, mask, border):
    """Drop points within `border` px of the fundus edge, where vessels are
    cut off by the field of view and give fake ends/junctions."""
    if mask is None or len(points) == 0:
        return points
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if border > 0:
        m = cv2.erode(m, np.ones((2 * border + 1, 2 * border + 1), np.uint8))
    xs = np.clip(points[:, 0].round().astype(int), 0, m.shape[1] - 1)
    ys = np.clip(points[:, 1].round().astype(int), 0, m.shape[0] - 1)
    return points[m[ys, xs] > 0]


def find_points(vessel_map, mask=None, spur_len=10, merge_dist=15, border=10,
                thinning="zhangsuen"):
    """Vessel map -> skeleton -> bifurcation and crossover points.

    vessel_map : binary vessel segmentation (Person A's output, or a DRIVE
                 hand-marked map). Vessels white, background black.
    mask       : optional field-of-view mask (white inside the fundus).
    spur_len   : side branches shorter than this (px) are removed.
    merge_dist : two bifurcations closer than this (px) become one crossover.
    border     : points this close (px) to the mask edge are ignored.

    Returns a dict with
      "skeleton"     : uint8 0/1 skeleton after pruning
      "bifurcations" : (N, 2) float array of (x, y)
      "crossovers"   : (M, 2) float array of (x, y)
    """
    sk = skeletonise(vessel_map, method=thinning)
    sk = prune_spurs(sk, max_len=spur_len)
    if mask is not None:
        sk = sk * (np.asarray(mask) > 0).astype(np.uint8)

    pts = find_junctions(sk)
    bifs = [(x, y) for x, y, b in pts if b == 3]
    direct = [(x, y) for x, y, b in pts if b >= 4]   # crossing at one spot

    bifs, merged = _merge_close_bifurcations(bifs, merge_dist)
    crosses = np.vstack([np.array(direct).reshape(-1, 2), merged])

    return {"skeleton": sk,
            "bifurcations": _inside(bifs, mask, border),
            "crossovers": _inside(crosses, mask, border)}


def draw_points(background, result, radius=4):
    """Colour image of the skeleton with bifurcations (green circles) and
    crossovers (red squares). `background` can be the fundus image or the
    vessel map; the skeleton is drawn on top in white."""
    bg = np.asarray(background)
    if bg.ndim == 2:
        bg = cv2.cvtColor((bg > 0).astype(np.uint8) * 80, cv2.COLOR_GRAY2BGR)
    out = bg.copy()
    out[result["skeleton"] > 0] = (255, 255, 255)
    for x, y in result["bifurcations"]:
        cv2.circle(out, (int(round(x)), int(round(y))), radius, (0, 255, 0), 1)
    for x, y in result["crossovers"]:
        c = (int(round(x)), int(round(y)))
        cv2.rectangle(out, (c[0] - radius, c[1] - radius),
                      (c[0] + radius, c[1] + radius), (0, 0, 255), 1)
    return out


def match_point_sets(a, b, tol=6):
    """Greedy one-to-one pairing of two point sets: a point in `a` counts as
    found if an unused point in `b` lies within `tol` px.
    Returns (n_matched, precision, recall) treating `b` as the reference."""
    a, b = np.asarray(a).reshape(-1, 2), np.asarray(b).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return 0, 0.0, 0.0
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    used_a, used_b = set(), set()
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > tol:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    m = len(used_a)
    return m, m / len(a), m / len(b)
