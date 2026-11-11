import cv2
import numpy as np

# (dy, dx) offsets of the 8 neighbours of a pixel, clockwise from top-left.
RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def binarise(vessel_map, thresh=127):
    """Any vessel map (0/1, 0/255, bool) -> uint8 0/255."""
    v = np.asarray(vessel_map)
    if v.ndim == 3:
        v = v[:, :, 0]
    if v.dtype == bool or v.max() <= 1:
        v = v.astype(np.uint8) * 255
    return np.where(v > thresh, 255, 0).astype(np.uint8)


def skeletonise(vessel_map, method="zhangsuen", min_area=30):
    """Thin a binary vessel map to 1-pixel-wide centre lines.

    Small blobs (noise, < min_area pixels) are removed first, because each
    one would otherwise become a tiny skeleton with fake end points.
    Returns uint8 0/1.
    """
    v = binarise(vessel_map)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(v, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            v[labels == i] = 0

    kind = {"zhangsuen": cv2.ximgproc.THINNING_ZHANGSUEN,
            "guohall": cv2.ximgproc.THINNING_GUOHALL}[method]
    sk = cv2.ximgproc.thinning(v, thinningType=kind)
    return remove_staircase((sk > 0).astype(np.uint8))


# Corner of an "L" step: P with two perpendicular 4-neighbours (here N and E)
# and nothing on the opposite side. 1 = vessel, -1 = background, 0 = any.
_STAIR = np.array([[0, 1, 0],
                   [-1, 1, 1],
                   [-1, -1, 0]], dtype=np.int32)


def remove_staircase(sk):
    """Make the skeleton strictly 8-connected (1 pixel thick everywhere).

    Zhang-Suen leaves "staircase" steps on diagonal vessels, so a pixel can
    touch the next one both directly and through a corner. At junctions this
    breaks the crossing number (an X shape scores 2 instead of 4). Deleting
    the corner pixel of each step keeps the line connected because its two
    neighbours already touch diagonally.
    """
    sk = sk.copy()
    for k in range(4):
        kernel = np.rot90(_STAIR, k).copy()
        hits = cv2.morphologyEx(sk, cv2.MORPH_HITMISS, kernel)
        sk[hits > 0] = 0
    return sk


def neighbour_stack(sk):
    """8 shifted copies of the skeleton, one per neighbour in RING order.
    Pixels past the image border count as background."""
    p = np.pad(sk, 1)
    h, w = sk.shape
    return np.stack([p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in RING])


def neighbour_count(sk):
    return neighbour_stack(sk).sum(axis=0) * sk


def prune_spurs(sk, max_len=10, iterations=2):
    """Remove short side branches ("spurs") left by thinning.

    Thinning turns small bumps on a vessel edge into short branches, and each
    one creates a fake bifurcation. Walk in from every end point; if a branch
    reaches a junction within max_len pixels, delete it. Branches that never
    reach a junction (free-standing vessel pieces) are kept.
    """
    sk = sk.copy()
    h, w = sk.shape
    for _ in range(iterations):
        nc = neighbour_count(sk)
        ends = np.argwhere((sk == 1) & (nc == 1))
        removed = False
        for y, x in ends:
            path, prev, cur = [], None, (y, x)
            while len(path) <= max_len:
                path.append(cur)
                nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in RING
                       if 0 <= cur[0] + dy < h and 0 <= cur[1] + dx < w
                       and sk[cur[0] + dy, cur[1] + dx]
                       and (cur[0] + dy, cur[1] + dx) != prev
                       and (cur[0] + dy, cur[1] + dx) not in path]
                if len(nxt) != 1:          # 0 = isolated piece, >1 = junction
                    break
                prev, cur = cur, nxt[0]
            if len(nxt) > 1 and len(path) <= max_len:
                for py, px in path:
                    sk[py, px] = 0
                removed = True
        if not removed:
            break
        # Deleting a spur can leave a 2-pixel corner at the old junction
        sk = remove_staircase((cv2.ximgproc.thinning(sk * 255) > 0).astype(np.uint8))
    return sk
