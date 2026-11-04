import numpy as np


def far_frr(genuine, impostor, thresholds):
    """Accept if score >= threshold (higher score = better match)."""
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    return far, frr


def eer(genuine, impostor, n=1000):
    """Return (eer, threshold, thresholds, far, frr)."""
    lo = min(np.min(genuine), np.min(impostor))
    hi = max(np.max(genuine), np.max(impostor))
    t = np.linspace(lo, hi, n)
    far, frr = far_frr(genuine, impostor, t)
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, t[i], t, far, frr