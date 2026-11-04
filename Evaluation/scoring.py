import numpy as np


def dummy_match(a, b):
    """Normalised correlation (higher = more similar).
    Placeholder until Person B's matcher is ready."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float(np.mean(a * b))


def score_all(templates, probes, impostors, extract_features, match=dummy_match):
    """Compare every probe/impostor image with every enrolled template.

    genuine     : probe vs its own person's template
    zero_effort : probe vs another enrolled person's template
    unknown     : never-enrolled person vs any template
    """
    genuine, zero_effort, unknown = [], [], []
    for p, img in probes:
        f = extract_features(img)
        for q, t in templates.items():
            (genuine if q == p else zero_effort).append(match(f, t))
    for _, img in impostors:
        f = extract_features(img)
        unknown += [match(f, t) for t in templates.values()]
    return np.array(genuine), np.array(zero_effort), np.array(unknown)