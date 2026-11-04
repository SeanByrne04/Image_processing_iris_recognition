import pickle
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = ROOT / "Templates" / "templates.pkl"


def dummy_features(img):
    """Placeholder until Person B's feature extraction is ready."""
    return cv2.resize(img[:, :, 1], (64, 64)).flatten()


def enrol_all(enrol_imgs, extract_features=dummy_features, path=TEMPLATE_PATH):
    templates = {p: extract_features(img) for p, img in enrol_imgs.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(templates, f)
    return templates


def load_templates(path=TEMPLATE_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)