from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DRIVE_DIR = ROOT / "Data" / "DRIVE"


def _gif(path):
    # cv2.imread cannot read GIFs, and DRIVE stores its maps and masks as GIF
    return np.array(Image.open(path).convert("L"))


def load_drive(idx, folder=DRIVE_DIR, observer=1):
    """Load one DRIVE image by number (1-20 = test, 21-40 = training).

    Returns (fundus_rgb, vessel_map, fov_mask). observer=2 gives the second
    human's hand-marked map, which only exists for the test set.
    """
    split = "test" if idx <= 20 else "training"
    d = Path(folder) / split
    img = np.array(Image.open(d / "images" / f"{idx:02d}_{split}.tif").convert("RGB"))
    vessels = _gif(d / f"{'1st' if observer == 1 else '2nd'}_manual"
                   / f"{idx:02d}_manual{observer}.gif")
    mask = _gif(d / "mask" / f"{idx:02d}_{split}_mask.gif")
    return img, vessels, mask
