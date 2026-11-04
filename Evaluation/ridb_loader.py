import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RIDB_DIR = ROOT / "Data" / "RIDB"


def load_ridb(folder=RIDB_DIR):
    """Return {person_id: [(image_no, image), ...]} sorted by image number.
    Filename IM00000X_Y: X = image number, Y = person."""
    data = defaultdict(list)
    for fname in sorted(os.listdir(folder)):
        nums = re.findall(r"\d+", fname)
        if len(nums) < 2:
            continue
        image_no, person = int(nums[0]), int(nums[1])
        img = cv2.imread(str(Path(folder) / fname))
        if img is None:
            print("Could not read", fname)
            continue
        data[person].append((image_no, img))
    for p in data:
        data[p].sort(key=lambda t: t[0])
    return dict(data)


def disc_side(img):
    """Brightest point of a heavily blurred green channel ~ optic disc."""
    g = cv2.GaussianBlur(img[:, :, 1], (0, 0), 25)
    _, _, _, (x, _) = cv2.minMaxLoc(g)
    return "left" if x < img.shape[1] / 2 else "right"


def check_same_eye(ridb, show=True):
    """Print optic disc side for each person's images; flag mixed sets."""
    flagged = []
    for p, imgs in sorted(ridb.items()):
        sides = [disc_side(im) for _, im in imgs]
        mixed = len(set(sides)) > 1
        if mixed:
            flagged.append(p)
        print(f"Person {p:2d}: {sides}{'  <-- CHECK' if mixed else ''}")

        if show:
            fig, axes = plt.subplots(1, len(imgs), figsize=(15, 3))
            for ax, (n, im) in zip(axes, imgs):
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                ax.set_title(f"img {n}")
                ax.axis("off")
            fig.suptitle(f"Person {p}")
            plt.show()
    return flagged