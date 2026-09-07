"""
Scene discovery, image loading, and patch extraction for the Helsinki
(HDC2021-derived) defocus dataset: paired CAM01 (sharp) / CAM02 (blurred)
captures at several focus steps.

Layout expected under a data root (see `find_data_root`):
    <root>/CAM01_focused/focusStep_<N>_<scene_id>.tif
    <root>/CAM02_blurred/focusStep_<N>_<scene_id>.tif

Scenes fall into three categories:
    natural  -- focusStep_<N>_Image_*.tif           (15 scenes)
    text     -- focusStep_<N>_(timesR|verdanaRef)_size_30_sample_*.tif (40 scenes)
    qr       -- focusStep_<N>_QRcode.tif             (1 scene)
`LSF_X`, `LSF_Y`, `PSF` are calibration captures, not scene content, and are
never returned by `discover_scenes`.
"""

import re
from pathlib import Path

import numpy as np
import tifffile
import torch

# Candidate relative locations of the dataset content folder (the one that
# directly contains CAM01_focused/ and CAM02_blurred/), tried in order.
# The dataset ships as a zip whose top-level folder duplicates its own name,
# so a naive unzip leaves it double-nested; a plain rsync of its contents
# does not. Both layouts show up across machines this project runs on.
_CANDIDATE_SUBPATHS = [
    "steps_0_to_4/steps_0_to_4",
    "steps_0_to_4",
]

_NATURAL_RE = "(?P<id>Image_.+)"
_TEXT_RE = "(?P<id>(?:timesR|verdanaRef)_size_30_sample_\\d+)"
_QR_RE = "(?P<id>QRcode)"


def find_data_root(data_root=None, repo_root=None):
    """Resolve the folder that directly contains CAM01_focused/CAM02_blurred.

    Tries `data_root` first if given, then a couple of known layouts under
    `<repo_root>/data` (default: this file's repo). Raises FileNotFoundError
    with a clear message if none match, rather than silently picking wrong.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    candidates = []
    if data_root is not None:
        candidates.append(Path(data_root))
    else:
        base = Path(repo_root) / "data"
        candidates.extend(base / sub for sub in _CANDIDATE_SUBPATHS)

    for cand in candidates:
        if (cand / "CAM01_focused").is_dir() and (cand / "CAM02_blurred").is_dir():
            return cand

    raise FileNotFoundError(
        "Could not find a data root containing CAM01_focused/ and "
        f"CAM02_blurred/. Tried: {[str(c) for c in candidates]}. "
        "Pass --data-root explicitly."
    )


def discover_scenes(data_root, step):
    """Return {'natural': [...], 'text': [...], 'qr': [...]} scene ids for one focus step."""
    cam1_dir = Path(data_root) / "CAM01_focused"
    patterns = {
        "natural": re.compile(rf"^focusStep_{step}_{_NATURAL_RE}\.tif$"),
        "text": re.compile(rf"^focusStep_{step}_{_TEXT_RE}\.tif$"),
        "qr": re.compile(rf"^focusStep_{step}_{_QR_RE}\.tif$"),
    }
    scenes = {k: [] for k in patterns}
    for f in sorted(cam1_dir.glob(f"focusStep_{step}_*.tif")):
        for category, pat in patterns.items():
            m = pat.match(f.name)
            if m:
                scenes[category].append(m.group("id"))
                break
    return scenes


def load_pair(data_root, step, scene_id, dtype=torch.float32):
    """Load one (sharp, blurred) image pair, each independently min-max normalized to [0, 1]."""
    data_root = Path(data_root)
    fname = f"focusStep_{step}_{scene_id}.tif"
    sharp = tifffile.imread(str(data_root / "CAM01_focused" / fname)).astype(np.float32)
    blurred = tifffile.imread(str(data_root / "CAM02_blurred" / fname)).astype(np.float32)

    def normalize(img):
        img = torch.from_numpy(img).to(dtype)
        return (img - img.min()) / (img.max() - img.min())

    return normalize(sharp), normalize(blurred)


def extract_patch_pairs(sharp, blurred, center, neighborhood, patch, kernel_size, stride, min_var=0.0):
    """Densely tile overlapping patch pairs in a square neighborhood around `center`.

    For each tile, the sharp crop is padded by `kernel_size // 2` on every side
    relative to the blurred crop, so that a 'valid'-mode convolution of the
    sharp crop with a `kernel_size`-sized kernel lands exactly on the blurred
    crop's footprint -- no circular-boundary artifacts, no cropping needed
    downstream.

    Returns a list of (sharp_crop, blurred_crop) tensor pairs. Patches whose
    blurred crop has variance below `min_var` (near-flat content -- almost no
    information about kernel shape) are dropped.
    """
    cy, cx = center
    pad = kernel_size // 2
    half_nb = neighborhood // 2

    tops = range(cy - half_nb, cy + half_nb - patch + 1, stride)
    lefts = range(cx - half_nb, cx + half_nb - patch + 1, stride)

    pairs = []
    for top in tops:
        for left in lefts:
            blurred_crop = blurred[top : top + patch, left : left + patch]
            if blurred_crop.var().item() < min_var:
                continue
            sharp_crop = sharp[
                top - pad : top + patch + pad,
                left - pad : left + patch + pad,
            ]
            pairs.append((sharp_crop, blurred_crop))
    return pairs
