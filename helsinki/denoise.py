"""
Frozen, pretrained cnFFDNet denoiser used as a preprocessing step for
cleaning blurred targets before PSF forward-model fitting.

Rationale (see the noise term in the forward model discussed in
`.claude/skills/helsinki-forward-model/SKILL.md`:
`blurred = a*conv(warp(sharp), h) + b + noise`): real sensor noise isn't
representable by a smooth convolution kernel, so asking a blur fit to also
explain it wastes fit precision on something structurally outside the
model's range. This module denoises once, per full image, before patch
extraction, using a FROZEN checkpoint -- deliberately not jointly optimized
with the kernel, since adding more free parameters to an already
under-constrained fit (see the coma/astigmatism non-identifiability found
when fitting ZernikePSF -- coefficients that shift substantially between
random scene subsets) would likely make identifiability worse, not better.
"""

import sys
from pathlib import Path

import torch

_CNFFDNET_CANDIDATE_SUBPATHS = ["cnffdnet"]  # sibling to this repo, on every machine seen so far
_DEFAULT_CHECKPOINT = "cnffd_sca2_bs128_p66_lr3_depth15.pth"


def find_cnffdnet_root(cnffdnet_root=None, repo_root=None):
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    if cnffdnet_root is not None:
        candidates.append(Path(cnffdnet_root))
    else:
        candidates.extend(Path(repo_root).parent / sub for sub in _CNFFDNET_CANDIDATE_SUBPATHS)
    for cand in candidates:
        if (cand / "models.py").is_file() and (cand / "checkpoints").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not find the cnffdnet repo (a directory with models.py + checkpoints/). "
        f"Tried: {[str(c) for c in candidates]}. Pass --cnffdnet-root explicitly."
    )


def load_cnffdnet(cnffdnet_root=None, checkpoint=None, repo_root=None, device=None):
    root = find_cnffdnet_root(cnffdnet_root, repo_root=repo_root)
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    from cnffdnet import models  # deferred: only needed/importable once root is resolved

    ckpt_path = Path(checkpoint) if checkpoint is not None else root / "checkpoints" / _DEFAULT_CHECKPOINT
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"cnFFDNet checkpoint not found at {ckpt_path}")

    model = models.conicFFDNet(
        depth=15, image_channels=1, n_channels=64, sn=True, lip=1.0, sca=2, patch_shape=(66, 66)
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model


_LAPLACIAN_MASK = torch.tensor([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]]).view(1, 1, 3, 3)


def estimate_noise_sigma(image):
    """Fast noise-std estimate (Immerkaer 1996) via a Laplacian-like mask that
    nulls out linear gradients, isolating high-frequency noise even in
    textured content. Needed because this dataset has no literally-flat
    regions near the positions we fit at (checked directly: at the frame
    center, the lowest patch variance found was ~0.006 -- three orders of
    magnitude above a "near-flat" threshold like 1e-4 -- so a flat-patch
    search would silently never find anything and always fall back to a
    constant guess). `image`: (H, W) tensor in [0, 1].
    """
    from math import pi

    x = image.unsqueeze(0).unsqueeze(0)
    resp = torch.nn.functional.conv2d(x, _LAPLACIAN_MASK.to(dtype=x.dtype, device=x.device))
    h, w = image.shape
    return float((pi / 2) ** 0.5 / (6 * (w - 2) * (h - 2)) * resp.abs().sum())


@torch.no_grad()
def denoise_image(model, image, noise_level, device=None):
    """image: (H, W) tensor in [0, 1]. Returns a denoised (H, W) tensor on the input's original device."""
    original_device = image.device
    x = image.unsqueeze(0).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    nl = torch.tensor([noise_level], dtype=x.dtype, device=x.device)
    out = model(x, nl)
    return out.squeeze(0).squeeze(0).clamp(0, 1).to(original_device)
