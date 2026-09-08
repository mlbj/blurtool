"""
Fit a single small PSF for one fixed sensor position on the Helsinki defocus
dataset, from real (sharp, blurred) camera pairs -- no deconvolution, this is
a pure forward-model fit: find h such that conv(sharp, h) approx blurred.

Two PSF parametrizations are available via --model:
    gaussian (default) -- an anisotropic Gaussian with a free centroid offset
        (sigma_x, sigma_y, theta, dy, dx -- 5 parameters). Smooth and compact
        by construction, so it cannot produce noise-driven pixel artifacts;
        this was the reliable choice in earlier work on this exact dataset at
        this exact blur scale.
    freeform -- an unconstrained per-pixel array (softplus + sum-to-one +
        Hann taper). More expressive, but with real sensor noise and finite
        data it can park spurious weight at individual pixels. Kept for
        comparison against the gaussian model, not as the default.

Objective (per patch, averaged over a large pooled batch of patches from
every training scene at once):
    loss = 0.5 * mean((conv(sharp, h) - blurred)^2)      [pixel fidelity]
         + ssim_weight * (1 - SSIM(conv(sharp, h), blurred))  [structural fidelity]
         + regularizer(h)     [freeform only: a mild smoothness penalty]

Why pooling everything into one batch works with so few source images: see
the module docstring in `helsinki/data.py` -- one shared kernel constrained
by thousands of patches (dense tiles across many scenes) is a heavily
overdetermined problem, whereas fitting a kernel to any single patch alone
is not. Real sensor noise on the blurred images is additive and (mostly)
independent across patches, so it adds variance to the estimate but -- given
a low-dimensional model and/or enough pooled patches -- should not bias it
toward a specific spurious kernel shape; see --stability-check.

Usage:
    python -m helsinki.fit_psf_single_position
    python -m helsinki.fit_psf_single_position --model freeform --steps 800
    python -m helsinki.fit_psf_single_position --stability-check
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helsinki.data import discover_scenes, extract_patch_pairs, find_data_root, load_pair

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Differentiable SSIM (single-scale, Gaussian window). blurtool.ssim wraps
# skimage on detached numpy arrays -- fine for reporting, not usable as a
# training loss. This is the standard Wang et al. (2004) formulation.
# ---------------------------------------------------------------------------
def _gaussian_window(window_size, sigma, dtype, device):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window2d.unsqueeze(0).unsqueeze(0)  # (1, 1, W, W)


def ssim_map(pred, target, window, data_range=1.0):
    """pred, target: (N, 1, H, W). Returns the per-pixel SSIM map.

    Uses reflect padding (not zero padding) around each patch before the local
    windowed statistics: zero-padding would tell every patch border "there is
    nothing outside you", which is false and, since it applies identically to
    every pooled patch regardless of content, is a systematic bias rather than
    noise that averages out.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window.shape[-1] // 2

    def local_mean(x):
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), window)

    mu_p, mu_t = local_mean(pred), local_mean(target)
    mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

    sigma_p2 = local_mean(pred * pred) - mu_p2
    sigma_t2 = local_mean(target * target) - mu_t2
    sigma_pt = local_mean(pred * target) - mu_pt

    numerator = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    denominator = (mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2)
    return numerator / denominator


# ---------------------------------------------------------------------------
# PSF parametrizations
# ---------------------------------------------------------------------------
def hann_taper(kernel_size, dtype, device):
    """2D Hann window: 1 at the center, exactly 0 at the border."""
    n = torch.arange(kernel_size, dtype=dtype, device=device)
    w1d = 0.5 * (1 - torch.cos(2 * torch.pi * n / (kernel_size - 1)))
    return w1d.unsqueeze(1) * w1d.unsqueeze(0)


def kernel_smoothness(h):
    """Sum of squared finite differences -- a mild total-variation-style penalty."""
    dh = h[1:, :] - h[:-1, :]
    dw = h[:, 1:] - h[:, :-1]
    return (dh**2).sum() + (dw**2).sum()


class FreeformPSF(nn.Module):
    """Unconstrained per-pixel kernel: softplus (non-negative) * Hann taper, sum-to-one.

    Expressive, but with only a few thousand noisy real patches it can fit
    spurious structure at individual pixels -- see the module docstring.
    """

    def __init__(self, kernel_size, taper=True, smooth_weight=1e-3, seed=0, device=None, dtype=torch.float32):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        init = 0.01 * torch.randn(kernel_size, kernel_size, generator=gen)
        self.raw = nn.Parameter(init.to(device=device, dtype=dtype))
        self.taper = hann_taper(kernel_size, dtype, device) if taper else None
        self.smooth_weight = smooth_weight

    def forward(self):
        k = F.softplus(self.raw)
        if self.taper is not None:
            k = k * self.taper
        return k / k.sum()

    def regularizer(self, kernel):
        return self.smooth_weight * kernel_smoothness(kernel)

    def describe(self):
        return {}


class GaussianPSF(nn.Module):
    """Anisotropic Gaussian with a free centroid offset: sigma_x, sigma_y, theta, dy, dx.

    Smooth and compact by construction -- cannot produce a noise-driven pixel
    artifact, since it has no per-pixel degrees of freedom at all. This was
    the reliable PSF family in earlier work on this dataset at this blur scale.
    """

    def __init__(self, kernel_size, init_sigma=2.0, device=None, dtype=torch.float32):
        super().__init__()
        self.kernel_size = kernel_size
        init_raw_sigma = math.log(math.exp(init_sigma) - 1.0)  # softplus^{-1}
        self.raw_sigma_x = nn.Parameter(torch.tensor(init_raw_sigma, device=device, dtype=dtype))
        self.raw_sigma_y = nn.Parameter(torch.tensor(init_raw_sigma, device=device, dtype=dtype))
        self.theta = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.dy = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.dx = nn.Parameter(torch.zeros((), device=device, dtype=dtype))

    def _sigmas(self):
        return F.softplus(self.raw_sigma_x) + 0.3, F.softplus(self.raw_sigma_y) + 0.3

    def forward(self):
        sigma_x, sigma_y = self._sigmas()
        coords = torch.arange(self.kernel_size, dtype=self.theta.dtype, device=self.theta.device)
        coords = coords - (self.kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        yy, xx = yy - self.dy, xx - self.dx
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        u = cos_t * yy + sin_t * xx
        v = -sin_t * yy + cos_t * xx
        g = torch.exp(-0.5 * ((u / sigma_x) ** 2 + (v / sigma_y) ** 2))
        return g / g.sum()

    def regularizer(self, kernel):
        return torch.zeros((), device=kernel.device, dtype=kernel.dtype)

    def describe(self):
        sigma_x, sigma_y = self._sigmas()
        return {
            "sigma_x": sigma_x.item(),
            "sigma_y": sigma_y.item(),
            "theta_deg": math.degrees(self.theta.item()),
            "dy": self.dy.item(),
            "dx": self.dx.item(),
        }


def make_psf_model(args, device, seed=None):
    dtype = torch.float32
    if args.model == "gaussian":
        return GaussianPSF(args.kernel_size, init_sigma=args.init_sigma, device=device, dtype=dtype)
    return FreeformPSF(
        args.kernel_size,
        taper=not args.no_taper,
        smooth_weight=args.smooth_weight,
        seed=args.seed if seed is None else seed,
        device=device,
        dtype=dtype,
    )


def valid_conv(sharp_batch, kernel):
    """sharp_batch: (N, Hs, Ws) already padded by kernel_size//2 per side. Returns (N, 1, H, W)."""
    x = sharp_batch.unsqueeze(1)
    w = kernel.unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, w)


def fit_psf(model, sharp_batch, blurred_batch, window, args, log_prefix=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    target = blurred_batch.unsqueeze(1)
    for step in range(args.steps):
        optimizer.zero_grad()
        kernel = model()
        pred = valid_conv(sharp_batch, kernel)
        l2 = 0.5 * F.mse_loss(pred, target)
        ssim_term = 1.0 - ssim_map(pred, target, window).mean()
        loss = l2 + args.ssim_weight * ssim_term + model.regularizer(kernel)
        loss.backward()
        optimizer.step()

        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"{log_prefix}step {step:4d}  loss={loss.item():.5f}  l2={l2.item():.5f}  "
                  f"ssim={1 - ssim_term.item():.4f}")
    return model().detach()


def kernel_correlation(a, b):
    a = a.flatten() - a.mean()
    b = b.flatten() - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


# ---------------------------------------------------------------------------
# Data pooling / evaluation
# ---------------------------------------------------------------------------
def gather_patches(data_root, step, scene_ids, center, neighborhood, patch, kernel_size, stride, min_var):
    sharp_crops, blurred_crops = [], []
    for scene_id in scene_ids:
        sharp, blurred = load_pair(data_root, step, scene_id)
        for sharp_crop, blurred_crop in extract_patch_pairs(
            sharp, blurred, center, neighborhood, patch, kernel_size, stride, min_var
        ):
            sharp_crops.append(sharp_crop)
            blurred_crops.append(blurred_crop)
    return torch.stack(sharp_crops), torch.stack(blurred_crops)


def evaluate(sharp_batch, blurred_batch, kernel, window, pad):
    """No-grad evaluation: fitted-kernel fidelity vs. a no-blur-model baseline."""
    with torch.no_grad():
        pred = valid_conv(sharp_batch, kernel)
        target = blurred_batch.unsqueeze(1)
        l2 = 0.5 * F.mse_loss(pred, target).item()
        ssim_val = ssim_map(pred, target, window).mean().item()

        # baseline: "no blur model" -- compare the sharp patch directly (center-cropped
        # to the same footprint) against the real blurred patch.
        sharp_center = sharp_batch[:, pad : pad + target.shape[-2], pad : pad + target.shape[-1]].unsqueeze(1)
        baseline_l2 = 0.5 * F.mse_loss(sharp_center, target).item()
        baseline_ssim = ssim_map(sharp_center, target, window).mean().item()
    return {
        "l2": l2,
        "ssim": ssim_val,
        "baseline_l2": baseline_l2,
        "baseline_ssim": baseline_ssim,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=str, default=None, help="Folder containing CAM01_focused/CAM02_blurred")
    p.add_argument("--focus-step", type=int, default=1, help="Blur level (0-4); 'level 1 blur' = 1")
    p.add_argument("--position", type=str, default="730,1180", help="cy,cx in full-frame pixel coordinates")
    p.add_argument("--model", type=str, choices=["gaussian", "freeform"], default="gaussian")
    p.add_argument("--init-sigma", type=float, default=2.0, help="gaussian model: initial sigma_x=sigma_y")
    p.add_argument("--kernel-size", type=int, default=15)
    p.add_argument("--patch", type=int, default=48, help="Patch size compared in the loss")
    p.add_argument("--neighborhood", type=int, default=96, help="Side of the square tiling window around --position")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--min-var", type=float, default=1e-4, help="Drop patches with less variance than this (near-flat content)")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--ssim-weight", type=float, default=1.0)
    p.add_argument("--smooth-weight", type=float, default=1e-3, help="freeform model only")
    p.add_argument("--no-taper", action="store_true", help="freeform model only: disable the Hann border taper")
    p.add_argument("--stability-check", action="store_true",
                    help="Also fit independently on two random disjoint halves of the training scenes "
                         "and report kernel correlation, to check sensitivity to sensor noise vs. real signal")
    p.add_argument("--test-natural", type=str, default="Image_squirrel_200")
    p.add_argument("--test-text", type=str, default="timesR_size_30_sample_0001")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "results" / "psf_single_position"))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cy, cx = (int(v) for v in args.position.split(","))

    data_root = find_data_root(args.data_root, repo_root=REPO_ROOT)
    scenes = discover_scenes(data_root, args.focus_step)
    print(f"data root: {data_root}")
    print(f"scenes at focusStep_{args.focus_step}: "
          f"{len(scenes['natural'])} natural, {len(scenes['text'])} text, {len(scenes['qr'])} qr")

    assert args.test_natural in scenes["natural"], f"{args.test_natural!r} not found among natural scenes"
    assert args.test_text in scenes["text"], f"{args.test_text!r} not found among text scenes"

    train_scenes = (
        [s for s in scenes["natural"] if s != args.test_natural]
        + [s for s in scenes["text"] if s != args.test_text]
        + scenes["qr"]
    )
    print(f"train scenes: {len(train_scenes)}  |  held out: {args.test_natural!r}, {args.test_text!r}")
    print(f"psf model: {args.model}")

    common = dict(
        center=(cy, cx),
        neighborhood=args.neighborhood,
        patch=args.patch,
        kernel_size=args.kernel_size,
        stride=args.stride,
    )
    train_sharp, train_blurred = gather_patches(data_root, args.focus_step, train_scenes, min_var=args.min_var, **common)
    print(f"pooled training patches: {train_sharp.shape[0]} "
          f"(each {args.patch}x{args.patch}, {args.min_var=} filter)")

    train_sharp, train_blurred = train_sharp.to(device), train_blurred.to(device)
    window = _gaussian_window(11, 1.5, dtype=train_sharp.dtype, device=device)
    pad = args.kernel_size // 2

    stability = None
    if args.stability_check:
        print("\n--- stability check: fitting independently on two random disjoint scene halves ---")
        rng = np.random.RandomState(args.seed)
        shuffled = list(train_scenes)
        rng.shuffle(shuffled)
        half_a, half_b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]
        kernels = []
        for label, half in [("A", half_a), ("B", half_b)]:
            sharp_h, blurred_h = gather_patches(data_root, args.focus_step, half, min_var=args.min_var, **common)
            sharp_h, blurred_h = sharp_h.to(device), blurred_h.to(device)
            model_h = make_psf_model(args, device, seed=args.seed)
            kernel_h = fit_psf(model_h, sharp_h, blurred_h, window, args, log_prefix=f"  [half {label}] ")
            kernels.append(kernel_h)
            print(f"  [half {label}] {len(half)} scenes, {sharp_h.shape[0]} patches"
                  + (f"  params={model_h.describe()}" if model_h.describe() else ""))
        corr = kernel_correlation(kernels[0], kernels[1])
        print(f"--- stability check: correlation(half A, half B) = {corr:.4f} "
              f"(near 1.0 = stable/systematic signal; low/unstable = likely fitting noise) ---\n")
        stability = {"correlation_half_a_half_b": corr}

    model = make_psf_model(args, device)
    kernel = fit_psf(model, train_sharp, train_blurred, window, args)
    if model.describe():
        print(f"fitted params: {model.describe()}")

    train_metrics = evaluate(train_sharp, train_blurred, kernel, window, pad)
    print(f"\n[train, pooled] l2={train_metrics['l2']:.5f} (baseline {train_metrics['baseline_l2']:.5f})  "
          f"ssim={train_metrics['ssim']:.4f} (baseline {train_metrics['baseline_ssim']:.4f})")

    test_results = {}
    test_examples = {}
    for label, scene_id in [("natural", args.test_natural), ("text", args.test_text)]:
        sharp, blurred = gather_patches(data_root, args.focus_step, [scene_id], min_var=0.0, **common)
        sharp, blurred = sharp.to(device), blurred.to(device)
        metrics = evaluate(sharp, blurred, kernel, window, pad)
        test_results[label] = {"scene_id": scene_id, "n_patches": sharp.shape[0], **metrics}
        print(f"[test/{label} {scene_id!r}] l2={metrics['l2']:.5f} (baseline {metrics['baseline_l2']:.5f})  "
              f"ssim={metrics['ssim']:.4f} (baseline {metrics['baseline_ssim']:.4f})")

        with torch.no_grad():
            pred = valid_conv(sharp[:1], kernel).squeeze().cpu().numpy()
            sharp_center = sharp[:1, pad : pad + args.patch, pad : pad + args.patch].squeeze().cpu().numpy()
            blurred_np = blurred[:1].squeeze().cpu().numpy()
        test_examples[label] = (sharp_center, pred, blurred_np)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"psf_focusStep{args.focus_step}_pos{cy}_{cx}.npy", kernel.cpu().numpy())

    report = {
        "args": vars(args),
        "data_root": str(data_root),
        "train_scenes": train_scenes,
        "train_patches": int(train_sharp.shape[0]),
        "train_metrics": train_metrics,
        "test_metrics": test_results,
        "psf_params": model.describe(),
        "stability": stability,
    }
    with open(out_dir / f"report_focusStep{args.focus_step}_pos{cy}_{cx}.json", "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes[0, 0].imshow(kernel.cpu().numpy(), cmap="viridis")
    axes[0, 0].set_title(f"fitted PSF ({args.model}, {args.kernel_size}x{args.kernel_size})")
    axes[1, 0].axis("off")
    for row, label in enumerate(["natural", "text"]):
        sharp_center, pred, blurred_np = test_examples[label]
        for col, (img, title) in enumerate(
            [(sharp_center, "sharp (input)"), (pred, "predicted blur"), (blurred_np, "real blurred")]
        ):
            ax = axes[row, col + 1]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{label}: {title}", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"panel_focusStep{args.focus_step}_pos{cy}_{cx}.png", dpi=120)
    print(f"\nsaved kernel, report, and panel to {out_dir}")


if __name__ == "__main__":
    main()
