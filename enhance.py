"""
Enhancement arms. Each takes a degraded image (H,W,3 in [0,1]) and returns the same shape.

Two backends for the classical arm, because the honest one is slow:

  nlm  - non-local means + unsharp mask. Faithful to what a real preprocessing
         pipeline does. Runs on CPU at roughly 0.5-1.5 s per image.
  fast - edge-preserving bilateral filter + unsharp mask, written in torch so it
         runs on the GPU. Same two operations (denoise, then sharpen), ~100x faster.

Both are "denoise then sharpen". If you have a GPU, use `fast` for the sweeps and
run `nlm` once at a couple of severities to confirm the conclusion does not depend
on the backend - that check belongs in the report.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means

try:                                    # estimate_sigma needs PyWavelets
    from skimage.restoration import estimate_sigma as _sk_sigma
except Exception:                       # pragma: no cover
    _sk_sigma = None


def _noise_sigma(x):
    """Robust noise estimate. Uses skimage when PyWavelets is available, else a
    median-absolute-deviation estimate off the image's high-frequency content,
    which needs no extra dependency and is close enough for setting a filter strength."""
    if _sk_sigma is not None:
        try:
            return float(np.mean(_sk_sigma(x, channel_axis=-1)))
        except Exception:
            pass
    d = np.diff(x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.ravel(), axis=0)
    return float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2)) or 0.01


def none_fn(x):
    return x


def _unsharp(x, radius=2.0, amount=1.0):
    """Unsharp mask, implemented here on purpose.

    skimage's unsharp_mask returns garbage (values ~1e300) when channel_axis=-1 is
    used on some versions - 0.26 among them - so we do it ourselves. It is three
    lines, it is version-proof, and it makes the CPU and GPU backends perform the
    identical operation, so the only difference between them is the denoiser."""
    blur = np.stack([gaussian_filter(x[..., c], radius) for c in range(x.shape[-1])], -1)
    return np.clip(x + amount * (x - blur), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# CPU backend: non-local means + unsharp
# --------------------------------------------------------------------------
def classical_nlm(x):
    s = max(_noise_sigma(x), 1e-3)
    d = denoise_nl_means(x, h=1.15 * s, sigma=s, fast_mode=True,
                         patch_size=5, patch_distance=6, channel_axis=-1)
    # NOTE: unsharp_mask on float32 input without preserve_range returns +/-inf
    # (skimage rescales float32 as if it were a different dtype range). Both the
    # float64 cast and preserve_range=True avoid it; we do both, deliberately.
    return _unsharp(np.clip(d, 0, 1), radius=2.0, amount=1.0)


# --------------------------------------------------------------------------
# GPU backend: bilateral filter + unsharp, in torch
# --------------------------------------------------------------------------
def _gauss_kernel(sigma, device, dtype=torch.float32):
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return (k / k.sum()), r


def _gaussian_blur(t, sigma):
    """Separable Gaussian blur on (1,C,H,W)."""
    k, r = _gauss_kernel(sigma, t.device, t.dtype)
    C = t.shape[1]
    kx = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), kx, groups=C)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), ky, groups=C)
    return t


def _bilateral(t, win=5, sigma_s=1.5, sigma_r=0.08):
    """Edge-preserving denoise on (1,C,H,W). Keeps boundaries, removes grain."""
    B, C, H, W = t.shape
    p = win // 2
    patches = F.unfold(F.pad(t, (p, p, p, p), mode="reflect"), kernel_size=win)  # B, C*win*win, H*W
    patches = patches.view(B, C, win * win, H * W)
    centre = t.view(B, C, 1, H * W)
    yy, xx = torch.meshgrid(torch.arange(win, device=t.device) - p,
                            torch.arange(win, device=t.device) - p, indexing="ij")
    ws = torch.exp(-(xx ** 2 + yy ** 2).float() / (2 * sigma_s ** 2)).view(1, 1, win * win, 1)
    wr = torch.exp(-((patches - centre) ** 2) / (2 * sigma_r ** 2))
    w = ws * wr
    out = (w * patches).sum(2) / w.sum(2).clamp_min(1e-8)
    return out.view(B, C, H, W)


def classical_fast(x, device=None, amount=1.0, radius=2.0):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)
    with torch.no_grad():
        d = _bilateral(t)
        sharp = (d + amount * (d - _gaussian_blur(d, radius))).clamp(0, 1)
    return sharp[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def classical(x, backend="auto"):
    """backend: 'auto' (gpu->fast, cpu->nlm), 'fast', or 'nlm'."""
    if backend == "auto":
        backend = "fast" if torch.cuda.is_available() else "nlm"
    return classical_fast(x) if backend == "fast" else classical_nlm(x)


def make_classical(backend="auto"):
    if backend == "auto":
        backend = "fast" if torch.cuda.is_available() else "nlm"
    fn = classical_fast if backend == "fast" else classical_nlm
    fn.backend_name = backend
    return fn


def learned(ckpt, device=None):
    """An enhance_fn backed by a trained restoration U-Net."""
    from unet import UNet
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(3, 3).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)); net.eval()
    def fn(x):
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)
        with torch.no_grad():
            y = net(t)[0].cpu().numpy().transpose(1, 2, 0)
        return np.clip(y, 0, 1).astype(np.float32)
    return fn
