"""
Physically-motivated simulation of low-field / low-spec MRI acquisition.

Everything here operates on float images in [0,1], shape (H,W) or (H,W,C).
No torch needed - numpy/scipy/skimage only.

The four effects, and why each one is defensible in a viva:

1. k-space truncation   -> a smaller acquisition matrix. This is literally what a
                           scanner does when you lower the resolution: it samples
                           fewer phase-encode lines. We crop the centre of the FFT
                           and zero-fill back, which is the standard simulation.
2. Rician noise         -> lower field strength means lower SNR. MRI magnitude
                           images are Rician (magnitude of complex Gaussian), not
                           Gaussian. Using Gaussian noise here would be the single
                           easiest thing for an examiner to attack.
3. Bias field           -> B1 inhomogeneity, worse on older / cheaper coils.
                           Smooth multiplicative low-frequency field.
4. Through-plane blur   -> thicker slices (5mm vs 1mm) average tissue together.
                           Approximated in-plane here as mild Gaussian blur.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# Single-precision FFT when scipy is available: numpy's np.fft always computes in
# double precision and returns complex128, which costs twice the memory for no
# benefit here. scipy.fft keeps float32 -> complex64. On a memory-tight machine
# this is the difference between running and an ArrayMemoryError.
try:
    from scipy import fft as _sfft

    def _fft2(a):
        return _sfft.fft2(np.asarray(a, dtype=np.float32))

    def _ifft2(A):
        return _sfft.ifft2(A)

    _fftshift, _ifftshift = _sfft.fftshift, _sfft.ifftshift
except ImportError:                                     # pragma: no cover
    _fft2, _ifft2 = np.fft.fft2, np.fft.ifft2
    _fftshift, _ifftshift = np.fft.fftshift, np.fft.ifftshift

# scipy's FFT keeps single precision (float32 -> complex64). numpy's always upcasts
# to complex128, which doubles the transient memory of every degraded image and is
# slower. On a 256x256x3 slice that is the difference between ~3 MiB and ~6 MiB per
# call, and it is called once per training sample when --augment is on.
try:
    from scipy.fft import fft2 as _fft2, ifft2 as _ifft2, fftshift as _fftshift, ifftshift as _ifftshift
except ImportError:                                           # pragma: no cover
    from numpy.fft import fft2 as _fft2, ifft2 as _ifft2, fftshift as _fftshift, ifftshift as _ifftshift


# --------------------------------------------------------------------------
# individual effects
# --------------------------------------------------------------------------
def kspace_truncate(img, keep_frac):
    """Keep only the central `keep_frac` of k-space, then zero-fill back."""
    if keep_frac >= 0.999:
        return img.copy()
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[-1])]
    done = []
    for ch in chans:
        F = _fftshift(_fft2(np.ascontiguousarray(ch, dtype=np.float32)))
        H, W = F.shape
        kh, kw = int(H * keep_frac) // 2, int(W * keep_frac) // 2
        # zero outside the kept rectangle in place - no second full-size mask array
        F[:H // 2 - kh, :] = 0
        F[H // 2 + kh:, :] = 0
        F[:, :W // 2 - kw] = 0
        F[:, W // 2 + kw:] = 0
        out = np.abs(_ifft2(_ifftshift(F))).astype(np.float32)
        del F
        done.append(out)
    return done[0] if img.ndim == 2 else np.stack(done, -1)


def rician_noise(img, sigma, rng):
    """Magnitude of a complex Gaussian - the correct noise model for MRI magnitude.

    Written in float32 and in place. rng.normal() returns float64 by default, and the
    naive expression builds six temporaries of it; on a memory-starved machine that is
    what fails. Same maths, a third of the peak memory.
    """
    if sigma <= 0:
        return img.copy()
    re = rng.standard_normal(img.shape, dtype=np.float32)
    re *= sigma
    re += img
    im = rng.standard_normal(img.shape, dtype=np.float32)
    im *= sigma
    np.square(re, out=re)
    np.square(im, out=im)
    re += im
    del im
    return np.sqrt(re, out=re)


def bias_field(img, strength, rng):
    """Smooth multiplicative B1 inhomogeneity."""
    if strength <= 0:
        return img.copy()
    H, W = img.shape[:2]
    low = rng.standard_normal((8, 8), dtype=np.float32)
    field = np.kron(low, np.ones((H // 8 + 1, W // 8 + 1), dtype=np.float32))[:H, :W]
    field = gaussian_filter(field, sigma=max(H, W) / 12.0).astype(np.float32, copy=False)
    field /= (np.abs(field).max() + 1e-8)
    field *= strength
    field += 1.0
    if img.ndim == 3:
        field = field[..., None]
    return np.multiply(img, field, dtype=np.float32)


def slice_blur(img, sigma):
    if sigma <= 0:
        return img.copy()
    if img.ndim == 2:
        return gaussian_filter(img, sigma).astype(np.float32)
    return np.stack([gaussian_filter(img[..., c], sigma) for c in range(img.shape[-1])], -1)


# --------------------------------------------------------------------------
# severity ladder
# --------------------------------------------------------------------------
# s = 0.0 -> untouched (research-grade 3T)
# s = 1.0 -> heavily degraded (old / low-field scanner, thick slices)
def params_for_severity(s):
    s = float(np.clip(s, 0.0, 1.0))
    return dict(
        keep_frac=1.0 - 0.65 * s,     # 1.00 -> 0.35 of k-space
        sigma=0.10 * s,               # Rician sigma, image scaled to [0,1]
        bias=0.30 * s,                # +/- 30% shading at worst
        blur=1.2 * s,                 # thick-slice partial volume proxy
    )


# Named presets. REPLACE THESE NUMBERS with parameters you get from a real
# radiologist / DICOM header - that is what makes the study about India rather
# than about an arbitrary simulator.
PROTOCOLS = {
    "research_3T":        0.00,
    "metro_1p5T_3mm":     0.30,
    "district_1p5T_5mm":  0.55,
    "older_0p35T_6mm":    0.85,
}


def degrade(img, severity, seed=0):
    """Apply the full acquisition chain at a given severity."""
    rng = np.random.default_rng(seed)
    p = params_for_severity(severity)
    x = np.asarray(img, dtype=np.float32)
    x = kspace_truncate(x, p["keep_frac"])
    x = slice_blur(x, p["blur"])
    x = bias_field(x, p["bias"], rng)
    x = rician_noise(x, p["sigma"], rng)
    return np.clip(x, 0, 1, out=x if x.dtype == np.float32 else None).astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# B1: one effect at a time, matched severity
# --------------------------------------------------------------------------
COMPONENTS = ("resolution", "noise", "bias", "blur")

def degrade_component(img, severity, component, seed=0):
    """Apply ONE degradation at the strength it would have at this severity."""
    rng = np.random.default_rng(seed)
    p = params_for_severity(severity)
    if component == "resolution":
        return np.clip(kspace_truncate(img, p["keep_frac"]), 0, 1).astype(np.float32)
    if component == "blur":
        return np.clip(slice_blur(img, p["blur"]), 0, 1).astype(np.float32)
    if component == "bias":
        return np.clip(bias_field(img, p["bias"], rng), 0, 1).astype(np.float32)
    if component == "noise":
        return np.clip(rician_noise(img, p["sigma"], rng), 0, 1).astype(np.float32)
    raise ValueError(f"unknown component {component!r}; expected one of {COMPONENTS}")
