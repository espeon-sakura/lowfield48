"""Environment plumbing: find the data, pick the device, pick the worker count.

Every one of these exists because it is a thing that wastes an hour if it goes wrong.
"""
import os, glob, sys

# --------------------------------------------------------------------------
# 1. find the dataset wherever it was unzipped
# --------------------------------------------------------------------------
_MARKER = "*_mask.tif"

def _looks_like_root(p):
    """A valid root directly contains patient folders that contain *_mask.tif."""
    if not os.path.isdir(p):
        return False
    for d in glob.glob(os.path.join(p, "*")):
        if os.path.isdir(d) and glob.glob(os.path.join(d, _MARKER)):
            return True
    return False

def find_data_root(given=None, quiet=False):
    """Return a usable dataset root, searching the obvious places if needed."""
    candidates = []
    if given:
        candidates += [given, os.path.join(given, "kaggle_3m"),
                       os.path.join(given, "lgg-mri-segmentation", "kaggle_3m")]
    candidates += [
        "data/kaggle_3m", "kaggle_3m",
        "data/lgg-mri-segmentation/kaggle_3m", "lgg-mri-segmentation/kaggle_3m",
        os.path.expanduser("~/data/kaggle_3m"),
        os.path.expanduser("~/Downloads/kaggle_3m"),
        os.path.expanduser("~/Downloads/lgg-mri-segmentation/kaggle_3m"),
        "/kaggle/input/lgg-mri-segmentation/kaggle_3m",   # Kaggle notebooks
        "../input/lgg-mri-segmentation/kaggle_3m",
    ]
    for c in candidates:
        if _looks_like_root(c):
            if not quiet and c != given:
                print(f"[data] using {os.path.abspath(c)}")
            return c
    # last resort: walk a couple of levels looking for the marker
    for base in (".", "data", os.path.expanduser("~/Downloads")):
        if not os.path.isdir(base):
            continue
        for d in glob.glob(os.path.join(base, "*", "*")) + glob.glob(os.path.join(base, "*")):
            if _looks_like_root(d):
                if not quiet:
                    print(f"[data] found dataset at {os.path.abspath(d)}")
                return d
    raise FileNotFoundError(
        "Could not find the LGG dataset.\n"
        "  Expected a folder whose subfolders are TCGA_* and contain *_mask.tif files.\n"
        "  The Kaggle zip unpacks to lgg-mri-segmentation/kaggle_3m/ - pass that with --root,\n"
        f"  or move it to ./data/kaggle_3m/. Searched from {os.path.abspath('.')}.")

# --------------------------------------------------------------------------
# 2. device, with a useful diagnosis instead of a silent fallback
# --------------------------------------------------------------------------
def pick_device(verbose=True):
    import torch
    if torch.cuda.is_available():
        if verbose:
            print(f"[device] cuda: {torch.cuda.get_device_name(0)}")
        return "cuda"
    has_driver = any(os.path.exists(p) for p in ("/proc/driver/nvidia", "/dev/nvidia0")) \
                 or _which("nvidia-smi") is not None
    if verbose:
        if has_driver:
            print("[device] cpu - BUT an NVIDIA driver is present, so you installed the CPU-only"
                  " build of PyTorch.\n"
                  "         Fix: pip uninstall torch, then install the CUDA build from pytorch.org"
                  " for your CUDA version.")
        else:
            print("[device] cpu - no GPU detected; everything still runs, just slower.")
    return "cpu"

def _which(cmd):
    import shutil
    return shutil.which(cmd)

# --------------------------------------------------------------------------
# 3. dataloader workers - 0 on Windows, where forking breaks
# --------------------------------------------------------------------------
def n_workers(requested=None):
    if os.name == "nt" or sys.platform == "darwin" and sys.version_info < (3, 8):
        return 0
    if requested is not None:
        return requested
    return min(4, (os.cpu_count() or 2))
