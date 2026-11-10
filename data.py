"""LGG MRI dataset (Kaggle: mateuszbuda/lgg-mri-segmentation), patient-level splits."""
import os, glob, numpy as np, torch
from torch.utils.data import Dataset
from skimage.io import imread
from degrade import degrade

def patient_dirs(root):
    ds = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    return [d for d in ds if glob.glob(os.path.join(d, "*_mask.tif"))]

def split_patients(root, seed=0, frac=(0.7, 0.15, 0.15)):
    ds = patient_dirs(root)
    rng = np.random.default_rng(seed); rng.shuffle(ds)
    n = len(ds); a = int(frac[0]*n); b = a + int(frac[1]*n)
    return ds[:a], ds[a:b], ds[b:]

class LGG(Dataset):
    """severity: 0 = clean. enhance_fn: optional callable(np HWC)->np HWC applied after degradation."""
    def __init__(self, pdirs, severity=0.0, enhance_fn=None, tumour_only=False, seed=0, augment=False):
        """augment=True (B2): draw a random severity per sample instead of a fixed one."""
        self.items, self.severity, self.enhance_fn, self.seed = [], severity, enhance_fn, seed
        self.augment = augment
        self._rng = None
        for d in pdirs:
            for m in sorted(glob.glob(os.path.join(d, "*_mask.tif"))):
                img = m.replace("_mask.tif", ".tif")
                if os.path.exists(img):
                    self.items.append((img, m))
        if tumour_only:
            keep = []
            for img, m in self.items:
                if imread(m).max() > 0: keep.append((img, m))
            self.items = keep

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        x = imread(ip).astype(np.float32) / 255.0          # H,W,3
        y = (imread(mp) > 0).astype(np.float32)            # H,W
        if self.augment:
            if self._rng is None:                      # one Generator per worker, not per sample
                self._rng = np.random.default_rng(self.seed * 100003 + 7919)
            sev = float(self._rng.uniform(0.0, 0.85))
            if sev > 0.02:
                x = degrade(x, sev, seed=int(self._rng.integers(1 << 30)))
        elif self.severity > 0:
            x = degrade(x, self.severity, seed=self.seed * 100003 + i)
        if self.enhance_fn is not None:
            x = self.enhance_fn(x)
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(y[None].copy()))
