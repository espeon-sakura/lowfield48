"""Figure 0 for the report: one real slice across the degradation ladder.
  python make_ladder_figure.py --root data/kaggle_3m --outdir figures
Falls back to a synthetic phantom if --root is missing, so you can check the
simulator before the download finishes."""
import argparse, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from degrade import degrade, PROTOCOLS

def phantom():
    img = np.zeros((256, 256), np.float32); yy, xx = np.mgrid[0:256, 0:256]
    img[((yy-128)**2 + (xx-128)**2) < 100**2] = 0.55
    img[((yy-110)**2 + (xx-140)**2) < 22**2] = 0.90
    img[((yy-150)**2 + (xx-100)**2) < 9**2] = 0.85
    return np.stack([img]*3, -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None); ap.add_argument("--outdir", default="figures")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    try:
        from utils import find_data_root
        a.root = find_data_root(a.root)
    except Exception as err:
        print(f'[data] {err}\n[data] falling back to a synthetic phantom'); a.root = '__none__'
    img, src = phantom(), "synthetic phantom"
    masks = sorted(glob.glob(os.path.join(a.root, "*", "*_mask.tif")))
    if masks:
        from skimage.io import imread
        for m in masks:
            if imread(m).max() > 0:
                img = imread(m.replace("_mask.tif", ".tif")).astype(np.float32)/255.0
                src = os.path.basename(m).replace("_mask.tif", ""); break
    items = list(PROTOCOLS.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3.1*len(items), 3.5), dpi=200, facecolor="#FCFCFB")
    for ax, (name, sev) in zip(axes, items):
        ax.imshow(degrade(img, sev, seed=3)[..., 1], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name.replace('_',' ')}\nseverity {sev:g}", fontsize=9.5, color="#1A202C")
        ax.axis("off")
    fig.suptitle(f"Simulated acquisition ladder  ({src})", fontsize=12.5,
                 fontweight="bold", color="#1A202C", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"{a.outdir}/fig0_ladder.png", facecolor="#FCFCFB", bbox_inches="tight")
    print("wrote", f"{a.outdir}/fig0_ladder.png")

if __name__ == "__main__":
    main()
