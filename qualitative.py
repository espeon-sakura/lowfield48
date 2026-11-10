"""
The pictures your report and slides need. Numbers persuade reviewers; images persuade
everyone else, and a professor looks at these before any table.

    python qualitative.py --ckpt_clean ckpt/seg_clean.pt \
        --ckpt_aug ckpt/seg_aug.pt --ckpt_restore ckpt/restore_s55.pt

Writes:
  fig7_qualitative_ladder.png  3 patients x 4 severities, outlines overlaid
  fig8_qualitative_arms.png    one patient at one severity, every strategy side by side
"""
import argparse, os, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.io import imread
from data import split_patients
from unet import UNet
from degrade import degrade, PROTOCOLS
from utils import find_data_root, pick_device
import enhance as E

GT, PRED, SURF, INK, MUTED = "#0E8F6F", "#C2410C", "#FCFCFB", "#1A202C", "#5A6B72"


def load(ckpt, out_ch, dev):
    net = UNet(3, out_ch).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)); net.eval()
    return net


def predict(net, x, dev):
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)
    with torch.no_grad():
        return (torch.sigmoid(net(t))[0, 0].cpu().numpy() > 0.5).astype(np.float32)


def dice(p, g, eps=1.0):
    return float((2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps))


def panel(ax, img, gt, pred, title=None, sub=None):
    ax.imshow(img[..., 1], cmap="gray", vmin=0, vmax=1)
    if gt.max() > 0:
        ax.contour(gt, levels=[0.5], colors=[GT], linewidths=1.6)
    if pred is not None and pred.max() > 0:
        ax.contour(pred, levels=[0.5], colors=[PRED], linewidths=1.6)
    elif pred is not None:
        ax.text(0.5, 0.06, "predicted nothing", transform=ax.transAxes, ha="center",
                fontsize=8, color=PRED, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, pad=4)
    if sub:
        # inside the axes, not below it - captions under images collide with the next row
        ax.text(0.03, 0.04, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#00000099", ec="none"))
    ax.axis("off")


def pick_cases(root, k=3):
    _, _, test = split_patients(root, seed=0)
    cases = []
    for d in test:
        best = None
        for m in sorted(glob.glob(os.path.join(d, "*_mask.tif"))):
            area = float((imread(m) > 0).sum())
            if area > 300 and (best is None or area > best[0]):
                best = (area, m)
        if best:
            cases.append(best)
    cases.sort(reverse=True)
    step = max(1, len(cases) // k)
    return [c[1] for c in cases[::step]][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--ckpt_clean", default="ckpt/seg_clean.pt")
    ap.add_argument("--ckpt_aug", default="")
    ap.add_argument("--ckpt_restore", default="")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--severity", type=float, default=0.55)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    a.root = find_data_root(a.root)
    dev = pick_device()
    seg = load(a.ckpt_clean, 1, dev)
    cases = pick_cases(a.root, 3)
    print(f"{len(cases)} example slices")

    # ---------------- figure 7: the ladder, with outlines ----------------
    sev = list(PROTOCOLS.items())
    fig, axes = plt.subplots(len(cases), len(sev),
                             figsize=(2.7 * len(sev), 2.95 * len(cases)),
                             dpi=190, facecolor=SURF, squeeze=False)
    for r, mp in enumerate(cases):
        clean = imread(mp.replace("_mask.tif", ".tif")).astype(np.float32) / 255.0
        gt = (imread(mp) > 0).astype(np.float32)
        for c, (name, s) in enumerate(sev):
            x = clean if s == 0 else degrade(clean, s, seed=7)
            p = predict(seg, x, dev)
            panel(axes[r][c], x, gt, p,
                  title=name.replace("_", " ") if r == 0 else None,
                  sub=f"Dice {dice(p, gt):.2f}")
    fig.suptitle("What the model sees, and what it outlines", x=0.012, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=GT, lw=2, label="expert outline"),
                        Line2D([], [], color=PRED, lw=2, label="model prediction")],
               loc="upper right", frameon=False, fontsize=9.5, ncol=2,
               bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{a.outdir}/fig7_qualitative_ladder.png", facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig7_qualitative_ladder.png")

    # ---------------- figure 8: every strategy, one scan ----------------
    mp = cases[0]
    clean = imread(mp.replace("_mask.tif", ".tif")).astype(np.float32) / 255.0
    gt = (imread(mp) > 0).astype(np.float32)
    deg = degrade(clean, a.severity, seed=7)

    cols = [("clean scan", clean, seg), ("no mitigation", deg, seg),
            ("classical enhancement", E.make_classical("auto")(deg), seg)]
    if a.ckpt_restore:
        cols.append(("learned restoration", E.learned(a.ckpt_restore, dev)(deg), seg))
    if a.ckpt_aug:
        cols.append(("degradation-augmented", deg, load(a.ckpt_aug, 1, dev)))

    fig, axes = plt.subplots(1, len(cols), figsize=(2.7 * len(cols), 3.3),
                             dpi=190, facecolor=SURF, squeeze=False)
    for c, (name, img, net) in enumerate(cols):
        p = predict(net, img, dev)
        panel(axes[0][c], img, gt, p, title=name, sub=f"Dice {dice(p, gt):.2f}")
    fig.suptitle(f"Same scan at severity {a.severity:g}, five strategies",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=GT, lw=2, label="expert outline"),
                        Line2D([], [], color=PRED, lw=2, label="model prediction")],
               loc="upper right", frameon=False, fontsize=9.5, ncol=2,
               bbox_to_anchor=(0.99, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(f"{a.outdir}/fig8_qualitative_arms.png", facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig8_qualitative_arms.png")


if __name__ == "__main__":
    main()
