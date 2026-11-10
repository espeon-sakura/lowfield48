"""
Run every arm across the degradation ladder and write one long-format CSV.

  # main run (H1, H2, B2, B4, B5)
  python evaluate.py --ckpt_clean ckpt/seg_clean.pt --ckpt_restore ckpt/restore_s55.pt \
      --ckpt_aug ckpt/seg_aug.pt \
      --ckpt_ft 0.55:5:ckpt/ft_s55_n5.pt,0.55:10:ckpt/ft_s55_n10.pt,0.55:20:ckpt/ft_s55_n20.pt,0.55:40:ckpt/ft_s55_n40.pt

  # branch B1: one degradation at a time
  python evaluate.py --components --ckpt_clean ckpt/seg_clean.pt --out results/components.csv

Columns: arm, n_labels, severity, component, patient, slice,
         dice, psnr, ssim, pred_area, gt_area, mean_conf, entropy
pred_area/gt_area give B5 (volume bias, size strata); mean_conf/entropy give B4
(does the model know it is failing). Both are free - no extra forward passes.
"""
import argparse, os, csv, glob, numpy as np, torch
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread
from data import split_patients
from utils import find_data_root, pick_device
from unet import UNet
from degrade import degrade, degrade_component, COMPONENTS, PROTOCOLS
import enhance as E

SEVERITIES = [0.0, 0.15, 0.30, 0.45, 0.55, 0.70, 0.85]
COLS = ["arm", "n_labels", "ft_sev", "severity", "component", "patient", "slice",
        "dice", "psnr", "ssim", "pred_area", "gt_area", "mean_conf", "entropy"]

def safe_psnr(a, b, cap=100.0):
    """PSNR, but finite. Identical images give MSE 0 and therefore inf, which would
    propagate through every mean and break the plots. Cap it instead."""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return cap
    return min(float(10 * np.log10(1.0 / mse)), cap)


def dice(pred, gt, eps=1.0):
    return float((2 * (pred * gt).sum() + eps) / (pred.sum() + gt.sum() + eps))

def load_net(ckpt, out_ch, dev):
    net = UNet(3, out_ch).to(dev)
    net.load_state_dict(torch.load(ckpt, map_location=dev)); net.eval()
    return net

def seg_one(net, x, dev):
    """Return binary mask, mean confidence inside the prediction, and mean entropy."""
    t = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(dev)
    with torch.no_grad():
        p = torch.sigmoid(net(t))[0, 0]
    mask = (p > 0.5).float()
    conf = float((p * mask).sum() / mask.sum()) if float(mask.sum()) > 0 else 0.0
    ent = float(-(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean())
    return mask.cpu().numpy(), conf, ent

def test_items(root):
    _, _, test = split_patients(root, seed=0)
    items = []
    for d in test:
        for m in sorted(glob.glob(os.path.join(d, "*_mask.tif"))):
            ip = m.replace("_mask.tif", ".tif")
            if os.path.exists(ip) and imread(m).max() > 0:
                items.append((os.path.basename(d), ip, m))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="auto-detected if omitted")
    ap.add_argument("--ckpt_clean", required=True)
    ap.add_argument("--ckpt_restore", default="")
    ap.add_argument("--ckpt_aug", default="", help="B2: degradation-augmented model")
    ap.add_argument("--ckpt_ft", default="", help="comma list sev:n:path")
    ap.add_argument("--ft_all_severities", action="store_true",
                    help="B3: test every fine-tuned model at every severity (transfer matrix)")
    ap.add_argument("--components", action="store_true", help="B1 mode")
    ap.add_argument("--enhance", choices=["auto", "nlm", "fast"], default="auto",
                    help="classical backend: nlm (faithful, CPU) or fast (bilateral, GPU)")
    ap.add_argument("--out", default="results/results.csv")
    a = ap.parse_args()
    dev = pick_device()
    a.root = find_data_root(a.root)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    items = test_items(a.root)
    print(f"test slices with tumour: {len(items)}   device: {dev}")
    seg_clean = load_net(a.ckpt_clean, 1, dev)

    f = open(a.out, "w", newline=""); w = csv.writer(f); w.writerow(COLS)

    def row(arm, n, sev, comp, pid, ip, clean, x, net, gt, ft_sev=-1.0):
        mask, conf, ent = seg_one(net, x, dev)
        w.writerow([arm, n, ft_sev, sev, comp, pid, os.path.basename(ip),
                    dice(mask, gt),
                    safe_psnr(clean, x),
                    ssim(clean, x, channel_axis=-1, data_range=1.0),
                    float(mask.sum()), float(gt.sum()), conf, ent])

    # ---------------- B1: one degradation at a time ----------------
    if a.components:
        for sev in SEVERITIES:
            for pid, ip, mp in items:
                clean = imread(ip).astype(np.float32) / 255.0
                gt = (imread(mp) > 0).astype(np.float32)
                sd = abs(hash(ip)) % 10000
                for comp in COMPONENTS:
                    x = clean if sev == 0 else degrade_component(clean, sev, comp, seed=sd)
                    row("component", 0, sev, comp, pid, ip, clean, x, seg_clean, gt)
            print("B1 done severity", sev)
        f.close(); print("wrote", a.out); return

    # ---------------- main run ----------------
    cls = E.make_classical(a.enhance)
    print(f"[enhance] classical backend: {cls.backend_name}")
    arms = {"none": E.none_fn, "classical": cls}
    if a.ckpt_restore:
        arms["learned"] = E.learned(a.ckpt_restore, dev)
    aug = load_net(a.ckpt_aug, 1, dev) if a.ckpt_aug else None
    ft = {}
    if a.ckpt_ft:
        for spec in a.ckpt_ft.split(","):
            s, n, p = spec.split(":"); ft[(float(s), int(n))] = load_net(p, 1, dev)

    for sev in SEVERITIES:
        for pid, ip, mp in items:
            clean = imread(ip).astype(np.float32) / 255.0
            gt = (imread(mp) > 0).astype(np.float32)
            x = clean if sev == 0 else degrade(clean, sev, seed=abs(hash(ip)) % 10000)
            for name, fn in arms.items():
                row(name, 0, sev, "all", pid, ip, clean, fn(x), seg_clean, gt)
            if aug is not None:
                row("augmented", 0, sev, "all", pid, ip, clean, x, aug, gt)
            for (s_ft, n_ft), net in ft.items():
                if a.ft_all_severities or abs(s_ft - sev) < 1e-6:
                    row("finetune", n_ft, sev, "all", pid, ip, clean, x, net, gt, ft_sev=s_ft)
        print("done severity", sev)
    f.close()
    print("wrote", a.out)
    print("protocol landmarks:", PROTOCOLS)

if __name__ == "__main__":
    main()
