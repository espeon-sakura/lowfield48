"""
Train the segmenter (task=seg) or the learned enhancement net (task=restore).

  # baseline, clean data
  python train.py --task seg --severity 0 --epochs 25 --out ckpt/seg_clean.pt
  # fine-tune the clean model on N degraded patients
  python train.py --task seg --severity 0.55 --n_patients 10 \
      --init ckpt/seg_clean.pt --epochs 15 --out ckpt/seg_ft_s55_n10.pt
  # B2: degradation-augmented training (no local labels needed)
  python train.py --task seg --augment --epochs 30 --out ckpt/seg_aug.pt
  # learned restoration: degraded -> clean
  python train.py --task restore --severity 0.55 --epochs 20 --out ckpt/restore_s55.pt
"""
import argparse, os, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from data import LGG, split_patients
from utils import find_data_root, pick_device, n_workers
from unet import UNet
from degrade import degrade

def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * y).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps
    return 1 - (num / den).mean()

class RestorePairs(torch.utils.data.Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        from skimage.io import imread
        ip, _ = self.base.items[i]
        clean = imread(ip).astype(np.float32) / 255.0
        deg = degrade(clean, self.base.severity, seed=i)
        t = lambda a: torch.from_numpy(a.transpose(2, 0, 1).copy())
        return t(deg), t(clean)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="auto-detected if omitted")
    ap.add_argument("--task", choices=["seg", "restore"], default="seg")
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--n_patients", type=int, default=0, help="0 = all training patients")
    ap.add_argument("--augment", action="store_true", help="B2: train with random degradation")
    ap.add_argument("--init", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.root = find_data_root(a.root)
    nw = n_workers()

    tr, va, _ = split_patients(a.root, seed=0)
    if a.n_patients > 0:
        tr = tr[:a.n_patients]                       # deterministic subset = the label-budget arm
    print(f"train patients: {len(tr)}   val patients: {len(va)}")

    tr_ds = LGG(tr, severity=a.severity, tumour_only=(a.task == "seg"), augment=a.augment)
    va_ds = LGG(va, severity=a.severity, tumour_only=(a.task == "seg"))
    if a.task == "restore":
        tr_ds, va_ds = RestorePairs(tr_ds), RestorePairs(va_ds)

    dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, num_workers=nw, drop_last=True)
    vl = DataLoader(va_ds, batch_size=a.bs, num_workers=nw)

    dev = pick_device()
    net = UNet(3, 1 if a.task == "seg" else 3).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)); print("initialised from", a.init)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    best = 1e9; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    for ep in range(a.epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            out = net(x)
            loss = (bce(out, y) + dice_loss(out, y)) if a.task == "seg" else l1(out, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            del x, y, out, loss
        net.eval(); tot = n = 0
        if dev == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev); out = net(x)
                v = (bce(out, y) + dice_loss(out, y)) if a.task == "seg" else l1(out, y)
                tot += float(v) * len(x); n += len(x)
        vloss = tot / max(n, 1)
        print(f"epoch {ep+1:3d}/{a.epochs}  val {vloss:.4f}")
        if vloss < best:
            best = vloss; torch.save(net.state_dict(), a.out)
    print("best val", round(best, 4), "->", a.out)

if __name__ == "__main__":
    main()
