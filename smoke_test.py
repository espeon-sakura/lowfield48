"""
Five-minute end-to-end check. RUN THIS BEFORE COMMITTING TO THE REAL RUN.

  python smoke_test.py --root data/kaggle_3m

It walks the entire pipeline on a tiny subset, times each stage, and then
extrapolates how long the full run will take on YOUR machine. If this passes,
nothing downstream will surprise you except the numbers.
"""
import argparse, os, sys, time, glob

def ok(msg):   print(f"  [ok]   {msg}")
def bad(msg):  print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="auto-detected if omitted")
    a = ap.parse_args()
    t_all = time.time()

    # ---------------------------------------------------------------- 1. imports
    print("\n1. dependencies")
    try:
        import numpy, scipy, skimage, pandas, matplotlib
        ok(f"numpy {numpy.__version__}, scipy {scipy.__version__}, skimage {skimage.__version__}")
    except ImportError as e:
        bad(f"{e}\n  fix: pip install numpy scipy scikit-image pandas matplotlib")
        sys.exit(1)
    try:
        import torch
        ok(f"torch {torch.__version__}")
    except ImportError:
        bad("torch missing\n  fix: pip install torch torchvision")
        sys.exit(1)

    from utils import pick_device, n_workers, find_data_root
    dev = pick_device(verbose=False)
    if dev == "cuda":
        ok(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        import shutil
        if shutil.which("nvidia-smi"):
            bad("GPU hardware present but torch.cuda is False - you have the CPU-only build")
            info("fix: pip uninstall -y torch && install the CUDA build from pytorch.org")
        else:
            info("no GPU detected - everything still runs, just slower")
    ok(f"dataloader workers: {n_workers()}" + ("  (0 on Windows, by design)" if os.name == "nt" else ""))

    # ---------------------------------------------------------------- 2. data
    print("\n2. dataset")
    from data import patient_dirs, split_patients, LGG
    try:
        a.root = find_data_root(a.root, quiet=True)
        ok(f"dataset found at {os.path.abspath(a.root)}")
    except FileNotFoundError as e:
        bad(str(e)); sys.exit(1)
    pdirs = patient_dirs(a.root)
    ok(f"{len(pdirs)} patient folders")
    tr, va, te = split_patients(a.root, seed=0)
    ok(f"split: {len(tr)} train / {len(va)} val / {len(te)} test patients")
    probe = LGG(te[:2], tumour_only=True)
    ok(f"{len(probe)} tumour slices in 2 test patients")
    x, y = probe[0]
    ok(f"sample shapes: image {tuple(x.shape)}, mask {tuple(y.shape)}")

    # ---------------------------------------------------------------- 3. simulator
    print("\n3. simulator")
    import numpy as np
    from degrade import degrade, degrade_component, COMPONENTS
    img = x.numpy().transpose(1, 2, 0)
    t0 = time.time(); d = degrade(img, 0.55, seed=0); t_deg = time.time() - t0
    ok(f"full degradation {t_deg*1000:.0f} ms/image, output range [{d.min():.2f}, {d.max():.2f}]")
    for c in COMPONENTS:
        degrade_component(img, 0.55, c, seed=0)
    ok(f"all four components run: {', '.join(COMPONENTS)}")

    # ---------------------------------------------------------------- 4. model
    print("\n4. model")
    from unet import UNet
    net = UNet(3, 1).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    ok(f"U-Net built, {n_par/1e6:.1f}M parameters")
    xb = x[None].to(dev)
    for _ in range(3):
        with torch.no_grad(): net(xb)
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad(): net(xb)
    if dev == "cuda": torch.cuda.synchronize()
    t_fwd = (time.time() - t0) / 10
    ok(f"forward pass {t_fwd*1000:.1f} ms/image")

    # ---------------------------------------------------------------- 5. training step
    print("\n5. training")
    import torch.nn as nn
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    xb = torch.stack([probe[i][0] for i in range(min(8, len(probe)))]).to(dev)
    yb = torch.stack([probe[i][1] for i in range(min(8, len(probe)))]).to(dev)
    out = net(xb); loss = bce(out, yb); opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        out = net(xb); loss = bce(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    if dev == "cuda": torch.cuda.synchronize()
    t_step = (time.time() - t0) / 5
    ok(f"training step {t_step*1000:.0f} ms for a batch of {len(xb)}")

    # ---------------------------------------------------------------- 6. enhancement
    print("\n6. enhancement backends")
    from enhance import classical_nlm, classical_fast
    t0 = time.time(); n_out = classical_nlm(d);  t_nlm  = time.time() - t0
    t0 = time.time(); f_out = classical_fast(d); t_fast = time.time() - t0
    import numpy as _np
    for nm, out in (("nlm", n_out), ("fast", f_out)):
        if _np.isnan(out).any() or _np.isinf(out).any():
            bad(f"{nm} backend produced non-finite values - stop and report this")
            sys.exit(1)
    ok(f"nlm  (CPU, faithful)      {t_nlm:.2f} s/image")
    ok(f"fast (bilateral, {dev})   {t_fast:.3f} s/image   -> {t_nlm/max(t_fast,1e-6):.0f}x faster")
    t_enh = t_fast if dev == "cuda" else t_nlm
    info(f"evaluate.py will use '{'fast' if dev == 'cuda' else 'nlm'}' by default; override with --enhance")

    # ---------------------------------------------------------------- 7. metrics
    print("\n7. metrics")
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    ok(f"PSNR {psnr(img, d, data_range=1.0):.1f} dB, "
       f"SSIM {ssim(img, d, channel_axis=-1, data_range=1.0):.3f} at severity 0.55")

    # ---------------------------------------------------------------- estimate
    print("\n" + "=" * 62)
    n_tr_slices = len(LGG(tr, tumour_only=True))
    n_te_slices = len(LGG(te, tumour_only=True))
    steps_per_epoch = max(1, n_tr_slices // 16)
    train_25 = t_step * steps_per_epoch * 25 * (16 / max(len(xb), 1)) / 60
    n_sev = 7
    eval_fwd = t_fwd * n_te_slices * n_sev * 5 / 60
    eval_enh = t_enh * n_te_slices * n_sev / 60
    comp_run = t_fwd * n_te_slices * n_sev * 4 / 60 + t_deg * n_te_slices * n_sev * 4 / 60

    print(f"ESTIMATE FOR YOUR MACHINE   ({n_tr_slices} train / {n_te_slices} test tumour slices)")
    print("=" * 62)
    print(f"  baseline segmenter, 25 epochs .......... {train_25:5.0f} min")
    print(f"  augmented model (B2), 30 epochs ........ {train_25*1.2:5.0f} min")
    print(f"  restoration net, 20 epochs ............. {train_25*0.8:5.0f} min")
    print(f"  4 fine-tune runs (H2) .................. {train_25*0.25:5.0f} min")
    print(f"  main evaluation: model forward ......... {eval_fwd:5.0f} min")
    print(f"  main evaluation: classical enhancement . {eval_enh:5.0f} min")
    print(f"  component ablation (B1) ................ {comp_run:5.0f} min")
    total = train_25 * 3.25 + eval_fwd + eval_enh + comp_run
    print("  " + "-" * 56)
    print(f"  TOTAL COMPUTE .......................... {total:5.0f} min  ({total/60:.1f} h)")
    print("=" * 62)
    if total > 480:
        print("  Over 8 hours. Cut the restoration arm and halve the test set with")
        print("  SEVERITIES = [0, 0.3, 0.55, 0.85] in evaluate.py.")
    else:
        print("  Comfortable. Run it.")
    print(f"\nsmoke test finished in {time.time()-t_all:.0f} s - the pipeline is sound.\n")

if __name__ == "__main__":
    main()
