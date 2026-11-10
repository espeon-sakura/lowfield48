"""
ONE COMMAND. Runs the whole project and tells you what it is doing.

    python run_all.py --quick     # ~15 min, proves everything works end to end
    python run_all.py             # the real run, every arm and every branch

It is resumable: a stage whose output already exists is skipped, so if something
dies at hour three you re-run the same command and it picks up. --force redoes
everything.

Stages
  1  baseline segmenter on clean data                     -> ckpt/seg_clean.pt
  2  B2: degradation-augmented model (no local labels)    -> ckpt/seg_aug.pt
  3  learned restoration net (the first thing to cut)     -> ckpt/restore_s55.pt
  4  H2: fine-tune on 5 / 10 / 20 / 40 local patients     -> ckpt/ft_s55_n*.pt
  5  main evaluation across the degradation ladder        -> results/results.csv
  6  B1: one degradation at a time                        -> results/components.csv
  7  figures and summary table                            -> figures/
"""
import argparse, os, subprocess, sys, time

PY_EXE = sys.executable


def run(cmd, label):
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")
    print("  $ " + " ".join(str(c) for c in cmd[1:]), flush=True)
    t0 = time.time()
    r = subprocess.run([PY_EXE] + cmd[1:])
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f"\n  FAILED after {dt:.1f} min: {label}")
        print("  Nothing after this will work. Fix it, then re-run the same command -")
        print("  finished stages are skipped, so you lose nothing.")
        sys.exit(r.returncode)
    print(f"  done in {dt:.1f} min")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset root (auto-detected if omitted)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny version: few epochs, two label budgets - for proving it runs")
    ap.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    ap.add_argument("--skip_restore", action="store_true", help="drop the learned enhancement arm")
    ap.add_argument("--skip_components", action="store_true", help="drop branch B1")
    ap.add_argument("--enhance", choices=["auto", "nlm", "fast"], default="auto")
    ap.add_argument("--transfer", action="store_true", help="B3: full severity transfer matrix")
    ap.add_argument("--cpu", action="store_true",
                    help="CPU-friendly preset: fewer epochs, no restoration arm (~half the time)")
    ap.add_argument("--ep_seg", type=int, default=None, help="override epochs: baseline")
    ap.add_argument("--ep_aug", type=int, default=None, help="override epochs: augmented")
    ap.add_argument("--ep_res", type=int, default=None, help="override epochs: restoration")
    ap.add_argument("--ep_ft", type=int, default=None, help="override epochs: fine-tune")
    a = ap.parse_args()

    if a.quick:
        ep_seg, ep_aug, ep_res, ep_ft = 3, 3, 2, 2
    elif a.cpu:
        ep_seg, ep_aug, ep_res, ep_ft = 12, 15, 10, 8
        a.skip_restore = True
    else:
        ep_seg, ep_aug, ep_res, ep_ft = 25, 30, 20, 15
    ep_seg = a.ep_seg or ep_seg
    ep_aug = a.ep_aug or ep_aug
    ep_res = a.ep_res or ep_res
    ep_ft  = a.ep_ft  or ep_ft
    budgets = [5, 20] if a.quick else [5, 10, 20, 40]
    root_args = ["--root", a.root] if a.root else []

    os.makedirs("ckpt", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    total = 0.0
    t_start = time.time()

    def need(path):
        return a.force or not os.path.exists(path)

    mode = "QUICK (proof it runs)" if a.quick else ("FULL RUN, CPU preset" if a.cpu else "FULL RUN")
    print(f"\nMode: {mode}   epochs: seg {ep_seg}, aug {ep_aug}, restore {ep_res}, finetune {ep_ft}")
    stale = [p for p in ("ckpt/seg_clean.pt", "results/results.csv") if os.path.exists(p)]
    if stale and not a.force:
        print("\n  !! Found output from an earlier run: " + ", ".join(stale))
        print("     Those will be REUSED, not redone. If they came from --quick, that is wrong:")
        print("     your final numbers would come from a 3-epoch model.")
        print("     Delete ckpt/ results/ figures/ first, or re-run with --force.\n")
    print("Tip: run `python smoke_test.py` first if you have not - it estimates all of this.\n")

    # 0 ------------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    if need("figures/fig0_ladder.png"):
        total += run([PY_EXE, "make_ladder_figure.py", *root_args, "--outdir", "figures"],
                     "0/7  simulator sanity figure - LOOK at this one")
    else:
        print("0/7  ladder figure - already done, skipping")

    # 1 ------------------------------------------------------------------
    if need("ckpt/seg_clean.pt"):
        total += run([PY_EXE, "train.py", *root_args, "--task", "seg", "--severity", "0",
                      "--epochs", str(ep_seg), "--out", "ckpt/seg_clean.pt"],
                     "1/7  baseline segmenter, clean data")
    else:
        print("1/7  baseline segmenter - already done, skipping")

    # 2 ------------------------------------------------------------------
    if need("ckpt/seg_aug.pt"):
        total += run([PY_EXE, "train.py", *root_args, "--task", "seg", "--augment",
                      "--epochs", str(ep_aug), "--out", "ckpt/seg_aug.pt"],
                     "2/7  B2: degradation-augmented model")
    else:
        print("2/7  augmented model - already done, skipping")

    # 3 ------------------------------------------------------------------
    restore = ""
    if not a.skip_restore:
        if need("ckpt/restore_s55.pt"):
            total += run([PY_EXE, "train.py", *root_args, "--task", "restore", "--severity", "0.55",
                          "--epochs", str(ep_res), "--out", "ckpt/restore_s55.pt"],
                         "3/7  learned restoration net")
        else:
            print("3/7  restoration net - already done, skipping")
        restore = "ckpt/restore_s55.pt"
    else:
        print("3/7  restoration net - skipped by request")

    # 4 ------------------------------------------------------------------
    ft_specs = []
    for n in budgets:
        out = f"ckpt/ft_s55_n{n}.pt"
        if need(out):
            total += run([PY_EXE, "train.py", *root_args, "--task", "seg", "--severity", "0.55",
                          "--n_patients", str(n), "--init", "ckpt/seg_clean.pt",
                          "--epochs", str(ep_ft), "--out", out],
                         f"4/7  H2: fine-tune on {n} local patients")
        else:
            print(f"4/7  fine-tune n={n} - already done, skipping")
        ft_specs.append(f"0.55:{n}:{out}")

    # 5 ------------------------------------------------------------------
    cmd = [PY_EXE, "evaluate.py", *root_args, "--ckpt_clean", "ckpt/seg_clean.pt",
           "--ckpt_aug", "ckpt/seg_aug.pt", "--ckpt_ft", ",".join(ft_specs),
           "--enhance", a.enhance, "--out", "results/results.csv"]
    if restore:
        cmd += ["--ckpt_restore", restore]
    if a.transfer:
        cmd += ["--ft_all_severities"]
    if need("results/results.csv"):
        total += run(cmd, "5/7  main evaluation across the degradation ladder")
    else:
        print("5/7  main evaluation - results.csv exists, skipping (use --force to redo)")

    # 6 ------------------------------------------------------------------
    if not a.skip_components:
        if need("results/components.csv"):
            total += run([PY_EXE, "evaluate.py", *root_args, "--components",
                          "--ckpt_clean", "ckpt/seg_clean.pt", "--out", "results/components.csv"],
                         "6/7  B1: one degradation at a time")
        else:
            print("6/7  component ablation - already done, skipping")
    else:
        print("6/7  component ablation - skipped by request")

    # 7 ------------------------------------------------------------------
    total += run([PY_EXE, "analyse.py", "--csv", "results/results.csv",
                  "--components_csv", "results/components.csv", "--outdir", "figures"],
                 "7/7  figures and summary table")

    wall = (time.time() - t_start) / 60
    print(f"\n{'=' * 70}")
    print(f"  FINISHED in {wall:.0f} min of wall time")
    print(f"{'=' * 70}")
    print("""
  Look at these, in this order:

    figures/fig0_ladder.png          do the simulated scans look like real bad scans?
    figures/fig1_dice_vs_severity    how fast accuracy falls, and which arm holds up
    figures/fig2_decoupling.png      THE HEADLINE - does better PSNR mean better Dice?
    figures/fig3_label_efficiency    how many local labels beat enhancement
    figures/fig4_components.png      which degradation does the damage (B1)
    figures/fig5_confidence.png      does the model know it is failing (B4)
    figures/fig6_failure_modes.png   who fails first, and in which direction (B5)
    figures/summary.csv              every number behind the figures

  Then read 'Reading the results' in the handbook before deciding what they mean.
""")


if __name__ == "__main__":
    main()
