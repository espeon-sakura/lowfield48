"""
Turn results.csv into the three report figures + a summary table.

  python analyse.py --csv results/results.csv --outdir figures

Figures:
  fig1_dice_vs_severity.png   how each arm holds up as the scan gets worse
  fig2_decoupling.png         image-quality gain vs task gain  (the headline)
  fig3_label_efficiency.png   Dice vs number of locally labelled patients
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# validated categorical palette (light surface) - fixed order, never cycled
C = {"none": "#C2410C", "classical": "#2C6FAE", "learned": "#0E8F6F",
     "finetune": "#8B5CF6", "augmented": "#B08900", "clean": "#5A6B72"}
COMP = {"resolution": "#2C6FAE", "noise": "#C2410C", "bias": "#0E8F6F", "blur": "#8B5CF6"}
INK, MUTED, GRID, SURF = "#1A202C", "#5A6B72", "#E3E8EA", "#FCFCFB"
PROTO = {"metro 1.5T/3mm": 0.30, "district 1.5T/5mm": 0.55, "older 0.35T/6mm": 0.85}

def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)

def protocol_marks(ax):
    for label, s in PROTO.items():
        ax.axvline(s, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
        ax.text(s, ax.get_ylim()[1], " " + label, rotation=90, va="top", ha="left",
                fontsize=7.5, color=MUTED)

def fig1(df, outdir):
    d = df[df.arm != "finetune"]
    g = d.groupby(["arm", "severity"]).dice.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200, facecolor=SURF)
    for arm in ["none", "classical", "learned", "augmented"]:
        sub = g[g.arm == arm]
        if sub.empty: continue
        ax.plot(sub.severity, sub.dice, lw=2, color=C[arm], marker="o", ms=5, zorder=3)
        ax.annotate(arm, (sub.severity.iloc[-1], sub.dice.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=C[arm], fontsize=9,
                    va="center", fontweight="bold")
    ft = df[df.arm == "finetune"]
    if not ft.empty:
        best = ft.groupby("severity").dice.mean().reset_index()
        ax.plot(best.severity, best.dice, lw=2, color=C["finetune"], marker="s", ms=5, zorder=3)
        ax.annotate("fine-tuned", (best.severity.iloc[-1], best.dice.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=C["finetune"], fontsize=9,
                    va="center", fontweight="bold")
    style(ax)
    ax.set_xlabel("acquisition degradation severity", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Dice", color=MUTED, fontsize=9.5)
    ax.set_title("Segmentation accuracy as the scan degrades", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(-0.03, 1.02)
    protocol_marks(ax)
    ax.legend([], frameon=False)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig1_dice_vs_severity.png",
                                    facecolor=SURF, bbox_inches="tight"); plt.close(fig)

def fig2(df, outdir):
    """Headline: does an image-quality gain buy a task gain?"""
    base = df[df.arm == "none"].groupby(["severity", "patient"])[["dice", "psnr", "ssim"]].mean()
    rows = []
    for arm in ["classical", "learned"]:
        sub = df[df.arm == arm]
        if sub.empty: continue
        a = sub.groupby(["severity", "patient"])[["dice", "psnr", "ssim"]].mean()
        try:
            j = a.join(base, how="inner", suffixes=("", "_base")).reset_index()
        except TypeError:                       # older pandas
            j = a.join(base, how="inner", rsuffix="_base").reset_index()
        j["d_psnr"] = j.psnr - j.psnr_base
        j["d_dice"] = j.dice - j.dice_base
        j["arm"] = arm
        rows.append(j)
    if not rows:
        print("fig2 skipped - no enhancement arms"); return
    j = pd.concat(rows)
    j = j[j.severity > 0]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200, facecolor=SURF)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for arm in j.arm.unique():
        s = j[j.arm == arm]
        ax.scatter(s.d_psnr, s.d_dice, s=26, color=C[arm], alpha=0.75,
                   edgecolor=SURF, linewidth=1.2, zorder=3, label=arm)
    rho, p = spearmanr(j.d_psnr, j.d_dice)
    style(ax)
    ax.set_xlabel("change in image quality after enhancement  (ΔPSNR, dB)", color=MUTED, fontsize=9.5)
    ax.set_ylabel("change in Dice", color=MUTED, fontsize=9.5)
    ax.set_title("Does a better-looking image segment better?", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left", pad=16)
    ax.text(0, 1.02, f"Spearman ρ = {rho:.2f}   (p = {p:.1e})   n = {len(j)} patient×severity points",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts(): t.set_color(MUTED)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig2_decoupling.png",
                                    facecolor=SURF, bbox_inches="tight"); plt.close(fig)

def fig3(df, outdir):
    ft = df[df.arm == "finetune"]
    if ft.empty:
        print("fig3 skipped - no fine-tune arms"); return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200, facecolor=SURF)
    for sev in sorted(ft.severity.unique()):
        s = ft[ft.severity == sev].groupby("n_labels").dice.mean().reset_index()
        ax.plot(s.n_labels, s.dice, lw=2, marker="o", ms=5, color=C["finetune"], zorder=3)
        ax.annotate(f"severity {sev:g}", (s.n_labels.iloc[-1], s.dice.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=C["finetune"], fontsize=9, va="center", fontweight="bold")
        no_ft = df[(df.arm == "none") & (df.severity == sev)].dice.mean()
        ax.axhline(no_ft, color=C["none"], lw=1.5, ls=(0, (4, 3)), zorder=2)
        ax.annotate("no adaptation", (s.n_labels.min(), no_ft), xytext=(0, 5),
                    textcoords="offset points", color=C["none"], fontsize=8.5)
        best_enh = df[(df.arm == "classical") & (df.severity == sev)].dice.mean()
        ax.axhline(best_enh, color=C["classical"], lw=1.5, ls=(0, (1, 3)), zorder=2)
        ax.annotate("enhancement only", (s.n_labels.min(), best_enh), xytext=(0, 5),
                    textcoords="offset points", color=C["classical"], fontsize=8.5)
    style(ax)
    ax.set_xlabel("locally labelled patients used for fine-tuning", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Dice", color=MUTED, fontsize=9.5)
    ax.set_title("How many local labels buy what enhancement cannot", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig3_label_efficiency.png",
                                    facecolor=SURF, bbox_inches="tight"); plt.close(fig)


def fig4_components(outdir, csv_components):
    """B1: which physical degradation causes the damage?"""
    import os
    if not csv_components or not os.path.exists(csv_components):
        print("fig4 skipped - no component CSV"); return
    d = pd.read_csv(csv_components)
    g = d.groupby(["component", "severity"]).dice.mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=200, facecolor=SURF)
    for comp in ["resolution", "noise", "bias", "blur"]:
        sub = g[g.component == comp]
        if sub.empty: continue
        ax.plot(sub.severity, sub.dice, lw=2, marker="o", ms=5, color=COMP[comp], zorder=3)
        ax.annotate(comp, (sub.severity.iloc[-1], sub.dice.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=COMP[comp], fontsize=9,
                    va="center", fontweight="bold")
    style(ax)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("severity of the single degradation applied", color=MUTED, fontsize=9.5)
    ax.set_ylabel("Dice", color=MUTED, fontsize=9.5)
    ax.set_title("Which part of a bad scan actually breaks the model", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig4_components.png",
                                    facecolor=SURF, bbox_inches="tight"); plt.close(fig)


def fig5_confidence(df, outdir):
    """B4: does the model know it is failing?

    Three curves, all on a 0-1 axis so no second y-axis is needed:
      Dice              - how right it is
      uncertainty       - mean pixel entropy / ln2, defined for every image
      blank predictions - fraction of images where it predicted nothing at all

    Mean confidence inside the predicted mask is NOT used as the headline: when the
    model predicts nothing, that quantity is 0 by definition, so the curve would be
    measuring how often the prediction is empty rather than how sure the model is.
    """
    d = df[(df.arm == "none") & (df.component == "all")] if "component" in df else df[df.arm == "none"]
    if d.empty or "entropy" not in d:
        print("fig5 skipped"); return
    d = d.copy()
    d["uncertainty"] = d.entropy / np.log(2)
    d["blank"] = (d.pred_area == 0).astype(float)
    g = d.groupby("severity").agg(dice=("dice", "mean"),
                                  unc=("uncertainty", "mean"),
                                  blank=("blank", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.3), dpi=200, facecolor=SURF)
    series = [("dice", C["none"], "o", "Dice (how right it is)"),
              ("unc", C["classical"], "s", "uncertainty (mean entropy)"),
              ("blank", C["augmented"], "^", "predicted nothing at all")]
    for col, col_hex, mk, lab in series:
        ax.plot(g.severity, g[col], lw=2, marker=mk, ms=5, color=col_hex, zorder=3)
        ax.annotate(lab, (g.severity.iloc[-1], g[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", color=col_hex, fontsize=9,
                    va="center", fontweight="bold")
    style(ax)
    ax.set_ylim(-0.03, 1.05); ax.set_xlim(-0.03, 1.28)
    ax.set_xlabel("acquisition degradation severity", color=MUTED, fontsize=9.5)
    ax.set_ylabel("score (0-1)", color=MUTED, fontsize=9.5)
    ax.set_title("Does the model know when it is failing?", color=INK,
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig5_confidence.png",
                                    facecolor=SURF, bbox_inches="tight"); plt.close(fig)


def fig6_failure(df, outdir):
    """B5: who gets hurt, and in which direction?"""
    d = df[df.arm == "none"].copy()
    if d.empty or "gt_area" not in d:
        print("fig6 skipped"); return
    base = d[d.severity == 0].groupby("patient").gt_area.mean()
    if base.empty: return
    terc = pd.qcut(base, 3, labels=["small", "medium", "large"])
    d["size"] = d.patient.map(terc)
    # median, not mean: dividing by a tiny gt_area makes the mean explode on small lesions
    d["vol_err"] = (d.pred_area - d.gt_area) / d.gt_area.replace(0, np.nan)
    d.loc[d.gt_area < 50, "vol_err"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), dpi=200, facecolor=SURF)
    cols = {"small": "#C2410C", "medium": "#2C6FAE", "large": "#0E8F6F"}
    for sz in ["small", "medium", "large"]:
        g = d[d["size"] == sz].groupby("severity").dice.mean().reset_index()
        if g.empty: continue
        axes[0].plot(g.severity, g.dice, lw=2, marker="o", ms=5, color=cols[sz], zorder=3)
        axes[0].annotate(sz, (g.severity.iloc[-1], g.dice.iloc[-1]), xytext=(6, 0),
                         textcoords="offset points", color=cols[sz], fontsize=9,
                         va="center", fontweight="bold")
    style(axes[0]); axes[0].set_xlim(-0.03, 1.05)
    axes[0].set_xlabel("severity", color=MUTED, fontsize=9.5)
    axes[0].set_ylabel("Dice", color=MUTED, fontsize=9.5)
    axes[0].set_title("Small lesions fail first", color=INK, fontsize=12,
                      fontweight="bold", loc="left", pad=10)

    v = d.groupby("severity").vol_err.median().reset_index()
    axes[1].axhline(0, color=MUTED, lw=1, zorder=1)
    axes[1].bar(v.severity, v.vol_err, width=0.06,
                color=["#2C6FAE" if e >= 0 else "#C2410C" for e in v.vol_err], zorder=3)
    style(axes[1]); axes[1].set_xlim(-0.06, 0.95)
    axes[1].set_xlabel("severity", color=MUTED, fontsize=9.5)
    axes[1].set_ylabel("median relative volume error", color=MUTED, fontsize=9.5)
    axes[1].set_title("and the error has a direction", color=INK, fontsize=12,
                      fontweight="bold", loc="left", pad=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig6_failure_modes.png",
                                    facecolor=SURF, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--components_csv", default="results/components.csv")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.csv)
    if "component" not in df.columns:
        df["component"] = "all"
    if "ft_sev" not in df.columns:
        df["ft_sev"] = -1.0
    # for the headline figures, a fine-tuned model is only used at the severity it was tuned for
    df = df[(df.arm != "finetune") | (df.ft_sev < 0) | (abs(df.ft_sev - df.severity) < 1e-6)]
    fig1(df, a.outdir); fig2(df, a.outdir); fig3(df, a.outdir)
    fig4_components(a.outdir, a.components_csv)
    fig5_confidence(df, a.outdir); fig6_failure(df, a.outdir)
    summ = df.groupby(["arm", "n_labels", "severity"]).agg(
        dice=("dice", "mean"), dice_sd=("dice", "std"),
        psnr=("psnr", "mean"), ssim=("ssim", "mean"), n=("dice", "size")).reset_index()
    summ.to_csv(f"{a.outdir}/summary.csv", index=False)
    print(summ.to_string(index=False))
    print("\nfigures + summary.csv written to", a.outdir)

if __name__ == "__main__":
    main()
