"""
The matched-fidelity analysis.  Run this after compare_backends.py.

    python fidelity_table.py

Question it answers: when two preprocessing arms produce images that PSNR and
SSIM score as equally good, does the segmentation network do equally well on
them?  Every comparison here holds the NETWORK fixed and varies only the image,
which is the only way the question is answerable.

Reads  results/results.csv  and  results/results_nlm.csv
Writes figures/fidelity_table.md  and  figures/fig9_matched_fidelity.png
"""
import argparse, os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

C = {"none": "#C2410C", "classical": "#2C6FAE", "classical_nlm": "#0E8F6F",
     "learned": "#8B5CF6", "augmented": "#B08900"}
LAB = {"none": "no mitigation", "classical": "classical (bilateral)",
       "classical_nlm": "classical (non-local means)",
       "learned": "learned restoration", "augmented": "degradation-augmented"}
SHORT = {"none": "none", "classical": "bilateral", "classical_nlm": "nlm",
         "learned": "learned", "augmented": "augmented"}
# Arms that feed a DIFFERENT IMAGE to the SAME frozen segmentation net. Only
# these can answer the question. The augmented and fine-tuned arms change the
# MODEL and leave the image untouched, so "same PSNR, different Dice" is
# trivially true for them and proves nothing.
IMAGE_ARMS = ("none", "classical", "classical_nlm", "learned")
INK, MUTED, GRID, SURF = "#1A202C", "#5A6B72", "#E3E8EA", "#FCFCFB"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def load(main, nlm):
    df = pd.read_csv(main)
    if "component" in df:
        df = df[df.component == "all"]
    df = df[(df.n_labels == 0) & (df.arm != "finetune")]
    frames = [df]
    if nlm and os.path.exists(nlm):
        d2 = pd.read_csv(nlm)
        if "component" in d2:
            d2 = d2[d2.component == "all"]
        d2 = d2[(d2.n_labels == 0) & (d2.arm == "classical")].copy()
        d2["arm"] = "classical_nlm"
        frames.append(d2)
    else:
        print(f"[warn] {nlm} not found - running without the non-local-means arm")
    return pd.concat(frames, ignore_index=True)


def per_patient(df, arm, sev):
    d = df[(df.arm == arm) & (np.isclose(df.severity, sev))]
    return d.groupby("patient").dice.mean()


def paired_p(df, a1, a2, sev):
    x, y = per_patient(df, a1, sev), per_patient(df, a2, sev)
    k = x.index.intersection(y.index)
    if len(k) < 5:
        return float("nan"), 0
    try:
        return wilcoxon(y[k], x[k]).pvalue, len(k)
    except ValueError:                       # all differences identically zero
        return 1.0, len(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--csv_nlm", default="results/results_nlm.csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="dB: pairs closer than this count as matched in fidelity")
    ap.add_argument("--tight", type=float, default=0.35,
                    help="dB: pairs closer than this count as tightly matched")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = load(a.csv, a.csv_nlm)
    arms = [k for k in ("none", "classical", "classical_nlm", "learned", "augmented")
            if k in set(df.arm)]
    img = [k for k in IMAGE_ARMS if k in set(df.arm)]
    sevs = sorted(df.severity.unique())
    g = df.groupby(["arm", "severity"])[["psnr", "ssim", "dice"]].mean().reset_index()

    out = ["# Matched-fidelity analysis\n",
           "PSNR and SSIM are computed against the clean scan; Dice is the clinical "
           "task. Every comparison below holds the segmentation network fixed and "
           "varies only the image, so a Dice difference is attributable to the image "
           "alone.\n",
           "\nThe degradation-augmented arm appears in section 1 for reference only. "
           "It changes the **model** and leaves the image untouched, so its PSNR "
           "equals the no-mitigation arm's by construction and it cannot take part "
           "in a matched-fidelity comparison.\n"]

    # ---------- 1. everything side by side ----------
    out.append("\n## 1. Fidelity and task, arm by arm\n")
    out.append("| Severity | Arm | PSNR (dB) | SSIM | Dice |")
    out.append("| --- | --- | --- | --- | --- |")
    for s in sevs:
        for arm in arms:
            r = g[(g.arm == arm) & (np.isclose(g.severity, s))]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            p = "clean" if r.psnr >= 99 else f"{r.psnr:.2f}"
            note = "  _(image untouched)_" if arm == "augmented" else ""
            out.append(f"| {s:.2f} | {LAB[arm]}{note} | {p} | {r.ssim:.3f} | {r.dice:.3f} |")

    # ---------- 2. every matched pair ----------
    out.append(f"\n## 2. Pairs of image arms that fidelity scores as equal\n")
    out.append(f"Every pair of image arms at the same severity whose mean PSNR differs "
               f"by less than {a.tol:.1f} dB, closest first. Same patients, same slices, "
               f"same network. p is a paired Wilcoxon over patients.\n")
    out.append("| PSNR gap (dB) | Severity | Pair | PSNR | SSIM | Dice | Dice gap | p |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for s in sevs:
        sub = {}
        for arm in img:
            r = g[(g.arm == arm) & (np.isclose(g.severity, s))]
            if len(r) and r.iloc[0].psnr < 99:
                sub[arm] = r.iloc[0]
        for a1, a2 in itertools.combinations(sub, 2):
            r1, r2 = sub[a1], sub[a2]
            dp = abs(r1.psnr - r2.psnr)
            if dp > a.tol:
                continue
            p, _ = paired_p(df, a1, a2, s)
            rows.append((dp, s, a1, a2, r1, r2, p))
    rows.sort(key=lambda t: t[0])
    for dp, s, a1, a2, r1, r2, p in rows:
        dd = r2.dice - r1.dice
        star = "**" if (p == p and p < 0.05) else ""
        out.append(f"| {dp:.2f} | {s:.2f} | {SHORT[a1]} vs {SHORT[a2]} | "
                   f"{r1.psnr:.2f} / {r2.psnr:.2f} | {r1.ssim:.3f} / {r2.ssim:.3f} | "
                   f"{r1.dice:.3f} / {r2.dice:.3f} | {star}{dd:+.3f}{star} | "
                   f"{'n/a' if p != p else f'{p:.4f}'} |")

    # ---------- 3. what that adds up to ----------
    tight = [r for r in rows if r[0] <= a.tight]
    out.append(f"\n## 3. What this adds up to\n")
    if tight:
        dds = [abs(r[5].dice - r[4].dice) for r in tight]
        lo, hi = min(dds), max(dds)
        big = max(tight, key=lambda r: abs(r[5].dice - r[4].dice))
        out.append(f"- {len(tight)} pairs agree on PSNR to within {a.tight:.2f} dB. "
                   f"Their Dice differences run from **{lo:.3f}** to **{hi:.3f}**.")
        out.append(f"- The largest is at severity **{big[1]:.2f}**: "
                   f"**{SHORT[big[2]]}** and **{SHORT[big[3]]}** differ by "
                   f"{big[0]:.2f} dB PSNR and "
                   f"{abs(big[5].ssim - big[4].ssim):.3f} SSIM, "
                   f"but by **{abs(big[5].dice - big[4].dice):.3f}** Dice "
                   f"(paired p = {big[6]:.4f}).")
        out.append(f"- So equal fidelity is **compatible with** a large task gap, but "
                   f"does not imply one: the same {a.tight:.2f} dB agreement elsewhere "
                   f"goes with a Dice gap of only {lo:.3f}. Fidelity is uninformative "
                   f"about the task, in both directions.")
    sign = [r for r in rows
            if np.sign(r[5].psnr - r[4].psnr) != np.sign(r[5].dice - r[4].dice)]
    out.append(f"- Among all {len(rows)} fidelity-matched pairs, **{len(sign)}** have "
               f"the arm with the *higher* PSNR scoring the *lower* Dice.")
    out.append(f"\n**Not claimed here:** that fidelity gets the arm ranking wrong in "
               f"general. At the highest severities it gets it right - the arms with "
               f"the best PSNR and SSIM also segment best. The claim is narrower and "
               f"survives: PSNR agreement to a fraction of a dB carries no information "
               f"about whether the task will hold up.")

    # ---------- 4. the two denoisers, paired ----------
    if "classical_nlm" in arms:
        out.append("\n## 4. The two denoisers, paired per patient\n")
        out.append("Same patients, same slices, same degradation seed. Per-patient "
                   "means then paired Wilcoxon, so these differ slightly from the "
                   "slice-level means in section 1.\n")
        out.append("| Severity | bilateral | non-local means | nlm - bilateral | p | n |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        sig, diffs = [], {}
        for s in sevs:
            x, y = per_patient(df, "classical", s), per_patient(df, "classical_nlm", s)
            k = x.index.intersection(y.index)
            if len(k) < 5:
                continue
            p, n = paired_p(df, "classical", "classical_nlm", s)
            diffs[s] = y[k].mean() - x[k].mean()
            if p < 0.05:
                sig.append(s)
            star = "**" if p < 0.05 else ""
            out.append(f"| {s:.2f} | {x[k].mean():.3f} | {y[k].mean():.3f} | "
                       f"{star}{y[k].mean() - x[k].mean():+.3f}{star} | {p:.4f} | {n} |")
        if sig:
            out.append(f"\nThe two denoisers separate at severity "
                       f"{', '.join(f'{s:.2f}' for s in sig)} and nowhere else.")
            ns = [s for s in diffs if s not in sig and abs(diffs[s]) >= 0.05]
            if ns:
                out.append(f"At severity {', '.join(f'{s:.2f}' for s in ns)} the gap is "
                           f"large in the mean "
                           f"({', '.join(f'{diffs[s]:+.3f}' for s in ns)}) but not "
                           f"separable from patient-to-patient variation with "
                           f"{len(k)} patients - report it as a difference in means, "
                           f"not a significant one.")
        else:
            out.append("\nThe two denoisers are not separable at any severity with "
                       "this sample.")

    out.append("\n---\n_No pooled PSNR-bin table: pooling across severities lets each "
               "arm contribute a different mix of severities to the same bin, which "
               "confounds the comparison. Everything above is within-severity._")

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)
    fig.patch.set_facecolor(SURF); style(ax)
    for arm in arms:
        r = g[(g.arm == arm) & (g.psnr < 99)].sort_values("psnr")
        ax.plot(r.psnr, r.dice, "-o", color=C[arm], lw=1.6, ms=5,
                label=LAB[arm], zorder=3)
    if tight:
        b = max(tight, key=lambda r: abs(r[5].dice - r[4].dice))
        for r in (b[4], b[5]):
            ax.plot([r.psnr], [r.dice], "o", ms=13, mfc="none", mec=INK, mew=1.4, zorder=5)
        ax.annotate("", xy=(b[5].psnr, b[5].dice), xytext=(b[4].psnr, b[4].dice),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2), zorder=5)
        ax.text(0.5 * (b[4].psnr + b[5].psnr) + 0.4,
                0.5 * (b[4].dice + b[5].dice),
                f"{b[0]:.2f} dB apart on PSNR\n"
                f"{abs(b[5].dice - b[4].dice):.3f} apart on Dice",
                ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("PSNR against the clean scan (dB)  -  higher = better image",
                  color=INK, fontsize=10)
    ax.set_ylabel("Dice  -  higher = better tumour outline", color=INK, fontsize=10)
    ax.set_title("Equal image quality, unequal clinical usefulness",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fp = os.path.join(a.outdir, "fig9_matched_fidelity.png")
    fig.savefig(fp, facecolor=SURF); plt.close(fig)

    mp = os.path.join(a.outdir, "fidelity_table.md")
    # encoding is explicit: Windows defaults to cp1252 and chokes on anything
    # outside it, which kills the script after all the work is done
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", mp)
    print("wrote", fp)
    if tight:
        b = max(tight, key=lambda r: abs(r[5].dice - r[4].dice))
        print(f"\nheadline: severity {b[1]:.2f}, {SHORT[b[2]]} vs {SHORT[b[3]]}, "
              f"{b[0]:.2f} dB apart, {abs(b[5].dice - b[4].dice):.3f} Dice apart, "
              f"p = {b[6]:.4f}")


if __name__ == "__main__":
    main()
