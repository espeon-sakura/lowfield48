"""
The statistics your report needs. Reads results.csv, writes a markdown table.

    python stats.py                      # headline severity 0.55
    python stats.py --severity 0.7

Why this exists: summary.csv gives means and standard deviations, and the standard
deviations here are huge (0.24-0.36). Without paired tests you cannot tell which
differences are real. Slices within a patient are not independent, so everything
below aggregates per patient FIRST, then tests across patients.
"""
import argparse, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr


def per_patient(df, arm, sev, n_labels=None):
    d = df[(df.arm == arm) & (np.isclose(df.severity, sev))]
    if n_labels is not None:
        d = d[d.n_labels == n_labels]
    if "component" in d:
        d = d[d.component == "all"]
    return d.groupby("patient").dice.mean()


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--severity", type=float, default=0.55)
    ap.add_argument("--out", default="figures/stats.md")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    if "component" not in df:
        df["component"] = "all"
    sev = a.severity

    base = per_patient(df, "none", sev)
    n_pat = len(base)
    lines = [f"# Statistics at severity {sev:g}", "",
             f"Per-patient means, {n_pat} held-out patients. "
             f"Paired Wilcoxon signed-rank against the no-mitigation arm; "
             f"95% CI from 10,000 bootstrap resamples over patients.", "",
             "| Arm | Dice (mean) | 95% CI | vs. none | p (paired) |",
             "| --- | --- | --- | --- | --- |"]

    lo, hi = boot_ci(base.values)
    lines.append(f"| none | {base.mean():.3f} | [{lo:.3f}, {hi:.3f}] | - | - |")

    arms = [("classical", None), ("learned", None), ("augmented", None)]
    for n in sorted(df[df.arm == "finetune"].n_labels.unique()):
        arms.append(("finetune", int(n)))

    for arm, n_lab in arms:
        s = per_patient(df, arm, sev, n_lab)
        if s.empty:
            continue
        common = base.index.intersection(s.index)
        if len(common) < 5:
            continue
        b, t = base.loc[common].values, s.loc[common].values
        diff = t - b
        lo, hi = boot_ci(t)
        dlo, dhi = boot_ci(diff)
        try:
            p = wilcoxon(t, b).pvalue
        except ValueError:
            p = float("nan")
        name = arm if n_lab is None else f"finetune (n={n_lab})"
        sig = "**" if p < 0.05 else ""
        lines.append(f"| {name} | {t.mean():.3f} | [{lo:.3f}, {hi:.3f}] | "
                     f"{sig}{diff.mean():+.3f}{sig} [{dlo:+.3f}, {dhi:+.3f}] | {p:.4f} |")

    # ---- H1: does fidelity gain predict task gain? ----
    lines += ["", "## H1 - fidelity gain versus task gain", ""]
    rows = []
    for arm in ("classical", "learned"):
        sub = df[(df.arm == arm) & (df.component == "all")]
        if sub.empty:
            continue
        nb = df[(df.arm == "none") & (df.component == "all")]
        aa = sub.groupby(["severity", "patient"])[["dice", "psnr"]].mean()
        bb = nb.groupby(["severity", "patient"])[["dice", "psnr"]].mean()
        j = aa.join(bb, rsuffix="_b", how="inner").reset_index()
        j = j[j.severity > 0]
        j["d_psnr"] = j.psnr - j.psnr_b
        j["d_dice"] = j.dice - j.dice_b
        j["arm"] = arm
        rows.append(j)
    if rows:
        j = pd.concat(rows)
        rho, p = spearmanr(j.d_psnr, j.d_dice)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(j), size=(2000, len(j)))
        rhos = [spearmanr(j.d_psnr.values[i], j.d_dice.values[i]).statistic for i in idx[:500]]
        rlo, rhi = np.percentile(rhos, [2.5, 97.5])
        frac_bad = float((j.d_dice < 0).mean())
        gain = j[j.d_psnr > 2]
        frac_bad_gain = float((gain.d_dice < 0).mean()) if len(gain) else float("nan")
        lines += [f"- Spearman rho = **{rho:.2f}** [{rlo:.2f}, {rhi:.2f}], p = {p:.2e}, n = {len(j)}",
                  f"- Variance explained: about **{rho**2*100:.0f}%**",
                  f"- Enhancement made Dice **worse** in **{frac_bad*100:.0f}%** of patient-severity cases",
                  f"- Even where PSNR improved by more than 2 dB, Dice got worse in "
                  f"**{frac_bad_gain*100:.0f}%** of cases"]

    # ---- B4: the blank-prediction monitor ----
    if "pred_area" in df:
        lines += ["", "## B4 - failure signal without labels", "",
                  "| Severity | Dice | blank predictions | mean entropy |",
                  "| --- | --- | --- | --- |"]
        d = df[(df.arm == "none") & (df.component == "all")].copy()
        d["blank"] = (d.pred_area == 0).astype(float)
        g = d.groupby("severity").agg(dice=("dice", "mean"), blank=("blank", "mean"),
                                      ent=("entropy", "mean")).reset_index()
        for _, r in g.iterrows():
            lines.append(f"| {r.severity:.2f} | {r.dice:.3f} | {r.blank*100:.0f}% | {r.ent:.4f} |")
        corr = spearmanr(g.blank, g.dice).statistic
        lines.append("")
        lines.append(f"- Blank-prediction rate versus Dice: Spearman rho = **{corr:.2f}** "
                     f"- a label-free deployment monitor.")

    txt = "\n".join(lines)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt)
    print(txt)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
