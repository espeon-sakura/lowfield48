"""
Does the enhancement conclusion depend on which denoiser we used?

    python evaluate.py --ckpt_clean ckpt/seg_clean.pt --enhance nlm --out results/results_nlm.csv
    python compare_backends.py

The main run used the GPU bilateral filter ('fast'). The faithful CPU method is
non-local means ('nlm'). Both are "denoise, then sharpen" with the same unsharp step,
so if the classical arm behaves the same way under both, the finding is about
enhancement rather than about our choice of filter. That sentence belongs in the report.
"""
import argparse, numpy as np, pandas as pd


def arm_by_severity(csv, arm):
    d = pd.read_csv(csv)
    if "component" in d:
        d = d[d.component == "all"]
    d = d[d.arm == arm]
    return d.groupby("severity").dice.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", default="results/results.csv")
    ap.add_argument("--nlm", default="results/results_nlm.csv")
    ap.add_argument("--out", default="figures/backend_check.md")
    a = ap.parse_args()

    none_f = arm_by_severity(a.fast, "none")
    cls_f = arm_by_severity(a.fast, "classical")
    cls_n = arm_by_severity(a.nlm, "classical")

    lines = ["# Backend robustness check", "",
             "Classical enhancement with two different denoisers, same unsharp step.",
             "`fast` = bilateral filter (GPU). `nlm` = non-local means (CPU, the faithful one).", "",
             "| Severity | no mitigation | classical (fast) | classical (nlm) | fast - none | nlm - none |",
             "| --- | --- | --- | --- | --- | --- |"]
    agree = 0
    total = 0
    for s in sorted(set(none_f.index) & set(cls_f.index) & set(cls_n.index)):
        n, f, m = none_f[s], cls_f[s], cls_n[s]
        df_, dm = f - n, m - n
        total += 1
        if np.sign(df_) == np.sign(dm) or (abs(df_) < 0.01 and abs(dm) < 0.01):
            agree += 1
        lines.append(f"| {s:.2f} | {n:.3f} | {f:.3f} | {m:.3f} | {df_:+.3f} | {dm:+.3f} |")

    lines += ["", f"- The two backends agree on the **direction** of the effect at "
                  f"**{agree}/{total}** severity levels."]
    if agree == total:
        lines.append("- Conclusion: the finding is about enhancement, not about which filter we chose.")
    else:
        lines.append("- Conclusion: the two backends disagree somewhere - say where, and say which "
                     "one the headline numbers come from.")

    txt = "\n".join(lines)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(txt)
    print(txt)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
