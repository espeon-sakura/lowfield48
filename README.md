# Clean the Scan, or Teach the Model?

A task-based evaluation of mitigation strategies for degraded brain-tumour MRI.

Segmentation models are trained on research-grade MRI and deployed on whatever scanner a
hospital owns. There are two standard responses — clean up the image, or adapt the model on
local data. This project tests both under controlled conditions, and asks a second question
underneath: **can image-quality metrics tell you which one worked?**

Not reliably. That is the result.

---

## What was found

| | Result |
|---|---|
| **Degradation** | Dice falls 0.664 → 0.263 from research-grade to the worst simulated tier |
| **Cause** | Noise alone reproduces nearly all of it (0.24); resolution, bias field and blur applied alone each leave Dice near 0.65 |
| **Fidelity vs task** | Spearman ρ = 0.43 between ΔPSNR and ΔDice (≈19% of variance). Enhancement lowered Dice in 42% of cases — and in 29% even when PSNR rose by more than 2 dB |
| **Matched pair** | Two images 0.25 dB apart on PSNR and 0.002 on SSIM differ by **0.144 Dice** (paired p = 0.011). Control: pairs matched to 0.3 dB at mild severity differ by ≤ 0.023 |
| **What works** | Fine-tuning on 20 locally labelled patients: **+0.176 Dice, p = 0.0003** — the only arm clearing significance comfortably. Best enhancement: +0.128, p = 0.057 |
| **Silent failure** | Blank-prediction rate tracks Dice at ρ = −0.99, while mean predictive entropy *falls* from 0.0113 to 0.0038 as accuracy collapses |
| **Backend dependence** | The same pipeline with a different denoiser flips the sign: at the worst tier, bilateral −0.144 against non-local means +0.219 |
| **Hidden cost** | The degradation-augmented model predicts nothing on 23.3% of tumour slices **even on clean scans** |

---

## The degradation model

One severity parameter `s ∈ [0,1]` scales four physical effects, applied in this order:

| Operation | Stands for | Strength |
|---|---|---|
| k-space truncation | Lower spatial resolution | keep `1 − 0.65s` of k-space |
| Gaussian blur | Thicker slices, partial volume | `σ = 1.2s` px |
| Multiplicative smooth field | B₁ coil inhomogeneity | `± 0.30s` |
| Rician noise | Lower signal-to-noise ratio | `σ = 0.10s` |

The order matters: resolution and partial-volume effects belong to the acquisition, coil
shading multiplies the signal that results, thermal noise enters last at the receiver.

**Rician, not Gaussian.** MR signal is complex and the displayed image is its magnitude, so
noise enters as `√((I + n₁)² + n₂²)`. A magnitude cannot be negative, so dark regions lift
towards grey rather than going speckled. Verified numerically — background mean came out at
0.1251 against the 0.1253 the Rayleigh distribution predicts.

**The four scaling constants and the four severity tiers were chosen, not calibrated**
against measured acquisition parameters. See Limitations.

---

## The five arms

| Arm | Changes | Implementation |
|---|---|---|
| `none` | nothing | baseline |
| `classical` | image | non-local means (5×5 patches, 6 px search) **or** bilateral (5×5, σ_spatial 1.5, σ_range 0.08), then an unsharp mask (radius 2, amount 1.0) |
| `learned` | image | second U-Net, 3→3 channels, L1 pixel loss, trained at s = 0.55 |
| `augmented` | model | segmenter retrained with a random severity in [0, 0.85] per sample |
| `fine-tune` | model | clean weights, then continued training on 5/10/20/40 local patients at s = 0.55 |

The segmenter is **trained once on clean scans and frozen**. All three image-side arms load
identical weights, so any Dice difference between them is attributable to the pixels alone.
That is what makes the matched-fidelity comparison meaningful — and why the two model-side
arms are excluded from it.

---

## Data

[LGG Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
(Buda, Saha & Mazurowski, 2019), sourced from TCGA-LGG via The Cancer Imaging Archive.
110 patients, 256×256, three channels (pre-contrast T1, FLAIR, post-contrast T1), with
manual FLAIR-abnormality masks verified by a board-certified radiologist.

Split **70/15/15 by patient** (seed 0) → 77 / 16 / 17. Evaluation uses the 202
tumour-bearing slices belonging to the 17 held-out patients. Splitting by slice instead
would place near-duplicate images on both sides and inflate every number reported here.

---

## Reproducing

```bash
pip install torch scikit-image pandas matplotlib scipy
# download the Kaggle dataset and unzip it anywhere — the scripts locate it themselves

python smoke_test.py          # ~5 min: verifies the install and times your machine
python make_ladder_figure.py  # look at the simulated scans before trusting any number
python run_all.py             # full pipeline, resumable, seven stages
```

Then the analysis:

```bash
python analyse.py             # figures 1-6
python stats.py               # paired tests, bootstrap CIs  -> figures/stats.md
python fidelity_table.py      # matched-fidelity analysis    -> figures/fidelity_table.md
python compare_backends.py    # denoiser robustness check    -> figures/backend_check.md
python qualitative.py         # contour overlays
```

You do not need to pass `--root`. `utils.find_data_root()` searches the usual locations and
prints what it picked; pass `--root` only to override.

**Determinism.** The split seed is 0 and training uses seed 0. The per-image degradation
seed derives from the filename, so every arm sees identical degraded images within a run.
Fix `PYTHONHASHSEED` to reproduce exact values across separate runs.

---

## Modules

| File | Purpose |
|---|---|
| `degrade.py` | The four degradation operations and the severity ladder |
| `data.py` | Dataset loading and patient-level splitting |
| `unet.py` | Segmentation and restoration network (4 levels, 32 base channels, ~2M params) |
| `train.py` | Training for both the segmentation and restoration tasks |
| `enhance.py` | Classical enhancement backends and the learned restoration wrapper |
| `evaluate.py` | Evaluation across arms and severities; writes `results.csv` |
| `analyse.py` | Figures 1–6 |
| `stats.py` | Per-patient aggregation, paired Wilcoxon, bootstrap intervals |
| `fidelity_table.py` | Matched-fidelity analysis and the PSNR-against-Dice figure |
| `compare_backends.py` | Denoiser robustness check |
| `qualitative.py` | Contour overlay figures |
| `run_all.py` | Full pipeline, resumable |
| `smoke_test.py` | Environment check and runtime estimate |

### Things already handled

| Classic time-waster | What the code does |
|---|---|
| Wrong dataset path | auto-detected, with a clear message naming the expected layout |
| `num_workers` crashing on Windows | `utils.n_workers()` returns 0 on Windows |
| CPU-only torch on a GPU machine | `pick_device()` detects the driver and says so |
| Slow CPU denoising | two backends; the bilateral one runs in torch on the GPU |
| `estimate_sigma` needing PyWavelets | falls back to a median-absolute-deviation estimate |
| `unsharp_mask` returning ~1e300 on some skimage versions | unsharp implemented here, three lines, version-proof |
| PSNR = inf on identical images | `safe_psnr()` caps at 100 dB so means stay finite |

---

## Limitations

- Degradation is **simulated, not acquired**. No image here came from a low-field scanner.
- The severity axis is a monotone stress parameter, not a physically achievable protocol
  change — a real thicker slice costs resolution but *gains* SNR, while this worsens
  everything together.
- The four scaling constants and the four tier values were chosen by hand.
- 17 test patients, one dataset, 2D slices rather than volumes.
- Single training seed for most arms. The 40-patient fine-tune falls below the 20-patient
  one, which is almost certainly run-to-run variation.
- The restoration network is trained at s = 0.55 and principally evaluated there, which
  flatters it.
- Neither matched-pair result survives Bonferroni across nine comparisons; the claim rests
  on effect size and consistency of direction.
- Clean baseline Dice of 0.664 is below published pipelines on this dataset. A small untuned
  2D U-Net was chosen deliberately as a stable instrument, not as a competitive result.
- The severity transfer matrix (every fine-tuned checkpoint at every severity) is
  implemented but was not run.

Not a clinical tool. No regulatory claim. Not validated for patient care.

---

## Citation

Dataset — Buda, M., Saha, A. and Mazurowski, M.A. (2019). *Association of genomic subtypes
of lower-grade gliomas with shape features automatically extracted by a deep learning
algorithm.* Computers in Biology and Medicine, 109, 218–225.

Architecture — Ronneberger, O., Fischer, P. and Brox, T. (2015). *U-Net: Convolutional
Networks for Biomedical Image Segmentation.* MICCAI 2015, 234–241.

---

Bharathi Sankar R · Department of Medical Science and Technology, IIT Madras
AI in Medicine course project, 2026
