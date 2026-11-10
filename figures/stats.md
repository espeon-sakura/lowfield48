# Statistics at severity 0.55

Per-patient means, 17 held-out patients. Paired Wilcoxon signed-rank against the no-mitigation arm; 95% CI from 10,000 bootstrap resamples over patients.

| Arm | Dice (mean) | 95% CI | vs. none | p (paired) |
| --- | --- | --- | --- | --- |
| none | 0.474 | [0.338, 0.601] | - | - |
| classical | 0.481 | [0.345, 0.608] | +0.007 [-0.022, +0.035] | 0.2435 |
| learned | 0.602 | [0.517, 0.684] | +0.128 [+0.011, +0.242] | 0.0569 |
| augmented | 0.476 | [0.358, 0.585] | +0.002 [-0.104, +0.103] | 0.6777 |
| finetune (n=5) | 0.508 | [0.382, 0.622] | +0.034 [-0.040, +0.103] | 0.1594 |
| finetune (n=10) | 0.554 | [0.439, 0.657] | **+0.080** [+0.007, +0.152] | 0.0348 |
| finetune (n=20) | 0.650 | [0.567, 0.723] | **+0.176** [+0.103, +0.254] | 0.0003 |
| finetune (n=40) | 0.572 | [0.498, 0.642] | +0.098 [-0.026, +0.227] | 0.1202 |

## H1 - fidelity gain versus task gain

- Spearman rho = **0.43** [0.30, 0.56], p = 1.22e-10, n = 204
- Variance explained: about **19%**
- Enhancement made Dice **worse** in **42%** of patient-severity cases
- Even where PSNR improved by more than 2 dB, Dice got worse in **29%** of cases

## B4 - failure signal without labels

| Severity | Dice | blank predictions | mean entropy |
| --- | --- | --- | --- |
| 0.00 | 0.664 | 1% | 0.0113 |
| 0.15 | 0.656 | 1% | 0.0107 |
| 0.30 | 0.601 | 3% | 0.0091 |
| 0.45 | 0.503 | 7% | 0.0080 |
| 0.55 | 0.441 | 19% | 0.0072 |
| 0.70 | 0.346 | 34% | 0.0054 |
| 0.85 | 0.263 | 54% | 0.0038 |

- Blank-prediction rate versus Dice: Spearman rho = **-0.99** - a label-free deployment monitor.