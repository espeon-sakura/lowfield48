# Matched-fidelity analysis

Fidelity metrics (PSNR, SSIM) are computed against the clean scan. 
Dice is the clinical task. If fidelity were a good proxy for the task, 
then two images with the same PSNR, fed to the same network, would score the same Dice.


Sections 2-4 use only the arms that change the **image** and share one frozen segmentation net (none, bilateral, nlm, learned). The degradation-augmented arm is excluded from them: it changes the **model** and leaves the image untouched, so its PSNR is identical to the no-mitigation arm by construction and a Dice difference at equal PSNR would prove nothing.


## 1. Fidelity and task, arm by arm

| Severity | Arm | PSNR (dB) | SSIM | Dice |
| --- | --- | --- | --- | --- |
| 0.00 | no mitigation | clean | 1.000 | 0.664 |
| 0.00 | classical (bilateral) | 33.06 | 0.968 | 0.631 |
| 0.00 | classical (non-local means) | 31.50 | 0.953 | 0.597 |
| 0.00 | learned restoration | 32.57 | 0.725 | 0.589 |
| 0.00 | degradation-augmented  _(image untouched)_ | clean | 1.000 | 0.461 |
| 0.15 | no mitigation | 35.78 | 0.771 | 0.656 |
| 0.15 | classical (bilateral) | 31.82 | 0.815 | 0.628 |
| 0.15 | classical (non-local means) | 30.81 | 0.804 | 0.616 |
| 0.15 | learned restoration | 32.43 | 0.698 | 0.592 |
| 0.15 | degradation-augmented  _(image untouched)_ | 35.78 | 0.771 | 0.462 |
| 0.30 | no mitigation | 29.37 | 0.538 | 0.601 |
| 0.30 | classical (bilateral) | 29.07 | 0.650 | 0.610 |
| 0.30 | classical (non-local means) | 29.24 | 0.669 | 0.624 |
| 0.30 | learned restoration | 32.83 | 0.716 | 0.603 |
| 0.30 | degradation-augmented  _(image untouched)_ | 29.37 | 0.538 | 0.464 |
| 0.45 | no mitigation | 25.53 | 0.385 | 0.503 |
| 0.45 | classical (bilateral) | 26.24 | 0.493 | 0.539 |
| 0.45 | classical (non-local means) | 27.25 | 0.579 | 0.632 |
| 0.45 | learned restoration | 32.92 | 0.735 | 0.616 |
| 0.45 | degradation-augmented  _(image untouched)_ | 25.53 | 0.385 | 0.465 |
| 0.55 | no mitigation | 23.66 | 0.315 | 0.441 |
| 0.55 | classical (bilateral) | 24.29 | 0.392 | 0.446 |
| 0.55 | classical (non-local means) | 25.68 | 0.531 | 0.609 |
| 0.55 | learned restoration | 32.86 | 0.850 | 0.617 |
| 0.55 | degradation-augmented  _(image untouched)_ | 23.66 | 0.315 | 0.463 |
| 0.70 | no mitigation | 21.45 | 0.239 | 0.346 |
| 0.70 | classical (bilateral) | 21.66 | 0.268 | 0.288 |
| 0.70 | classical (non-local means) | 23.53 | 0.471 | 0.533 |
| 0.70 | learned restoration | 27.15 | 0.550 | 0.586 |
| 0.70 | degradation-augmented  _(image untouched)_ | 21.45 | 0.239 | 0.453 |
| 0.85 | no mitigation | 19.68 | 0.186 | 0.263 |
| 0.85 | classical (bilateral) | 19.44 | 0.184 | 0.119 |
| 0.85 | classical (non-local means) | 21.72 | 0.421 | 0.481 |
| 0.85 | learned restoration | 23.10 | 0.427 | 0.525 |
| 0.85 | degradation-augmented  _(image untouched)_ | 19.68 | 0.186 | 0.444 |

## 2. Image arms within 1.5 dB of each other

Each row groups the image arms whose mean PSNR agrees to within the band, at one severity. Same network, same patients, same slices - fidelity says these images are equally good.

| Severity | PSNR range (dB) | arms in band | Dice range | Dice spread |
| --- | --- | --- | --- | --- |
| 0.00 | 32.57 - 33.06 | bilateral, learned | 0.589 - 0.631 | **0.042** |
| 0.15 | 31.82 - 32.43 | bilateral, learned | 0.592 - 0.628 | **0.035** |
| 0.30 | 29.07 - 29.37 | bilateral, nlm, none | 0.601 - 0.624 | **0.023** |
| 0.45 | 26.24 - 27.25 | bilateral, nlm | 0.539 - 0.632 | **0.094** |
| 0.55 | 24.29 - 25.68 | bilateral, nlm | 0.446 - 0.609 | **0.163** |

Largest disagreement: at severity **0.55**, arms whose PSNR agrees to within 1.38 dB score Dice **0.446** to **0.609** - a spread of **0.163**.

For scale, the whole effect this study measures - Dice lost from the cleanest scan to the worst with no mitigation - is **0.401**. So at fixed image fidelity, the choice of preprocessing moves the task metric less than the degradation itself does.

## 3. Pooled: same PSNR band, different image arm

Image arms and severities pooled, then binned by PSNR. Within a bin every slice is of equal measured quality, whatever produced it, and every one is read by the same frozen network.

| PSNR bin (dB) | n slices | arms | Dice per arm | spread |
| --- | --- | --- | --- | --- |
| 18 - 20 | 157 | 2 | bilateral 0.183, none 0.314 | **0.131** |
| 20 - 21 | 250 | 2 | bilateral 0.018, none 0.255 | **0.237** |
| 21 - 22 | 585 | 3 | bilateral 0.288, nlm 0.485, none 0.345 | **0.197** |
| 22 - 24 | 600 | 4 | bilateral 0.635, nlm 0.528, learned 0.527, none 0.413 | **0.222** |
| 24 - 26 | 383 | 3 | bilateral 0.408, nlm 0.563, none 0.498 | **0.154** |
| 26 - 27 | 545 | 4 | bilateral 0.517, nlm 0.623, learned 0.611, none 0.544 | **0.106** |
| 27 - 28 | 384 | 3 | bilateral 0.638, nlm 0.637, learned 0.564 | **0.074** |
| 28 - 30 | 590 | 3 | bilateral 0.629, nlm 0.623, none 0.602 | **0.027** |
| 30 - 32 | 515 | 3 | bilateral 0.607, nlm 0.640, learned 0.609 | **0.033** |
| 32 - 33 | 615 | 3 | bilateral 0.669, nlm 0.687, learned 0.622 | **0.066** |
| 33 - 34 | 373 | 3 | bilateral 0.704, nlm 0.601, learned 0.644 | **0.103** |
| 34 - 36 | 284 | 3 | bilateral 0.639, learned 0.430, none 0.642 | **0.211** |

## 4. The two denoisers, paired per patient

Same patients, same slices, same degradation seed. Paired Wilcoxon.

| Severity | bilateral | non-local means | nlm - bilateral | p |
| --- | --- | --- | --- | --- |
| 0.00 | 0.617 | 0.604 | -0.013 | 0.6777 |
| 0.15 | 0.615 | 0.609 | -0.006 | 0.8900 |
| 0.30 | 0.603 | 0.607 | +0.005 | 0.7467 |
| 0.45 | 0.552 | 0.604 | +0.052 | 0.4586 |
| 0.55 | 0.481 | 0.572 | +0.091 | 0.3289 |
| 0.70 | 0.306 | 0.487 | **+0.181** | 0.0129 |
| 0.85 | 0.114 | 0.435 | **+0.321** | 0.0011 |

n = 17 patients.
