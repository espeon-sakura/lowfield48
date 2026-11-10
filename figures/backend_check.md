# Backend robustness check

Classical enhancement with two different denoisers, same unsharp step.
`fast` = bilateral filter (GPU). `nlm` = non-local means (CPU, the faithful one).

| Severity | no mitigation | classical (fast) | classical (nlm) | fast - none | nlm - none |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.664 | 0.631 | 0.597 | -0.033 | -0.066 |
| 0.15 | 0.656 | 0.628 | 0.616 | -0.029 | -0.040 |
| 0.30 | 0.601 | 0.610 | 0.624 | +0.009 | +0.023 |
| 0.45 | 0.503 | 0.539 | 0.632 | +0.036 | +0.129 |
| 0.55 | 0.441 | 0.446 | 0.609 | +0.005 | +0.168 |
| 0.70 | 0.346 | 0.288 | 0.533 | -0.058 | +0.188 |
| 0.85 | 0.263 | 0.119 | 0.481 | -0.144 | +0.219 |

- The two backends agree on the **direction** of the effect at **5/7** severity levels.
- Conclusion: the two backends disagree somewhere - say where, and say which one the headline numbers come from.