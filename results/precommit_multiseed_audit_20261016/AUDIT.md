# Corrected precommit VeraShield-FL multi-seed audit

Status: PASS WITH LIMITATIONS. Read-only audit; raw training outputs were not modified.

|Dataset|Seed|Final accuracy|Tamper rejection|Recovery|Recall|FPR|Precision|
|---|---:|---:|---|---:|---:|---:|---:|
|CIFAR-10|7|46.07%|[20, 40, 59]|240/240|100.00%|0.44%|98.28%|
|CIFAR-10|21|48.07%|[20, 40, 59]|240/240|93.86%|0.00%|100.00%|
|CIFAR-10|42|47.79%|[20, 40, 59]|240/240|100.00%|3.95%|86.36%|
|MNIST|7|98.65%|[20, 40, 59]|240/240|99.56%|6.14%|80.21%|
|MNIST|21|98.60%|[20, 40, 59]|240/240|95.61%|4.82%|83.21%|
|MNIST|42|98.52%|[20, 40, 59]|240/240|89.04%|2.30%|90.62%|

## Three-seed summary

|Dataset|Metric|Mean|Sample SD|
|---|---|---:|---:|
|CIFAR-10|final_accuracy|0.473100|0.010830|
|CIFAR-10|best_accuracy|0.476433|0.012815|
|CIFAR-10|recall|0.979532|0.035451|
|CIFAR-10|fpr|0.014620|0.021635|
|CIFAR-10|precision|0.948798|0.074255|
|CIFAR-10|recovery_rate|1.000000|0.000000|
|MNIST|final_accuracy|0.985900|0.000656|
|MNIST|best_accuracy|0.986100|0.000700|
|MNIST|recall|0.947368|0.053177|
|MNIST|fpr|0.044225|0.019502|
|MNIST|precision|0.846810|0.053609|
|MNIST|recovery_rate|1.000000|0.000000|

## Interpretation boundaries

- All six runs use the corrected precommit logic, logical dual-server sequential simulation, and post-share logical dropout.
- Each dataset has three seeds, but no statistical significance test is claimed.
- The four baseline methods are not rerun in this corrected batch; baseline comparisons must use their matching existing records and protocol audit.
- These results support complete-system utility and the tested rejection/recovery behavior, not physical two-server privacy or detector statistical correctness.
- Seed 7 and seed 21/42 may have different recovery-record counts in older runs; report each run from its own raw files before pooling mechanism totals.
