# IID 20% Attack Matrix: Final Analysis

Status: complete. The three attacks, five methods, and three seeds (7, 21,
42) contain 45/45 completed runs. Each run has 60 rounds and 20 clients.
Values below are final CIFAR-10 test accuracy in percent, reported as
mean +/- sample standard deviation across seeds. Baselines use the fixed
settings Krum `f=2` and Trimmed Mean `trim_ratio=0.2`. The three scaling
VeraShield-FL runs use the current `unified_v2_candidate` and `precommit`
profile; the other selected runs remain unchanged.

## Final Accuracy

| Attack | VeraShield-FL | FedAvg | Krum (`f=2`) | Trimmed Mean (`0.2`) | FedDMC |
|---|---:|---:|---:|---:|---:|
| Sign-flip | **49.81 +/- 0.68** | 10.00 +/- 0.00 | 34.86 +/- 0.95 | 10.01 +/- 0.01 | 10.00 +/- 0.00 |
| Random-gradient | **50.44 +/- 0.38** | 37.23 +/- 1.00 | 38.69 +/- 0.93 | 42.76 +/- 0.67 | 48.55 +/- 1.40 |
| Scaling | **50.00 +/- 0.87** | 10.00 +/- 0.00 | 43.36 +/- 1.83 | 49.87 +/- 1.35 | 15.82 +/- 10.08 |

Per-seed final accuracies, ordered as seeds 7 / 21 / 42:

| Attack | VeraShield-FL | FedAvg | Krum | Trimmed Mean | FedDMC |
|---|---|---|---|---|---|
| Sign-flip | 50.30 / 49.04 / 50.10 | 10.00 / 10.00 / 10.00 | 34.14 / 35.94 / 34.51 | 10.02 / 10.00 / 10.00 | 10.00 / 10.00 / 10.00 |
| Random-gradient | 50.01 / 50.61 / 50.71 | 36.49 / 38.37 / 36.83 | 37.81 / 38.61 / 39.66 | 42.67 / 43.47 / 42.15 | 50.08 / 47.33 / 48.24 |
| Scaling | 49.45 / 51.00 / 49.54 | 10.00 / 10.00 / 10.00 | 44.84 / 41.31 / 43.92 | 48.41 / 50.11 / 51.08 | 10.00 / 27.45 / 10.00 |

## Detection And Protocol Checks

- VeraShield-FL active-attack recall is 1.00 in all nine attack/seed runs.
- Active FPR is 0-1.44%. The maximum is sign-flip, seed 21 (TP=228,
  FP=14, FN=0, TN=958). The other runs have at most one false positive.
- Crypto-VL verification has no recorded failures in the applicable rounds;
  aggregation diagnostics show no empty accepted set or failed aggregation.
- Client dropout is disabled. These experiments therefore do not test secret
  recovery under dropout, and they do not include malicious-server behavior.

## Interpretation

- **Sign-flip:** VeraShield-FL is consistently strong: its mean final accuracy
  exceeds Krum by 14.95 percentage points and the other three baselines by
  about 39.8 points. FedAvg, Trimmed Mean, and FedDMC finish at the 10% chance
  level in all seeds; Krum retains partial utility but is substantially lower.
- **Random-gradient:** VeraShield-FL is stable and has the highest three-seed
  mean, 50.44%. Its advantage over FedDMC is modest (1.89 points), and seed 7
  slightly favors FedDMC. The result supports a qualified advantage, not a
  claim of large separation from every robust baseline.
- **Scaling:** VeraShield-FL reaches a three-seed mean of 50.00%, compared
  with 49.87% for Trimmed Mean. The per-seed results vary: VeraShield-FL is
  higher for seeds 7 and 21, while Trimmed Mean is higher for seed 42.

These are three-seed results for IID CIFAR-10, 20 clients, 20% persistent
malicious clients, attacks from round 4, and the tested attack strengths
(sign-flip 5, random-gradient 10, scaling 10). The scaling VeraShield-FL runs
use the current detector and precommit profile. They support a clear advantage
for sign-flip and a smaller advantage for random-gradient, but not uniform
superiority across attacks. They do not establish performance under non-IID
data, other attack strengths, dropout, or server tampering.
