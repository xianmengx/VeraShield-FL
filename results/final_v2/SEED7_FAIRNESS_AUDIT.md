# Seed-7 Five-Method Fairness Audit

## Scope

Three attacks (`sign_flip`, `random_gradient`, and `scaling`) and five methods
(VeraShield-FL, FedAvg, Krum, Trimmed Mean, and FedDMC), with 20 clients and
four fixed Byzantine clients (`[1, 8, 12, 13]`).

## Passed checks

- Every completed cell contains a 60-round `summary.json`.
- Within each attack, all five methods have the same `scenario.sha256`.
- All methods use seed 7, IID CIFAR-10, full participation, attack ratio 0.2,
  persistent attacks starting at round 4, and malicious-selection seed 42.
- FedAvg, Krum, Trimmed Mean, and FedDMC have SSDF, secret recovery, dynamic
  weighting, and Crypto-VL disabled. VeraShield-FL has the complete mechanism
  enabled and server tampering disabled.
- FedDMC uses its documented PCA + HDBSCAN + cumulative-majority default path.

## Required corrections

The first Krum and Trimmed Mean runs are pilot results and must not enter the
formal table:

- Krum used the legacy `f=2`, while the frozen scenario contains four
  Byzantine clients. Formal Krum must use `f=4`; with 20 available clients,
  its validity condition `n >= 2f + 3` is satisfied.
- The implementation defines `trim_ratio` as the total fraction removed across
  both tails (`k = int(trim_ratio * n / 2)`). Thus `trim_ratio=0.2` removed only
  two values per tail. The formal 20%-Byzantine comparison uses
  `trim_ratio=0.4`, which removes four values per tail (20% per tail).

FedAvg, FedDMC, and VeraShield-FL do not require reruns. Only the six corrected
Krum/Trimmed-Mean cells are rerun, with all other fields and scenario hashes
unchanged.
