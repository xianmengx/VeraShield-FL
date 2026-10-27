# Seed 7 Five-Method Client-Poisoning Comparison

Protocol: 20 clients, 60 rounds, IID CIFAR-10, full participation, 20% persistent attackers, attack starts at round 4, no server tampering. Krum uses `f=4`; Trimmed Mean uses `trim_ratio=0.4` (four values per tail).

| Attack | Method | Final accuracy | Best accuracy | Best round | Final loss | Total time (s) | Recall | FPR | Precision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sign_flip | VeraShield-FL | 0.5030 | 0.5049 | 54 | 1.5675 | 1141.0773 | 1.0000 | 0.0010 | 0.9956 |
| sign_flip | FedAvg | 0.1000 | 0.1791 | 4 | 50.3752 | 1068.7725 | N/A | N/A | N/A |
| sign_flip | Krum | 0.3854 | 0.3888 | 58 | 1.8426 | 763.7607 | N/A | N/A | N/A |
| sign_flip | Trimmed Mean | 0.4537 | 0.4545 | 56 | 1.8344 | 761.9419 | N/A | N/A | N/A |
| sign_flip | FedDMC | 0.1000 | 0.2310 | 3 | None | 657.7938 | N/A | N/A | N/A |
| random_gradient | VeraShield-FL | 0.5001 | 0.5009 | 58 | 1.5581 | 1153.0145 | 1.0000 | 0.0000 | 1.0000 |
| random_gradient | FedAvg | 0.3649 | 0.3649 | 60 | 1.7039 | 755.0313 | N/A | N/A | N/A |
| random_gradient | Krum | 0.3854 | 0.3888 | 58 | 1.8426 | 775.4918 | N/A | N/A | N/A |
| random_gradient | Trimmed Mean | 0.5065 | 0.5070 | 59 | 1.4817 | 787.1157 | N/A | N/A | N/A |
| random_gradient | FedDMC | 0.5008 | 0.5029 | 50 | 1.5271 | 794.0612 | N/A | N/A | N/A |
| scaling | VeraShield-FL | 0.4945 | 0.4945 | 60 | 1.5555 | 1314.3274 | 1.0000 | 0.0000 | 1.0000 |
| scaling | FedAvg | 0.1000 | 0.2284 | 4 | 2.3029 | 766.7687 | N/A | N/A | N/A |
| scaling | Krum | 0.3854 | 0.3888 | 58 | 1.8426 | 789.4774 | N/A | N/A | N/A |
| scaling | Trimmed Mean | 0.5233 | 0.5233 | 60 | 1.4225 | 792.7565 | N/A | N/A | N/A |
| scaling | FedDMC | 0.1000 | 0.1998 | 4 | 2.3029 | 792.9102 | N/A | N/A | N/A |
