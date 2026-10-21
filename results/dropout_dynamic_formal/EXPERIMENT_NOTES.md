# Dynamic-threshold dropout-recovery experiment

## Design

- Dataset/model: CIFAR-10 and the existing project CNN.
- Clients: 20; full participation; IID partitioning; one local epoch.
- Seeds: 7, 21, and 42.
- Attacks, Dynamic Weight, and VL: disabled to isolate dropout recovery.
- Dropout timing: logical `post_share`; the unavailable clients completed share
  provisioning before becoming unavailable.
- Availability stages: 20, 16, 14, and 12 clients over four rounds, corresponding
  to 0, 4, 6, and 8 unavailable clients.
- Dynamic policy: `Normal/A/16` for availability at least 16; `Emergency/B/12`
  for availability from 12 to 15.
- Paired baseline: static `A/16`, with the same seed, clients, unavailable-client
  manifest, and training parameters.

## Audited results

- Dynamic policy: 54/54 dropped-client updates recovered across three seeds.
- Static A/16: 12/54 recovered overall; 0/42 recovered when availability was 14
  or 12 clients.
- Dynamic policy switched exactly once per seed, when availability changed from
  16 to 14 clients.
- Maximum reconstruction error among successful recoveries: 7.62939453125e-06.
- Emergency B-share redistribution events: 0; B was selected from the
  pre-provisioned share set.
- Mean dynamic recovery time was 51.83 s at 4 dropped clients, 81.00 s at 6,
  and 111.99 s at 8. These are sequential logical-simulation timings, not
  distributed-network latency.
- At the separate 11-client boundary, the controller entered `failure` mode and
  rejected all 9 attempted recoveries, as required below the emergency threshold.

Accuracy is retained only as an auxiliary training-continuity measurement. It is
not used to claim that the threshold controller improves model utility, because
the short paired runs show runtime-order variability despite identical seeds.

The aggregate and per-round machine-readable results are in
`aggregate_summary.csv` and `round_summary.csv`.
