# Five-method joint-threat main experiment

Status: seed 7 five-method pilot completed; statistics reviewed. Seeds 21/42 not started.

## Fixed setting

- CIFAR-10, existing Net backbone, 20 clients, 60 rounds, all clients scheduled.
- Main partition: Dirichlet alpha=0.5. Seeds: 7 first, then 21/42 following audit.
- Local epochs=1, batch size=32, learning rate=0.1, cosine schedule, minimum LR=0.001.
- Client poisoning: persistent sign_flip, strength=5, four malicious clients (20%), from round 4.
- Malicious selection seed=42, identical malicious IDs and partitions across methods.
- Server tampering: unchanged existing after-aggregation fault injection at rounds 20/40/59.
  Same random noise generator seed=[training_seed, round, 77], per-coordinate
  noise SD=0.1*(honest aggregate delta L2 norm + 1e-9). This is the existing
  coordinate-noise rule, not an L2-normalized relative-noise specification.
- Dropout: logical post_share unavailability. Rounds 1-20: no unavailable IDs;
  rounds 21-40: four fixed unavailable IDs; rounds 41-60: eight fixed unavailable IDs.
- Fixed nested dropout IDs: rounds 21-40 use [16,17,18,19]; rounds 41-60 use
  [12,13,14,15,16,17,18,19]. No randomized dropout selection was used.
  Keep these IDs unchanged for seeds 21/42. With malicious IDs [1,8,12,13],
  scheduled online malicious counts are 4/20, 4/16 and 2/12 across stages.
- Post-share means a fresh update and its shares exist in each affected round;
  this does not model permanently offline devices generating fresh updates.
- Threshold profile: share_n=20, share_t=16, normal A=16, emergency B=12.

## Methods

|Method|SSDF / weights|Recovery|Server verification|
|---|---|---|---|
|VeraShield-FL|unified_v2_candidate, norm_k=2.70, dynamic weight blend=0.25|dynamic A/B 16/12|crypto verification and enforcement|
|FedAvg|standard sample-weighted mean|none|none|
|Krum|f=2 fixed|none|none|
|Trimmed Mean|total trim_ratio=0.2 fixed|none|none|
|FedDMC|existing PCA/HDBSCAN/history implementation|none|none|

Baselines aggregate surviving updates; VeraShield attempts to recover the
post-share unavailable updates before screening. Baselines receive no Crypto-VL.
This compares complete systems and cannot isolate an individual module's contribution.
Trimmed Mean's 0.2 ratio is the implementation's total removal fraction; record
actual per-side counts as survivor population changes. Krum never lowers f on failure.

## Execution scope

1. Create shared manifests and save exact commands. The pilot did not save a
   launch-time source-hash snapshot; do not claim retrospective code hashes
   establish the code version used at launch. Freeze source hashes before extension.
2. Dry-run each method's configuration; verify partition, dropout IDs and thresholds.
3. Run five methods for seed 7; inspect full 60-round joint trajectories.
4. Only after main-method functional audit passes, extend seeds 21 and 42.

Five pilot runs, fifteen total if expanded. No extra attack-type or IID matrix
is required at this stage; earlier independent experiments cover those axes.

## Audit and reporting

- Final accuracy, per-round accuracy, and recovery after each tamper event.
- Actual aggregation failures, missing diagnostics, empty accepted/aggregate sets.
- Planned/applied/skipped tamper events; accept skipped injection only with
  explicit evidence that no aggregate candidate existed. Report actual exposure.
- Client verification occurs next round: expected receipt checks at 21/41/60;
  detection and rollback denominators use applied tamper events only.
- Detection recall and FPR over ground-truth-labeled submitted or recovered
  candidates, distinguishing rounds rejected by client verification from genuine
  statistical screening failures. Report candidate coverage and excluded rounds.
- Recovery successes/attempts per client-round, quantization reconstruction error,
  and failures. Never treat unavailable or unsupported baseline recovery as a
  measured zero recovery-success rate.
- Normal/emergency mode counts deduplicated by round; switch expected at round 41
  only if actual availability matches schedule. Derive from actual records,
  not from planned dropout percentages.
- Unexpected crypto verification failures, separately from expected rejection
  of deliberately corrupted model deliveries. Receipt-rejected rounds can skip
  normal screening/recovery; report those explicitly, not as successful recoveries.
- Runtime and recovery overhead; separate actual wire measurements from logical
  payload estimates.

Pilot functional gate: no unexplained verification failure, each applied tamper
detected and rolled back, each attempted recoverable update reconstructed, no
unexplained empty aggregation. Recall>=0.9 and benign FPR<=0.05 on eligible
active-attack screening events are targets to audit, not guaranteed outcomes.

Interpretation: logical two-server joint fault-injection evidence; not independent
server deployment or a cryptographic security proof. Fix implementation failures
before expanding, and retain all outcomes without tuning to obtain a ranking.
