# V2 closed-loop false-negative and reference-lifecycle audit

Read-only audit of the completed 60-round development run. No detector change
or training was performed. Sources: `verashield_fl/detection_statistics.csv`,
`dynamic_weight_results.csv`, `protocol_trace.jsonl`, `results.csv`,
`summary.json`, and batch `replay.json`.

## Outcome

Active rounds 4–60: TP=214, TN=912, FP=0, FN=14; recall=93.8596%,
FPR=0%, precision=100%. Final accuracy=47.93%, best=48.45% at round 59.
Replay has zero decision mismatches. Crypto-VL passed all 60 rounds and
there were no empty accepted sets. This is a development seed and one
sign-flip strength, not a cross-seed or cross-attack guarantee.

## All false-negative instances

Weights below are actual quantized aggregation weights, not raw confidence.

|Round|Client|Norm z|Updated S|Confidence|Actual weight|
|---:|---:|---:|---:|---:|---:|
|5|8|2.62162|0.40106|0.01183|0.00247192|
|5|13|2.26733|0.37436|0.05013|0.00886536|
|6|1|2.62951|0.60187|0.00584|0.00140381|
|6|8|2.20841|0.48444|0.05731|0.01202393|
|6|12|2.51339|0.62556|0.01672|0.00407410|
|6|13|2.07977|0.49949|0.05749|0.01202393|
|7|13|2.62666|0.59959|0.00544|0.00088501|
|8|8|2.30663|0.64090|0.03348|0.00669861|
|8|12|2.58240|0.75165|0.00946|0.00213623|
|8|13|2.33448|0.66602|0.02415|0.00466919|
|9|8|2.55884|0.70226|0.01103|0.00250244|
|9|12|2.55620|0.79067|0.00877|0.00205994|
|9|13|2.34307|0.70638|0.02457|0.00524902|
|10|13|2.46734|0.76510|0.01012|0.00196838|

All 14 have norm z below 2.70 and S below 0.8. Negative reference cosine is
present in every instance, but v2 deliberately gives direction evidence no
immediate rejection path. Some cohort evidence reaches 1, yet accumulation
is still below the persistent boundary. Recomputed decisions match the logs.
No integrity bypass or forced-retention behavior explains these cases.

All four attackers were rejected in round 4 by unit norm evidence. They were
then intermittently readmitted when norm z fell below its threshold, until
the accumulated score became sufficient. Consequently this is not merely
an initial detection delay: early detection is followed by temporary
readmission. Client 1 is readmitted once, client 8 four times, client 12 three
times, and client 13 six times. No permanent blacklist is used.

Rounds 4–10 have TP=14/FN=14 (50% recall); rounds 11–60 have TP=200/FN=0
(100% recall). Both intervals have FP=0. Reporting only post-round-10 recall
would hide the early vulnerability; the primary reported recall must remain
214/228=93.86%.

## Actual aggregation exposure

|Round|Total malicious weight|Sum(lambda*malicious norm) / sum(lambda*benign norm)|
|---:|---:|---:|
|5|1.13373%|6.22395%|
|6|2.95258%|17.65198%|
|7|0.08850%|0.46462%|
|8|1.35040%|8.11657%|
|9|0.98114%|5.48717%|
|10|0.19684%|1.08431%|

The last column compares sums of weighted norms, not norms of aggregate
vectors. It cannot determine cancellation, the actual model perturbation,
or causal accuracy damage. Original update vectors are not saved here.
Small weight does not mean zero poisoning, especially with 5x sign flipping.
Accuracy at rounds 5–10 is 19.80%, 19.79%, 26.25%, 24.72%, 26.01%, 31.32%;
without a paired intervention these changes cannot be attributed to the
accepted malicious updates alone.

Detection pass sets agree with the trace accepted sets in every round;
weights sum to one; weight CSV values agree with trace values. Every rejected
client has zero actual weight. Trace weight dictionaries include explicit
zero-weight entries for rejected clients; dictionary presence is not evidence
that a rejected client participates in aggregation.

## Trusted-state hygiene and remaining channel limitations

- No malicious update is admitted to history or anchor in active rounds.
  Four malicious histories expire rather than being refreshed by attacks.
  Temporal evidence is unavailable for all 14 FN instances; insufficient
  calibration/expiration limits that channel's contribution.
- Clients 8 and 13 were anchor-qualified in round 3 while behaving benignly.
  The round-4 decision can therefore consult an anchor built partly from
  their previous benign updates. The precise claim is no *active malicious
  update* admitted, not an anchor containing no eventually malicious identity
  throughout the run. The active-membership gate does not identify future
  attackers during benign behavior.
- The anchor updates in 49 rounds. It has insufficient members in rounds
  1–2 and 52–60. Round 52 may use the valid previous-round anchor before
  invalidating it at the end of the round; cohort is unavailable in 53–60.
  This matches the lifecycle semantics and avoids frozen-anchor evidence.
  It is graceful degradation, not proof that anchor starvation is solved.
- Benign histories expire three times: client 15 at rounds 5 and 14, client
  18 at round 58. All remain accepted; the round-5 instance refreshes history,
  the other two do not meet refresh criteria. Maximum observed benign history
  age is 4 at expiration. Maximum active benign S is 0.73880, below 0.8.

## Decision

The registered single-run gate passes. The repair removes this run's late
false positives but trades immediate directional rejection for early
intermittent malicious acceptance. There is no evidence requiring an
identity-specific blacklist, attack-specific threshold, or reverting to the
previous direction hard gate. Nor does this run prove zero-miss robustness.

Keep v2 frozen for additional validation. Before a full multi-seed matrix,
use a clean non-IID check to assess defense cost, then a direction-only
equal-norm sign-flip check to measure the EMA latency when the immediate norm
channel contributes little. The current -5x setting tests combined direction
and scale anomalies. These checks are recommendations, not launched runs.
Other attack types/seeds must retain the same policy and parameters, with
per-round latency, FPR, actual malicious weight and cohort availability
reported alongside aggregate recall and accuracy.
