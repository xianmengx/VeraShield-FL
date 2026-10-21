# Joint seed 42 audit

PASS WITH LIMITATIONS; no training or raw-output modifications.

|Method|Final accuracy|Applied tamper rounds|Nonfinite loss rounds|Missing aggregation diagnostics|
|---|---:|---|---:|---:|
|verashield_fl|47.89%|[20, 40, 59]|0|0|
|fedavg|9.96%|[20]|47|47|
|krum|20.64%|[20, 40, 59]|0|0|
|trimmed_mean|10.00%|[20]|38|37|
|feddmc|10.00%|[]|54|53|

Corrected main-method counts: normal 40 rounds, emergency 20 rounds; recovery 220/220; rejection/rollback 21/41/60.
Detection: TP=216 TN=863 FP=1 FN=0; recall=100%, FPR=0.1157%, precision=99.5392%.

Limitations:
- Individual seed report, not pooled multi-seed evidence.
- FedAvg/FedDMC have no evidence of scheduled tamper execution after early instability.
- Three rollback rounds skip recovery and statistical screening; 220 attempts, not 240.
- Logical post-share simulation: fresh updates/shares are available each affected round.
- Runtime thresholds are share availability, not a guarantee against Byzantine share holders.
- Fixed nested dropout IDs 16-19 then 12-19; plan updated to match actual execution.
- Online malicious counts are 4/20, 4/16 and 2/12. Recovered candidates still include malicious updates.
- System comparison cannot isolate benefits of individual modules.
