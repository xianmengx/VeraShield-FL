# Staged dropout results

Threshold modes count rounds; recovery counts client-round events, not unique clients.

|Method|Final accuracy (mean ± sample SD)|
|---|---:|
|verashield_fl|51.00 ± 0.55%|
|without_recovery|50.91 ± 0.59%|

|Seed|Normal rounds|Emergency rounds|Switch round|Recovery successes/attempts|Empty accepted rounds|
|---|---:|---:|---:|---:|---:|
|7|40|20|41|240/240|0|
|21|40|20|41|240/240|0|
|42|40|20|41|240/240|0|

Comparison limitation: without_recovery also disables SSDF and dynamic weighting; this is not an isolated recovery-only ablation.
