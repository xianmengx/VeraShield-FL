# Final-V2 Matrix Audit

Windows `results/final_v2` asset records scanned: 27.
Additionally incorporated confirmed WSL formal GPU assets under `<WORKSPACE>/final_v2_gpu`.
Reduced matrix counts: MAIN=72, SECONDARY=8, SUPPLEMENTARY=17, OPTIONAL=1.
Total planned cells: 98.
Confirmed reusable Final-V2 GPU MAIN cells: 16 (D0 seed42, D0 seed2024, D02 seed42, D03 seed42; four methods each).
Remaining MAIN cells requiring a valid 30-round run: 56.

## Reuse
PASS and REPAIRED records with matching scenario/method/30R manifest may be reused; VALIDATION_ONLY and HISTORICAL are excluded from main statistics.

## Do not rerun
Existing PASS, REPAIRED, HISTORICAL, VALIDATION_ONLY and DUPLICATE assets are protected and must not be rerun.

## Must rerun
Only matrix cells with no matching valid 30-round asset are required. The next approved batch is D03/seed2024: four methods. Existing CPU, smoke, boundary, diagnostic, duplicate, and superseded VL assets are not reusable for the GPU main table.

## Figure dependencies
- Main accuracy curves: MAIN 30R results.csv grouped by scenario/method/seed.
- Robustness plots: aggregate final/best accuracy over dropout, attack, and partition tiers.
- Overhead plots: aggregation_diagnostics.csv, recovery_statistics.csv, overhead.csv.
- VL verification: VL verification/consistency records joined by scenario and round.
- Statistical bars: only PASS/REPAIRED final candidates, never smoke or CPU historical assets.

## Status totals

Known Windows asset classifications: HISTORICAL=14, VALIDATION_ONLY=13. WSL formal GPU assets are counted separately above because they live outside the Windows `results/final_v2` tree.
