# B6C.3a — read-only preflight of a real canonical municipality

This is the first stage of real-data B6C.3 validation. It does **not** yet
execute a scenario, create outputs, or change any baseline. Existing B6C.0,
B6C.1 and B6C.2 engines remain unchanged.

The preflight resolves the output path conventions from
`build_population_network_origins`, `build_service_layer_v2` and
`build_network_attachments_v2` and the OSM network layer. It verifies
that all five canonical Parquet files exist, hashes all source bytes, checks
schema and municipality identity, checks per-mode attachment graph checksum,
ensures node references are present in the network, and invokes the existing
B6A3 fail-closed service Quality Gate to distinguish validated / unvalidated
service destinations. No candidate is approved merely because it snaps to OSM.

Run on the **unchanged WSL repository**, after staging the new module/tests:

```bash
cd /mnt/c/Users/faval/Desktop/urban-digital-twin
conda activate hpc_env
PYTHONPATH=src python -m unittest tests.unit.test_preflight_real_scenario_b6c3 -v
PYTHONPATH=src python src/quality/preflight_real_scenario_b6c3.py \
  --municipality-code 034027 --census-year 2023 \
  --school-year 202425 --health-reference-date 2025-06-30 \
  --mode walk --service-type pharmacy
```

The preflight prints a few already-eligible service IDs as **potential** removal
candidates for a controlled simulation. The exact removal must be chosen and
reviewed before execution. The two unresolved ANNCSU health candidates are not
included unless they were independently validated in the canonical upstream
layer, which must *not* be done through this tool.

After this preflight succeeds, the next stage will construct a frozen
`AnalysisSpec` and a `ScenarioSpec` for one selected service and invoke
`run_verified_scenario` (B6C.2) on the real GeoParquet snapshots, inspecting
metrics and invariants in read-only mode before persisting a report.

Limits: hash identity and routing eligibility do not independently establish
authoritative source provenance, historical operation, physically verified
entrance or clinical service scope. This does not certify correctness of the
road network or the national zero-touch pipeline. No frontend, Gold layer or
full B10 integration is implemented here.
