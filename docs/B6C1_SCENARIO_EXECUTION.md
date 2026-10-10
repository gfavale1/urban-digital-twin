# B6C.1 — Paired service scenario execution (in-memory)

**Stage:** synthetic research engine; **no baseline mutation, files, DB writes or Git actions**.

`analysis.service_scenarios_v2.execute_service_scenario` applies only `add_service` and `remove_service` from the B6C.0 `ScenarioSpec`, then runs the **same** `compute_accessibility_from_canonical` engine twice with the same origins, graph, population selector, mode, thresholds and request. Only the service/attachment overlay changes. Both results and per-origin deltas are returned in memory. `scenario_id` is the immutable B6C.0 ID.

## Safety boundaries

- Caller MUST verify the actual bytes of the canonical origins, services, attachments and graph node/edge files against the SHA-256 references, using the existing manifest procedure. Passing a dictionary of expected hashes **is not independent verification**, and cannot prove in-memory frames correspond to these files. The graph digest is the existing `graph_checksum(nodes_path, edges_path)` scheme, not an in-memory hash of NetworkX.
- Each add must provide a *separate, externally curated* ServiceV2 row and mode-specific NetworkAttachmentV2 row. `scenario_row_sha256` hashes all row fields (including timestamps and geometry when present). Both row hashes must match the B6C.0 operation. The candidate must have the same municipality, mode, service type and graph checksum, and an attachment node in the graph.
- **B6A3 routing eligibility is mandatory**, but it trusts its upstream `legacy_usable_for_accessibility` field: the Scenario Engine **cannot independently authenticate** that claim. The upstream validated registry/catalogue must be independently governed, versioned and audited. Do NOT flip false/unresolved records to true merely to simulate a facility; Farmacia Parigi and Don Gnocchi are still `hold` and unavailable for this path.
- Removal may target only B6A3-eligible services in the baseline. Both additions and removals are deterministic at the canonical ID level; service ordering is sorted to stabilize ties.
- No edge removal/closure, drive speed change, capacity-aware modeling or hypothetical unvalidated facilities. Additions outside current municipality are not admitted; B7 will address nearby external services.

## Indicators and interpretation

- Coverage fraction is taken from the existing engine (10/15/20 or request thresholds). `delta` is scenario minus baseline in **fractions**, not percent points; multiply by 100 to report percentage points.
- Cumulative opportunities are returned per origin; summary includes a population-weighted mean opportunity count at each threshold.
- Nearest travel time is per origin. The summary `mean_nearest_time_reachable_min` is weighted **only over reachable population**; it is `null` if no reachable population remains. It is not a population-wide travel-time mean when some origins are unreachable; the `reachable_population` denominator is always reported. When reachability changes, do not infer monotonicity from differences between these reachable-only means.
- For add-only scenarios, each origin's reachability and nearest time cannot worsen and opportunity counts cannot decrease. Remove-only scenarios have the reversed constraints. Mixed adds/removes have no unconditional monotonicity guarantee.

## Provenance

The returned `provenance` contains policy, scenario ID and SHA-256, operations, input **claims**, and deterministic result hashes. No CSV/JSON is written. A subsequent runner must verify source files, write immutable artifacts + manifest, link the trusted candidate catalogue and independently recheck the persisted output hashes.

## Verify in WSL

```bash
cd /mnt/c/Users/faval/Desktop/urban-digital-twin
conda activate hpc_env
PYTHONPATH=src python -m unittest tests.unit.test_service_scenarios_b6c1 -v
PYTHONPATH=src python -m unittest discover -s tests/unit -p 'test_*.py'
git status --short
```

Do **not** commit until both test results are reviewed.
