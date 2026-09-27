# External technical validation

## Independent LLM review

The refactored Urban Digital Twin pipeline was subjected to an
independent technical review focused on:

- zero-touch municipality transfer;
- GIS and network-analysis correctness;
- missing/unknown handling;
- reproducibility;
- provenance;
- OSM-only vs Institutional+OSM experimental validity;
- national generalizability.

## Review outcome

No CRITICAL finding was identified.

Two findings were initially classified as MAJOR and subsequently
verified against the implementation and data.

### M1 — Semantic asymmetry between OSM-only and institutional education

A possible difference in semantic inclusion rules between MIM and OSM
was identified.

A targeted audit searched the OSM-only school inventories for terms
associated with restricted/special services such as correctional,
hospital and evening schools.

Results:

- Matera (077014): 50 OSM school sites, 0 suspected special services.
- Parma (034027): 111 OSM school sites, 0 suspected special services.

Therefore the proposed issue is not observed in the two validated
municipalities.

Different semantic schemas between OSM and institutional registries
remain a methodological limitation to monitor in future municipalities.

### M2 — Disconnected pedestrian-network components

The finding was determined to be a false positive.

The pipeline preserves all network components and preserves origins
without a directed path to a service as unreachable rather than
dropping them.

Population outside reachable service components therefore remains in
the demographic denominator.

### Minor finding — census-section fallback

The review referred to centroid-based allocation.

The implementation instead:

1. distributes observed population over pedestrian-network nodes
   located inside each census section;
2. only when no internal node exists, uses the polygon
   representative point;
3. snaps that point to the nearest network node in a metric CRS.

The remaining approximation is documented as a methodological
limitation for sparse/rural census sections.

### Minor finding — co-located schools

Co-located educational entities are not considered an error for
nearest-service accessibility.

They may represent separate educational services at the same physical
site. The distinction between service entities and physical sites is
therefore intentionally preserved.

## Final assessment

CRITICAL findings: 0

Demonstrated MAJOR findings: 0

The Matera/Parma implementation is considered ready for thesis
validation.

Known future work before national-scale execution includes:

- national versioned ISTAT municipality bootstrap;
- national/regional OSM bulk extraction strategy;
- monitoring semantic comparability of OSM service categories;
- sensitivity analysis for population allocation and geolocation
  uncertainty.
