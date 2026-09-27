-- ============================================================
-- OSM NETWORK ENRICHMENT
-- ============================================================

ALTER TABLE network_node
ADD COLUMN IF NOT EXISTS attributes JSONB
NOT NULL DEFAULT '{}'::jsonb;


ALTER TABLE network_edge
ADD COLUMN IF NOT EXISTS osm_way_ids TEXT;

ALTER TABLE network_edge
ADD COLUMN IF NOT EXISTS attributes JSONB
NOT NULL DEFAULT '{}'::jsonb;


-- Rende anche gli archi idempotenti.
-- source_record_id sarà u:v:key del MultiDiGraph OSMnx.

CREATE UNIQUE INDEX IF NOT EXISTS
uq_network_edge_source_record
ON network_edge (
    source_system,
    source_record_id
)
WHERE source_record_id IS NOT NULL;