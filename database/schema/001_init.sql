CREATE EXTENSION IF NOT EXISTS postgis;

-- Municipality

CREATE TABLE municipality (
    id                  BIGSERIAL PRIMARY KEY,
    istat_code          VARCHAR(10) UNIQUE NOT NULL,
    name                TEXT NOT NULL,
    province_code       VARCHAR(10),
    region_code         VARCHAR(10),

    geometry            GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,

    source_system       TEXT NOT NULL,
    source_record_id    TEXT,
    reference_date      DATE,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_municipality_geometry
ON municipality
USING GIST (geometry);


-- Census area

CREATE TABLE census_area (
    id                      BIGSERIAL PRIMARY KEY,
    census_section_code     TEXT UNIQUE NOT NULL,

    municipality_id         BIGINT NOT NULL
        REFERENCES municipality(id)
        ON DELETE CASCADE,

    geometry                GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,

    section_type_code       INTEGER,
    locality_type           INTEGER,

    source_system           TEXT NOT NULL,
    source_record_id        TEXT,
    reference_date          DATE,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_census_area_geometry
ON census_area
USING GIST (geometry);

CREATE INDEX idx_census_area_municipality
ON census_area(municipality_id);

CREATE TABLE census_observation (
    id BIGSERIAL PRIMARY KEY,

    census_area_id BIGINT NOT NULL
        REFERENCES census_area(id)
        ON DELETE CASCADE,

    reference_date DATE NOT NULL,

    population INTEGER,
    population_density DOUBLE PRECISION,
    families INTEGER,

    age_0_14 INTEGER,
    age_15_64 INTEGER,
    age_65_plus INTEGER,

    foreign_population INTEGER,
    employed_population INTEGER,
    unemployed_population INTEGER,

    housing_units INTEGER,

    source_system TEXT NOT NULL,
    source_record_id TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(census_area_id, reference_date)
);

CREATE INDEX idx_census_observation_area
ON census_observation(census_area_id);

CREATE INDEX idx_census_observation_date
ON census_observation(reference_date);

-- Network node

CREATE TABLE network_node (
    id                  BIGSERIAL PRIMARY KEY,

    geometry            GEOMETRY(POINT, 4326) NOT NULL,

    source_system       TEXT NOT NULL DEFAULT 'OSM',
    source_record_id    TEXT NOT NULL,

    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(source_system, source_record_id)
);

CREATE INDEX idx_network_node_geometry
ON network_node
USING GIST (geometry);


-- Network edge

CREATE TABLE network_edge (
    id                  BIGSERIAL PRIMARY KEY,

    source_node_id      BIGINT NOT NULL
        REFERENCES network_node(id),

    target_node_id      BIGINT NOT NULL
        REFERENCES network_node(id),

    geometry            GEOMETRY(LINESTRING, 4326) NOT NULL,

    length_m            DOUBLE PRECISION NOT NULL,
    walking_time_s      DOUBLE PRECISION,

    road_type           TEXT,

    foot_access         BOOLEAN,
    vehicle_access      BOOLEAN,
    oneway              BOOLEAN,

    source_system       TEXT NOT NULL DEFAULT 'OSM',
    source_record_id    TEXT,

    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_network_edge_geometry
ON network_edge
USING GIST (geometry);

CREATE INDEX idx_network_edge_source
ON network_edge(source_node_id);

CREATE INDEX idx_network_edge_target
ON network_edge(target_node_id);


-- Service

CREATE TABLE service (
    id                  BIGSERIAL PRIMARY KEY,

    municipality_id     BIGINT
        REFERENCES municipality(id),

    census_area_id      BIGINT
        REFERENCES census_area(id),

    category            TEXT NOT NULL,
    subcategory         TEXT,

    name                TEXT,

    geometry            GEOMETRY(POINT, 4326) NOT NULL,

    address             TEXT,

    capacity            DOUBLE PRECISION,
    usage_value         DOUBLE PRECISION,
    usage_unit          TEXT,

    quality_score       DOUBLE PRECISION
        CHECK (
            quality_score IS NULL
            OR quality_score BETWEEN 0 AND 1
        ),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_service_geometry
ON service
USING GIST (geometry);

CREATE INDEX idx_service_category
ON service(category);

CREATE INDEX idx_service_census_area
ON service(census_area_id);


-- Service source / Provenance

CREATE TABLE service_source (
    id                  BIGSERIAL PRIMARY KEY,

    service_id          BIGINT NOT NULL
        REFERENCES service(id)
        ON DELETE CASCADE,

    source_system       TEXT NOT NULL,
    source_record_id    TEXT,

    source_dataset      TEXT,
    source_url          TEXT,

    license             TEXT,

    reference_date      DATE,
    retrieved_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    geometry_source     TEXT,
    geocoding_method    TEXT,

    completeness_score  DOUBLE PRECISION,
    reliability_score   DOUBLE PRECISION,
    freshness_score     DOUBLE PRECISION
);


-- Temporal Observation

CREATE TABLE observation (
    id                  BIGSERIAL PRIMARY KEY,

    entity_type         TEXT NOT NULL,
    entity_id           TEXT NOT NULL,

    event_time          TIMESTAMPTZ NOT NULL,
    ingestion_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    variable            TEXT NOT NULL,
    value               DOUBLE PRECISION,
    unit                TEXT,

    source_system       TEXT,
    quality_flag        TEXT
);

CREATE INDEX idx_observation_entity_time
ON observation(entity_type, entity_id, event_time);

CREATE INDEX idx_observation_event_time
ON observation(event_time);


-- Accessibility

CREATE TABLE accessibility_result (
    id                      BIGSERIAL PRIMARY KEY,

    census_area_id          BIGINT NOT NULL
        REFERENCES census_area(id)
        ON DELETE CASCADE,

    service_category        TEXT NOT NULL,

    nearest_service_id      BIGINT
        REFERENCES service(id),

    distance_m              DOUBLE PRECISION,
    travel_time_s           DOUBLE PRECISION,

    services_5min           INTEGER,
    services_10min          INTEGER,
    services_15min          INTEGER,

    population_served       DOUBLE PRECISION,
    population_unserved     DOUBLE PRECISION,

    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_accessibility_area
ON accessibility_result(census_area_id);

CREATE INDEX idx_accessibility_category
ON accessibility_result(service_category);