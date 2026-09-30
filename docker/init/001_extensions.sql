CREATE EXTENSION IF NOT EXISTS postgis;

-- pgmq extension is not bundled in the postgis/postgis image and installing it adds
-- build friction disproportionate to this thin prototype. Fallback (documented deviation,
-- see PROJECT_SPEC.md / results.md limitations): a plain Postgres table used as a queue
-- (id SERIAL, payload JSONB, status TEXT, enqueued_at TIMESTAMPTZ, processed_at TIMESTAMPTZ).
-- Data/Model agents: create this table in your own migration if you need a queue at all;
-- for the thin slice, direct table reads/writes without a queue are acceptable.
