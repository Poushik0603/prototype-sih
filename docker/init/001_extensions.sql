CREATE EXTENSION IF NOT EXISTS postgis;

-- pgmq extension is not bundled in the postgis/postgis image and installing it adds
-- build friction disproportionate to this thin prototype. Fallback, if a queue is ever
-- needed: a plain Postgres table used as a queue (id SERIAL, payload JSONB, status TEXT,
-- enqueued_at TIMESTAMPTZ, processed_at TIMESTAMPTZ). For the current pipeline, direct
-- table/file reads and writes without a queue are sufficient.
