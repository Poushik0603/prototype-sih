# Running Postgres/PostGIS

This project uses **Docker inside WSL Ubuntu** (native `docker` CLI + daemon in WSL), NOT
Docker Desktop for Windows. Docker Desktop should stay closed.

## One-time setup (run yourself in a WSL Ubuntu terminal)
```bash
sudo service docker start
```
(Optional, to avoid repeating this: add passwordless sudo for the docker service, or enable
it via `sudo systemctl enable docker` if your WSL distro runs systemd.)

## Start the database
From WSL Ubuntu, in this directory (translate the Windows path to its `/mnt/c/...`
WSL equivalent):
```bash
cd "/mnt/c/path/to/this/repo/docker"
docker compose up -d
docker compose ps
```

Postgres will be reachable from Windows (and this repo's Python code) at
`localhost:5432`, user `cassandra`, password `cassandra`, db `cassandra`.

## Verify
```bash
docker compose exec postgres psql -U cassandra -d cassandra -c "\dx"
```
Should list `postgis`. If `pgmq` extension fails to install (not bundled in the
`postgis/postgis` image), fall back to a plain Postgres table-based queue — see
`docker/init/001_extensions.sql` for the documented fallback; not required for
the current pipeline, which reads Parquet directly.
