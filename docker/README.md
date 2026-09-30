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
From WSL Ubuntu, in this directory (translate the Windows path, e.g.
`/mnt/c/Users/royal/OneDrive/Documents/SIH 2026/prototype/docker`):
```bash
cd "/mnt/c/Users/royal/OneDrive/Documents/SIH 2026/prototype/docker"
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
`postgis/postgis` image), fall back to a plain Postgres table-based queue — note the
deviation in results.md, do not block the pipeline on it.
