# Cassandra AI — Thin Prototype

SIH26184, Team NeferX. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full spec and
[SCHEMA_CONTRACT.md](SCHEMA_CONTRACT.md) for the data contracts between pipeline stages.

## Setup
```bash
# Python env (uses Anaconda Python 3.11, not system Python)
C:/Users/royal/anaconda3/python.exe -m venv .venv
./.venv/Scripts/pip install -r requirements.txt

# Postgres/PostGIS (via Docker inside WSL Ubuntu — see docker/README.md)
# from a WSL terminal:
#   sudo service docker start
#   cd "/mnt/c/Users/royal/OneDrive/Documents/SIH 2026/prototype/docker" && docker compose up -d

cp .env.example .env
```

## Run the full demo
```bash
./scripts/run_demo.sh      # or scripts/run_demo.ps1 on native Windows PowerShell
```

## Status
Scaffold in progress. See PROJECT_SPEC.md §Agent assignments for build status per module.
