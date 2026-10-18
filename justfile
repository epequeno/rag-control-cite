# justfile for rag-control-cite
# `just`, not `make`. `just down` tears everything down for cents.

# default: show available recipes
default:
    @just --list

# sync dependencies
sync:
    uv sync

# bring up the local stack (postgres + pgvector + app)
up:
    docker compose -f infra/docker/docker-compose.yml up -d

# tear down everything — local stack AND AWS stack (cents, not data)
# teardown.sh is idempotent and always runs; it verifies zero resources remain.
down:
    docker compose -f infra/docker/docker-compose.yml down --remove-orphans || true
    AWS_PROFILE=${AWS_PROFILE:-arch} bash scripts/teardown.sh

# prove teardown: list anything still tagged for this project
verify-teardown:
    AWS_PROFILE=${AWS_PROFILE:-arch} bash scripts/verify_teardown.sh

# deploy to AWS: build image, cdk deploy, resolve task IP (Stage 7)
deploy:
    AWS_PROFILE=${AWS_PROFILE:-arch} bash scripts/deploy.sh

# load-test the deployed endpoint: just load 1.2.3.4 --rps 2 --duration 120
load IP *ARGS:
    uv run python scripts/load_test.py "http://{{IP}}:8000" {{ARGS}}

# seed pgvector (ingest → embed → upsert)
seed:
    uv run rcc seed

# run a demo query (Stages 1-4)
demo:
    uv run rcc demo

# run ingestion (Stage 0)
ingest:
    uv run rcc ingest

# run the eval gate (Stage 5)
eval *ARGS:
    uv run rcc eval {{ARGS}}

# promote a passing candidate baseline
accept:
    uv run rcc accept

# run the test suite
test:
    uv run pytest

# format
fmt:
    uv run ruff format src tests scripts infra

# lint
lint:
    uv run ruff check src tests scripts infra