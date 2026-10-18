# Pinned base and uv version — reproducible builds.
FROM python:3.14-slim

WORKDIR /app

# uv, pinned (fast, lockfile-based install)
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

# sentence-transformers pulls CUDA torch by default (~3 GB) — useless on a
# 0.25-vCPU Fargate task. Force the CPU wheel (~200 MB).
ENV UV_TORCH_BACKEND=cpu \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependency layer: lockfile + metadata only, so a src/ edit does not
# reinstall the world.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Project layer
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["python", "-m", "rag_control_cite.app"]
