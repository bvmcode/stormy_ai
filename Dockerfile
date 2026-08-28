# Stormy AI weather briefing agent — linux/arm64 (see Makefile build target).
# Builder: compile packages without wheels for Python 3.14 (e.g. cartopy).
FROM python:3.14-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libeccodes-dev \
        libgeos-dev \
        libhdf5-dev \
        libnetcdf-dev \
        libproj-dev \
        proj-bin \
        proj-data \
        libfreetype6-dev \
        libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY pyproject.toml uv.lock README.md ./
COPY main.py config.yaml ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Runtime image — shared libs only, no compilers.
FROM python:3.14-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libeccodes0 \
        libgeos-c1v5 \
        libhdf5-103-1 \
        libnetcdf19 \
        libproj25 \
        libstdc++6 \
        proj-bin \
        proj-data \
        libfreetype6 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

RUN mkdir -p briefings radar_plots model_plots /home/app/.cache/metpy /home/app/.cache/matplotlib \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && chown -R app:app /app /home/app

ENV MPLCONFIGDIR=/home/app/.cache/matplotlib \
    XDG_CACHE_HOME=/home/app/.cache

USER app

# Optional location arg: docker run ... image "Denver, CO"
ENTRYPOINT ["python", "main.py"]
CMD []
