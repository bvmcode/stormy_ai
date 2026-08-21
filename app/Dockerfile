FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PATH="/srv/stormy_ai/.venv/bin:$PATH" \
    RADAR_PLOT_DIR=/data/radar_plots \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=gemma4:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gfortran \
    libeccodes-dev \
    libeccodes0 \
    libfreetype6-dev \
    libgeos-dev \
    libhdf5-dev \
    libnetcdf-dev \
    libpng-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /srv/stormy_ai

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app

RUN uv sync --frozen --no-dev --group web

RUN mkdir -p /data/radar_plots

WORKDIR /srv/stormy_ai/app

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "900", "web:app"]
