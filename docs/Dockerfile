FROM ghcr.io/astral-sh/uv:latest AS uv

FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential libboost-dev python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY . .
RUN uv sync --locked --python /usr/bin/python3 --group docs --group notebook --group dev --no-default-groups
RUN uv run --no-sync python .github/scripts/normalize_notebooks.py --check notebooks/*.ipynb \
    && uv run --no-sync ruff check --no-fix notebooks/*.ipynb
RUN uv run --no-sync sphinx-build -b html -W --keep-going docs docs/_build/html \
    && uv run --no-sync python docs/check.py

FROM nginx:stable-alpine AS preview
COPY docs/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/docs/_build/html /usr/share/nginx/html/benchmax-preview
EXPOSE 80
