# Builds a self-contained image: the DuckDB warehouse and ChromaDB vector
# index are baked in at build time (both are reproducible from the checked-in
# synthetic data via a fixed seed), so the container needs no setup step and
# no network access at startup -- only for the Claude API itself at request time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . .

RUN pip install ".[api]"

RUN python -m app.utils.generate_synthetic_data

RUN cd dbt && dbt build --profiles-dir .

RUN python -m app.services.vector_store

EXPOSE 8000

# Render (and most container hosts) inject $PORT; default to 8000 for
# `docker run -p 8000:8000` outside of that.
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
