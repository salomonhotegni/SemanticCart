FROM python:3.12.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install \
        --yes \
        --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt ./

RUN python -m pip install \
        --upgrade \
        pip==26.2 \
    && python -m pip install \
        --requirement requirements-serving.txt

RUN useradd \
    --create-home \
    --uid 10001 \
    --shell /usr/sbin/nologin \
    semanticcart

COPY --chown=semanticcart:semanticcart \
    src/ \
    /app/src/

USER semanticcart

EXPOSE 8000

CMD ["uvicorn", "semanticcart.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]