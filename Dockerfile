# Planora backend + dashboard image (same image, different command)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential postgresql-client curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/db-opt-platform

# CPU-only PyTorch wheel keeps the image small; torch_geometric is pure Python.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \
 && pip install -r requirements.txt

COPY . .
RUN chmod +x scripts/entrypoint.sh \
 && useradd --create-home --uid 10001 qs \
 && chown -R qs /srv/db-opt-platform
USER qs

EXPOSE 8000 8501
ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
