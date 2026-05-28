FROM python:3.11-slim AS build

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim

RUN useradd -m appuser

WORKDIR /app

COPY --from=build /install /usr/local

COPY backend/ backend/
COPY ml/ ml/
COPY db/ db/

# Bake the approved model
COPY models/approved/ models/approved/

# Split is needed by config paths
COPY data/raw/ data/raw/

ENV DB_PATH=/data/house_price.db

RUN mkdir -p /data && chown -R appuser:appuser /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]