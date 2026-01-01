FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mlops_rakuten/ ./mlops_rakuten/

RUN pip install --no-cache-dir .

# Optionnel: créer des dossiers si notre code s’attend à ce qu’ils existent
# RUN mkdir -p /app/models /app/reports /app/uploads

EXPOSE 8000

CMD ["python", "-m", "mlops_rakuten.main", "--help"]