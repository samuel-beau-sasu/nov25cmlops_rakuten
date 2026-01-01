FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mlops_rakuten/ ./mlops_rakuten/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "mlops_rakuten.api:app", "--host", "0.0.0.0", "--port", "8000"]