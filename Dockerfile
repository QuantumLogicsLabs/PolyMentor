FROM python:3.11-slim

WORKDIR /app

# Copy and install API runtime dependencies only.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

# Copy project
COPY . .
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]