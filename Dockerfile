FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies needed for local PDF partitioning
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies. Ensure your project has a requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --ignore-installed -r /app/requirements.txt

# Copy application code
COPY . /app

# Default command: adjust if your project uses a different entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
