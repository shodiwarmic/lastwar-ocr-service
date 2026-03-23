# Use slim Python image to minimize container size and cold start time
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
# Unbuffered output ensures logs reach Cloud Run / Cloud Logging immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install dependencies first (separate layer for better Docker cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user for security best practice on Cloud Run
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Expose the port Cloud Run expects
EXPOSE 8080

# Gunicorn:
#   - 1 worker: Cloud Run scales via instances, not workers
#   - timeout 120s: allows time for large batch OCR processing
#   - access-logfile -: routes access logs to stdout for Cloud Logging
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "main:app"]
