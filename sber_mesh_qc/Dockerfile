# Multi-Stage Production Dockerfile for 3D Mesh Quality Control REST API
FROM python:3.11-slim as base

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn onnxruntime

# Copy solution source code
COPY solution/ /app/solution/
COPY README.md /app/

# Expose HTTP port for REST API
EXPOSE 8000

# Set Python path to include solution directory
ENV PYTHONPATH="/app/solution:${PYTHONPATH}"

# Health check endpoint for AWS ALB / GCP Kubernetes
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI Uvicorn REST microservice server
CMD ["python", "solution/app.py"]
