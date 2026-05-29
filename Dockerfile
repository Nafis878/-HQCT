# Dockerfile — Reproducible HQCT environment (CPU-only PennyLane)
FROM python:3.10-slim

LABEL maintainer="hqct-pipeline"
LABEL description="Hybrid Quantum-Classical Transformer for CKD + FHS classification"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p data results results/figures results/latex_tables logs models

# Default: run the full CKD pipeline skipping quantum (fast mode)
# Override with: docker run hqct python main.py [--flags]
CMD ["python", "main.py", "--skip-quantum"]
