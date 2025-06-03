FROM python:3.11-slim

# Install system dependencies for image processing
RUN apt-get update && apt-get install -y \
    libraw-dev \
    libexiv2-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libheif-dev \
    python3-dev \
    build-essential \
    libmagic1 \
    libmagic-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create user to match host user (john with UID 1000)
RUN useradd -m -u 1000 -s /bin/bash john

# Create directories and set ownership
RUN mkdir -p /app/inbox /app/processed /app/processed/originals /app/temp && \
    chown -R john:john /app

# Switch to non-root user
USER john

CMD ["python", "main.py"]