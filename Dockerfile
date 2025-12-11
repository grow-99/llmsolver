# Dockerfile — Playwright-ready, includes ffmpeg
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install system packages Playwright + Chrome need (and ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    wget \
    ca-certificates \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libfreetype6 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libharfbuzz0b \
    libicu66 || true \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libgbm1 \
    libxcb-render0 \
    libxcb-shm0 \
    ffmpeg \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy only requirements first to leverage build cache
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Install Playwright browsers and required libs
# Use a separate command so failures are visible in build logs
RUN python -m playwright install --with-deps

# Copy app code
COPY . /app

# Ensure downloads directory is writeable
RUN mkdir -p /app/downloads && chmod -R 777 /app/downloads

EXPOSE 8000

# Use PORT env variable if available
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
