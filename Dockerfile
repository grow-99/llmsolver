# Dockerfile — deploy-ready for Render (Debian slim)
FROM python:3.11-slim

# avoid interactive tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

# install system deps (chromium dependencies + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg ffmpeg \
    fonts-liberation libnss3 libatk1.0-0 libatk-bridge2.0-0 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 libgtk-3-0 libgbm-dev libasound2 \
    libpangocairo-1.0-0 libxss1 libxcursor1 libxkbcommon0 libx11-6 \
    build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# set workdir
WORKDIR /app

# copy requirements and source
COPY requirements.txt .

# install python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# install Playwright and browsers
# (use sync playwright used by our utils; this installs browsers)
RUN python -m playwright install --with-deps

# copy app code
COPY . .

# create writable downloads dir
RUN mkdir -p /app/downloads && chmod -R 777 /app/downloads

# render uses $PORT env var; default to 8000 locally
ENV PORT=8000

# recommended non-root user (optional)
# RUN useradd -m appuser && chown -R appuser /app
# USER appuser

# expose port
EXPOSE 8000

# start uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
