FROM python:3.10-slim

# System deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + pretrained model weights
COPY . .

# Render injects $PORT at runtime
CMD uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000}
