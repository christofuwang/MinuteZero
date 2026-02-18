# MinuteZero

## Backend API

This repo now includes a FastAPI backend for:
- `POST /api/transcription` (proxy upload to Archia transcription API)
- `POST /api/audio-process` (local emotion + dispatch concern scoring)
- `GET /health`

### 1. Environment setup

Copy env template and add your key:

```bash
cp .env.example .env
```

Set at least:

```env
ARCHIA_API_KEY=your_real_archia_key
```

Optional overrides:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
ARCHIA_BASE_URL=https://api.archia.ai
ARCHIA_TRANSCRIPTION_PATH=/v1/transcriptions
ARCHIA_TRANSCRIPTION_MODEL=whisper-1
```

### 2. Install deps

If using conda environment file:

```bash
conda env create -f environment.yml
conda activate distress_audio
```

### 3. Run API server

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### 4. Example calls

Transcription:

```bash
curl -X POST "http://localhost:8000/api/transcription" \
  -F "audio=@/path/to/audio.wav"
```

Audio processing:

```bash
curl -X POST "http://localhost:8000/api/audio-process" \
  -F "audio=@/path/to/audio.wav"
```

Note: the local audio processing endpoint expects a decodable audio format (WAV/FLAC/OGG/AIFF).

## Frontend wiring

The Next.js frontend now sends recorded audio to both backend endpoints and renders JSON responses.

Set frontend API base URL (optional, default is `http://localhost:8000`):

```bash
cp frontend/.env.example frontend/.env.local
```
