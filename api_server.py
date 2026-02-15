from __future__ import annotations

from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from audio_process import score_audio_file_bytes
from settings import settings


app = FastAPI(title="MinuteZero API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/audio-process")
async def audio_process_endpoint(audio: UploadFile = File(...)) -> dict[str, Any]:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    try:
        result = score_audio_file_bytes(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}") from exc

    return {
        "filename": audio.filename,
        "content_type": audio.content_type,
        "result": result,
    }


@app.post("/api/transcription")
async def transcription_endpoint(audio: UploadFile = File(...)) -> dict[str, Any]:
    if not settings.archia_api_key:
        raise HTTPException(
            status_code=500,
            detail="ARCHIA_API_KEY is not configured. Add it to .env before using this endpoint.",
        )

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    if settings.archia_transcription_url:
        url = settings.archia_transcription_url
    elif settings.archia_base_url:
        url = f"{settings.archia_base_url.rstrip('/')}{settings.archia_transcription_path}"
    else:
        raise HTTPException(
            status_code=500,
            detail="Archia URL is not configured. Set ARCHIA_TRANSCRIPTION_URL or ARCHIA_BASE_URL in .env.",
        )

    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=500,
            detail=f"Archia URL must start with http:// or https://. Current value: {url}",
        )

    headers = {"Authorization": f"Bearer {settings.archia_api_key}"}
    files = {
        "file": (
            audio.filename or "audio.wav",
            data,
            audio.content_type or "application/octet-stream",
        )
    }
    payload = {"model": settings.archia_transcription_model}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, data=payload, files=files)
            response.raise_for_status()
            upstream = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Archia transcription request failed: {detail}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Archia API at {url}: {exc}",
        ) from exc

    text = (
        upstream.get("text")
        or upstream.get("transcript")
        or upstream.get("output_text")
        or upstream.get("result")
    )

    return {
        "filename": audio.filename,
        "content_type": audio.content_type,
        "text": text,
        "archia_response": upstream,
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
