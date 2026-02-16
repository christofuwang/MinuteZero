# api_server.py
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import speech_recognition as sr
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import soundfile as sf

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from settings import settings

# -----------------------------
# Config (match dispatch_assistant.py)
# -----------------------------
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# Make LOCAL_DIR stable regardless of uvicorn working directory:
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_audeering")

SAMPLE_RATE = 16000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# -----------------------------
# Threat reference (match script)
# -----------------------------
def load_threat_reference(filepath: str = "Threat_Level.xlsx") -> list[dict[str, Any]]:
    try:
        df = pd.read_excel(filepath)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Warning: Could not load threat reference file: {e}")
        return []


# -----------------------------
# Emotion model (match script)
# -----------------------------
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


def load_or_download():
    """
    Loads processor+model from LOCAL_DIR if present; otherwise downloads from HF and saves locally.
    Robust: if local load fails, wipe and re-download (prevents the config issues you hit earlier).
    """
    if os.path.isdir(LOCAL_DIR):
        try:
            print(f"Loading processor/model from local dir: {LOCAL_DIR}")
            processor = Wav2Vec2Processor.from_pretrained(LOCAL_DIR)
            model = EmotionModel.from_pretrained(LOCAL_DIR)
            return processor, model
        except Exception as e:
            print(f"Local load failed ({e}); re-downloading...")
            import shutil

            shutil.rmtree(LOCAL_DIR, ignore_errors=True)

    print(f"Downloading processor/model from HF: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = EmotionModel.from_pretrained(MODEL_NAME)

    print(f"Saving processor/model to local dir: {LOCAL_DIR}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    processor.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)

    return processor, model


processor, emotion_model = load_or_download()
emotion_model = emotion_model.to(device)
emotion_model.eval()


@torch.no_grad()
def predict_dims(audio: np.ndarray) -> dict[str, float]:
    proc = processor(audio, sampling_rate=SAMPLE_RATE)
    x = torch.tensor(proc["input_values"], dtype=torch.float32).to(device)
    _, logits = emotion_model(x)
    dims = logits.squeeze().detach().cpu().tolist()
    return {"arousal": float(dims[0]), "dominance": float(dims[1]), "valence": float(dims[2])}


def rms_loudness(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def zero_crossing_rate(audio: np.ndarray) -> float:
    signs = np.sign(audio)
    return float(np.mean(signs[1:] != signs[:-1]))


def smoothstep(x: float, edge0: float, edge1: float) -> float:
    if edge0 == edge1:
        return 0.0
    t = (x - edge0) / (edge1 - edge0)
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def compute_dispatch_concern(audio: np.ndarray, dims: dict[str, float]) -> dict[str, float]:
    ar = float(np.clip(dims["arousal"], 0.0, 1.0))
    va = float(np.clip(dims["valence"], 0.0, 1.0))
    do = float(np.clip(dims["dominance"], 0.0, 1.0))

    rms = rms_loudness(audio)
    zcr = zero_crossing_rate(audio)

    loud = smoothstep(rms, 0.02, 0.12)
    very_quiet = 1.0 - smoothstep(rms, 0.01, 0.03)

    yell_score = smoothstep(loud, 0.6, 0.9)
    breathy = smoothstep(zcr, 0.08, 0.18)
    whisper_score = very_quiet * breathy

    negativity = (1.0 - va)
    low_control = (1.0 - do)

    concern = (
        0.35 * ar
        + 0.20 * negativity
        + 0.10 * low_control
        + 0.20 * yell_score
        + 0.15 * whisper_score
    )
    concern = float(np.clip(concern, 0.0, 1.0))

    return {
        "dispatch_concern": concern,
        "rms": rms,
        "zcr": zcr,
        "yell_score": float(np.clip(yell_score, 0.0, 1.0)),
        "whisper_score": float(np.clip(whisper_score, 0.0, 1.0)),
    }


def transcribe_google(audio_f32: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True

    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
    pcm_bytes = audio_i16.tobytes()

    audio_data = sr.AudioData(pcm_bytes, sample_rate, sample_width=2)

    try:
        text = r.recognize_google(audio_data)
        return text.strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return f"[Transcription API error] {e}"


def call_archia_dispatch_agent(
    agent_name: str,
    transcript: str,
    dims: dict[str, float],
    scores: dict[str, float],
    threat_data: list[dict[str, Any]],
) -> str:
    token = os.environ["ARCHIA_TOKEN"]

    client = OpenAI(
        base_url="https://registry.archia.app/v1",
        api_key="not-used",
        default_headers={"Authorization": f"Bearer {token}"},
    )

    payload = {
        "transcript": transcript if transcript else "",
        "emotion_dims": dims,
        "acoustic_scores": scores,
        "reference_manual": {
            "threat_definitions": threat_data,
            "instruction": (
                "Step 1. Scan threat_definitions' for the 'NatureOfReport' that best matches the transcript. "
                "Step 2. Use the associated 'threat_level' to determine the priority. "
                "Step 3. If the transcript describes a situation rated 'High' or 'Critical' in the definitions, the priority Must be HIGH or IMMEDIATE. "
            ),
        },
    }

    resp = client.responses.create(model=f"agent:{agent_name}", input=json.dumps(payload))
    return resp.output[0].content[0].text


# -----------------------------
# Audio decoding: accept WAV (frontend sends WAV) + fallback to ffmpeg for webm
# -----------------------------
def _ffmpeg_available() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode == 0
    except Exception:
        return False


def decode_audio_bytes(data: bytes) -> np.ndarray:
    """
    Returns float32 mono audio at SAMPLE_RATE.
    If input is WAV (from your frontend), soundfile will decode it.
    If decoding fails and ffmpeg is installed, use ffmpeg.
    """
    # Try soundfile first (works for WAV/FLAC/etc)
    try:
        audio, sr_in = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        # If sample rate differs, use ffmpeg to resample reliably (keeps code simple)
        if int(sr_in) != SAMPLE_RATE and _ffmpeg_available():
            raise RuntimeError("Resample via ffmpeg")
        if int(sr_in) != SAMPLE_RATE and not _ffmpeg_available():
            raise RuntimeError(f"Expected {SAMPLE_RATE}Hz but got {sr_in}Hz; install ffmpeg or upload 16k audio.")

        return audio
    except Exception:
        pass

    if not _ffmpeg_available():
        raise RuntimeError("Could not decode audio. Install ffmpeg (macOS: `brew install ffmpeg`) or upload WAV.")

    # ffmpeg fallback
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as fin:
        fin.write(data)
        in_path = fin.name
    out_path = in_path + ".wav"

    try:
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(SAMPLE_RATE), out_path]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg decode failed: {err}")

        audio, _sr = sf.read(out_path, dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32)
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="MinuteZero API", version="0.3.0")

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


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)) -> dict[str, Any]:
    """
    EXACT pipeline from dispatch_assistant.py:
      decode -> dims -> concern -> transcribe -> threat ref -> agent -> print blocks
    Returns the same pieces to the frontend.
    """
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    try:
        audio_f32 = decode_audio_bytes(data)

        dims = predict_dims(audio_f32)
        scores = compute_dispatch_concern(audio_f32, dims)

        # Print EXACT style blocks
        print("Emotion Dimensions (approx 0..1):")
        for k, v in dims.items():
            print(f"  {k}: {v:.4f}")

        print("\nAcoustic signals:")
        print(f"  rms: {scores['rms']:.5f} | zcr: {scores['zcr']:.3f}")
        print(f"  yell_score: {scores['yell_score']:.3f} | whisper_score: {scores['whisper_score']:.3f}")
        print(f"  dispatch_concern: {scores['dispatch_concern']:.4f}")

        print("\nTranscribing (Google Speech Recognition)...")
        transcript = transcribe_google(audio_f32)
        print(f"Transcript: {transcript if transcript else '[No clear transcript]'}")

        threat_data = load_threat_reference("Threat_Level.xlsx")

        print("\nCalling dispatcher assistant (OpenAI)...")
        recommendation = call_archia_dispatch_agent(
            agent_name=os.environ.get("ARCHIA_AGENT_NAME", "Summary Generation"),
            transcript=transcript,
            dims=dims,
            scores=scores,
            threat_data=threat_data,
        )

        print("\n================ DISPATCH RECOMMENDATION ================")
        print(recommendation)
        print("========================================================\n")

        return {
            "filename": audio.filename,
            "content_type": audio.content_type,
            "transcript": transcript,
            "emotion_dims": dims,
            "acoustic_scores": scores,
            "recommendation": recommendation,
        }

    except KeyError:
        raise HTTPException(
            status_code=500,
            detail="ARCHIA_TOKEN is not set. Export ARCHIA_TOKEN in your environment.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
