from __future__ import annotations

import io
import os
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

try:
    import sounddevice as sd
except Exception:
    sd = None

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
LOCAL_DIR = "./pretrained_audeering"
SAMPLE_RATE = 16000
DURATION_SEC = 5


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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
    if os.path.isdir(LOCAL_DIR) and os.path.exists(os.path.join(LOCAL_DIR, "config.json")):
        print(f"Loading processor/model from local dir: {LOCAL_DIR}")
        processor = Wav2Vec2Processor.from_pretrained(LOCAL_DIR)
        model = EmotionModel.from_pretrained(LOCAL_DIR)
        return processor, model

    print(f"Downloading processor/model from HF: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = EmotionModel.from_pretrained(MODEL_NAME)

    print(f"Saving processor/model to local dir: {LOCAL_DIR}")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    processor.save_pretrained(LOCAL_DIR)
    model.save_pretrained(LOCAL_DIR)

    return processor, model


@lru_cache(maxsize=1)
def get_model_components():
    processor, model = load_or_download()
    model = model.to(device)
    model.eval()
    return processor, model


def record_audio(duration=DURATION_SEC, sample_rate=SAMPLE_RATE):
    if sd is None:
        raise RuntimeError("sounddevice is unavailable in this environment")

    print("Recording...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Done.")
    return audio.squeeze()


def _ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    return np.ascontiguousarray(audio)


def decode_audio_file_bytes(file_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(io.BytesIO(file_bytes), dtype="float32")
    except Exception as exc:
        raise ValueError(
            "Unsupported audio format. Please upload WAV/FLAC/OGG/AIFF or convert to WAV before sending."
        ) from exc

    audio = _ensure_mono_float32(np.asarray(audio))
    if audio.size == 0:
        raise ValueError("Audio payload decoded to empty signal")

    return audio, int(sample_rate)


def resample_audio(audio: np.ndarray, original_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if original_rate == target_rate:
        return audio
    if original_rate <= 0:
        raise ValueError(f"Invalid sample rate: {original_rate}")

    gcd = np.gcd(original_rate, target_rate)
    up = target_rate // gcd
    down = original_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


@torch.no_grad()
def predict_dims(audio: np.ndarray):
    processor, model = get_model_components()
    proc = processor(audio, sampling_rate=SAMPLE_RATE)
    x = torch.tensor(proc["input_values"], dtype=torch.float32).to(device)
    _, logits = model(x)
    dims = logits.squeeze().detach().cpu().tolist()
    return {
        "arousal": float(dims[0]),
        "dominance": float(dims[1]),
        "valence": float(dims[2]),
    }


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


def compute_dispatch_concern(audio: np.ndarray, dims: dict) -> dict:
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

    negativity = 1.0 - va
    low_control = 1.0 - do

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


def score_audio_array(audio: np.ndarray, sample_rate: int) -> dict:
    audio = _ensure_mono_float32(audio)
    audio = resample_audio(audio, sample_rate, SAMPLE_RATE)

    dims = predict_dims(audio)
    scores = compute_dispatch_concern(audio, dims)
    return {"emotion_dimensions": dims, "acoustic_signals": scores}


def score_audio_file_bytes(file_bytes: bytes) -> dict:
    audio, sample_rate = decode_audio_file_bytes(file_bytes)
    return score_audio_array(audio, sample_rate)


if __name__ == "__main__":
    audio = record_audio()
    result = score_audio_array(audio, SAMPLE_RATE)

    print("\nEmotion Dimensions (approx 0..1):")
    for k, v in result["emotion_dimensions"].items():
        print(f"{k}: {v:.4f}")

    scores = result["acoustic_signals"]
    print("\nAcoustic signals:")
    print(f"rms: {scores['rms']:.5f} | zcr: {scores['zcr']:.3f}")
    print(f"yell_score: {scores['yell_score']:.3f} | whisper_score: {scores['whisper_score']:.3f}")
    print(f"\nDispatch Concern Score (0-1): {scores['dispatch_concern']:.4f}")
