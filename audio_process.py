import os
import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
LOCAL_DIR = "./pretrained_audeering"   # <- saved model lives here
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
        hidden_states = outputs.last_hidden_state          # (B, T, H)
        pooled = hidden_states.mean(dim=1)                 # (B, H)
        logits = self.classifier(pooled)                   # (B, 3)
        return pooled, logits

def load_or_download():
    """
    Loads processor+model from LOCAL_DIR if present; otherwise downloads from HF and saves locally.
    """
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

processor, model = load_or_download()
model = model.to(device)
model.eval()

def record_audio(duration=DURATION_SEC, sample_rate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("Done.")
    return audio.squeeze()

@torch.no_grad()
def predict_dims(audio: np.ndarray):
    proc = processor(audio, sampling_rate=SAMPLE_RATE)
    x = torch.tensor(proc["input_values"], dtype=torch.float32).to(device)  # (1, L)
    _, logits = model(x)
    dims = logits.squeeze().detach().cpu().tolist()
    return {"arousal": float(dims[0]), "dominance": float(dims[1]), "valence": float(dims[2])}

# -----------------------------
# Dispatch-oriented scoring
# -----------------------------
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

    negativity = (1.0 - va)
    low_control = (1.0 - do)

    concern = (
        0.35 * ar +
        0.20 * negativity +
        0.10 * low_control +
        0.20 * yell_score +
        0.15 * whisper_score
    )
    concern = float(np.clip(concern, 0.0, 1.0))

    return {
        "dispatch_concern": concern,
        "rms": rms,
        "zcr": zcr,
        "yell_score": float(np.clip(yell_score, 0.0, 1.0)),
        "whisper_score": float(np.clip(whisper_score, 0.0, 1.0)),
    }

if __name__ == "__main__":
    audio = record_audio()
    dims = predict_dims(audio)

    print("\nEmotion Dimensions (approx 0..1):")
    for k, v in dims.items():
        print(f"{k}: {v:.4f}")

    scores = compute_dispatch_concern(audio, dims)

    print("\nAcoustic signals:")
    print(f"rms: {scores['rms']:.5f} | zcr: {scores['zcr']:.3f}")
    print(f"yell_score: {scores['yell_score']:.3f} | whisper_score: {scores['whisper_score']:.3f}")

    print(f"\nDispatch Concern Score (0-1): {scores['dispatch_concern']:.4f}")

