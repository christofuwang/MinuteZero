# dispatch_assistant.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import sounddevice as sd
import speech_recognition as sr
import json
import pandas as pd

from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# OpenAI (Responses API)
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
LOCAL_DIR = "./pretrained_audeering"
SAMPLE_RATE = 16000
DURATION_SEC = 5

# Pick device (Apple Silicon MPS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the reference table
def load_threat_reference(filepath="Threat_Level.xlsx"):
    try:
        # Assuming columns are: 'NatureofReport', 'countofReports', 'threat_level'
        df = pd.read_excel(filepath)
        
        # Convert to a list of dicts for easy JSON serialization
        # Example output: [{'NatureofReport': 'Fire', 'threat_level': 'High', ...}, ...]
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Warning: Could not load threat reference file: {e}")
        return []
    
# -----------------------------
# Emotion model (same as yours)
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
        hidden_states = outputs.last_hidden_state          # (B, T, H)
        pooled = hidden_states.mean(dim=1)                 # (B, H)
        logits = self.classifier(pooled)                   # (B, 3)
        return pooled, logits

def load_or_download():
    """Loads processor+model from LOCAL_DIR if present; otherwise downloads from HF and saves locally."""
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

processor, emotion_model = load_or_download()
emotion_model = emotion_model.to(device)
emotion_model.eval()


# -----------------------------
# Audio capture
# -----------------------------
def record_audio(duration=DURATION_SEC, sample_rate=SAMPLE_RATE) -> np.ndarray:
    print(f"\nRecording {duration}s...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("Done.\n")
    return audio.squeeze()


# -----------------------------
# Emotion dims + dispatch concern (same as yours)
# -----------------------------
@torch.no_grad()
def predict_dims(audio: np.ndarray):
    proc = processor(audio, sampling_rate=SAMPLE_RATE)
    x = torch.tensor(proc["input_values"], dtype=torch.float32).to(device)  # (1, L)
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


# -----------------------------
# Transcription (SpeechRecognition + Google) for *the recorded 5s*
# -----------------------------
def transcribe_google(audio_f32: np.ndarray, sample_rate=SAMPLE_RATE) -> str:
    """
    Convert float32 [-1,1] audio to 16-bit PCM and transcribe via recognize_google.
    Note: This sends audio to Google and requires internet.
    """
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True

    # float32 -> int16 PCM bytes
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


# -----------------------------
# Chatbot call (OpenAI Responses API)
# -----------------------------
'''
def call_dispatch_chatbot(transcript: str, dims: dict, scores: dict) -> str:
    """
    Calls an LLM with dispatcher-oriented instructions and returns a structured recommendation.
    Requires OPENAI_API_KEY in env.
    """
    client = OpenAI()

    # Keep it readable + structured for dispatch usage
    instructions = (
        "You are an assistant that helps 911 dispatchers triage calls. "
        "Given a short transcript and coarse emotional/acoustic scores, you produce:\n"
        "1) A 2-4 sentence situation summary (no speculation beyond evidence)\n"
        "2) Key risk flags (bullets)\n"
        "3) Suggested dispatcher actions (bullets, practical and safety-focused)\n"
        "4) 3-6 high-yield follow-up questions to ask the caller\n"
        "5) A suggested priority level: LOW / MEDIUM / HIGH / IMMEDIATE, with 1-sentence rationale\n\n"
        "Rules:\n"
        "- Do NOT invent facts not present in transcript.\n"
        "- Treat emotion scores as noisy signals.\n"
        "- If the transcript is empty/unclear, say so and focus on questions.\n"
        "- Keep it concise and operational.\n"
    )

    # Provide all the model outputs as context
    input_text = f"""
TRANSCRIPT (may be empty or partial):
{transcript if transcript else "[No clear transcript detected]"}

EMOTION DIMS (approx 0..1, noisy):
- arousal: {dims.get("arousal", None)}
- dominance: {dims.get("dominance", None)}
- valence: {dims.get("valence", None)}

ACOUSTIC SIGNALS (engineered features):
- dispatch_concern: {scores.get("dispatch_concern", None)}
- rms: {scores.get("rms", None)}
- zcr: {scores.get("zcr", None)}
- yell_score: {scores.get("yell_score", None)}
- whisper_score: {scores.get("whisper_score", None)}
"""

    # Use Responses API (recommended shape)
    resp = client.responses.create(
        model="gpt-5",
        instructions=instructions,
        input=input_text,
    )
    return resp.output_text
'''
def call_archia_dispatch_agent(agent_name: str, transcript: str, dims: dict, scores: dict, threat_data: list) -> str:
    """
    Calls an Archia Agent (OpenAI-compatible Responses API).
    Requires: ARCHIA_TOKEN env var.
    """
    token = os.environ["ARCHIA_TOKEN"]

    client = OpenAI(
        base_url="https://registry.archia.app/v1",          # Archia base URL :contentReference[oaicite:3]{index=3}
        api_key="not-used",                                 # per Archia guide :contentReference[oaicite:4]{index=4}
        default_headers={"Authorization": f"Bearer {token}"} # Bearer token :contentReference[oaicite:5]{index=5}
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
            )
        }
    }

    resp = client.responses.create(
        model=f"agent:{agent_name}",                         # invoke agent :contentReference[oaicite:6]{index=6}
        input=json.dumps(payload)
    )

    # Same access pattern shown in guide :contentReference[oaicite:7]{index=7}
    return resp.output[0].content[0].text

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Record
    audio = record_audio()

    # 2) Emotion dims + engineered concern score
    dims = predict_dims(audio)
    scores = compute_dispatch_concern(audio, dims)

    print("Emotion Dimensions (approx 0..1):")
    for k, v in dims.items():
        print(f"  {k}: {v:.4f}")

    print("\nAcoustic signals:")
    print(f"  rms: {scores['rms']:.5f} | zcr: {scores['zcr']:.3f}")
    print(f"  yell_score: {scores['yell_score']:.3f} | whisper_score: {scores['whisper_score']:.3f}")
    print(f"  dispatch_concern: {scores['dispatch_concern']:.4f}")


    threat_data = load_threat_reference("Threat_Level.xlsx")
    # 3) Transcribe the same 5s chunk
    print("\nTranscribing (Google Speech Recognition)...")
    transcript = transcribe_google(audio)
    print(f"Transcript: {transcript if transcript else '[No clear transcript]'}")

    # 4) Call chatbot for dispatcher recommendation
    print("\nCalling dispatcher assistant (OpenAI)...")
    try:
        recommendation = call_archia_dispatch_agent(
        agent_name="Summary Generation",   # <-- use your exact agent name from Archia UI
        transcript=transcript,
        dims=dims,
        scores=scores,
        threat_data = threat_data
        )
    except Exception as e:
        print(f"\n[OpenAI call error] {e}")
        print("Make sure you set ARCHIA_TOKEN in your environment.")
        return

    print("\n================ DISPATCH RECOMMENDATION ================")
    print(recommendation)
    print("========================================================\n")


if __name__ == "__main__":
    main()