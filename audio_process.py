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
        # ✅ REQUIRED in Transformers v5 so tied-weight metadata gets created
        self.post_init()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state          # (B, T, H)
        pooled = hidden_states.mean(dim=1)                 # (B, H)
        logits = self.classifier(pooled)                   # (B, 3)
        return pooled, logits

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = EmotionModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def record_audio(duration=DURATION_SEC, sample_rate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype="float32")
    sd.wait()
    print("Done.")
    return audio.squeeze()

@torch.no_grad()
def predict_dims(audio: np.ndarray):
    proc = processor(audio, sampling_rate=SAMPLE_RATE)
    # Avoid torch<->numpy bridge issues: build tensor directly
    x = torch.tensor(proc["input_values"], dtype=torch.float32).to(device)  # shape (1, L)
    _, logits = model(x)
    dims = logits.squeeze().detach().cpu().tolist()

    # Model outputs arousal, dominance, valence approx 0..1 :contentReference[oaicite:1]{index=1}
    return {"arousal": float(dims[0]), "dominance": float(dims[1]), "valence": float(dims[2])}

if __name__ == "__main__":
    audio = record_audio()
    dims = predict_dims(audio)

    print("\nEmotion Dimensions (approx 0..1):")
    for k, v in dims.items():
        print(f"{k}: {v:.4f}")

    distress = (1.0 - dims["valence"]) * dims["arousal"]
    distress = float(np.clip(distress, 0.0, 1.0))
    print(f"\nDistress Proxy Score (0-1): {distress:.4f}")



