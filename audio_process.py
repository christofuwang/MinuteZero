import torch
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)

model.eval()

def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype="float32")
    sd.wait()
    print("Done.")
    return audio.squeeze()
    
def predict(audio):
    audio = np.asarray(audio, dtype=np.float32).squeeze()  # ensure 1D float32

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(inputs.input_values.to(device))

    scores = outputs.logits.squeeze().cpu().numpy()
    labels = model.config.id2label
    return {labels[i]: float(scores[i]) for i in range(len(scores))}

    return results

if __name__ == "__main__":
    audio = record_audio()

    emotion_scores = predict(audio)

    print("\nEmotion Dimensions:")
    for k, v in emotion_scores.items():
        print(f"{k}: {v:.4f}")

    valence = emotion_scores.get("valence", 0)
    arousal = emotion_scores.get("arousal", 0)

    distress = (1 - valence) * arousal

    print(f"\nDistress Proxy Score (0-1 approx): {distress:.4f}")

