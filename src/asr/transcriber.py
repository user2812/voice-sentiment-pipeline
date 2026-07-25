"""
Transcription audio -> texte via Wav2Vec 2.0 fine-tuné pour le français.
Modèle : jonatasgrosman/wav2vec2-large-xlsr-53-french
"""

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.asr.preprocessing import load_and_preprocess, TARGET_SR

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-french"


class Transcriber:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Chargement du modèle ASR '{model_name}' sur {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def transcribe_waveform(self, waveform) -> str:
        """Transcrit un waveform numpy (déjà prétraité à 16kHz, mono) en texte."""
        inputs = self.processor(
            waveform, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription.strip()

    def transcribe_file(self, filepath: str) -> str:
        """Charge, prétraite, et transcrit un fichier audio directement."""
        waveform = load_and_preprocess(filepath)
        return self.transcribe_waveform(waveform)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage : python transcriber.py <chemin_audio>")
        sys.exit(1)

    transcriber = Transcriber()
    text = transcriber.transcribe_file(sys.argv[1])
    print(f"Transcription : {text}")
