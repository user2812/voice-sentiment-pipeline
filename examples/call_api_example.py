"""
Exemple de script Python pour appeler l'API voice-sentiment-pipeline.
Usage : python examples/call_api_example.py chemin/vers/audio.mp3
"""

import sys
import requests

API_URL = "http://127.0.0.1:8000/predict"


def call_api(audio_path: str):
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path, f, "audio/mpeg")}
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        print("Réponse de l'API :")
        print(response.json())
    else:
        print(f"Erreur ({response.status_code}) :")
        print(response.json())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python call_api_example.py <chemin_audio>")
        sys.exit(1)

    call_api(sys.argv[1])
