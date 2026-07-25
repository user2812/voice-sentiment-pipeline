# ============================================================
# À COPIER DANS DES CELLULES GOOGLE COLAB (une section = une cellule)
# ============================================================

# --- Cellule 1 : cloner ton repo ---
# !git clone https://github.com/user2812/voice-sentiment-pipeline.git
# %cd voice-sentiment-pipeline

# --- Cellule 2 : installer les dépendances ---
# !pip install -q -r requirements.txt

# --- Cellule 3 : uploader un fichier audio de test ---
# from google.colab import files
# uploaded = files.upload()  # sélectionne un .wav ou .mp3 depuis ton PC

# --- Cellule 4 : tester le pipeline complet ---
import sys
sys.path.append(".")  # pour que "from src..." fonctionne depuis Colab

from src.pipeline import VoiceSentimentPipeline

pipeline = VoiceSentimentPipeline()

# Remplace par le nom exact du fichier uploadé
audio_path = "test_audio/exemple_positif.wav"

result = pipeline.predict(audio_path)
print(result)
