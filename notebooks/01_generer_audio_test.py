# ============================================================
# À COPIER DANS DES CELLULES GOOGLE COLAB (une section = une cellule)
# Génère 3 fichiers audio de test (positif / négatif / neutre) avec gTTS
# ============================================================

# --- Cellule 1 : installer gTTS ---
# !pip install -q gtts

# --- Cellule 2 : générer les 3 fichiers de test ---
from gtts import gTTS
import os

os.makedirs("test_audio", exist_ok=True)

phrases = {
    "exemple_positif": (
        "Bonjour, je vous appelle pour vous dire que je suis vraiment très satisfait "
        "du service que j'ai reçu la semaine dernière. Le technicien a été ponctuel, "
        "professionnel, et le problème a été réglé rapidement. Merci beaucoup, c'était parfait."
    ),
    "exemple_negatif": (
        "Bonjour, j'appelle car je suis extrêmement déçu par votre service. "
        "J'attends une réponse depuis trois semaines et personne ne me rappelle. "
        "Le service était catastrophique et je ne compte plus jamais commander chez vous."
    ),
    "exemple_neutre": (
        "Bonjour, je vous appelle simplement pour savoir à quelle heure ouvre le magasin "
        "demain matin, et si vous êtes ouverts le dimanche également. Merci de votre réponse."
    ),
}

for filename, text in phrases.items():
    tts = gTTS(text=text, lang="fr")
    path = f"test_audio/{filename}.mp3"
    tts.save(path)
    print(f"Créé : {path}")

# --- Cellule 3 : écouter un fichier pour vérifier (optionnel) ---
# from IPython.display import Audio
# Audio("test_audio/exemple_positif.mp3")

# --- Cellule 4 : tester le pipeline complet sur les 3 fichiers ---
import sys
sys.path.append(".")
from src.pipeline import VoiceSentimentPipeline

pipeline = VoiceSentimentPipeline()

for filename in phrases:
    path = f"test_audio/{filename}.mp3"
    result = pipeline.predict(path)
    print(f"\n--- {filename} ---")
    print(result)
