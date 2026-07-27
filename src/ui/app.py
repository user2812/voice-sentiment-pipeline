"""
Interface Gradio pour tester le pipeline complet :
audio -> transcription (affichée) -> sentiment + score de confiance.
"""

import gradio as gr

from src.pipeline import VoiceSentimentPipeline

# Chargé une seule fois au démarrage de l'app (pas à chaque appel)
print("Initialisation du pipeline (peut prendre 1-2 minutes)...")
pipeline = VoiceSentimentPipeline()
print("Pipeline prêt.")

SENTIMENT_EMOJIS = {
    "positif": "😊 Positif",
    "négatif": "😞 Négatif",
    "neutre": "😐 Neutre",
}


def analyze_audio(audio_filepath):
    """
    Callback appelé par Gradio quand l'utilisateur soumet un fichier audio.
    Retourne (transcription, sentiment_affiché, score_de_confiance).
    """
    if audio_filepath is None:
        return "Aucun fichier fourni.", "-", 0.0

    result = pipeline.predict(audio_filepath)

    if "error" in result:
        return f"⚠️ Erreur : {result['error']}", "-", 0.0

    sentiment_display = SENTIMENT_EMOJIS.get(result["sentiment"], result["sentiment"])
    return result["transcription"], sentiment_display, result["confidence"]


demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath", label="Fichier audio (.wav ou .mp3, max 5 min)"),
    outputs=[
        gr.Textbox(label="Transcription (ASR)", lines=3),
        gr.Textbox(label="Sentiment détecté"),
        gr.Number(label="Score de confiance"),
    ],
    title="🎙️ Détection de Sentiment dans des Appels Vocaux",
    description=(
        "Uploade un extrait audio en français. Le pipeline le transcrit "
        "(Wav2Vec 2.0) puis analyse le sentiment exprimé (DistilCamemBERT) : "
        "positif, négatif ou neutre."
    ),
    examples=[
        ["test_audio/exemple_positif.mp3"],
        ["test_audio/exemple_negatif.mp3"],
        ["test_audio/exemple_neutre.mp3"],
    ],
)


if __name__ == "__main__":
    demo.launch()
