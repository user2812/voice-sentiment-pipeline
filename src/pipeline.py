"""
Pipeline complet : audio -> transcription (ASR) -> sentiment (NLP).
Point d'entrée unique utilisé par l'API et l'interface Gradio.
"""

from src.asr.transcriber import Transcriber
from src.asr.preprocessing import AudioValidationError
from src.sentiment.analyzer import SentimentAnalyzer


class VoiceSentimentPipeline:
    def __init__(self):
        self.transcriber = Transcriber()
        self.analyzer = SentimentAnalyzer()

    def predict(self, filepath: str) -> dict:
        """
        Exécute le pipeline complet sur un fichier audio.
        Retourne {"transcription": str, "sentiment": str, "confidence": float}
        ou {"error": str} en cas de problème.
        """
        try:
            transcription = self.transcriber.transcribe_file(filepath)
        except AudioValidationError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Erreur lors de la transcription : {e}"}

        if not transcription:
            return {"error": "Transcription vide, impossible d'analyser le sentiment."}

        try:
            sentiment_result = self.analyzer.predict(transcription)
        except Exception as e:
            return {"error": f"Erreur lors de l'analyse de sentiment : {e}"}

        return {
            "transcription": transcription,
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
        }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage : python -m src.pipeline <chemin_audio>")
        sys.exit(1)

    pipeline = VoiceSentimentPipeline()
    result = pipeline.predict(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
