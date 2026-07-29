"""
Tests unitaires pour src/pipeline.py.
Les classes Transcriber et SentimentAnalyzer sont mockées pour éviter de
charger les vrais modèles (Wav2Vec2 / DistilCamemBERT) à chaque test -> rapide,
aucun téléchargement nécessaire.
"""

from unittest.mock import MagicMock, patch

from src.asr.preprocessing import AudioValidationError
from src.pipeline import VoiceSentimentPipeline


def _build_pipeline_with_mocks(transcription="bonjour", sentiment_result=None,
                                 transcribe_side_effect=None):
    """Construit un VoiceSentimentPipeline dont les 2 modèles sont mockés."""
    sentiment_result = sentiment_result or {"sentiment": "positif", "confidence": 0.9}

    with patch("src.pipeline.Transcriber") as MockTranscriber, \
         patch("src.pipeline.SentimentAnalyzer") as MockAnalyzer:

        mock_transcriber_instance = MagicMock()
        if transcribe_side_effect:
            mock_transcriber_instance.transcribe_file.side_effect = transcribe_side_effect
        else:
            mock_transcriber_instance.transcribe_file.return_value = transcription
        MockTranscriber.return_value = mock_transcriber_instance

        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.predict.return_value = sentiment_result
        MockAnalyzer.return_value = mock_analyzer_instance

        pipeline = VoiceSentimentPipeline()

    return pipeline


def test_pipeline_happy_path():
    pipeline = _build_pipeline_with_mocks(
        transcription="je suis très content",
        sentiment_result={"sentiment": "positif", "confidence": 0.95},
    )

    result = pipeline.predict("fake_path.wav")

    assert result == {
        "transcription": "je suis très content",
        "sentiment": "positif",
        "confidence": 0.95,
    }


def test_pipeline_handles_audio_validation_error():
    pipeline = _build_pipeline_with_mocks(
        transcribe_side_effect=AudioValidationError("Fichier audio vide.")
    )

    result = pipeline.predict("fake_path.wav")

    assert "error" in result
    assert "vide" in result["error"]


def test_pipeline_handles_empty_transcription():
    pipeline = _build_pipeline_with_mocks(transcription="")

    result = pipeline.predict("fake_path.wav")

    assert "error" in result
    assert "vide" in result["error"].lower() or "transcription" in result["error"].lower()


def test_pipeline_handles_unexpected_transcription_error():
    pipeline = _build_pipeline_with_mocks(
        transcribe_side_effect=RuntimeError("erreur inattendue du modèle")
    )

    result = pipeline.predict("fake_path.wav")

    assert "error" in result
    assert "transcription" in result["error"].lower()
