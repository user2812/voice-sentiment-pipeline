"""
Tests unitaires pour src/asr/preprocessing.py
Génère de petits fichiers audio synthétiques avec numpy/soundfile (rapide,
aucun téléchargement de modèle nécessaire).
"""

import numpy as np
import soundfile as sf
import pytest

from src.asr.preprocessing import (
    load_and_preprocess,
    validate_file_format,
    AudioValidationError,
    TARGET_SR,
)


def _write_wav(path, duration_sec=1.0, sr=8000, silent=False, amplitude=0.5):
    """Génère un fichier .wav de test : un ton pur, ou du silence."""
    n_samples = int(duration_sec * sr)
    if silent:
        signal = np.zeros(n_samples, dtype=np.float32)
    else:
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        signal = amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(path, signal, sr)


def test_validate_file_format_accepts_wav_and_mp3():
    validate_file_format("audio.wav")
    validate_file_format("audio.mp3")
    validate_file_format("AUDIO.WAV")  # insensible à la casse


def test_validate_file_format_rejects_other_formats():
    with pytest.raises(AudioValidationError):
        validate_file_format("audio.ogg")
    with pytest.raises(AudioValidationError):
        validate_file_format("audio.txt")


def test_load_and_preprocess_valid_audio(tmp_path):
    filepath = tmp_path / "test.wav"
    _write_wav(str(filepath), duration_sec=1.0, sr=8000)

    waveform = load_and_preprocess(str(filepath))

    assert isinstance(waveform, np.ndarray)
    # Doit être rééchantillonné à TARGET_SR (16kHz), donc ~16000 échantillons pour 1s
    assert abs(len(waveform) - TARGET_SR) < 10
    # Normalisé : l'amplitude max doit être proche de 1
    assert np.max(np.abs(waveform)) <= 1.0 + 1e-6


def test_load_and_preprocess_rejects_silent_audio(tmp_path):
    filepath = tmp_path / "silence.wav"
    _write_wav(str(filepath), duration_sec=1.0, sr=8000, silent=True)

    with pytest.raises(AudioValidationError):
        load_and_preprocess(str(filepath))


def test_load_and_preprocess_rejects_wrong_format(tmp_path):
    filepath = tmp_path / "test.txt"
    filepath.write_text("ceci n'est pas un fichier audio")

    with pytest.raises(AudioValidationError):
        load_and_preprocess(str(filepath))


def test_load_and_preprocess_rejects_too_long_audio(tmp_path, monkeypatch):
    import src.asr.preprocessing as preprocessing_module

    # Réduit temporairement la durée max autorisée pour ne pas générer un vrai fichier de 5 minutes
    monkeypatch.setattr(preprocessing_module, "MAX_DURATION_SEC", 1)

    filepath = tmp_path / "too_long.wav"
    _write_wav(str(filepath), duration_sec=2.0, sr=8000)  # 2s > 1s (limite réduite)

    with pytest.raises(AudioValidationError):
        preprocessing_module.load_and_preprocess(str(filepath))
