"""
Prétraitement audio : chargement, conversion mono, rééchantillonnage 16kHz,
normalisation, et validation (format, durée max, silence).
"""

import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000
MAX_DURATION_SEC = 5 * 60  # 5 minutes
SUPPORTED_FORMATS = (".wav", ".mp3")
SILENCE_RMS_THRESHOLD = 1e-4  # en dessous de ça, on considère l'audio comme silencieux


class AudioValidationError(Exception):
    """Levée quand un fichier audio ne respecte pas les contraintes du pipeline."""
    pass


def validate_file_format(filepath: str) -> None:
    if not filepath.lower().endswith(SUPPORTED_FORMATS):
        raise AudioValidationError(
            f"Format non supporté pour '{filepath}'. Formats acceptés : {SUPPORTED_FORMATS}"
        )


def load_and_preprocess(filepath: str) -> np.ndarray:
    """
    Charge un fichier audio, le convertit en mono, le rééchantillonne à 16kHz,
    normalise l'amplitude, et vérifie qu'il respecte les contraintes du projet.

    Retourne un np.ndarray 1D (waveform) prêt à être passé au modèle ASR.
    """
    validate_file_format(filepath)

    try:
        # sr=None garde le sample rate d'origine, mono=True force la conversion mono
        waveform, original_sr = librosa.load(filepath, sr=None, mono=True)
    except Exception as e:
        raise AudioValidationError(f"Impossible de lire le fichier audio : {e}")

    if waveform is None or len(waveform) == 0:
        raise AudioValidationError("Le fichier audio est vide.")

    duration_sec = len(waveform) / original_sr
    if duration_sec > MAX_DURATION_SEC:
        raise AudioValidationError(
            f"Durée ({duration_sec:.1f}s) supérieure à la limite autorisée "
            f"({MAX_DURATION_SEC}s)."
        )

    # Rééchantillonnage à 16kHz si nécessaire
    if original_sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=TARGET_SR)

    # Normalisation de l'amplitude entre -1 et 1
    max_amplitude = np.max(np.abs(waveform))
    if max_amplitude > 0:
        waveform = waveform / max_amplitude

    # Détection de silence (audio quasi vide après normalisation)
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms < SILENCE_RMS_THRESHOLD:
        raise AudioValidationError("Le fichier audio semble silencieux (aucun signal détecté).")

    return waveform


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage : python preprocessing.py <chemin_audio>")
        sys.exit(1)

    audio = load_and_preprocess(sys.argv[1])
    print(f"Audio chargé : {len(audio)} échantillons, {len(audio) / TARGET_SR:.2f}s à {TARGET_SR}Hz")
