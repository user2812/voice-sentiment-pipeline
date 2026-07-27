# Résultats de démonstration

| Fichier audio | Transcription | Sentiment attendu | Sentiment prédit | Confiance |
|---|---|---|---|---|
| exemple_positif.mp3 | je suis vraiment très satisfait du service merci beaucoup cétait parfait | Positif | Positif | 0.9897 |
| exemple_negatif.mp3 | je suis extrêmement déçu le service était catastrophique et je ne reviendrai jamais | Négatif | Négatif | 0.9933 |
| exemple_neutre.mp3  | bonjour je voudrais savoir à quelle heure ouvre le magasin demain matin sil vous plait | Neutre  | Neutre | 0.5298 |

## Remarques

- Les 3 fichiers ont été générés par synthèse vocale (Google Text-to-Speech, `gTTS`) à partir de phrases rédigées manuellement pour représenter chaque classe de sentiment.
- La transcription (Wav2Vec 2.0) est fidèle au texte original dans les 3 cas (quelques fautes mineures liées à la ponctuation/accents, normales pour un modèle ASR).
- Le cas neutre a nécessité l'ajout d'un seuil de confiance (voir `src/sentiment/analyzer.py`, `CONFIDENCE_THRESHOLD_FOR_NEUTRAL`) : le modèle de sentiment, entraîné sur des avis clients (5 étoiles), a tendance à mal classer les phrases purement informatives sans opinion exprimée. Sous ce seuil, la prédiction bascule automatiquement vers "neutre".
