# Résultats de démonstration

| Fichier audio | Transcription | Sentiment attendu | Sentiment prédit | Confiance |
|---|---|---|---|---|
| exemple_positif.mp3 | (à re-générer et re-tester) | Positif | (à compléter) | (à compléter) |
| exemple_negatif.mp3 | (à re-générer et re-tester) | Négatif | (à compléter) | (à compléter) |
| exemple_neutre.mp3  | (à re-générer et re-tester) | Neutre  | (à compléter) | (à compléter) |

## Remarques

- Les 3 fichiers ont été générés par synthèse vocale (Google Text-to-Speech, `gTTS`) à partir de phrases rédigées manuellement, formulées comme de véritables appels clients (formule d'ouverture, contexte, formule de clôture) pour chaque classe de sentiment.
- Le cas neutre nécessite un seuil de confiance (voir `src/sentiment/analyzer.py`, `CONFIDENCE_THRESHOLD_FOR_NEUTRAL`) : le modèle de sentiment, entraîné sur des avis clients (5 étoiles), a tendance à mal classer les phrases purement informatives sans opinion exprimée. Sous ce seuil, la prédiction bascule automatiquement vers "neutre".
