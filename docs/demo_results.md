# Résultats de démonstration

| Fichier audio | Transcription | Sentiment attendu | Sentiment prédit | Confiance |
|---|---|---|---|---|
| exemple_positif.mp3 | bonjour je vous appelle pour vous dire que je suis vraiment très satisfait du service que jai reçu la semaine dernière le technicien a été ponctuel professionnel et le problème a été réglé rapidement merci beaucoup cétait parfait | Positif | Positif | 0.9798 |
| exemple_negatif.mp3 | bonjour jappelle car je suis extrêmement déçu par votre service jattends une réponse depuis trois semaines et personne ne me rappelle le service était catastrophiques et je ne compte plus jamais commande chez vous | Négatif | Négatif | 0.9886 |
| exemple_neutre.mp3  | bonjour je vous appelle simplement pour savoir à quelle heure ouvre le magasin demain matin et si vous êtes ouvert le dimanche également merci de votre réponse | Neutre  | Neutre | 0.4655 |

## Remarques

- Les 3 fichiers ont été générés par synthèse vocale (Google Text-to-Speech, `gTTS`) à partir de phrases rédigées manuellement, formulées comme de véritables appels clients (formule d'ouverture, contexte, formule de clôture) pour chaque classe de sentiment.
- Ces résultats ont été obtenus en interrogeant l'API FastAPI (`POST /predict`) exposée publiquement via un tunnel ngrok depuis Colab, validant ainsi le pipeline de bout en bout dans des conditions proches de la production.
- Le cas neutre nécessite un seuil de confiance (voir `src/sentiment/analyzer.py`, `CONFIDENCE_THRESHOLD_FOR_NEUTRAL`) : le modèle de sentiment, entraîné sur des avis clients (5 étoiles), a tendance à mal classer les phrases purement informatives sans opinion exprimée. Sous ce seuil, la prédiction bascule automatiquement vers "neutre".
