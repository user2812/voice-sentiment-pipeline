# 🎙️ Détection Automatique de Sentiment dans des Appels Vocaux

**Module :** Deep Learning 2 — DIT (Dakar Institute of Technology)
**Auteur :** Fatim [Nom]
**Encadrant :** Pr. Abdouaziz
**Date limite :** 31 juillet 2026, 23h59 GMT

Pipeline automatisé qui transcrit un appel vocal (ASR) puis analyse le sentiment
du client (NLP) pour le classer en **Positif**, **Négatif** ou **Neutre**, avec un
score de confiance.

---

## 📐 Architecture

```
Audio (.wav/.mp3)
      │
      ▼
Prétraitement (mono, 16kHz, normalisation)
      │
      ▼
ASR — Wav2Vec 2.0 (jonatasgrosman/wav2vec2-large-xlsr-53-french)
      │
      ▼
Sentiment — CamemBERT/DistilBERT fine-tuné
      │
      ▼
{ transcription, sentiment, confidence }
```

---

## 🗂️ Structure du projet

```
voice-sentiment-pipeline/
├── src/
│   ├── asr/            # Prétraitement audio + transcription Wav2Vec 2.0
│   ├── sentiment/      # Modèle de classification de sentiment
│   ├── api/            # API REST (FastAPI)
│   └── ui/             # Interface Gradio
├── tests/               # Tests unitaires (pytest)
├── notebooks/           # Exploration / expérimentation
├── test_audio/          # 3 fichiers audio de démo (1 par classe)
├── docs/                # Documentation complémentaire, schémas
├── requirements.txt
├── Dockerfile           # (bonus)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le repo
git clone https://github.com/<username>/voice-sentiment-pipeline.git
cd voice-sentiment-pipeline

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate   # ou venv\Scripts\activate sous Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

### Interface Gradio

```bash
python src/ui/app.py
```

Ouvre l'interface dans le navigateur, upload un fichier `.wav` ou `.mp3`,
et affiche la transcription intermédiaire ainsi que le sentiment détecté.

### API REST

```bash
uvicorn src.api.main:app --reload
```

Exemple d'appel :

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@test_audio/exemple_positif.mp3"
```

Réponse JSON attendue :

```json
{
  "transcription": "Je suis très satisfait du service, merci beaucoup.",
  "sentiment": "positif",
  "confidence": 0.94
}
```

---

## 🧠 Modèles utilisés

| Tâche      | Modèle                                                      | Justification                                                                 |
|------------|--------------------------------------------------------------|--------------------------------------------------------------------------------|
| ASR        | `jonatasgrosman/wav2vec2-large-xlsr-53-french`               | Fine-tuné spécifiquement pour le français, bonnes performances (WER) publiées sur Hugging Face, facilement utilisable via `transformers`. |
| Sentiment  | `[à compléter — ex. CamemBERT fine-tuné]`                    | [à compléter — pourquoi ce modèle plutôt qu'un autre]                          |

Liens Hugging Face : *(à ajouter)*

---

## ✅ Gestion des erreurs

Le pipeline gère proprement :
- Formats de fichiers non supportés (autre que `.wav`/`.mp3`)
- Fichiers audio vides ou corrompus
- Audio silencieux (aucune transcription détectée)
- Fichiers dépassant la durée maximale (5 minutes)

---

## 🧪 Démonstration

3 fichiers audio de test sont fournis dans `test_audio/`, un par classe de
sentiment (positif, négatif, neutre), avec les résultats attendus documentés
dans `docs/demo_results.md`.

---

## 📊 Évaluation quantitative (bonus)

*(à compléter si réalisé : WER pour l'ASR, accuracy/F1 pour le sentiment sur un
petit jeu de données annoté)*

---

## 🐳 Docker (bonus)

```bash
docker build -t voice-sentiment-pipeline .
docker run -p 8000:8000 voice-sentiment-pipeline
```

---

## ⚠️ Limites connues

- Le pipeline est optimisé pour le français uniquement.
- La qualité de la transcription dépend fortement du bruit de fond et de la qualité d'enregistrement.
- Le modèle de sentiment n'a pas été entraîné spécifiquement sur du langage oral transcrit (peut contenir des disfluences, hésitations, etc.).
- Durée maximale supportée : 5 minutes par fichier.

---

## 📄 Licence

Projet académique — Deep Learning 2, DIT, 2026.
