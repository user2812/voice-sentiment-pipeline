# Évaluation Quantitative (bonus)

Évaluation du pipeline sur le jeu de données annoté fourni (`test_audio/`, 3 fichiers,
un par classe de sentiment). Script reproductible : `src/evaluate.py`.

⚠️ **Note de méthodologie** : ce jeu de données est volontairement petit (n=3).
Il sert à démontrer la méthodologie d'évaluation (calcul du WER, de l'accuracy
et du F1) plutôt qu'à produire une mesure statistiquement robuste, qui
nécessiterait un corpus de test bien plus large et plus varié.

## 1. WER (Word Error Rate) — Transcription ASR

Le WER compare, mot à mot, la transcription réellement obtenue par Wav2Vec 2.0
à la phrase de référence (le texte utilisé pour générer l'audio via gTTS).

| Fichier | WER | Erreurs | Mots (référence) |
|---|---|---|---|
| exemple_positif.mp3 | 5.26% | 2 | 38 |
| exemple_negatif.mp3 | 5.88% | 2 | 34 |
| exemple_neutre.mp3 | 3.70% | 1 | 27 |
| **Global (moyenne pondérée)** | **5.05%** | 5 | 99 |

Les erreurs observées sont mineures : absence de ponctuation et de majuscules
(normal pour un modèle CTC comme Wav2Vec 2.0, qui ne prédit ni l'une ni
l'autre), et quelques légères déformations phonétiques (ex. "la semaine" →
"làsemaine"). Le sens du message reste intact dans les 3 cas.

## 2. Accuracy / Précision / Rappel / F1 — Classification de Sentiment

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| Positif | 100.0% | 100.0% | 100.0% |
| Négatif | 100.0% | 100.0% | 100.0% |
| Neutre | 100.0% | 100.0% | 100.0% |

**Accuracy globale : 100% (3/3)**
**F1-macro : 100%**

Ce score parfait s'explique par la taille très réduite du jeu de test (3
exemples) : il confirme que le pipeline fonctionne correctement sur des cas
clairement tranchés, mais ne garantit pas une performance identique sur des
cas plus ambigus ou un volume de données plus important (voir la section
"Limites connues" du README).

## Reproduire cette évaluation

```bash
python src/evaluate.py
```
