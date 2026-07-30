"""
Évaluation quantitative du pipeline (bonus) :
- WER (Word Error Rate) pour la transcription ASR
- Accuracy / Précision / Rappel / F1 pour la classification de sentiment

Le jeu de données annoté est volontairement petit (3 fichiers, un par classe de
sentiment) : il sert à illustrer la méthodologie d'évaluation plutôt qu'à
produire une mesure statistiquement robuste (qui nécessiterait un corpus bien
plus large).
"""

import re


def normalize(text: str) -> str:
    """Normalise un texte pour la comparaison WER : minuscule, sans ponctuation."""
    text = text.lower()
    text = re.sub(r"[.,!?;:'\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_error_rate(reference: str, hypothesis: str):
    """Calcule le WER (distance de Levenshtein au niveau mot) entre une
    transcription de référence et la transcription prédite par le modèle."""
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()
    n, m = len(ref_words), len(hyp_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    distance = dp[n][m]
    wer = distance / n if n > 0 else 0.0
    return wer, distance, n


def precision_recall_f1(cls: str, samples: list):
    tp = sum(1 for s in samples if s["predicted_sentiment"] == cls and s["expected_sentiment"] == cls)
    fp = sum(1 for s in samples if s["predicted_sentiment"] == cls and s["expected_sentiment"] != cls)
    fn = sum(1 for s in samples if s["predicted_sentiment"] != cls and s["expected_sentiment"] == cls)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(samples: list):
    """
    samples : liste de dicts avec les clés
        name, reference (transcription attendue), hypothesis (transcription prédite),
        expected_sentiment, predicted_sentiment
    """
    print(f"{'Fichier':<20} {'WER':>8} {'Erreurs':>10} {'Mots ref':>10}")
    total_errors, total_words = 0, 0
    for s in samples:
        wer, dist, n = word_error_rate(s["reference"], s["hypothesis"])
        total_errors += dist
        total_words += n
        print(f"{s['name']:<20} {wer * 100:>6.2f}% {dist:>10} {n:>10}")

    global_wer = total_errors / total_words if total_words else 0
    print(f"\nWER global (moyenne pondérée) : {global_wer * 100:.2f}%")

    classes = sorted(set(s["expected_sentiment"] for s in samples))
    correct = sum(1 for s in samples if s["expected_sentiment"] == s["predicted_sentiment"])
    accuracy = correct / len(samples)
    print(f"\nAccuracy sentiment : {accuracy * 100:.1f}% ({correct}/{len(samples)})")

    print(f"\n{'Classe':<12}{'Précision':>12}{'Rappel':>10}{'F1':>8}")
    f1_scores = []
    for cls in classes:
        p, r, f1 = precision_recall_f1(cls, samples)
        f1_scores.append(f1)
        print(f"{cls:<12}{p * 100:>11.1f}%{r * 100:>9.1f}%{f1 * 100:>7.1f}%")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    print(f"\nF1-macro : {macro_f1 * 100:.1f}%")

    return {"wer": global_wer, "accuracy": accuracy, "macro_f1": macro_f1}


if __name__ == "__main__":
    # Jeu de données annoté : phrases de référence (utilisées pour générer les
    # audios via gTTS) vs transcriptions réellement obtenues par le pipeline.
    samples = [
        {
            "name": "exemple_positif",
            "reference": "Bonjour, je vous appelle pour vous dire que je suis vraiment très satisfait du service que j'ai reçu la semaine dernière. Le technicien a été ponctuel, professionnel, et le problème a été réglé rapidement. Merci beaucoup, c'était parfait.",
            "hypothesis": "bonjour je vous appelle pour vous dire que je suis vraiment très satisfait du service que jai reçu làsemaine dernière le technicien a été ponctuel professionnel et le problème a été réglé rapidement  merci beaucoup cétait parfait",
            "expected_sentiment": "positif",
            "predicted_sentiment": "positif",
        },
        {
            "name": "exemple_negatif",
            "reference": "Bonjour, j'appelle car je suis extrêmement déçu par votre service. J'attends une réponse depuis trois semaines et personne ne me rappelle. Le service était catastrophique et je ne compte plus jamais commander chez vous.",
            "hypothesis": "bonjour jappelle car je suis extrêmement déçu par votre service jattends une réponse depuis trois semaines et personne ne me rappelle  le service était catastrophiques et je ne compte plus jamais commande chez vous",
            "expected_sentiment": "négatif",
            "predicted_sentiment": "négatif",
        },
        {
            "name": "exemple_neutre",
            "reference": "Bonjour, je vous appelle simplement pour savoir à quelle heure ouvre le magasin demain matin, et si vous êtes ouverts le dimanche également. Merci de votre réponse.",
            "hypothesis": "bonjour je vous appelle simplement pour savoir à quelle heure ouvre le magasin demain matin et si vous êtes ouvert le dimanche également  merci de votre réponse",
            "expected_sentiment": "neutre",
            "predicted_sentiment": "neutre",
        },
    ]

    evaluate(samples)
