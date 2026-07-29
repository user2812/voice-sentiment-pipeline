"""
Analyse de sentiment via DistilCamemBERT fine-tuné (cmarkea/distilcamembert-base-sentiment).
Le modèle prédit une note sur 5 étoiles ; on la mappe vers 3 classes :
    1-2 étoiles -> négatif
    3 étoiles   -> neutre
    4-5 étoiles -> positif
"""

from transformers import pipeline

MODEL_NAME = "cmarkea/distilcamembert-base-sentiment"

STAR_TO_CLASS = {
    "1 star": "négatif",
    "2 stars": "négatif",
    "3 stars": "neutre",
    "4 stars": "positif",
    "5 stars": "positif",
}

# Le modèle est entraîné sur des avis (opinions fortes). Sur un texte purement
# informatif, sans opinion exprimée, le score gagnant reste souvent proche de
# 0.5 (le modèle hésite). Sous ce seuil, on force la classe "neutre" plutôt
# que de garder une prédiction positif/négatif peu fiable.
CONFIDENCE_THRESHOLD_FOR_NEUTRAL = 0.6


def aggregate_scores(raw_scores):
    """
    Agrège les scores des 5 classes du modèle (1 star, ..., 5 stars) en 3 classes
    finales (positif/négatif/neutre), en sommant les scores de chaque groupe.

    raw_scores : liste de dicts au format [{"label": "1 star", "score": 0.1}, ...]
    Retourne : {"positif": float, "négatif": float, "neutre": float}
    """
    aggregated = {"positif": 0.0, "négatif": 0.0, "neutre": 0.0}
    for item in raw_scores:
        final_class = STAR_TO_CLASS[item["label"]]
        aggregated[final_class] += item["score"]
    return aggregated


def decide_sentiment(aggregated, threshold=CONFIDENCE_THRESHOLD_FOR_NEUTRAL):
    """
    Prend les scores agrégés (positif/négatif/neutre) et applique le seuil de
    confiance : si la classe gagnante est positif ou négatif mais avec un score
    trop faible (< threshold), on bascule sur "neutre".

    Retourne : {"sentiment": str, "confidence": float}
    """
    best_class = max(aggregated, key=aggregated.get)
    best_score = aggregated[best_class]

    if best_class != "neutre" and best_score < threshold:
        # Confiance dans le "neutre" = à quel point le modèle était loin de trancher
        return {"sentiment": "neutre", "confidence": round(1 - best_score, 4)}

    return {"sentiment": best_class, "confidence": round(best_score, 4)}


class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME, device: int = -1):
        # device=-1 -> CPU, device=0 -> premier GPU (utile sur Colab)
        print(f"Chargement du modèle de sentiment '{model_name}'...")
        self.pipe = pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )

    def predict(self, text: str) -> dict:
        """
        Retourne {"sentiment": "positif|négatif|neutre", "confidence": float}
        """
        if not text or not text.strip():
            return {"sentiment": "neutre", "confidence": 0.0}

        raw_scores = self.pipe(text, top_k=None)  # scores pour les 5 classes
        aggregated = aggregate_scores(raw_scores)
        return decide_sentiment(aggregated)


if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Je suis très satisfait du service."
    analyzer = SentimentAnalyzer()
    result = analyzer.predict(text)
    print(result)
