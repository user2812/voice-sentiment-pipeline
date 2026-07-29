"""
Tests unitaires pour la logique pure de src/sentiment/analyzer.py
(agrégation des scores et décision de seuil), sans charger le vrai modèle
DistilCamemBERT -> tests rapides, aucun téléchargement nécessaire.
"""

from src.sentiment.analyzer import aggregate_scores, decide_sentiment


def test_aggregate_scores_groups_stars_correctly():
    raw_scores = [
        {"label": "1 star", "score": 0.05},
        {"label": "2 stars", "score": 0.05},
        {"label": "3 stars", "score": 0.10},
        {"label": "4 stars", "score": 0.30},
        {"label": "5 stars", "score": 0.50},
    ]
    result = aggregate_scores(raw_scores)

    assert result["négatif"] == 0.10  # 0.05 + 0.05
    assert result["neutre"] == 0.10
    assert result["positif"] == 0.80  # 0.30 + 0.50


def test_decide_sentiment_clear_positive():
    aggregated = {"positif": 0.95, "négatif": 0.03, "neutre": 0.02}
    result = decide_sentiment(aggregated)

    assert result["sentiment"] == "positif"
    assert result["confidence"] == 0.95


def test_decide_sentiment_clear_negative():
    aggregated = {"positif": 0.02, "négatif": 0.93, "neutre": 0.05}
    result = decide_sentiment(aggregated)

    assert result["sentiment"] == "négatif"
    assert result["confidence"] == 0.93


def test_decide_sentiment_low_confidence_falls_back_to_neutral():
    # Le modèle penche très légèrement vers "positif" (0.52) mais sans certitude
    aggregated = {"positif": 0.52, "négatif": 0.30, "neutre": 0.18}
    result = decide_sentiment(aggregated, threshold=0.6)

    assert result["sentiment"] == "neutre"
    assert result["confidence"] == 0.48  # round(1 - 0.52, 4)


def test_decide_sentiment_already_neutral_is_kept():
    aggregated = {"positif": 0.20, "négatif": 0.15, "neutre": 0.65}
    result = decide_sentiment(aggregated)

    assert result["sentiment"] == "neutre"
    assert result["confidence"] == 0.65


def test_decide_sentiment_respects_custom_threshold():
    aggregated = {"positif": 0.55, "négatif": 0.30, "neutre": 0.15}

    # Avec un seuil bas (0.5), 0.55 est suffisant pour garder "positif"
    result_low_threshold = decide_sentiment(aggregated, threshold=0.5)
    assert result_low_threshold["sentiment"] == "positif"

    # Avec un seuil haut (0.7), 0.55 est insuffisant -> bascule vers "neutre"
    result_high_threshold = decide_sentiment(aggregated, threshold=0.7)
    assert result_high_threshold["sentiment"] == "neutre"
