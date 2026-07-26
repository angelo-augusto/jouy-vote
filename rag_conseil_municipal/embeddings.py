"""Module d'embedding partagé entre le pipeline d'indexation (rag_conseil_municipal/index.py,
tourne dans le conteneur opencode) et l'outil de recherche du chatbot (chatbot_actions.py, tourne
dans jouyvote-web) — même modèle des deux côtés, sinon les vecteurs ne seraient pas comparables.

Modèle multilingue (les PV de conseil municipal sont en français) plutôt que all-MiniLM-L6-v2
(anglais, déjà utilisé côté RPG mais pour du texte anglais/générique) — paraphrase-multilingual-
MiniLM-L12-v2 reste local et léger (pas d'appel API, décision validée par le développeur le
2026-07-26 : cohérent avec le pattern RPG, pas de raison de payer/exposer ces documents publics à
un tiers externe).
"""
from __future__ import annotations

from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384

_model: Optional["SentenceTransformer"] = None


def is_available() -> bool:
    return _AVAILABLE


def get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    return get_model().encode(text).tolist()
