"""index_transcriptions.py — Indexe les transcriptions markdown produites manuellement par
angelobot (2026-07-26) pour les 25 PV existants au moment du chantier — meilleure qualité que
l'OCR automatique (voir index.py, gardé comme filet pour les futurs PV que personne ne
transcrit à la main). Un fichier .md par PV dans transcriptions/, frontmatter YAML minimal :

---
date_seance: 2026-06-05
source_url: https://jouy28.com/wp-content/uploads/sites/159/.../PV.pdf
---
texte transcrit...

Tourne dans le conteneur opencode (jamais jouyvote-web), comme index.py — mêmes dépendances déjà
installées, même collection Qdrant, même schéma de payload (recherche uniforme quelle que soit
l'origine du texte). Idempotent par défaut (skip si déjà indexé) ; --force réindexe en
supprimant d'abord les anciens points pour ce source_url (corrige une transcription sans laisser
de chunks obsolètes en doublon)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import embed  # noqa: E402
from index import COLLECTION, QDRANT_URL, _already_indexed, _ensure_collection, _make_point_id, chunk_text  # noqa: E402

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

TRANSCRIPTIONS_DIR = Path(__file__).parent / "transcriptions"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError("frontmatter manquant ou mal formé (attendu : --- ... --- en tête de fichier)")
    raw_meta, body = match.groups()
    meta: dict[str, str] = {}
    for line in raw_meta.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body.strip()


def _delete_existing(client: QdrantClient, source_url: str) -> None:
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=[FieldCondition(key="source_url", match=MatchValue(value=source_url))]),
    )


def index_transcription_file(client: QdrantClient, path: Path) -> dict:
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    source_url = meta.get("source_url")
    if not source_url:
        return {"file": path.name, "ok": False, "error": "source_url manquant dans le frontmatter"}
    if not body:
        return {"file": path.name, "ok": False, "error": "corps vide"}
    meeting_date = meta.get("date_seance")
    chunks = chunk_text(body)
    points = [
        PointStruct(
            id=_make_point_id(source_url, i),
            vector=embed(chunk),
            payload={
                "text": chunk,
                "source_url": source_url,
                "meeting_date": meeting_date,
                "chunk_index": i,
                "chunk_count": len(chunks),
                "origin": "transcription_manuelle",
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return {"file": path.name, "ok": True, "chunks": len(chunks), "source_url": source_url, "meeting_date": meeting_date}


def run() -> None:
    """TOUJOURS remplace le contenu existant pour chaque source_url trouvé ici (jamais de
    'skip si déjà indexé', contrairement à index.py) : une transcription manuelle est
    AUTORITAIRE — elle doit systématiquement l'emporter sur un éventuel doublon déjà indexé par
    l'OCR automatique (index.py) pour le même document, trouvé en réel le 2026-07-26 (la veille
    OCR a tourné en parallèle des livraisons de transcriptions et a indexé plusieurs des mêmes
    PV en moins bonne qualité, avant que la transcription correspondante n'arrive)."""
    client = QdrantClient(url=QDRANT_URL)
    _ensure_collection(client)
    if not TRANSCRIPTIONS_DIR.exists():
        print(f"Répertoire {TRANSCRIPTIONS_DIR} introuvable, rien à indexer.")
        return
    md_files = sorted(TRANSCRIPTIONS_DIR.glob("*.md"))
    print(f"{len(md_files)} fichier(s) de transcription trouvé(s).")
    for path in md_files:
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"ÉCHEC pour {path.name} : {e}")
            continue
        source_url = meta.get("source_url")
        if source_url and _already_indexed(client, source_url):
            _delete_existing(client, source_url)
        try:
            result = index_transcription_file(client, path)
            print(f"Indexé : {result}")
        except Exception as e:
            print(f"ÉCHEC pour {path.name} : {e}")


if __name__ == "__main__":
    run()
