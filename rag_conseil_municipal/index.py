"""index.py — Pipeline d'indexation des comptes-rendus de conseil municipal (RAG, 2026-07-26).

Tourne dans le conteneur jouyvote-opencode, JAMAIS dans jouyvote-web (volontairement minimal, voir
docker-compose.yml) — déclenché périodiquement par cron (rythme mensuel de publication des PV,
voir wiki:themes:ressources). Idempotent : ne retélécharge/ré-indexe jamais un PDF déjà présent
dans Qdrant (vérifié via une recherche par source_url avant tout traitement), donc un simple
relancement régulier suffit à détecter les nouveaux documents sans logique de diff séparée.

Source : le panneau d'affichage numérique officiel de la mairie (jouy28.com) — PAS l'ancienne
section "/category/informations-communales/le-conseil-municipal/" du même site, abandonnée depuis
2022 (voir wiki:themes:ressources, point de vigilance explicite du développeur).

Aucun PDF brut conservé sur disque : seul le texte extrait (par chunk) est stocké dans Qdrant,
avec l'URL source d'origine pour la citation — inutile de dupliquer un contenu déjà public et
accessible à cette URL.
"""
from __future__ import annotations

import hashlib
import io
import re
import sys
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import requests
from PIL import Image
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).parent))
from embeddings import embed, VECTOR_SIZE  # noqa: E402

SOURCE_PAGE = "https://jouy28.com/affichage/panneaux-daffichage/reunions/pv-conseils-municipaux/"
QDRANT_URL = "http://qdrant:6333"
COLLECTION = "conseil_municipal_pv"

CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

_UA = "Mozilla/5.0 (compatible; JouyVoteRAGBot/1.0; +https://jouyvote.fr)"


def _ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def list_pdf_urls() -> list[str]:
    """Scrape la page officielle du panneau d'affichage, renvoie les URLs de PDF trouvées."""
    resp = requests.get(SOURCE_PAGE, headers={"User-Agent": _UA}, timeout=30)
    resp.raise_for_status()
    return sorted(set(re.findall(r'href="([^"]+\.pdf)"', resp.text, re.IGNORECASE)))


def _already_indexed(client: QdrantClient, source_url: str) -> bool:
    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="source_url", match=MatchValue(value=source_url))]),
        limit=1,
    )
    return len(points) > 0


def _parse_date_label(url: str) -> str:
    """Repli d'affichage best-effort depuis le nom de fichier (non normalisé — dates de
    publication ET de séance mélangées selon les documents) — voir _extract_meeting_date pour la
    source plus fiable (le texte du PV lui-même)."""
    filename = url.rsplit("/", 1)[-1]
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if match:
        return f"{match.group(3)}/{match.group(2)}/{match.group(1)}"
    return filename


_MOIS = (
    "janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre"
)


_JOURS = "lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"


def _extract_meeting_date(text: str) -> str | None:
    """Cherche la date de SÉANCE (pas la date de convocation, souvent mentionnée juste avant dans
    le même paragraphe — bug réel trouvé en testant sur un vrai PV OCRisé : 'légalement convoqué
    le 29 mai 2026, s'est réuni... le vendredi 05 juin 2026' contient 2 dates, la 1re n'étant PAS
    celle de la séance). Sur ce corpus, la date de séance est systématiquement précédée d'un jour
    de la semaine ('le vendredi 05 juin 2026'), jamais la date de convocation — priorité à ce
    motif, repli sur la 1re date trouvée si absent (OCR trop dégradé pour capturer le jour)."""
    head = text[:3000]
    with_weekday = re.search(
        rf"(?:{_JOURS})\s+(\d{{1,2}}(?:er)?\s+(?:{_MOIS})\s+\d{{4}})",
        head, re.IGNORECASE,
    )
    if with_weekday:
        return with_weekday.group(1)
    any_date = re.search(rf"(\d{{1,2}}(?:er)?\s+(?:{_MOIS})\s+\d{{4}})", head, re.IGNORECASE)
    return any_date.group(1) if any_date else None


def download_pdf(url: str) -> bytes:
    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=60)
    resp.raise_for_status()
    return resp.content


OCR_DPI = 200  # compromis qualité/vitesse — suffisant pour du texte scanné A4 net


def _ocr_pdf(pdf_bytes: bytes) -> str:
    """Repli OCR (2026-07-26, trouvé en réel : 22 des 25 PV publiés sont des scans purs, zéro
    texte embarqué — sans ça le RAG ne couvrirait que 3 documents sur 25). Rastérise chaque page
    via PyMuPDF (pas besoin de poppler-utils séparé, contrairement à pdf2image) puis passe
    l'image à tesseract en français."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages_text.append(pytesseract.image_to_string(img, lang="fra"))
    doc.close()
    return "\n\n".join(pages_text)


def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    # Les PV publiés par la mairie sont chiffrés AES (restrictions d'édition/impression, pas un
    # vrai secret — mot de passe utilisateur vide, trouvé en réel : les 25 PDF de la source
    # échouaient tous sans ce déchiffrement). pypdf ne décrypte jamais automatiquement.
    if reader.is_encrypted:
        reader.decrypt("")
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if text.strip():
        return text
    # Aucun texte embarqué : très probablement un scan (confirmé en réel sur ce corpus) — repli OCR.
    return _ocr_pdf(pdf_bytes)


def chunk_text(text: str) -> list[str]:
    """Découpage par paragraphe, fusionnés jusqu'à une taille cible — garde des citations
    précises (pas tout le document d'un coup), léger chevauchement pour ne pas perdre le contexte
    à la frontière de deux chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if current and len(current) + len(p) > CHUNK_TARGET_CHARS:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP_CHARS:] + "\n\n" + p
        else:
            current = (current + "\n\n" + p) if current else p
    if current:
        chunks.append(current)
    return chunks


def _make_point_id(url: str, chunk_index: int) -> str:
    raw = f"{url}:{chunk_index}"
    return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))


def index_pdf(client: QdrantClient, url: str) -> dict:
    pdf_bytes = download_pdf(url)
    text = extract_text(pdf_bytes)
    if not text.strip():
        return {"url": url, "ok": False, "error": "aucun texte extrait (PDF scanné/image ?)"}
    meeting_date = _extract_meeting_date(text) or _parse_date_label(url)
    chunks = chunk_text(text)
    points = [
        PointStruct(
            id=_make_point_id(url, i),
            vector=embed(chunk),
            payload={
                "text": chunk,
                "source_url": url,
                "meeting_date": meeting_date,
                "chunk_index": i,
                "chunk_count": len(chunks),
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return {"url": url, "ok": True, "chunks": len(chunks), "meeting_date": meeting_date}


def run() -> None:
    client = QdrantClient(url=QDRANT_URL)
    _ensure_collection(client)
    try:
        urls = list_pdf_urls()
    except Exception as e:
        print(f"ÉCHEC récupération de la page source ({SOURCE_PAGE}) : {e}")
        return
    print(f"{len(urls)} PDF trouvés sur la page source.")
    new_count = 0
    for url in urls:
        if _already_indexed(client, url):
            continue
        new_count += 1
        try:
            result = index_pdf(client, url)
            print(f"Indexé : {result}")
        except Exception as e:
            print(f"ÉCHEC pour {url} : {e}")
    print(f"{new_count} nouveau(x) PDF traité(s) sur {len(urls)}.")


if __name__ == "__main__":
    run()
