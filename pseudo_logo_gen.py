"""Génération d'un logo silhouette par MOT de pseudo (2026-07-31, décision développeur via
angelobot : silhouette neutre par mot, recolorée dynamiquement côté client via CSS mask-image —
pas une image par combinaison mot+couleur).

Module autonome (comme sprite_gen.py du plan RPG jamais implémenté) : aucune dépendance à
main.py/chatbot_actions.py, ne lève jamais, dégrade en None en cas d'échec à n'importe quelle
étape (clé API absente, timeout, réponse inattendue, échec Pillow).

Format d'appel FLUX.2 Klein (2026-07-31, corrige un faux diagnostic de la veille — voir mémoire
KhadasBot "FLUX.2 Klein disponible sur OpenRouter") : `modalities: ["image"]` SEUL, ce modèle ne
sait produire QUE de l'image, jamais de texte en sortie — `["image", "text"]` renvoie 404.
"""
from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "black-forest-labs/flux.2-klein-4b"


def _call_flux_image(prompt: str) -> bytes | None:
    """Appel nu à FLUX.2 Klein via OpenRouter. Ne lève jamais — retourne None au moindre souci
    (clé absente, timeout, réponse inattendue), à charge de l'appelant de gérer proprement."""
    if not OPENROUTER_API_KEY:
        return None
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image"],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        image_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
        header, b64data = image_url.split(",", 1)
        import base64
        return base64.b64decode(b64data)
    except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


def _whiten_to_transparent(png_bytes: bytes) -> bytes | None:
    """Convertit le fond blanc (généré, plus fiable qu'un fond "transparent" souvent ignoré par
    le modèle) en véritable alpha=0, pour permettre le masquage CSS (mask-image) côté client —
    même logique de détourage que le plan RPG jamais implémenté (sprite_gen.py), simplifiée :
    silhouette pleine (pas de sous-régions à recadrer), donc pas besoin d'autocrop/découpe."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        pixels = img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = pixels[x, y]
                if r > 235 and g > 235 and b > 235:
                    pixels[x, y] = (r, g, b, 0)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return None


def generate_word_silhouette(word: str) -> bytes | None:
    """Génère une silhouette noire pure (fond transparent) représentant le MOT donné — sans
    référence de couleur, la couleur du pseudo est appliquée à l'affichage (CSS mask-image +
    background-color), pas cuite dans l'image. Retourne les octets PNG (avec alpha) ou None en
    cas d'échec à n'importe quelle étape.

    2e version du prompt (2026-07-31, retour réel Angelo après test sur 6 comptes) : le 1er
    prompt ("shadow-puppet silhouette" naturaliste) donnait des formes illisibles/informes une
    fois réduites à la petite taille d'affichage ("des cacas ou des tas de terre"), et 2 animaux
    morphologiquement proches (Faucon/Hibou) se confondaient en silhouette corps entier. Testé et
    validé avec angelobot : (1) style icône/pictogramme plat façon emoji plutôt que silhouette
    photo-réaliste — bien plus lisible et net ; (2) exiger explicitement AUCUN trou/zone blanche
    interne (un œil blanc devenait un TROU transparent une fois masqué en CSS, bug réel corrigé
    ici) ; (3) cadrage tête/buste testé pour mieux distinguer les espèces à tête caractéristique —
    ÉCARTÉ, résultat PIRE (têtes méconnaissables, ex. Faucon = tête à cornes, Hibou = oreilles de
    chauve-souris) ; le corps entier en vue de profil reste la meilleure option trouvée à ce jour.

    Ambiguïté franco-anglaise (2026-07-31, trouvé par Angelo sur son propre pseudo "Chat") : le
    prompt étant en anglais, le mot français "chat" a été compris comme le mot ANGLAIS "chat"
    (conversation en ligne) plutôt que le félin — logo généré = bulle de discussion avec des
    oreilles, pas un chat. "Chat" n'est pas dans PSEUDO_WORDS (mot LIBRE proposé par Angelo via
    propose_custom_pseudo) — les 24 mots curés de la liste officielle n'ont pas d'homographe
    anglais évident (vérifié), mais le risque reste réel et imprévisible pour tout mot libre.
    Fix générique (pas une liste de cas particuliers à maintenir) : préciser explicitement dans le
    prompt que le mot est FRANÇAIS et doit être interprété selon son sens français littéral, avec
    "chat" cité en exemple concret pour ancrer l'instruction."""
    prompt = (
        f"A bold, minimalist flat icon/pictogram depicting a {word} — {word} is a French word "
        f"naming a concrete animal, plant, or physical object; interpret its natural French "
        f"meaning, never any unrelated English word that happens to share the same spelling "
        f"(for instance a French cat/feline animal, never an online chat/messaging bubble icon). "
        f"Depict the OBJECT ITSELF as a picture — never render the word as text, letters, or a "
        f"logotype. In the style of a simple app icon or emoji logo — NOT a realistic silhouette. "
        "Solid thick black shape on a plain white "
        "background, heavily simplified and geometric, thick rounded chunky outlines, no thin "
        "lines, no spindly legs — designed to stay clearly recognizable even at a very small "
        "size like 24x24 pixels. Exaggerate the ONE OR TWO most distinctive features that make "
        "this specific animal/object recognizable and different from similar-looking ones (e.g. "
        "ear tufts for an owl, a hooked beak for a falcon, antlers for a deer) so it cannot be "
        "confused with a generic shape of the same category. CRITICAL: the shape must be a "
        "single continuous solid silhouette with ZERO holes, zero gaps, zero white areas inside "
        "it — no eyes, no facial features, no internal details of any kind, nothing but one "
        "uninterrupted flat black blob outline. High contrast, bold positive shape. Single "
        "object/being, centered, nothing else in the image, no color, no shading, no gradient, "
        "no texture, no background elements."
    )
    raw = _call_flux_image(prompt)
    if raw is None:
        return None
    return _whiten_to_transparent(raw)
