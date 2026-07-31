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
    cas d'échec à n'importe quelle étape."""
    prompt = (
        f"A naive, minimalist shadow-puppet silhouette (ombre chinoise style) of a {word}, "
        "solid flat pure black silhouette on a plain white background. Simple, childlike, "
        "hand-cut-paper look, clean smooth outline, no texture, no gradient, no details inside "
        "the silhouette, single object/being, nothing else in the image — pure flat black "
        "paper-cutout shape only, no color, no shading."
    )
    raw = _call_flux_image(prompt)
    if raw is None:
        return None
    return _whiten_to_transparent(raw)
