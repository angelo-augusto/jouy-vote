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
TRANSLATE_MODEL = "deepseek/deepseek-v4-flash"
VISION_MODEL = "google/gemini-2.5-flash-image"
MAX_DRAW_ATTEMPTS = 2


def _translate_to_english(word: str) -> str:
    """Traduit un mot français en un mot/expression anglaise courte AVANT de construire le prompt
    d'image (2026-07-31, proposition d'Angelo suite au bug "chat" -> bulle de discussion) : plutôt
    que de demander au modèle d'IMAGE d'ignorer un sens anglais parasite (fragile — a déjà échoué
    une fois, voir generate_word_silhouette), on élimine l'ambiguïté À LA RACINE en ne lui montrant
    jamais le mot français litigieux. Dégrade proprement : retourne le mot original si la
    traduction échoue à n'importe quelle étape (mieux qu'un plantage, quitte à retomber sur le
    risque d'ambiguïté résiduel)."""
    if not OPENROUTER_API_KEY:
        return word
    body = json.dumps({
        "model": TRANSLATE_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                f'Translate this single French word to English: "{word}". It names a concrete '
                "animal, plant, or physical object (never an abstract/unrelated meaning, even if "
                "the spelling coincidentally matches an unrelated English word). Reply with ONLY "
                "the English word or short phrase, nothing else — no punctuation, no quotes, no "
                "explanation."
            ),
        }],
        # max_tokens=200, pas 20 (2026-07-31, bug réel trouvé au premier test) : deepseek-v4-flash
        # est un modèle "raisonnement", certains routages OpenRouter consomment des tokens de
        # raisonnement AVANT la réponse finale — avec une limite trop basse, la génération
        # s'arrête à mi-raisonnement (finish_reason="length") et content=None, jamais la traduction
        # elle-même. 200 laisse assez de marge pour le raisonnement ET la réponse (coût négligeable,
        # quelques centièmes de cent).
        "max_tokens": 200,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        # Filet de sécurité en plus du max_tokens=200 ci-dessus (2026-07-31) : si un routage
        # renvoie quand même content=None (raisonnement pas terminé malgré la marge), retomber
        # sur le mot original plutôt que planter sur .strip() appelé sur None.
        if not content:
            return word
        translated = content.strip().strip(".\"'")
        return translated if translated else word
    except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError, AttributeError):
        return word


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


def _build_icon_prompt(english_word: str) -> str:
    """Prompt de génération — extrait de generate_word_silhouette (2026-07-31) pour être réutilisé
    à chaque tentative de la boucle dessin/relecture ci-dessous."""
    return (
        f"A bold, minimalist flat icon/pictogram depicting a {english_word}. Depict the OBJECT "
        f"ITSELF as a picture — never render any word as text, letters, or a logotype. In the "
        f"style of a simple app icon or emoji logo — NOT a realistic silhouette. "
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


def _read_and_moderate_image(png_bytes: bytes) -> tuple[str | None, bool]:
    """Modèle "lecteur" (2026-07-31, contrôle qualité demandé par Angelo) : demande à un modèle
    vision ce que représente l'image générée, SANS lui donner le mot attendu — la comparaison se
    fait après coup (voir _words_match), jamais en soufflant la réponse au modèle qui relit.

    Modération de l'image elle-même (2026-07-31, ajout Angelo) : même appel, demande AUSSI si
    l'image comporte une connotation politique/religieuse/sexuelle/déplacée non voulue — le mot
    lui-même est déjà jugé par le LLM du chatbot (voir "appropriate" dans propose_custom_pseudo),
    mais l'IMAGE générée est un artefact séparé qui pourrait en théorie dériver même à partir d'un
    mot inoffensif ; double filet plutôt qu'un seul point de contrôle.

    Retourne (description_ou_None, contenu_problematique) — en cas d'échec technique, retourne
    (None, True) : par prudence, un échec de lecture doit être traité comme un échec de
    vérification (donc une nouvelle tentative ou un abandon), jamais comme une validation
    implicite."""
    if not OPENROUTER_API_KEY:
        return None, True
    import base64
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    body = json.dumps({
        "model": VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {
                "type": "text",
                "text": (
                    "Look at this simple black icon/silhouette. Reply with EXACTLY two lines, "
                    "nothing else:\n"
                    "NAME: <in one or two words, the animal/plant/object it represents>\n"
                    "FLAG: <yes if this image has any unintended political, religious, sexual, "
                    "or otherwise inappropriate/offensive connotation, otherwise no>"
                ),
            },
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}],
        "max_tokens": 60,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        if not content:
            return None, True
        name = None
        flagged = True
        for line in content.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("NAME:"):
                name = line.split(":", 1)[1].strip()
            elif line.upper().startswith("FLAG:"):
                flagged = line.split(":", 1)[1].strip().lower().startswith("yes")
        return name, flagged
    except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError, AttributeError):
        return None, True


def _words_match(expected_word: str, described: str) -> bool:
    """Compare sémantiquement (pas une égalité de chaîne stricte — "cat" vs "cat head" vs
    "feline" doivent tous compter comme une correspondance) le mot attendu et la description
    donnée par le modèle lecteur. En cas d'échec technique, retourne False par prudence (mieux
    vaut une régénération ou un refus inutile qu'un faux positif qui laisse passer un mauvais
    logo)."""
    if not OPENROUTER_API_KEY:
        return False
    body = json.dumps({
        "model": TRANSLATE_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                f'Do these two both refer to the same or a closely related animal/plant/object? '
                f'word A: "{expected_word}" — word B: "{described}". Reply with ONLY "yes" or '
                f'"no", nothing else.'
            ),
        }],
        "max_tokens": 200,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        if not content:
            return False
        return content.strip().lower().startswith("yes")
    except (OSError, KeyError, IndexError, ValueError, json.JSONDecodeError, AttributeError):
        return False


def generate_word_silhouette(word: str) -> bytes | None:
    """Génère une silhouette noire pure (fond transparent) représentant le MOT donné — sans
    référence de couleur, la couleur du pseudo est appliquée à l'affichage (CSS mask-image +
    background-color), pas cuite dans l'image. Retourne les octets PNG (avec alpha) ou None en
    cas d'échec — y compris un échec de VALIDATION (voir boucle contrôle qualité ci-dessous),
    jamais juste un échec technique.

    2e version du prompt (2026-07-31, retour réel Angelo après test sur 6 comptes) : le 1er
    prompt ("shadow-puppet silhouette" naturaliste) donnait des formes illisibles/informes une
    fois réduites à la petite taille d'affichage ("des cacas ou des tas de terre"), et 2 animaux
    morphologiquement proches (Faucon/Hibou) se confondaient en silhouette corps entier. Testé et
    validé avec angelobot : (1) style icône/pictogramme plat façon emoji plutôt que silhouette
    photo-réaliste — bien plus lisible et net ; (2) exiger explicitement AUCUN trou/zone blanche
    interne (un œil blanc devenait un TROU transparent une fois masqué en CSS, bug réel corrigé
    ici) ; (3) cadrage tête/buste testé pour mieux distinguer les espèces à tête caractéristique —
    ÉCARTÉ, résultat PIRE (têtes méconnaissables) ; le corps entier en vue de profil reste la
    meilleure option trouvée à ce jour.

    Ambiguïté franco-anglaise (2026-07-31, trouvé par Angelo sur son propre pseudo "Chat") :
    corrigée en traduisant le mot en anglais AVANT de construire le prompt (_translate_to_english)
    — le modèle d'image ne voit alors plus jamais le mot français ambigu.

    Contrôle qualité par boucle dessin/relecture (2026-07-31, demande explicite d'Angelo : "tu
    peux boucler jusqu'à ce que cela fonctionne", donc une boucle DOIT être bornée) : après chaque
    génération, un modèle VISION distinct relit l'image SANS connaître le mot attendu et la
    modère au passage (_read_and_moderate_image), puis un 2e appel compare sémantiquement cette
    lecture au mot attendu (_words_match). Si ça correspond ET que rien n'est signalé comme
    problématique, terminé. Si non (mauvaise correspondance OU contenu signalé), UNE SEULE
    nouvelle tentative (redessiner + relire) — jamais plus, jamais de boucle illimitée. Si les 2
    tentatives échouent, retourne None : c'est à l'appelant (voir /pseudo/logo/preview) de dire au
    citoyen que ce mot ne donne pas un logo assez reconnaissable et de lui proposer d'en choisir
    un autre, plutôt que de publier un logo raté/problématique ou de boucler indéfiniment."""
    english_word = _translate_to_english(word)
    prompt = _build_icon_prompt(english_word)
    for _attempt in range(MAX_DRAW_ATTEMPTS):
        raw = _call_flux_image(prompt)
        if raw is None:
            continue
        png = _whiten_to_transparent(raw)
        if png is None:
            continue
        described, flagged = _read_and_moderate_image(png)
        if flagged:
            continue
        if described and _words_match(english_word, described):
            return png
    return None
