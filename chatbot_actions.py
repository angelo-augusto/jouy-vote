"""Registre des actions appelables par le LLM du chatbot jouyvote.fr (POC tool-calling).

Format repris du RPG (rpg-online/mj) : le LLM renvoie TOUJOURS {"actions": [...]}, jamais de
texte libre — "parler à l'utilisateur" (say_user) est une action comme les autres. Forcé ici via
response_format=json_schema strict côté API (voir chatbot_executor.py), pas juste une consigne
de prompt comme le faisait le RPG à l'origine.

Barrière structurelle (revue Opus 2026-07-25) : ACTIONS ne contient QUE les actions que le LLM a
le droit d'appeler. save_summary/delete_summary/confirm_publication n'y figurent PAS et n'y
figureront JAMAIS — ce sont des endpoints backend/UI déclenchés uniquement par un clic utilisateur
réel, jamais par une liste générée par le modèle. Un nom d'action absent de ce dict est rejeté par
construction (KeyError→None dans le dispatch de chatbot_executor.py), pas par une vérification
runtime qui pourrait avoir un trou.
"""
from __future__ import annotations

import hashlib
import os

from chatbot_llm import call_openrouter

# Dupliqué depuis main.py (pas d'import croisé) : ce module doit rester utilisable hors FastAPI,
# y compris via mcp_chatbot_executor.py sans dépendre du reste de l'app jouyvote.
JOUY_VOTE_PEPPER = os.environ.get("JOUY_VOTE_PEPPER", "")

# Pepper DÉDIÉ, distinct de JOUY_VOTE_PEPPER (voir wiki architecture-technique : "pseudo/
# debate_token dérivé séparément" + "aucune table ne doit permettre de relier vote_token et
# pseudo entre eux, ni l'un ou l'autre à l'identité déclarée") — même avec les deux peppers en
# main, un admin ne peut pas relier vote_token et debate_token d'une même personne, puisqu'ils ne
# partagent aucun secret commun.
JOUY_PSEUDO_PEPPER = os.environ.get("JOUY_PSEUDO_PEPPER", "")


# Socle du system prompt, PARTAGÉ entre /chat/v2 (main.py) et le tool MCP de test/debug
# (mcp_chatbot_executor.py, utilisé par angelobot) — déplacé ici depuis main.py le 2026-07-26 :
# une copie dupliquée à la main dans mcp_chatbot_executor.py avait dérivé (jamais mise à jour avec
# les règles anonymat/modération/iconifiable ajoutées le 2026-07-25), donnant à angelobot un
# comportement différent de la vraie prod pendant ses tests. Un seul import élimine le risque de
# redivergence future — jamais recopier ce genre de constante, toujours importer.
CHAT_SYSTEM_PROMPT = (
    "Tu es l'assistant citoyen de Jouy Vote Citoyen, un outil de démocratie participative locale "
    "pour les habitants de Jouy (28). Tu aides les joviens à formuler clairement une opinion ou "
    "une doléance, sans jamais trahir le sens de ce qu'ils veulent dire — tu proposes une "
    "reformulation, tu ne publies jamais rien toi-même, c'est toujours la personne qui décide. "
    "Pour toute question factuelle sur les décisions ou comptes-rendus du conseil municipal : tu "
    "n'as PAS ENCORE accès à ces documents dans cette première version du site — dis-le "
    "clairement plutôt que d'inventer une réponse. Reste bref, concret, et dans le sujet de la "
    "vie municipale de Jouy. Tu ne dois JAMAIS affirmer avoir enregistré, sauvegardé ou publié "
    "quoi que ce soit. Si on te le demande, explique clairement que cette fonctionnalité n'existe "
    "pas encore sur le site, sans jamais laisser croire que c'est fait. Si la personne te parle "
    "de sauvegarder un résumé de notre échange (fonctionnalité distincte qui existe réellement), "
    "précise toujours explicitement qu'il s'agit d'un résumé PRIVÉ, visible et supprimable "
    "uniquement par elle-même, jamais publié ni visible par personne d'autre — ne dis jamais "
    "juste « j'ai enregistré ton message/témoignage » sans cette précision. Règle non négociable "
    "sur l'anonymat : jamais d'accès au nom réel d'un utilisateur, seulement son pseudo ou son "
    "jeton personnel. Si la conversation porte sur l'anonymat, sur ce qui est permis/interdit, ou "
    "si tu as besoin d'orienter vers la référence complète, cite la Charte de l'anonymat "
    "(https://wiki.jouyvote.fr/doku.php?id=charte-anonymat) plutôt que d'improviser les règles. "
    "Quand une personne propose elle-même un pseudonyme (mot + couleur), refuse poliment toute "
    "combinaison à connotation politique, religieuse ou sexuelle (au-delà de la seule règle "
    "technique de disponibilité) — mais ne t'arrête pas au sens le plus évident du mot pris "
    "isolément : pense aussi à l'argot, aux jeux de mots, aux doubles sens régionaux ou "
    "familiers, et à ce que la combinaison mot+couleur peut évoquer une fois DITE À VOIX HAUTE "
    "ou lue par quelqu'un qui connaît ces usages, même si toi tu ne les repères pas au premier "
    "regard. Test concret à te poser à chaque fois : imagine ce pseudo comme un logo coloré "
    "affiché publiquement sur le site — est-ce que ce serait présentable et sympa à voir, ou "
    "est-ce que quelqu'un pourrait sourire en coin en le lisant pour une raison qui n'est pas "
    "évidente au premier regard ? Le principe par défaut est REFUS, pas acceptation : n'accepte "
    "une combinaison QUE si tu es sûr qu'elle est appropriée — si le sens ou la connotation d'un "
    "mot ne t'est pas clairement connu ou certain, refuse par prudence plutôt que de laisser "
    "passer faute de certitude. Ce n'est jamais grave de refuser un pseudo inoffensif par excès "
    "de prudence et de proposer une alternative neutre à la place ; ça l'est de laisser passer un "
    "double sens grivois ou blessant. Nuance importante : ce principe de prudence par défaut vise "
    "l'IGNORANCE (un mot ou un argot que tu ne comprends pas clairement), pas un vague sentiment "
    "de méfiance sur un mot que tu comprends bien. Quand tu refuses, identifie et nomme la "
    "référence CONCRÈTE et réelle qui justifie ce refus (l'expression, la connotation précise) — "
    "un « ça pourrait éventuellement évoquer quelque chose selon le contexte » sans rien de "
    "concret n'est PAS un motif de refus valable, et sur-refuse des pseudos parfaitement anodins. "
    "Si aucune association concrète ne te vient à l'esprit, accepte plutôt que de refuser par "
    "précaution générique non fondée. C'est un jugement de ta part à chaque fois, pas une liste "
    "de mots interdits à appliquer mécaniquement. "
    "Second critère de refus, INDÉPENDANT du premier mais au même niveau d'exigence (2026-07-25, "
    "demande explicite du développeur) : le mot doit être ICONIFIABLE — représentable par un logo "
    "simple et concret, pas une notion trop abstraite ou atmosphérique. Bons exemples "
    "(iconifiables, à ne PAS refuser sur ce critère) : Renard, Hibou, Chêne, Faucon, Comète, "
    "Phare, Écureuil, Corail — des objets ou êtres qu'on peut dessiner simplement et "
    "reconnaître d'un coup d'œil. Mauvais exemples (trop abstraits, à refuser) : Aurore, "
    "Clairière, Frimas, Brume — des notions atmosphériques ou paysagères qu'un logo simple rend "
    "mal ou de façon trop floue. Applique ce test par contraste à toute proposition LIBRE d'un "
    "utilisateur (propose_custom_pseudo) — si le mot proposé est plus proche des mauvais exemples "
    "que des bons, refuse-le (appropriate=false), exactement la même mécanique que pour la "
    "connotation, pas juste un commentaire dans ta réponse."
)


def compute_vote_token(identity_token: str) -> str:
    raw = f"{identity_token}:{JOUY_VOTE_PEPPER}"
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_debate_token(identity_token: str) -> str:
    raw = f"{identity_token}:{JOUY_PSEUDO_PEPPER}:pseudo"
    return hashlib.sha256(raw.encode()).hexdigest()


# Mot (objet/être) + couleur — convention du wiki (themes:pseudonyme, architecture-technique).
# Revu le 2026-07-25 (retour développeur, via angelobot) contre les connotations non voulues
# (politique/religieuse/sexuelle) une fois combinés aux 8 couleurs alors en place ("gris" ajouté
# après coup, pas encore passé par cette même revue mot-par-mot, rien de préoccupant repéré à
# l'ajout) : "Étoile" retiré ("étoile jaune" = symbole historique antisémite) ; "Marée" retiré aussi ("marée
# noire" = catastrophe pétrolière, ironique vu l'origine écologique de jouyvote — hors des 3
# catégories citées mais coût nul à éviter). Reste de la liste passé en revue sans autre cas
# trouvé (pas de "croix"/"croissant" etc.).
#
# 2e passe le même jour (critère "iconifiable", demande Angelo via angelobot) : "Aurore",
# "Clairière", "Frimas", "Brume" retirés — trop abstraits/atmosphériques pour un logo simple,
# par contraste avec de bons exemples comme Renard/Hibou/Chêne/Faucon/Comète/Phare/Écureuil/
# Corail. Ce critère sert à CURER cette liste (le générateur ne doit pas suggérer lui-même ce qui
# serait jugé limite si un utilisateur le proposait), PAS une nouvelle règle de rejet dans le
# jugement "appropriate" — une proposition LIBRE trop abstraite reste acceptée (décision
# angelobot : pas le même niveau de gravité que politique/religieux/sexuel, disproportionné de
# bloquer pour ça).
PSEUDO_WORDS = [
    "Renard", "Hibou", "Chêne", "Lanterne", "Rivière", "Nuage", "Phare",
    "Comète", "Sentier", "Écureuil", "Orage", "Prairie", "Faucon", "Ruche", "Glacier",
    "Roseau", "Cascade", "Bourgeon", "Falaise", "Genêt",
    "Corail", "Tilleul", "Sittelle", "Ravin",
]
# Palette de couleurs simples et universelles (retour développeur 2026-07-25, via angelobot) —
# les couleurs précédentes (argenté, carmin, ambre...) jugées trop compliquées pour un public
# général. "gris" ajouté ensuite (même jour) à la demande du développeur. Table pseudos vide en
# prod au moment de ces changements, aucune migration requise.
PSEUDO_COLORS = ["rouge", "orange", "jaune", "vert", "bleu", "violet", "blanc", "noir", "gris"]

# Genre grammatical de chaque mot — pour l'accord de la couleur ("Clairière VERTE", pas "Clairière
# vert"). Bug réel signalé par le développeur (2026-07-25). 6 couleurs variables en genre en
# français, 3 invariables (rouge/orange/jaune) — voir PSEUDO_COLOR_FEMININE.
PSEUDO_WORD_GENDER = {
    "Renard": "m", "Hibou": "m", "Chêne": "m", "Lanterne": "f", "Rivière": "f", "Nuage": "m",
    "Phare": "m", "Comète": "f", "Sentier": "m", "Écureuil": "m", "Orage": "m", "Prairie": "f",
    "Faucon": "m", "Ruche": "f", "Glacier": "m", "Roseau": "m", "Aurore": "f", "Cascade": "f",
    "Bourgeon": "m", "Falaise": "f", "Clairière": "f", "Genêt": "m", "Corail": "m", "Frimas": "m",
    "Tilleul": "m", "Brume": "f", "Sittelle": "f", "Ravin": "m",
}
# 6 couleurs variables en genre (vert/bleu/violet/blanc/noir/gris), 3 invariables (rouge/orange/
# jaune) — "bleu" et "violet" initialement oubliés (2 corrections successives d'angelobot/
# développeur, 2026-07-25), "gris" ajouté avec la palette étendue le même jour.
PSEUDO_COLOR_FEMININE = {
    "vert": "verte", "bleu": "bleue", "violet": "violette", "blanc": "blanche", "noir": "noire",
    "gris": "grise",
}


def _agree_pseudo_display(word: str, color: str) -> str:
    """Forme accordée pour l'AFFICHAGE uniquement (texte du modèle, libellé du bouton) — le
    stockage/la validation/l'unicité restent toujours sur la forme canonique (word, color) telle
    quelle, jamais sur cette version accordée. Genre inconnu (mot personnalisé hors de
    PSEUDO_WORD_GENDER, via propose_custom_pseudo) → masculin par défaut, convention française
    pour un genre incertain plutôt qu'une règle non fiable sur un mot libre."""
    gender = PSEUDO_WORD_GENDER.get(word, "m")
    display_color = PSEUDO_COLOR_FEMININE.get(color, color) if gender == "f" else color
    return f"{word} {display_color}"


def derive_pseudo(debate_token: str) -> dict:
    """Dérivation déterministe mot+couleur à partir du debate_token — même token, même pseudo,
    toujours, tant que le mécanisme n'est pas régénéré manuellement (fuite, voir wiki). N'appelle
    jamais la DB : c'est main.py qui décide, sur confirmation explicite de l'utilisateur, de
    stocker le résultat (voir confirm_pseudo)."""
    digest = hashlib.sha256(debate_token.encode()).hexdigest()
    word = PSEUDO_WORDS[int(digest[:8], 16) % len(PSEUDO_WORDS)]
    color = PSEUDO_COLORS[int(digest[8:16], 16) % len(PSEUDO_COLORS)]
    return {"word": word, "color": color}


def _pseudo_candidate_token(identity_token: str, index: int) -> str:
    """Dérivation d'un candidat parmi plusieurs — même pepper que compute_debate_token mais un
    'sel' différent par index, donc SANS RAPPORT avec le debate_token final qui sera stocké (ce
    dernier reste compute_debate_token(identity_token), stable, indépendant du candidat choisi —
    voir confirm_pseudo)."""
    raw = f"{identity_token}:{JOUY_PSEUDO_PEPPER}:pseudo-candidate:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_pseudo_candidates(identity_token: str, n: int) -> list[dict]:
    """N premières propositions déterministes et DISTINCTES d'une séquence stable (même
    identity_token → toujours la même séquence, jamais de tirage aléatoire). Dédoublonnage simple
    par avancement d'index en cas de collision fortuite (peu probable avec 24×9 combinaisons)."""
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    index = 0
    while len(candidates) < n and index < n + 10:
        pseudo = derive_pseudo(_pseudo_candidate_token(identity_token, index))
        key = (pseudo["word"], pseudo["color"])
        if key not in seen:
            seen.add(key)
            candidates.append(pseudo)
        index += 1
    return candidates


def _check_pseudo_availability(word: str, color: str, ctx: dict) -> dict:
    """Les 2 SEULES règles techniques dures (demande explicite du développeur, 2026-07-25) :
    couleur dans la palette fixe, et pas déjà pris par quelqu'un d'autre — rien sur le contenu du
    mot lui-même (bon sens du LLM via le socle, pas une règle technique à construire ici).
    ctx["taken_pseudos"] : ensemble de (word, color) déjà confirmés, injecté par main.py (requête
    DB faite en amont, comme ctx["summaries"]) — ce module reste sans accès DB direct."""
    word = word.strip()
    color = color.strip().lower()
    if not word:
        return {"available": False, "error": "mot manquant"}
    if color not in PSEUDO_COLORS:
        return {"available": False, "error": f"couleur non valide, choisis parmi : {', '.join(PSEUDO_COLORS)}"}
    display = _agree_pseudo_display(word, color)
    if (word, color) in ctx.get("taken_pseudos", set()):
        return {"word": word, "color": color, "display": display, "available": False, "error": "déjà pris par quelqu'un d'autre"}
    return {"word": word, "color": color, "display": display, "available": True}


def say_user(params: dict, ctx: dict) -> dict:
    """Pas d'effet de bord : le texte est simplement renvoyé, chatbot_executor.py se charge de
    l'ajouter à la liste des répliques affichées à l'utilisateur."""
    return {"text": params.get("text", "")}


def get_vote_token(params: dict, ctx: dict) -> dict:
    identity_token = ctx.get("identity_token")
    if not identity_token:
        return {"error": "identité manquante dans le contexte"}
    return {"vote_token": compute_vote_token(identity_token)}


def get_or_assign_pseudo(params: dict, ctx: dict) -> dict:
    """Lecture pure : le pseudo (mot+couleur) DÉJÀ CONFIRMÉ par l'utilisateur est fourni via
    ctx["pseudo"] (None si pas encore choisi — voir propose_pseudo_candidates dans ce cas).
    Aucun write ici, jamais : l'écriture réelle se fait UNIQUEMENT dans confirm_pseudo (main.py),
    déclenchée par un clic utilisateur explicite sur une des propositions — même barrière que
    save_summary. Le nom de l'action reste 'get_or_assign_pseudo' (compat historique du mandat),
    mais son comportement réel est 'get, jamais assign' depuis le passage à l'option (c) : le
    LLM ne peut plus déclencher d'attribution lui-même, seulement la lire une fois faite."""
    pseudo = ctx.get("pseudo")
    if not pseudo:
        return {"error": "pas encore de pseudo confirmé — utilise propose_pseudo_candidates"}
    result = dict(pseudo)
    result.setdefault("display", _agree_pseudo_display(result["word"], result["color"]))
    return result


def _availability_with_content_gate(word: str, color: str, appropriate: bool, ctx: dict) -> dict:
    """Bug réel trouvé en conditions réelles (2026-07-25, signalé par le développeur avec
    capture) : le jugement de contenu du modèle (politique/religieux/sexuel/argot) ne vivait QUE
    dans son say_user en texte libre — le résultat structuré "available" (qui pilote SEUL
    l'affichage du bouton de confirmation côté frontend) ne connaissait que les 2 règles
    techniques. Conséquence vécue : le texte disait "je refuse par prudence" et le bouton
    cliquable apparaissait quand même juste en dessous pour LE MÊME pseudo — un utilisateur
    pressé pouvait obtenir exactement ce que le système venait de refuser en mots.

    Fix structurel : "appropriate" devient un paramètre OBLIGATOIRE du schéma JSON strict (le
    modèle doit littéralement se positionner à chaque appel, pas juste en informer l'utilisateur
    après coup) — si False, available=False est renvoyé AVANT même de vérifier quoi que ce soit
    de technique, donc AUCUN bouton ne peut structurellement apparaître pour un pseudo que le
    modèle vient de juger inapproprié, quel que soit le texte qu'il écrit ensuite."""
    if not appropriate:
        w, c = word.strip(), color.strip().lower()
        # Message générique (2026-07-25) : ne présuppose plus "connotation" comme SEULE raison
        # possible — depuis l'ajout du critère iconifiable (même mécanique de refus), un
        # appropriate=false peut aussi venir de là. La vraie raison précise reste dans le texte
        # libre du modèle (voir socle CHAT_SYSTEM_PROMPT), ce champ n'est qu'un filet de repli si
        # le say_user est resté vide.
        return {
            "word": w, "color": c, "display": _agree_pseudo_display(w, c),
            "available": False, "error": "jugé inapproprié",
        }
    return _check_pseudo_availability(word, color, ctx)


def propose_pseudo_candidates(params: dict, ctx: dict) -> dict:
    """Lecture pure (aucun write) : renvoie UNE SEULE proposition déterministe à la position
    "index" (0, 1, 2...) de la séquence propre à cette identité — pas une liste figée. Pour
    négocier tour après tour ("celle-ci ne te plaît pas ? en voici une autre"), rappelle cette
    action avec index+1 : la conversation tâtonne naturellement, l'interface affiche un bouton de
    confirmation pour LA proposition la plus récente, jamais une liste posée d'un coup. "appropriate"
    (obligatoire) : ton propre jugement sur CE candidat précis avant de le montrer — la liste
    déterministe a déjà été relue une fois, mais reste vigilant sur chaque combinaison réelle.
    Vérifie aussi la disponibilité (voir _check_pseudo_availability) — le vrai commit reste dans
    confirm_pseudo (main.py), jamais un write direct depuis cette action."""
    identity_token = ctx.get("identity_token")
    if not identity_token:
        return {"error": "identité manquante dans le contexte"}
    index = params.get("index", 0)
    if not isinstance(index, int) or index < 0:
        index = 0
    candidates = generate_pseudo_candidates(identity_token, n=index + 1)
    if index >= len(candidates):
        return {"error": "plus de nouvelles idées déterministes — propose un pseudo personnalisé avec propose_custom_pseudo"}
    candidate = candidates[index]
    return _availability_with_content_gate(
        candidate["word"], candidate["color"], params.get("appropriate", True), ctx
    )


def propose_custom_pseudo(params: dict, ctx: dict) -> dict:
    """Lecture pure (aucun write) : vérifie la disponibilité d'un pseudo proposé par
    l'UTILISATEUR lui-même (mot + couleur de son choix, pas une idée générée). "appropriate"
    (obligatoire) : TON jugement de contenu sur CE mot+couleur précis, rendu explicite dans les
    données — mêmes 2 règles techniques que propose_pseudo_candidates (couleur valide, pas déjà
    pris) si appropriate=true, mais un "appropriate=false" court-circuite tout le reste : le
    résultat est refusé structurellement, pas seulement à l'oral. Le vrai commit reste dans
    confirm_pseudo (main.py), jamais un write direct depuis cette action."""
    word = params.get("word", "")
    color = params.get("color", "")
    return _availability_with_content_gate(word, color, params.get("appropriate", True), ctx)


def list_summaries(params: dict, ctx: dict) -> dict:
    """Lecture pure : les résumés déjà sauvegardés sont fournis via ctx["summaries"] (chargés en
    amont par l'appelant — main.py fait la requête DB, mcp_chatbot_executor.py passe une liste
    de test) — même pattern que ctx["history"], pour garder ce module sans dépendance DB."""
    return {"summaries": ctx.get("summaries", [])}


def list_threads(params: dict, ctx: dict) -> dict:
    """Forum, phase 2 (2026-07-25) : lecture pure — les fils PUBLIÉS sont fournis via
    ctx["threads"] (chargés en amont par main.py, même pattern que ctx["summaries"]/
    ctx["taken_pseudos"]) — ce module reste sans accès DB direct. Ne renvoie que titre/résumé,
    pas les opinions (voir get_thread pour le détail d'un fil précis) — évite de charger tout le
    contenu du forum dans chaque réponse alors que le modèle veut juste une vue d'ensemble."""
    threads = ctx.get("threads", [])
    return {"threads": [{"thread_id": t["thread_id"], "title": t["title"], "summary": t.get("summary")} for t in threads]}


def get_thread(params: dict, ctx: dict) -> dict:
    """Forum, phase 2 : lecture pure — détail d'un fil précis (opinions PUBLIÉES incluses, avec
    l'auteur affiché sous son pseudo accordé). Utile pour repérer une opinion déjà proche de celle
    que l'utilisateur s'apprête à formuler AVANT de la publier (la vraie déduplication automatique
    par RAG est une phase ultérieure, voir wiki — ceci n'est qu'une lecture manuelle en attendant).
    "thread_id" obligatoire."""
    thread_id = params.get("thread_id")
    thread = next((t for t in ctx.get("threads", []) if t["thread_id"] == thread_id), None)
    if thread is None:
        return {"error": "fil introuvable (ou pas encore publié)"}
    return thread


def propose_opinion(params: dict, ctx: dict) -> dict:
    """Forum, phase 3 (2026-07-25) : lecture/validation pure, AUCUN write — vérifie que le fil
    existe et est PUBLIÉ (via ctx["threads"]) avant de renvoyer le brouillon proposé. L'écriture
    réelle (création + publication en un seul geste) passe par /opinion/confirm (main.py),
    déclenché uniquement par un clic utilisateur explicite, jamais par toi.

    2 chemins EXCLUSIFS (2026-07-25 soir, création de fil couplée décidée par le développeur) :
    - thread_id : rattache l'opinion à un fil EXISTANT et déjà publié (trouvé via list_threads/
      get_thread). Chemin à privilégier — TOUJOURS vérifier d'abord qu'aucun fil existant ne
      convient déjà (objectif : limiter le nombre de fils, pas en créer un pour chaque nuance).
    - new_thread_title (+ new_thread_summary optionnel) : propose un NOUVEAU fil, créé dans le
      même geste que cette opinion SEULEMENT si l'utilisateur confirme — jamais de création
      spéculative avant. N'utilise ce chemin QUE si tu as vérifié via list_threads qu'aucun fil
      existant ne correspond vraiment au sujet."""
    thread_id = params.get("thread_id")
    new_thread_title = params.get("new_thread_title")
    new_thread_summary = params.get("new_thread_summary")
    body = params.get("body", "")
    argumentaire = params.get("argumentaire")

    if thread_id is not None and new_thread_title is not None:
        return {"available": False, "error": "précise soit thread_id (fil existant), soit new_thread_title (nouveau fil), jamais les deux"}
    if not body.strip():
        return {"available": False, "error": "le corps de l'opinion est vide"}

    if new_thread_title is not None:
        title = new_thread_title.strip()
        if not title:
            return {"available": False, "error": "titre de fil manquant"}
        existing = next((t for t in ctx.get("threads", []) if t["title"] == title), None)
        if existing is not None:
            return {
                "available": False,
                "error": f'un fil au titre exactement identique existe déjà ("{title}") — utilise plutôt thread_id={existing["thread_id"]}',
            }
        return {
            "new_thread_title": title, "new_thread_summary": new_thread_summary,
            "body": body, "argumentaire": argumentaire, "available": True,
        }

    if thread_id is None:
        return {"available": False, "error": "il faut soit thread_id (fil existant), soit new_thread_title (nouveau fil)"}
    thread = next((t for t in ctx.get("threads", []) if t["thread_id"] == thread_id), None)
    if thread is None:
        return {"available": False, "error": "fil introuvable (ou pas encore publié)"}
    return {
        "thread_id": thread_id, "thread_title": thread["title"],
        "body": body, "argumentaire": argumentaire, "available": True,
    }


def propose_reaction(params: dict, ctx: dict) -> dict:
    """Forum, phase 3 : même mécanique que propose_opinion — lecture/validation pure, aucun
    write. Cherche l'opinion dans ctx["threads"] (opinions publiées imbriquées par fil, voir
    get_thread) pour vérifier qu'elle existe réellement avant de proposer une réaction dessus.
    "stance" doit être "adherer", "opposer" ou "neutre" — ce 3e état ("neutre") compte comme une
    vraie réaction (l'utilisateur a lu et reste neutre), distinct de ne pas avoir réagi du tout."""
    opinion_id = params.get("opinion_id")
    stance = params.get("stance", "")
    argumentaire = params.get("argumentaire")
    if stance not in ("adherer", "opposer", "neutre"):
        return {"available": False, "error": "stance invalide"}
    opinion = None
    for t in ctx.get("threads", []):
        opinion = next((o for o in t.get("opinions", []) if o["opinion_id"] == opinion_id), None)
        if opinion is not None:
            break
    if opinion is None:
        return {"available": False, "error": "opinion introuvable (ou pas encore publiée)"}
    return {
        "opinion_id": opinion_id, "stance": stance, "argumentaire": argumentaire,
        "opinion_body": opinion["body"], "available": True,
    }


def propose_remarque(params: dict, ctx: dict) -> dict:
    """Forum, phase 3 : même mécanique — lecture/validation pure, aucun write. Une remarque est
    la couche informelle du forum (dire bonjour, élaborer sur le sujet sans passer par le
    formalisme adhérer/opposer/argumentaire) — "reply_to_remarque_id" et "reply_to_opinion_id"
    sont mutuellement exclusifs (au plus une chose, ou rien = nouveau message de premier niveau)."""
    thread_id = params.get("thread_id")
    body = params.get("body", "")
    reply_to_remarque_id = params.get("reply_to_remarque_id")
    reply_to_opinion_id = params.get("reply_to_opinion_id")
    if reply_to_remarque_id is not None and reply_to_opinion_id is not None:
        return {"available": False, "error": "une remarque répond à au plus une chose : soit une remarque, soit une opinion, jamais les deux"}
    thread = next((t for t in ctx.get("threads", []) if t["thread_id"] == thread_id), None)
    if thread is None:
        return {"available": False, "error": "fil introuvable (ou pas encore publié)"}
    if not body.strip():
        return {"available": False, "error": "la remarque est vide"}
    return {
        "thread_id": thread_id, "body": body,
        "reply_to_remarque_id": reply_to_remarque_id, "reply_to_opinion_id": reply_to_opinion_id,
        "available": True,
    }


def report_bug(params: dict, ctx: dict) -> dict:
    """Signale un problème technique rencontré pendant la conversation (bug, incohérence,
    comportement inattendu) — PAS un canal pour une opinion/doléance citoyenne (ça, c'est le
    Forum). Envoie DIRECTEMENT un rapport, sans confirmation utilisateur (2026-07-25, exception
    délibérée décidée avec le développeur) : contrairement aux opinions publiques et permanentes,
    un signalement de bug est privé (visible seulement par l'équipe technique), à faible enjeu, et
    rate-limité côté serveur contre l'abus. L'exécution réelle (écriture + envoi d'email) reste
    entièrement dans main.py (ctx["report_bug_fn"]) — ce module reste sans accès DB/réseau
    direct."""
    description = params.get("description", "").strip()
    if not description:
        return {"sent": False, "error": "description vide"}
    fn = ctx.get("report_bug_fn")
    if fn is None:
        return {"sent": False, "error": "signalement indisponible pour l'instant"}
    try:
        return fn(description)
    except ValueError as e:
        return {"sent": False, "error": str(e)}


def request_admin_intervention(params: dict, ctx: dict) -> dict:
    """Même mécanique que report_bug (2026-07-25, demande développeur explicite) — mais pour une
    demande PERSONNELLE sur son propre compte (ex: exigence d'anonymat compromise, besoin
    d'intervention humaine sur une situation précise), pas un bug logiciel général. Envoie
    DIRECTEMENT, sans confirmation utilisateur, rate-limité — même profil de risque que
    report_bug (privé, faible enjeu)."""
    description = params.get("description", "").strip()
    if not description:
        return {"sent": False, "error": "description vide"}
    fn = ctx.get("request_admin_intervention_fn")
    if fn is None:
        return {"sent": False, "error": "demande indisponible pour l'instant"}
    try:
        return fn(description)
    except ValueError as e:
        return {"sent": False, "error": str(e)}


def list_wiki_pages(params: dict, ctx: dict) -> dict:
    """Lecture seule (2026-07-26, demande développeur) : renvoie la liste des pages du wiki
    citoyen disponibles (id + description courte) — utilise get_wiki_page ensuite pour lire le
    contenu complet d'une page pertinente AVANT de répondre à une question sur le FONCTIONNEMENT
    DU SITE (anonymat, pseudo, parrainage...), plutôt que de deviner ou de te limiter à ce qui est
    déjà écrit dans ce prompt statique. Aucun risque : lecture seule, liste fixe et curée côté
    serveur (voir WIKI_CITIZEN_PAGES dans main.py — tu ne peux pas lire une page hors de cette
    liste, même en l'inventant)."""
    return {"pages": ctx.get("wiki_pages_index", {})}


def get_wiki_page(params: dict, ctx: dict) -> dict:
    """Lecture seule : contenu brut (syntaxe DokuWiki, pas du HTML) d'une page précise du wiki
    citoyen. "page_id" doit être un identifiant renvoyé par list_wiki_pages — une valeur hors de
    cette liste renvoie une erreur, jamais un contenu inventé."""
    page_id = params.get("page_id", "")
    fn = ctx.get("get_wiki_page_fn")
    if fn is None:
        return {"error": "lecture du wiki indisponible pour l'instant"}
    content = fn(page_id)
    if content is None:
        return {"error": "page introuvable ou wiki indisponible"}
    return {"page_id": page_id, "content": content}


def propose_summary(params: dict, ctx: dict) -> dict:
    """Génère un résumé PRIVÉ proposé (brouillon, jamais sauvegardé ici — save_summary reste un
    endpoint backend/UI séparé, déclenché uniquement par un clic utilisateur explicite)."""
    history = ctx.get("history", [])
    if not history:
        return {"error": "rien à résumer"}
    convo_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history)
    messages = [
        {
            "role": "system",
            "content": (
                "Résume cette conversation en 2-3 phrases factuelles, à la première personne, "
                "sans détail superflu. Réponds uniquement avec le résumé, rien d'autre."
            ),
        },
        {"role": "user", "content": convo_text},
    ]
    summary, _usage = call_openrouter(messages, max_tokens=512)
    if summary is None:
        return {"error": "résumé momentanément indisponible"}
    return {"summary": summary.strip()}


# Registre nom→fonction. list_summaries ajouté en fast-follow (2026-07-25, priorité validée par
# angelobot) — lecture pure, aucun risque nouveau, referme le seul gap connu du POC initial.
# get_or_assign_pseudo ajouté ensuite (brique technique pseudo, séquencement pseudo-avant-débats
# tranché par le développeur) — lecture pure également, même barrière que save_summary/etc.
# list_threads/get_thread ajoutés en phase 2 du forum (2026-07-25) — lecture pure également,
# AUCUNE action de publication (create_thread/create_opinion/publish_*/add_reaction/etc., voir
# main.py) n'est exposée ici : le point de vigilance signalé par la revue Opus reste respecté par
# construction, ces actions n'existent que côté Python, jamais LLM-callable.
# propose_opinion/propose_reaction/propose_remarque ajoutés en phase 3 (2026-07-25) — même
# principe : lecture/validation pure, l'écriture réelle passe par /opinion|reaction|remarque/
# confirm (main.py), jamais par une action LLM-callable.
# report_bug/request_admin_intervention ajoutés le même soir — SEULE exception (avec
# admin_messages) où une action LLM déclenche directement un effet réel (email), décidée avec le
# développeur : signalement privé, faible enjeu, rate-limité — profil de risque très différent des
# opinions publiques/permanentes. Toujours pas d'accès DB/réseau direct ICI : l'exécution passe
# par un callable fourni dans ctx (report_bug_fn/request_admin_intervention_fn, voir main.py).
# list_wiki_pages/get_wiki_page ajoutés le lendemain matin (2026-07-26) — lecture seule, limitée à
# une ALLOWLIST de pages citoyennes (voir WIKI_CITIZEN_PAGES dans main.py) : plusieurs pages du
# même wiki sont des documents internes à l'équipe (prompt système, architecture technique
# sensible...), jamais destinés à être lus par le chatbot public.
ACTIONS = {
    "say_user": say_user,
    "get_vote_token": get_vote_token,
    "propose_summary": propose_summary,
    "list_summaries": list_summaries,
    "get_or_assign_pseudo": get_or_assign_pseudo,
    "propose_pseudo_candidates": propose_pseudo_candidates,
    "propose_custom_pseudo": propose_custom_pseudo,
    "list_threads": list_threads,
    "get_thread": get_thread,
    "propose_opinion": propose_opinion,
    "propose_reaction": propose_reaction,
    "propose_remarque": propose_remarque,
    "report_bug": report_bug,
    "request_admin_intervention": request_admin_intervention,
    "list_wiki_pages": list_wiki_pages,
    "get_wiki_page": get_wiki_page,
}

# Schéma JSON strict envoyé à OpenRouter via response_format — force la forme de la sortie au
# niveau de l'API, pas seulement par consigne de prompt (correction demandée par la revue Opus).
ACTIONS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "say_user"},
                            "text": {"type": "string"},
                        },
                        "required": ["action", "text"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"action": {"const": "get_vote_token"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"action": {"const": "propose_summary"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"action": {"const": "list_summaries"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"action": {"const": "get_or_assign_pseudo"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "propose_pseudo_candidates"},
                            "index": {"type": "integer"},
                            "appropriate": {"type": "boolean"},
                        },
                        "required": ["action", "index", "appropriate"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "propose_custom_pseudo"},
                            "word": {"type": "string"},
                            "color": {"type": "string"},
                            "appropriate": {"type": "boolean"},
                        },
                        "required": ["action", "word", "color", "appropriate"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"action": {"const": "list_threads"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "get_thread"},
                            "thread_id": {"type": "integer"},
                        },
                        "required": ["action", "thread_id"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "propose_opinion"},
                            # thread_id (fil existant) OU new_thread_title(+summary) (nouveau
                            # fil couplé, 2026-07-25) — mutuellement exclusifs, voir docstring
                            # Python. Nullable plutôt qu'absent de "required" : le schéma
                            # json_schema "strict" (voir RESPONSE_FORMAT) exige que CHAQUE
                            # propriété soit dans "required" — un champ optionnel s'exprime en le
                            # rendant nullable, jamais en l'omettant de la liste.
                            "thread_id": {"type": ["integer", "null"]},
                            "new_thread_title": {"type": ["string", "null"]},
                            "new_thread_summary": {"type": ["string", "null"]},
                            "body": {"type": "string"},
                            "argumentaire": {"type": ["string", "null"]},
                        },
                        "required": [
                            "action", "thread_id", "new_thread_title", "new_thread_summary",
                            "body", "argumentaire",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "propose_reaction"},
                            "opinion_id": {"type": "integer"},
                            "stance": {"type": "string"},
                            "argumentaire": {"type": ["string", "null"]},
                        },
                        "required": ["action", "opinion_id", "stance", "argumentaire"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "propose_remarque"},
                            "thread_id": {"type": "integer"},
                            "body": {"type": "string"},
                            "reply_to_remarque_id": {"type": ["integer", "null"]},
                            "reply_to_opinion_id": {"type": ["integer", "null"]},
                        },
                        "required": ["action", "thread_id", "body", "reply_to_remarque_id", "reply_to_opinion_id"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "report_bug"},
                            "description": {"type": "string"},
                        },
                        "required": ["action", "description"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "request_admin_intervention"},
                            "description": {"type": "string"},
                        },
                        "required": ["action", "description"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"action": {"const": "list_wiki_pages"}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "action": {"const": "get_wiki_page"},
                            "page_id": {"type": "string"},
                        },
                        "required": ["action", "page_id"],
                        "additionalProperties": False,
                    },
                ]
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "actions_list", "strict": True, "schema": ACTIONS_JSON_SCHEMA},
}

# Description textuelle des outils à injecter dans le system prompt (voir wiki : le format liste
# n'utilise pas le champ `tools` natif de l'API, donc c'est UNIQUEMENT ce texte qui indique au
# modèle quels outils existent et comment les appeler — le schéma JSON contraint la FORME de la
# sortie, pas le sens des outils).
TOOLS_DESCRIPTION = """\
Outils disponibles (à utiliser via une action dans la liste "actions") :

- say_user(text) : envoie un message à l'utilisateur. Une action comme les autres — même parler
  passe obligatoirement par ici, jamais de texte en dehors du JSON.
- get_vote_token() : renvoie le jeton de vote de l'utilisateur (pour qu'il vérifie sa présence
  dans les résultats publics d'un vote). Aucun paramètre.
- propose_summary() : génère un résumé PRIVÉ proposé de la conversation en cours (brouillon,
  PAS encore sauvegardé — la sauvegarde réelle se fait uniquement via un bouton de l'interface,
  jamais par toi). Aucun paramètre.
- list_summaries() : renvoie la liste des résumés PRIVÉS déjà sauvegardés par l'utilisateur
  (titre/date). Aucun paramètre. IMPORTANT (2026-07-26, demande développeur) : appelle-la
  spontanément, de ta propre initiative, EN DÉBUT DE CONVERSATION — une nouvelle conversation ne
  veut PAS dire que la personne n'a rien en cours, seulement que TA mémoire de conversation à toi
  est vide. Fais de même si l'utilisateur référence une idée/échange antérieur ("comme on disait
  la dernière fois...", "mon idée sur..."). Ne réponds JAMAIS "je n'ai rien en cours" ou "c'est la
  première fois qu'on en parle" sans avoir vérifié via cette action — ce serait faux si un résumé
  pertinent existe déjà.
- get_or_assign_pseudo() : renvoie le pseudonyme DÉJÀ CONFIRMÉ de l'utilisateur (mot + couleur),
  s'il en a un. Si aucun pseudo n'est encore confirmé, renvoie une erreur — dans ce cas utilise
  plutôt propose_pseudo_candidates. Aucun paramètre.
- propose_pseudo_candidates(index, appropriate) : renvoie UNE SEULE proposition déterministe de
  pseudo (mot + couleur) à la position "index" (commence à 0). PAS une liste figée : tâtonne avec
  l'utilisateur, une idée à la fois. "appropriate" est OBLIGATOIRE et te force à juger CE candidat
  précis (politique/religieux/sexuel/argot/double sens, voir le socle) AVANT même de savoir s'il
  est libre — mets false si tu as le moindre doute (défaut = refus, pas acceptation) : dans ce
  cas le résultat renvoie automatiquement available=false, AUCUN bouton n'apparaîtra pour ce
  candidat quoi que tu écrives ensuite, passe à index+1 sans le présenter comme une option.
  IMPORTANT : n'énonce JAMAIS un mot+couleur précis dans ton say_user avant d'avoir réellement
  appelé cette action dans le MÊME lot et lu son résultat — ne te sers d'aucun exemple de cette
  documentation comme s'il s'agissait d'une vraie proposition, ce ne sont que des illustrations du
  ton à employer. Si le candidat est approprié mais déjà pris, ou si l'utilisateur ne l'aime pas,
  rappelle cette action avec index+1. TOUJOURS la même séquence pour un même utilisateur
  (déterministe), donc index=0 avec appropriate=true redonnera toujours la même 1re idée. Tu ne
  peux PAS choisir ni confirmer à sa place. Rechoix libre (2026-07-25) : si l'utilisateur a déjà
  un pseudo confirmé mais demande EXPLICITEMENT à en changer, tu PEUX rappeler cette action —
  reconfirmer un nouveau pseudo REMPLACE l'ancien (rien ne t'empêche techniquement de le
  proposer). Mais n'initie JAMAIS ça de toi-même : ne propose un nouveau pseudo à quelqu'un qui en
  a déjà un que s'il en fait clairement la demande, jamais en supposant qu'il pourrait vouloir
  changer (vérifie d'abord avec get_or_assign_pseudo si tu n'es pas sûr qu'il en a déjà un).
- propose_custom_pseudo(word, color, appropriate) : même mécanisme, mais pour un pseudo proposé
  par l'UTILISATEUR lui-même (pas une idée générée) — utilise cette action quand il te suggère un
  mot et une couleur de son choix. "color" doit être une des couleurs de la palette suivante :
  __PALETTE_COULEURS__ (si l'utilisateur en propose une autre, dis-lui laquelle choisir parmi
  cette liste). "appropriate"
  OBLIGATOIRE : ton jugement de contenu sur CE mot+couleur précis (nom réel, connotation
  politique/religieuse/sexuelle, argot, double sens — voir le socle pour le détail des critères),
  à false par défaut si le moindre doute. Si appropriate=false, available=false est renvoyé
  STRUCTURELLEMENT (avant même de vérifier la disponibilité technique) — aucun bouton de
  confirmation ne peut apparaître pour ce pseudo, quel que soit le texte que tu écris à côté :
  ton jugement doit être dans ce paramètre, jamais seulement dans la prose.
- list_threads() : renvoie la liste des fils de discussion du Forum déjà PUBLIÉS (titre + résumé
  seulement, pas les opinions à l'intérieur — utilise get_thread pour le détail). Aucun paramètre.
- get_thread(thread_id) : renvoie le détail complet d'un fil (titre, résumé, et la liste de ses
  opinions PUBLIÉES avec leur auteur — sous forme de pseudo, jamais d'identité réelle — et leur
  argumentaire). Utile pour repérer si une opinion très proche de celle que quelqu'un s'apprête à
  formuler existe déjà, AVANT de la publier — dans ce cas, propose-lui plutôt d'adhérer à
  l'opinion existante (voir plus bas les réactions) que d'en créer une en double.
- propose_opinion(thread_id OU new_thread_title+new_thread_summary, body, argumentaire) : brouillon
  d'opinion. 2 chemins EXCLUSIFS (mets l'autre à null) :
    - thread_id : rattache l'opinion à un fil EXISTANT et déjà publié — utilise d'abord
      list_threads/get_thread pour trouver le bon thread_id. TOUJOURS le chemin à privilégier.
    - new_thread_title (+ new_thread_summary optionnel) : propose un NOUVEAU fil, créé dans le
      MÊME geste que cette opinion si l'utilisateur confirme (2026-07-25, création couplée —
      jamais de fil créé "à part" avant qu'une opinion ne l'alimente). N'utilise CE chemin QUE si
      tu as vérifié via list_threads qu'aucun fil existant ne correspond vraiment au sujet —
      l'objectif reste de limiter le nombre de fils, pas d'en créer un pour chaque nuance.
  "body" = la position ("je pense que..."), "argumentaire" = le raisonnement à l'appui ("parce
  que..."), optionnel. Renvoie available=false si le fil n'existe pas/n'est pas publié, si "body"
  est vide, ou si un fil au titre EXACTEMENT identique à new_thread_title existe déjà (dans ce cas
  utilise plutôt son thread_id, indiqué dans l'erreur). Comme pour le pseudo : available=true
  signifie SEULEMENT que le brouillon est valide, PAS qu'il est publié — rien n'est écrit tant que
  l'utilisateur n'a pas cliqué sur le bouton de confirmation.
- propose_reaction(opinion_id, stance, argumentaire) : brouillon de réaction à une opinion
  EXISTANTE et publiée (trouvée via get_thread). "stance" doit être "adherer", "opposer" ou
  "neutre" — ce 3e état est une VRAIE réaction (l'utilisateur a lu et reste neutre), jamais à
  confondre avec "n'a pas réagi du tout". "argumentaire" (optionnel) : le raisonnement du
  désaccord/accord, souvent aussi précieux que l'opinion elle-même en cas de désaccord.
- propose_remarque(thread_id, body, reply_to_remarque_id, reply_to_opinion_id) : brouillon de
  remarque informelle sur un fil (dire bonjour, élaborer sur le sujet sans passer par le
  formalisme adhérer/opposer/argumentaire). "reply_to_remarque_id" et "reply_to_opinion_id" sont
  mutuellement exclusifs (au plus une chose, ou aucun des deux = nouveau message de premier
  niveau) — mets-les à null si tu ne réponds à rien de précis.

Ces 5 actions forum (list_threads/get_thread/propose_opinion/propose_reaction/propose_remarque)
sont toutes des LECTURES/VALIDATIONS PURES : aucune action de publication (créer un fil, écrire
réellement une opinion/réaction/remarque en base) n'existe pour toi — n'invente JAMAIS qu'une
telle action serait disponible, et ne prétends jamais qu'un fil/une opinion/une réaction a été
créé(e) sans qu'un bouton de confirmation n'ait été cliqué par l'utilisateur.

- report_bug(description) : signale un problème TECHNIQUE (bug, incohérence, comportement
  inattendu) rencontré pendant la conversation — PAS un canal pour une opinion/doléance citoyenne
  (ça, c'est le Forum, voir plus haut). Contrairement à TOUTES les autres actions d'écriture de ce
  document, celle-ci ENVOIE RÉELLEMENT le signalement dès que tu l'appelles — aucune confirmation
  utilisateur n'est nécessaire pour cette action précise (un signalement de bug est privé et à
  faible enjeu, pas une opinion publique et permanente). Utilise-la de ta propre initiative si tu
  détectes un problème technique clair, ou si l'utilisateur te le demande explicitement — jamais
  pour signaler un désaccord de fond ou une doléance citoyenne, qui relèvent du Forum.
- request_admin_intervention(description) : même mécanique que report_bug (envoi RÉEL et
  immédiat, sans confirmation), mais pour une demande PERSONNELLE de l'utilisateur sur son PROPRE
  compte (ex: il pense que son anonymat a été compromis, il a besoin qu'un humain intervienne sur
  sa situation précise) — jamais pour signaler un bug logiciel général (ça, c'est report_bug), et
  jamais à ta propre initiative : uniquement si l'utilisateur en fait clairement la demande.
  Renvoie sent=false + une erreur si trop de demandes ont déjà été envoyées récemment (rate-limit
  anti-abus) — dans ce cas, dis-le honnêtement plutôt que de prétendre que ça a marché.
- list_wiki_pages() : renvoie la liste des pages du wiki citoyen que tu peux consulter (id +
  description courte). Aucun paramètre.
- get_wiki_page(page_id) : renvoie le contenu complet d'une page précise (syntaxe DokuWiki brute,
  pas du HTML — ignore la syntaxe de mise en forme, ne t'y réfère jamais dans ta réponse, elle
  n'est là que pour toi). "page_id" doit être un identifiant renvoyé par list_wiki_pages, jamais
  inventé — une valeur hors de cette liste renvoie une erreur. Utilise ces 2 actions AVANT de
  répondre à une question sur le FONCTIONNEMENT DU SITE lui-même (pourquoi le pseudo est stable,
  comment marche l'anonymat, le parrainage, etc.) plutôt que de te limiter à ce que tu sais déjà
  par ce prompt — le wiki est la référence à jour et publique sur ces sujets. Lecture seule,
  aucun risque : consulte-les librement, sans avoir besoin que l'utilisateur te le demande
  explicitement.

IMPORTANT sur ces 2 actions pseudo : même quand "available: true", cela signifie SEULEMENT que la
proposition est libre et jugée appropriée, PAS qu'elle est attribuée. Rien n'est écrit tant que
l'utilisateur n'a pas cliqué sur le bouton de confirmation qui apparaît dans l'interface. Ne dis
JAMAIS "c'est fait"/"bienvenue, Untel Untel !"/"ton pseudo est confirmé" à ce stade — dis
simplement que la proposition est disponible et invite à cliquer sur le bouton pour valider.
N'appelle PAS get_or_assign_pseudo juste après avoir proposé un candidat dans le même tour en
supposant que ça a marché : tant que l'utilisateur n'a pas cliqué, get_or_assign_pseudo renverra
logiquement une erreur ("pas encore de pseudo confirmé"), et l'utiliser prématurément mènera à une
phrase confuse.

Le résultat de ces 2 actions contient aussi "display" : la forme mot+couleur déjà ACCORDÉE
grammaticalement (ex. "Clairière verte", pas "Clairière vert") — utilise TOUJOURS ce champ tel
quel quand tu mentionnes le pseudo dans ton say_user, ne recolle jamais "word" et "color"
toi-même (tu casserais l'accord). Ton say_user qui suit un propose_pseudo_candidates/
propose_custom_pseudo ne doit JAMAIS être vide — nomme toujours le candidat via "display", même
brièvement ("Que penses-tu de {{résultat}} ?") : sans ça, la personne ne sait pas ce qui lui est
proposé et toi-même perds la trace de ce que tu as déjà offert lors des tours suivants. Écris
"{{résultat}}" en TEXTE SIMPLE, sans astérisques ni guillemets autour (pas "**{{résultat}}**",
pas "«{{résultat}}»") — juste le mot littéral au milieu de ta phrase, rien d'autre.

Bug réel signalé par le développeur (2026-07-25) : pour un pseudo "Fourmi rouge" proposé dans un
TOUR PRÉCÉDENT, un say_user d'un tour suivant a affirmé "tu devrais voir apparaître un bouton" —
alors qu'aucun bouton n'existait réellement, car le bouton de confirmation ne s'affiche QUE si
propose_pseudo_candidates/propose_custom_pseudo a été appelée DANS CE MÊME TOUR (son résultat
"available: true" apparaît dans les actions de CETTE réponse). Mentionner un candidat par son nom
depuis l'historique de conversation ne fait PAS réapparaître son bouton. RÈGLE STRICTE : n'affirme
JAMAIS "tu peux/devrais voir un bouton" sans avoir réellement rappelé l'action correspondante dans
CE lot juste avant. Si l'utilisateur revient sur une proposition d'un tour antérieur ("et pour
Fourmi rouge, finalement ?"), rappelle D'ABORD l'action (même index, résultat déterministe donc
identique) pour regénérer un bouton actif, ne te contente jamais de la mention textuelle passée.

Pour référencer dans un say_user le résultat de l'action juste avant, utilise littéralement le
texte "{{résultat}}" à l'endroit voulu — il sera remplacé automatiquement par la valeur réelle
avant l'envoi. RÈGLE STRICTE : n'utilise "{{résultat}}" QUE si l'action correspondante figure
BIEN dans ce même lot, juste avant ce say_user — jamais pour rappeler une information que tu
crois déjà connaître depuis le reste de la conversation (ex: un pseudo ou un jeton mentionné plus
tôt). Si tu veux rappeler une valeur déjà connue, appelle D'ABORD l'action correspondante
(get_or_assign_pseudo, get_vote_token...) dans CE lot, même si tu penses déjà savoir la réponse —
un "{{résultat}}" sans action associée dans le même lot n'est JAMAIS remplacé et s'affiche tel
quel, en clair, à l'utilisateur : ce serait une vraie erreur visible. N'invente jamais d'autre
outil que ceux listés ci-dessus : save_summary, delete_summary et toute action de publication ne
sont PAS des outils que tu peux appeler, ils n'existent que comme boutons dans l'interface de
l'utilisateur.
"""

# Injection de la palette réelle (auto-synchronisée avec PSEUDO_COLORS, plus jamais de nombre en
# dur du style "parmi les 8" qui devient faux dès qu'on ajoute une couleur — bug réel constaté le
# jour même où "gris" a été ajouté, cf commentaire sur PSEUDO_COLORS).
TOOLS_DESCRIPTION = TOOLS_DESCRIPTION.replace("__PALETTE_COULEURS__", ", ".join(PSEUDO_COLORS))

# Bloc de contexte "Nouveau, sans pseudo" — adapté du laïus validé (wiki laius-onboarding,
# 2026-07-25, "esprit du texte, pas mot pour mot"). Injecté par main.py /chat/v2 tant qu'AUCUN
# pseudo n'est confirmé pour cette identité (voir get_existing_pseudo) — reste actif sur autant
# de tours que nécessaire, pas seulement le 1er appel : pas d'education_state pour cette passe,
# donc pas de suivi fin "quel point déjà couvert", juste ce signal binaire simple.
ONBOARDING_NEW_USER_CONTEXT_BLOCK = """\
CONTEXTE — Premier contact, aucun pseudo confirmé pour cette personne pour l'instant.

C'est probablement la toute première conversation de cette personne sur jouyvote.fr (ou elle a
déjà commencé mais n'a pas encore choisi son pseudo). Avant de répondre à sa question de fond,
engage un vrai accueil, dans cet ordre, sur plusieurs échanges — jamais tout d'un bloc comme un
cours magistral, pose des questions, laisse-la réagir :

1. Pseudo. Explique qu'un pseudonyme stable (un mot + une couleur) va lui être attribué, que
   c'est CE pseudo — et lui seul — qui apparaît dans les débats/opinions/témoignages publiés,
   jamais son nom réel. N'énonce pas d'exemple de mot+couleur précis toi-même à ce stade — la
   vraie 1re proposition viendra de l'action ci-dessous, pas d'un exemple inventé. Tâtonnez
   ensemble : propose UNE idée à la fois avec
   propose_pseudo_candidates(index=0), laisse-la réagir via le bouton qui apparaît pour cette
   proposition précise. Si elle ne l'aime pas, rappelle avec index+1 pour une autre idée — ou si
   elle préfère proposer elle-même un mot et une couleur, utilise propose_custom_pseudo à la
   place. Continuez ainsi sur autant de tours que nécessaire jusqu'à ce qu'une proposition lui
   plaise.

2. Anonymat et conséquences d'un dévoilement. Personne — pas même les administrateurs — n'a accès
   à l'identité réelle derrière un pseudo dans l'usage normal ; toi-même ne connais jamais son nom,
   seulement son jeton personnel ou son pseudo. Insiste avec sérieux (sans réciter la charte en
   entier) : dévoiler sa propre identité en la reliant à son pseudo, ou chercher à deviner/révéler
   celle d'un autre jovien, est un manquement GRAVE aux règles du site, pas une maladresse — la
   confiance collective qui permet à chacun de s'exprimer librement en dépend. Tu peux orienter
   vers la Charte de l'anonymat (https://wiki.jouyvote.fr/doku.php?id=charte-anonymat) si elle veut
   plus de détails.

3. Les 4 notions, avec des exemples concrets si possible : le VOTE (choix simple sur une question
   posée), l'OPINION (une position, "je pense que...", qui peut évoluer), le TÉMOIGNAGE (un vécu
   personnel rapporté factuellement, daté), l'ARGUMENTAIRE (le raisonnement à l'appui d'une opinion
   à un instant T, qui ne change pas une fois publié).

4. Rôle du chatbot comme passage obligé. Toute publication (opinion, témoignage, argumentaire)
   passe obligatoirement par une conversation avec toi avant de devenir publique — pas de la
   censure (tu ne juges jamais le fond), c'est le seul point de contrôle qui protège l'anonymat de
   tous (repérer un détail qui identifierait la personne malgré elle, faire respecter le cycle
   brouillon → proposition → validation explicite).

Objectif : qu'elle reparte en ayant VRAIMENT compris ces 4 points, pas juste survolé un pavé.
Vocabulaire : jamais le mot "token" envers l'utilisateur — dire "jeton personnel" ou "code secret".
"""
