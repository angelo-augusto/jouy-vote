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


def compute_vote_token(identity_token: str) -> str:
    raw = f"{identity_token}:{JOUY_VOTE_PEPPER}"
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_debate_token(identity_token: str) -> str:
    raw = f"{identity_token}:{JOUY_PSEUDO_PEPPER}:pseudo"
    return hashlib.sha256(raw.encode()).hexdigest()


# Mot (objet/être) + couleur — convention du wiki (themes:pseudonyme, architecture-technique).
# Liste volontairement modeste pour cette passe technique (pas de choix collaboratif avec
# l'utilisateur pour l'instant, voir docstring de get_or_assign_pseudo) : suffisant à l'échelle
# de Jouy, affinable plus tard sans changer le mécanisme de dérivation lui-même.
PSEUDO_WORDS = [
    "Renard", "Hibou", "Chêne", "Lanterne", "Étoile", "Rivière", "Nuage", "Phare",
    "Comète", "Sentier", "Écureuil", "Orage", "Prairie", "Faucon", "Ruche", "Glacier",
    "Roseau", "Aurore", "Cascade", "Bourgeon", "Falaise", "Marée", "Clairière", "Genêt",
    "Corail", "Frimas", "Tilleul", "Brume", "Sittelle", "Ravin",
]
# Palette réduite à 8 couleurs simples et universelles (retour développeur 2026-07-25, via
# angelobot) — les couleurs précédentes (argenté, carmin, ambre...) jugées trop compliquées pour
# un public général. Table pseudos vide en prod au moment du changement, aucune migration requise.
PSEUDO_COLORS = ["rouge", "orange", "jaune", "vert", "bleu", "violet", "blanc", "noir"]


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


def generate_pseudo_candidates(identity_token: str, n: int = 3) -> list[dict]:
    """N propositions déterministes et DISTINCTES (même identity_token → toujours les mêmes N
    propositions, y compris si redemandées plus tard dans la conversation — pas de tirage
    aléatoire qui changerait à chaque appel). Dédoublonnage simple par avancement d'index en cas
    de collision fortuite (peu probable avec 30×20 combinaisons, mais pas coûteux à éviter)."""
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
    return dict(pseudo)


def propose_pseudo_candidates(params: dict, ctx: dict) -> dict:
    """Lecture pure (aucun write) : génère 2-3 propositions déterministes de pseudo à partir de
    l'identity_token. L'utilisateur choisit parmi elles via l'interface (bouton), qui appelle le
    vrai point d'écriture confirm_pseudo (main.py) — jamais un commit direct depuis cette action,
    même logique que propose_summary/save_summary."""
    identity_token = ctx.get("identity_token")
    if not identity_token:
        return {"error": "identité manquante dans le contexte"}
    return {"candidates": generate_pseudo_candidates(identity_token)}


def list_summaries(params: dict, ctx: dict) -> dict:
    """Lecture pure : les résumés déjà sauvegardés sont fournis via ctx["summaries"] (chargés en
    amont par l'appelant — main.py fait la requête DB, mcp_chatbot_executor.py passe une liste
    de test) — même pattern que ctx["history"], pour garder ce module sans dépendance DB."""
    return {"summaries": ctx.get("summaries", [])}


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
ACTIONS = {
    "say_user": say_user,
    "get_vote_token": get_vote_token,
    "propose_summary": propose_summary,
    "list_summaries": list_summaries,
    "get_or_assign_pseudo": get_or_assign_pseudo,
    "propose_pseudo_candidates": propose_pseudo_candidates,
}

# Schéma JSON strict envoyé à OpenRouter via response_format — force la forme de la sortie au
# niveau de l'API, pas seulement par consigne de prompt (correction demandée par la revue Opus).
ACTIONS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
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
                        "properties": {"action": {"const": "propose_pseudo_candidates"}},
                        "required": ["action"],
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
  (titre/date). Aucun paramètre.
- get_or_assign_pseudo() : renvoie le pseudonyme DÉJÀ CONFIRMÉ de l'utilisateur (mot + couleur,
  ex. "Renard bleu"), s'il en a un. Si aucun pseudo n'est encore confirmé, renvoie une erreur —
  dans ce cas utilise plutôt propose_pseudo_candidates. Aucun paramètre.
- propose_pseudo_candidates() : génère 2-3 propositions de pseudo (mot + couleur) parmi
  lesquelles l'utilisateur va choisir via l'interface — TOUJOURS les mêmes propositions pour un
  même utilisateur si tu les redemandes plus tard dans la conversation. Tu ne peux PAS choisir ni
  confirmer à sa place : présente les propositions, laisse l'utilisateur cliquer sur celle qu'il
  préfère (ou en redemander d'autres plus tard s'il n'aime aucune — même mécanisme, mêmes
  propositions). N'appelle JAMAIS cette action si l'utilisateur a déjà un pseudo confirmé (vérifie
  d'abord avec get_or_assign_pseudo si tu n'es pas sûr) — un pseudo confirmé est stable, il ne se
  change pas à volonté ; si on te le redemande, rappelle simplement le pseudo déjà attribué au
  lieu d'en reproposer de nouveaux. IMPORTANT : les propositions sont déjà affichées SÉPARÉMENT
  sous forme de boutons cliquables dans l'interface — n'énumère JAMAIS toi-même les candidats
  dans ton say_user (ni en texte, ni via {{résultat}}, qui ne fonctionne de toute façon que pour
  une valeur simple, pas une liste). Écris seulement une courte phrase d'accroche du type "Voici
  quelques idées, clique sur celle qui te plaît". Aucun paramètre.

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

1. Pseudo. Explique qu'un pseudonyme stable (mot + couleur, ex. "Lapin jaune") va lui être
   attribué, que c'est CE pseudo — et lui seul — qui apparaît dans les débats/opinions/témoignages
   publiés, jamais son nom réel. Utilise l'action propose_pseudo_candidates pour lui proposer 2-3
   combinaisons ; laisse-la choisir via l'interface (tu ne peux pas choisir à sa place). Si aucune
   ne lui plaît, tu peux relancer propose_pseudo_candidates plus tard — les mêmes propositions
   reviendront, ce n'est pas un problème, explique-le simplement si elle demande pourquoi.

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
