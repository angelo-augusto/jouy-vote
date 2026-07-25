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


def compute_vote_token(identity_token: str) -> str:
    raw = f"{identity_token}:{JOUY_VOTE_PEPPER}"
    return hashlib.sha256(raw.encode()).hexdigest()


def say_user(params: dict, ctx: dict) -> dict:
    """Pas d'effet de bord : le texte est simplement renvoyé, chatbot_executor.py se charge de
    l'ajouter à la liste des répliques affichées à l'utilisateur."""
    return {"text": params.get("text", "")}


def get_vote_token(params: dict, ctx: dict) -> dict:
    identity_token = ctx.get("identity_token")
    if not identity_token:
        return {"error": "identité manquante dans le contexte"}
    return {"vote_token": compute_vote_token(identity_token)}


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
ACTIONS = {
    "say_user": say_user,
    "get_vote_token": get_vote_token,
    "propose_summary": propose_summary,
    "list_summaries": list_summaries,
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

Pour référencer dans un say_user le résultat de l'action juste avant, utilise littéralement le
texte "{{résultat}}" à l'endroit voulu — il sera remplacé automatiquement par la valeur réelle
avant l'envoi. N'invente jamais d'autre outil que ceux listés ci-dessus : save_summary,
delete_summary et toute action de publication ne sont PAS des outils que tu peux appeler, ils
n'existent que comme boutons dans l'interface de l'utilisateur.
"""
