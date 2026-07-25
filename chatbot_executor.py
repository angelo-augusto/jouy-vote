"""Boucle d'exécution du chatbot jouyvote.fr (POC tool-calling).

Cœur Python pur, réutilisable — pas de dépendance à FastAPI/main.py. Exposé en prod via un
appel direct depuis main.py (POST /chat/v2), et en test/debug via mcp_chatbot_executor.py (1 tool
MCP par-dessus, voir ce fichier).
"""
from __future__ import annotations

import json
import logging

from chatbot_actions import ACTIONS, RESPONSE_FORMAT, TOOLS_DESCRIPTION
from chatbot_llm import call_openrouter

log = logging.getLogger("chatbot_executor")

MAX_ITERATIONS = 5

FORMAT_REMINDER = """\
Rappel de format de sortie STRICT : la SEULE chose que tu as le droit de renvoyer est un appel à
l'outil de sortie structurée avec la liste d'actions en paramètre. Jamais de texte libre en
dehors de ce format, même pour parler à l'utilisateur — utilise say_user pour ça, à l'intérieur
de la liste."""


def build_system_prompt(base_prompt: str, context_block: str = "") -> str:
    """Assemble le system prompt dans l'ordre retenu (wiki themes:prompt-chatbot) : identité/
    règles (base_prompt) → outils → bloc de contexte situationnel (vide pour le POC, socle
    statique) → rappel de format en tout dernier."""
    parts = [base_prompt.strip(), TOOLS_DESCRIPTION.strip()]
    if context_block:
        parts.append(context_block.strip())
    parts.append(FORMAT_REMINDER)
    return "\n\n".join(parts)


def _substitute_placeholder(text: str, previous_result: dict | None) -> str:
    """Bug réel trouvé en conditions réelles (2026-07-25, test get_or_assign_pseudo) : un résultat
    à UNE clé (vote_token, summary...) donnait un texte naturel, mais un résultat à PLUSIEURS clés
    (pseudo: {word, color}) tombait dans le fallback JSON brut — "Ton pseudo est :
    {"word": "Clairière", "color": "corail"}" affiché tel quel à un vrai utilisateur. Fix : joindre
    les valeurs par un espace plutôt que sérialiser le dict, ce qui donne "Clairière corail" (et
    reste correct pour le cas à une seule clé, qui redonne simplement cette valeur seule)."""
    if "{{résultat}}" not in text or not previous_result:
        return text
    if "error" in previous_result:
        value = previous_result["error"]
    else:
        values = [str(v) for k, v in previous_result.items() if k != "text"]
        value = " ".join(values) if values else json.dumps(previous_result, ensure_ascii=False)
    return text.replace("{{résultat}}", str(value))


def run_turn(
    system_prompt: str,
    conversation_messages: list[dict],
    ctx: dict,
    model: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> dict:
    """Exécute un tour complet (potentiellement plusieurs allers-retours LLM si une action non-
    parole termine un lot). ctx : contexte métier (identity_token, history...) passé tel quel à
    chaque fonction d'action.

    Retourne {"replies": [str, ...], "actions_log": [...], "error": None|str}. N'échoue jamais
    par exception non capturée — une erreur devient "error" dans le retour."""
    messages = [{"role": "system", "content": system_prompt}] + conversation_messages
    replies: list[str] = []
    actions_log: list[dict] = []

    for _iteration in range(max_iterations):
        content, _usage = call_openrouter(messages, response_format=RESPONSE_FORMAT, max_tokens=4096, model=model)
        if content is None:
            return {"replies": replies, "actions_log": actions_log, "error": "llm_indisponible"}

        try:
            data = json.loads(content)
            actions = data.get("actions", []) if isinstance(data, dict) else []
        except json.JSONDecodeError:
            log.error("Réponse LLM non-JSON malgré response_format strict : %s", content[:300])
            return {"replies": replies, "actions_log": actions_log, "error": "json_invalide"}

        if not actions:
            return {"replies": replies, "actions_log": actions_log, "error": "liste_actions_vide"}

        previous_result: dict | None = None
        for cmd in actions:
            action = cmd.get("action")
            fn = ACTIONS.get(action)
            if fn is None:
                # Barrière structurelle : toute action hors du registre (ex: le LLM invente
                # "save_summary") est ignorée, jamais exécutée. Ce n'est pas censé arriver — le
                # schéma strict l'empêche déjà — mais on ne fait jamais confiance à une seule
                # couche de défense pour un commit sensible.
                log.warning("Action hors périmètre ignorée : %s", action)
                actions_log.append({"action": action, "error": "action non autorisée pour le LLM"})
                continue
            if action == "say_user":
                text = _substitute_placeholder(cmd.get("text", ""), previous_result)
                replies.append(text)
                actions_log.append({"action": action, "text": text})
                previous_result = None
            else:
                result = fn(cmd, ctx)
                actions_log.append({"action": action, "result": result})
                previous_result = result

        last_action = actions[-1].get("action")
        if last_action == "say_user":
            return {"replies": replies, "actions_log": actions_log, "error": None}

        # Le lot se termine par une action non-parole : le LLM a besoin du résultat pour
        # continuer à raisonner (sinon il aurait clos par un say_user) — on relance avec ce
        # résultat plutôt que de couper le tour court.
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": f"[Résultat de la dernière action : {json.dumps(previous_result, ensure_ascii=False)}] Continue.",
        })

    log.warning("Limite d'itérations (%d) atteinte sans clôture par say_user", max_iterations)
    return {"replies": replies, "actions_log": actions_log, "error": "max_iterations_atteinte"}
