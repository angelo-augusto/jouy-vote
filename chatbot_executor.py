"""Boucle d'exécution du chatbot jouyvote.fr (POC tool-calling).

Cœur Python pur, réutilisable — pas de dépendance à FastAPI/main.py. Exposé en prod via un
appel direct depuis main.py (POST /chat/v2), et en test/debug via mcp_chatbot_executor.py (1 tool
MCP par-dessus, voir ce fichier).
"""
from __future__ import annotations

import json
import logging
import re

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


# Bug réel #8 (2026-07-25, capture développeur : "Que dirais-tu de **** ?" recurrence après le
# fix #7) : la consigne TOOLS_DESCRIPTION interdit d'entourer "{{résultat}}" de "**"/guillemets,
# mais un LLM ne respecte pas toujours une consigne — le modèle a quand même écrit
# "**{{résultat}}**". Avant ce fix, un simple .replace("{{résultat}}", "") laissait les 2 paires de
# "**" adjacentes, qui se collent visuellement en "****" une fois le mot retiré entre elles. Fix :
# retirer le wrapping markdown/guillemets EN MÊME TEMPS que le token, pas seulement le token seul.
_WRAPPED_UNRESOLVED_PLACEHOLDER = re.compile(
    r'\*\*\{\{résultat\}\}\*\*|«\{\{résultat\}\}»|"\{\{résultat\}\}"|\{\{résultat\}\}'
)


def _strip_unresolved_placeholder(text: str) -> str:
    text = _WRAPPED_UNRESOLVED_PLACEHOLDER.sub("", text)
    return text.replace("  ", " ").strip()


def _render_result_value(previous_result: dict | None) -> str:
    """Meilleure représentation textuelle d'un résultat d'action — réutilisée à la fois pour
    substituer {{résultat}} et pour générer un texte de repli quand le LLM laisse un say_user
    vide juste après une action (voir bug #5 plus bas). Préfère "display" (forme déjà accordée
    grammaticalement, voir chatbot_actions._agree_pseudo_display) s'il existe, sinon joint les
    valeurs scalaires restantes (hors champs de statut interne)."""
    if not previous_result:
        return ""
    if "error" in previous_result:
        return str(previous_result["error"])
    display = previous_result.get("display")
    if isinstance(display, str) and display:
        return display
    scalar_values = [
        str(v) for k, v in previous_result.items()
        if k not in ("text", "available", "display") and isinstance(v, (str, int, float)) and not isinstance(v, bool)
    ]
    return " ".join(scalar_values)


def _substitute_placeholder(text: str, previous_result: dict | None) -> str:
    """Bug réel #1 trouvé en conditions réelles (2026-07-25, test get_or_assign_pseudo) : un
    résultat à UNE clé (vote_token, summary...) donnait un texte naturel, mais un résultat à
    PLUSIEURS clés (pseudo: {word, color}) tombait dans le fallback JSON brut — "Ton pseudo est :
    {"word": "Clairière", "color": "corail"}" affiché tel quel à un vrai utilisateur. Fix : joindre
    les valeurs par un espace plutôt que sérialiser le dict, ce qui donne "Clairière corail" (et
    reste correct pour le cas à une seule clé, qui redonne simplement cette valeur seule).

    Bug réel #2 trouvé en conditions réelles (même jour, un tour plus tard) : le LLM a écrit
    "{{résultat}}" dans un say_user SANS appeler l'action correspondante dans le même lot (il
    croyait connaître la valeur depuis le reste de la conversation) — previous_result était None,
    et l'ancien code renvoyait le texte INCHANGÉ, donc "Ton pseudo est {{résultat}}" littéral
    envoyé tel quel à l'utilisateur. Fix défensif (en plus du renforcement de la consigne dans
    TOOLS_DESCRIPTION) : si le placeholder n'a rien à substituer, on le retire proprement plutôt
    que de laisser fuiter une syntaxe technique interne vers un citoyen.

    Bug réel #3 trouvé en conditions réelles (2026-07-25, signalé par Angelo avec capture d'écran)
    : propose_pseudo_candidates renvoie {"candidates": [...]} — une clé unique dont la VALEUR est
    une LISTE de dicts, pas un scalaire. Le fix #1 ci-dessus ne filtrait que sur le nombre de clés,
    pas sur le TYPE de chaque valeur : str() d'une liste de dicts Python produit un repr avec des
    guillemets simples ("[{'word': 'Falaise', 'color': 'argenté'}, ...]"), affiché tel quel dans le
    chat. Fix : ne retenir que les valeurs SCALAIRES (str/int/float/bool) pour la jointure — toute
    valeur composite (liste/dict) est ignorée, et si aucune valeur scalaire ne reste, le
    placeholder est retiré proprement (même filet que le bug #2) plutôt que de risquer une autre
    forme de fuite de structure interne.

    Bug réel #4 trouvé en conditions réelles (2026-07-25, un tour plus tard encore) :
    propose_pseudo_candidates renvoie désormais {"word":.., "color":.., "available": True} — le
    fix #3 incluait les booléens dans la jointure ("bool" faisait partie des types scalaires
    acceptés), donc "available" s'est retrouvé littéralement collé au texte : "Roseau orange
    True". Fix : exclure explicitement les booléens ET la clé "available" (champ de statut interne,
    jamais un mot à afficher) — ne garder que des valeurs de CONTENU réel (mot, couleur, jeton,
    résumé...)."""
    if "{{résultat}}" not in text:
        return text
    value = _render_result_value(previous_result)
    if not value:
        return _strip_unresolved_placeholder(text)
    return text.replace("{{résultat}}", value)


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
    # Bug réel #7 (2026-07-25, cause profonde des bugs #5/#6 : mesuré à ~2/5 même après leurs
    # fixes) : quand un lot se termine par une action non-parole (relance LLM), le résultat de
    # cette action était PERDU au début de l'itération suivante — "previous_result" était
    # réinitialisé à None à chaque nouvelle complétion, alors que le say_user qui arrive dans LA
    # COMPLÉTION SUIVANTE en a justement besoin pour son fallback. Concrètement : le modèle
    # proposait un candidat dans un 1er appel (lot sans say_user final → relance), puis le
    # say_user "troué" du 2e appel n'avait plus aucune trace du résultat à substituer. Fix :
    # transporter le résultat en attente ("carried_result") d'une itération à l'autre.
    carried_result: dict | None = None

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

        previous_result: dict | None = carried_result
        carried_result = None
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
                fallback = _render_result_value(previous_result) if previous_result else ""
                # "display" (pseudo word+couleur accordé) n'existe que pour les actions pseudo —
                # c'est UNIQUEMENT là que le modèle est explicitement tenu de nommer le résultat
                # (voir TOOLS_DESCRIPTION) ; d'autres actions (ex: get_vote_token) tolèrent un
                # accusé de réception sans citer littéralement la valeur brute ("Voilà." est un
                # say_user valide après get_vote_token, pas la peine de forcer le jeton dedans).
                display_value = previous_result.get("display") if previous_result else None
                is_pseudo_result = isinstance(display_value, str) and bool(display_value)
                if fallback and not text.strip():
                    # Bug réel #5 (2026-07-25, root cause d'une répétition signalée par le
                    # développeur : "Clairière vert" reproposé 5 fois) : le LLM laisse parfois
                    # say_user vide juste après une action de proposition — sans texte, le filtre
                    # frontend anti-bulle-vide masque la bulle, effaçant toute trace du candidat
                    # proposé dans l'historique renvoyé au modèle au tour suivant. Sans mémoire de
                    # ce qu'il vient d'offrir, le modèle repart d'index=0 → répétition.
                    text = fallback
                elif is_pseudo_result and display_value not in text:
                    # Bug réel #6 (même jour, mesuré à ~2 tentatives sur 5 malgré une clarification
                    # de prompt) : le LLM ne laisse pas toujours le say_user TOTALEMENT vide — il
                    # écrit parfois du texte autour d'un "trou" ("Que penses-tu de **** ?"), sans
                    # jamais avoir inclus le token {{résultat}} littéral (donc rien à substituer).
                    # Un simple test "texte vide ?" ne détecte pas ce cas. Restreint aux actions
                    # pseudo (display présent) : c'est la seule famille où citer le résultat est
                    # une exigence explicite du prompt, pas une généralisation à toute action.
                    text = fallback
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

        # Reporté à l'itération suivante (voir "Bug réel #7" plus haut) : le say_user qui clôturera
        # potentiellement le tour prochain doit encore pouvoir substituer/vérifier CE résultat-ci.
        carried_result = previous_result

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
