"""Serveur MCP de test/debug pour l'exécuteur du chatbot jouyvote.fr (POC tool-calling).

PAS utilisé en production — le vrai /chat/v2 (main.py) appelle chatbot_executor.run_turn()
directement, sans passer par MCP (un appel Python suffit, pas besoin d'un process séparé). Ce
serveur existe uniquement pour permettre à un Claude (angelobot) de tester la boucle
d'exécution depuis l'extérieur (simuler une conversation) sans avoir à passer par une vraie
session HTTP/DB jouyvote à chaque essai.

Un seul tool volontairement générique ("exécute cette liste de messages") plutôt qu'un tool par
action — coller à l'esprit "1 tool" du mandat, la granularité par action n'apporte rien ici
puisque c'est justement la boucle interne (pas le choix d'action) qu'on veut pouvoir observer.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from chatbot_actions import ONBOARDING_NEW_USER_CONTEXT_BLOCK, compute_debate_token, compute_vote_token, derive_pseudo
from chatbot_executor import build_system_prompt, run_turn

mcp = FastMCP("jouyvote-chatbot-executor")

BASE_PROMPT = (
    "Tu es l'assistant citoyen de Jouy Vote Citoyen, un outil de démocratie participative locale "
    "pour les habitants de Jouy (28). Tu aides les joviens à formuler clairement une opinion ou "
    "une doléance, sans jamais trahir le sens de ce qu'ils veulent dire. Tu ne dois JAMAIS "
    "affirmer avoir enregistré, sauvegardé ou publié quoi que ce soit sans que ce soit réellement "
    "arrivé. Reste bref, concret, et dans le sujet de la vie municipale de Jouy."
)

# Identité de test fixe (pas de vraie base de données ici) : le POC MCP sert à observer le
# comportement de la boucle d'actions, pas à rejouer de vraies données citoyennes.
_TEST_IDENTITY_TOKEN = "test-identity-token-mcp-debug"
# Résumés de test fixes (pas de vraie DB ici) — juste de quoi observer list_summaries renvoyer
# quelque chose de non trivial plutôt qu'une liste toujours vide.
_TEST_SUMMARIES = [
    {"id": 1, "summary": "J'ai signalé un trottoir dangereux rue de la Mairie.", "created_at": "2026-07-20T10:00:00"},
]


@mcp.tool()
def run_chat_turn(user_message: str, history_json: str = "[]", has_pseudo: bool = False) -> dict:
    """Exécute un tour complet de la boucle d'actions du chatbot jouyvote (say_user/
    get_vote_token/propose_summary/list_summaries/propose_pseudo_candidates/
    get_or_assign_pseudo) avec une identité de test fixe, et retourne les répliques produites +
    le détail de chaque action exécutée.

    Args:
        user_message: le message envoyé par l'utilisateur de test.
        history_json: historique de conversation précédent, JSON d'une liste de
            {"role": "user"|"assistant", "content": "..."} (vide par défaut).
        has_pseudo: False (défaut) simule un nouvel utilisateur SANS pseudo confirmé — déclenche
            le bloc de contexte onboarding (laïus + propose_pseudo_candidates attendu). True
            simule un utilisateur qui a déjà confirmé un pseudo (pas de bloc onboarding,
            get_or_assign_pseudo renvoie une vraie valeur).
    """
    import json

    try:
        history = json.loads(history_json)
    except json.JSONDecodeError:
        history = []

    context_block = "" if has_pseudo else ONBOARDING_NEW_USER_CONTEXT_BLOCK
    system_prompt = build_system_prompt(BASE_PROMPT, context_block=context_block)
    conversation_messages = list(history) + [{"role": "user", "content": user_message}]
    test_debate_token = compute_debate_token(_TEST_IDENTITY_TOKEN)
    ctx = {
        "identity_token": _TEST_IDENTITY_TOKEN,
        "history": conversation_messages,
        "summaries": _TEST_SUMMARIES,
        "pseudo": derive_pseudo(test_debate_token) if has_pseudo else None,
    }
    result = run_turn(system_prompt, conversation_messages, ctx)
    result["_test_vote_token"] = compute_vote_token(_TEST_IDENTITY_TOKEN)
    return result


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY manquante dans l'environnement.")
    mcp.run()
