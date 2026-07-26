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

from chatbot_actions import (
    CHAT_SYSTEM_PROMPT, ONBOARDING_NEW_USER_CONTEXT_BLOCK, compute_debate_token, compute_vote_token,
    derive_pseudo,
)
from chatbot_executor import build_system_prompt, run_turn

mcp = FastMCP("jouyvote-chatbot-executor")

# Réutilise LE MÊME socle que la vraie prod (main.py) — plus de copie dupliquée à la main (2026-07-26,
# root cause d'un comportement observé différent de la prod pendant un test angelobot : l'ancienne
# copie n'avait jamais été mise à jour avec les règles anonymat/modération/iconifiable ajoutées la
# veille à CHAT_SYSTEM_PROMPT). Toujours importer, jamais recopier une constante de prompt.
BASE_PROMPT = CHAT_SYSTEM_PROMPT

# Identité de test fixe (pas de vraie base de données ici) : le POC MCP sert à observer le
# comportement de la boucle d'actions, pas à rejouer de vraies données citoyennes.
_TEST_IDENTITY_TOKEN = "test-identity-token-mcp-debug"
# Résumés de test fixes (pas de vraie DB ici) — juste de quoi observer list_summaries renvoyer
# quelque chose de non trivial plutôt qu'une liste toujours vide.
_TEST_SUMMARIES = [
    {"id": 1, "summary": "J'ai signalé un trottoir dangereux rue de la Mairie.", "created_at": "2026-07-20T10:00:00"},
]
# Fils de test fixes (2026-07-26, ajouté avec le fix de désynchronisation ci-dessus) — sans cette
# clé, list_threads/get_thread/propose_opinion voyaient toujours ctx["threads"]==[] et le modèle
# concluait à tort qu'aucun forum n'existait, révélant un comportement différent de la vraie prod.
_TEST_THREADS = [
    {
        "thread_id": 1,
        "title": "Faut-il plus de pistes cyclables à Jouy ?",
        "summary": None,
        "opinions": [
            {
                "opinion_id": 1, "auteur": "Renard bleu",
                "body": "Il faut plus de pistes cyclables, la route principale est dangereuse à vélo.",
                "argumentaire": None, "superseded_by_opinion_id": None,
            },
        ],
    },
]


def _mock_report_bug_fn(description: str) -> dict:
    """JAMAIS de vrai envoi email pendant un test MCP (2026-07-26) — évite de spammer
    ADMIN_BUG_EMAIL à chaque essai d'angelobot. Juste une trace locale."""
    print(f"[MCP TEST — jamais envoyé] report_bug: {description}")
    return {"sent": False, "note": "envoi désactivé en mode test MCP"}


def _mock_request_admin_intervention_fn(description: str) -> dict:
    """Même garde-fou que _mock_report_bug_fn."""
    print(f"[MCP TEST — jamais envoyé] request_admin_intervention: {description}")
    return {"sent": False, "note": "envoi désactivé en mode test MCP"}


@mcp.tool()
def run_chat_turn(user_message: str, history_json: str = "[]", has_pseudo: bool = False, taken_pseudos_json: str = "[]") -> dict:
    """Exécute un tour complet de la boucle d'actions du chatbot jouyvote (say_user/get_vote_token/
    propose_summary/list_summaries/get_or_assign_pseudo/propose_pseudo_candidates/
    propose_custom_pseudo/list_threads/get_thread/propose_opinion/propose_reaction/
    propose_remarque/report_bug/request_admin_intervention — TOUJOURS la vraie liste actuelle de
    chatbot_actions.ACTIONS, celle-ci est juste une note pour toi, pas une limite en dur) avec une
    identité de test fixe, et retourne les répliques produites + le détail de chaque action
    exécutée. report_bug/request_admin_intervention sont mockées ici (jamais de vrai envoi email
    pendant un test, voir _mock_report_bug_fn).

    Args:
        user_message: le message envoyé par l'utilisateur de test.
        history_json: historique de conversation précédent, JSON d'une liste de
            {"role": "user"|"assistant", "content": "..."} (vide par défaut).
        has_pseudo: False (défaut) simule un nouvel utilisateur SANS pseudo confirmé — déclenche
            le bloc de contexte onboarding (laïus + propose_pseudo_candidates attendu). True
            simule un utilisateur qui a déjà confirmé un pseudo (pas de bloc onboarding,
            get_or_assign_pseudo renvoie une vraie valeur).
        taken_pseudos_json: JSON d'une liste de [word, color] déjà "pris" par d'autres identités
            de test, pour observer le cas "déjà pris" via propose_pseudo_candidates/
            propose_custom_pseudo (vide par défaut).
    """
    import json

    try:
        history = json.loads(history_json)
    except json.JSONDecodeError:
        history = []
    try:
        taken_pseudos = {tuple(pair) for pair in json.loads(taken_pseudos_json)}
    except (json.JSONDecodeError, TypeError, ValueError):
        taken_pseudos = set()

    context_block = "" if has_pseudo else ONBOARDING_NEW_USER_CONTEXT_BLOCK
    system_prompt = build_system_prompt(BASE_PROMPT, context_block=context_block)
    conversation_messages = list(history) + [{"role": "user", "content": user_message}]
    test_debate_token = compute_debate_token(_TEST_IDENTITY_TOKEN)
    ctx = {
        "identity_token": _TEST_IDENTITY_TOKEN,
        "history": conversation_messages,
        "summaries": _TEST_SUMMARIES,
        "pseudo": derive_pseudo(test_debate_token) if has_pseudo else None,
        "taken_pseudos": taken_pseudos,
        "threads": _TEST_THREADS,
        "report_bug_fn": _mock_report_bug_fn,
        "request_admin_intervention_fn": _mock_request_admin_intervention_fn,
    }
    result = run_turn(system_prompt, conversation_messages, ctx)
    result["_test_vote_token"] = compute_vote_token(_TEST_IDENTITY_TOKEN)
    return result


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY manquante dans l'environnement.")
    mcp.run()
