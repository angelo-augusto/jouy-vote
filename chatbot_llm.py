"""Appel LLM nu vers OpenRouter, sans dépendance au reste de jouy-vote.

Module volontairement autonome (pas d'import de main.py) : chatbot_actions.py et
chatbot_executor.py doivent rester testables/exécutables hors FastAPI (via
mcp_chatbot_executor.py), sans lever au démarrage faute de JOUY_ADMIN_KEY/PEPPER comme le
ferait un import de main.py.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.environ.get("CHAT_MODEL", "deepseek/deepseek-v4-flash")


def call_openrouter(
    messages: list[dict],
    response_format: dict | None = None,
    max_tokens: int = 4096,
    model: str | None = None,
) -> tuple[str | None, dict]:
    """Appelle l'API chat/completions d'OpenRouter. Ne lève jamais.

    Retourne (contenu, usage) — contenu est None en cas d'échec (clé absente, erreur réseau,
    réponse inattendue), à charge de l'appelant de gérer proprement."""
    if not OPENROUTER_API_KEY:
        return None, {}
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        return content, data.get("usage", {})
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return None, {}
