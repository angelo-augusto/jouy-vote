#!/bin/bash
# Veille mensuelle des PV de conseil municipal (RAG, 2026-07-26) — déclenchée par cron sur le
# Khadas (voir crontab utilisateur angelo), pas par CronCreate (session-only, expire sous 7
# jours, inadapté à une tâche permanente). Relance index.py (OCR automatique, filet de sécurité
# pour les nouveaux PV que personne ne transcrit manuellement) — idempotent, ne retraite jamais un
# document déjà indexé (voir _already_indexed dans index.py), donc un lancement mensuel suffit à
# détecter les nouveaux documents sans logique de diff séparée.
set -euo pipefail
cd /home/angelo/codage/jouy-vote
docker compose exec -T opencode /home/coder/.venv/bin/python3 /workspace/rag_conseil_municipal/index.py
