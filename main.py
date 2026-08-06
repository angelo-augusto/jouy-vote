"""Prototype de vote citoyen local pour Jouy (28).

Sépare identité (nom/adresse/email) et vote (jeton/choix) pour garantir
l'anonymat du vote tout en gardant une vérification de résidence déclarative.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date, datetime
from zoneinfo import ZoneInfo
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot_actions import (
    ALL_CATEGORIES, CHAT_SYSTEM_PROMPT, FORUM_CATEGORIES, FORUM_REACTION_ACTIONS,
    ONBOARDING_ACTIONS, PSEUDO_COLORS, RESERVED_CATEGORIES,
    _agree_pseudo_display, build_onboarding_context_block, build_pseudo_rechoice_context_block,
    build_response_format, compute_debate_token, random_available_pseudo_candidates,
)
from chatbot_executor import build_system_prompt, run_turn
import pseudo_logo_gen

DB_PATH = os.environ.get("DB_PATH") or os.path.join(os.path.dirname(__file__), "vote.db")
ADMIN_KEY = os.environ.get("JOUY_ADMIN_KEY")
if not ADMIN_KEY:
    raise RuntimeError("La variable d'environnement JOUY_ADMIN_KEY doit être définie.")

JOUY_VOTE_PEPPER = os.environ.get("JOUY_VOTE_PEPPER")
if not JOUY_VOTE_PEPPER:
    raise RuntimeError("La variable d'environnement JOUY_VOTE_PEPPER doit être définie.")

# Pepper dédié au pseudo (distinct de JOUY_VOTE_PEPPER, voir chatbot_actions.compute_debate_token)
# — même exigence de présence que le pepper de vote : un pseudo dérivé d'un pepper vide/absent
# perdrait sa garantie de non-corrélation, mieux vaut un crash net au démarrage qu'une faille
# silencieuse.
JOUY_PSEUDO_PEPPER = os.environ.get("JOUY_PSEUDO_PEPPER")
if not JOUY_PSEUDO_PEPPER:
    raise RuntimeError("La variable d'environnement JOUY_PSEUDO_PEPPER doit être définie.")

# Envoi d'email (lien de réinitialisation de mot de passe) via l'API transactionnelle Brevo.
# BREVO_API_KEY absente = fonctionnalité désactivée proprement (pas de crash au démarrage,
# contrairement à ADMIN_KEY/PEPPER) : /forgot-password répond alors sans jamais rien envoyer
# ni révéler le token, cf. forgot_password() plus bas.
BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "noreply@jouyvote.fr")
SITE_URL = os.environ.get("SITE_URL", "https://jouyvote.fr")

# Destinataire des signalements de bug / demandes d'intervention admin (2026-07-25, demande
# développeur) — adresse confirmée par Angelo, déjà pontée email→Matrix côté angelobot. Même
# garde-fou qu'au-dessus : absente = fonctionnalité désactivée proprement, jamais un crash.
ADMIN_BUG_EMAIL = os.environ.get("ADMIN_BUG_EMAIL", "ab@angeloaugusto.fr")

# URL interne (réseau Docker, service "wiki" du même docker-compose.yml) pour aller chercher le
# contenu de la page d'accueil du wiki à afficher sur la page d'accueil de jouyvote.fr — plus
# rapide et plus fiable qu'un aller-retour par le tunnel Cloudflare public.
WIKI_INTERNAL_URL = os.environ.get("WIKI_INTERNAL_URL", "http://wiki:8080")
WIKI_PUBLIC_URL = os.environ.get("WIKI_PUBLIC_URL", "https://wiki.jouyvote.fr")

# Nombre maximum de filleuls par parrain (voir wiki.jouyvote.fr/themes:representation) : limite
# l'impact d'un parrain complaisant ou compromis qui ferait entrer un grand nombre de faux
# comptes d'un coup. Contrôlé à la fois à la création de l'invitation et à l'inscription.
REFERRAL_MAX = 5

# Coupe-circuit temporaire (faille Sybil : rien n'empêche aujourd'hui de créer un faux compte
# résident) — fermé par défaut tant que le parrainage n'est pas construit. Flag d'env plutôt
# qu'en dur pour pouvoir rouvrir sans redéployer de code le moment venu.
REGISTRATIONS_OPEN = os.environ.get("REGISTRATIONS_OPEN", "false").lower() == "true"

# Chatbot v1 (voir wiki.jouyvote.fr/themes:chatbot) — même fournisseur que le conteneur dev
# (OpenRouter/DeepSeek), pas de nouvelle dépendance HTTP (urllib, comme send_reset_email).
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "deepseek/deepseek-v4-flash")
# CHAT_SYSTEM_PROMPT déplacé dans chatbot_actions.py le 2026-07-26 (import partagé avec
# mcp_chatbot_executor.py, voir ce module pour la justification — une copie dupliquée avait dérivé).

_keepalive_conn: sqlite3.Connection | None = None

app = FastAPI(title="Jouy Vote Citoyen")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@contextmanager
def db():
    global _keepalive_conn
    if DB_PATH == ":memory:":
        uri = "file::memory:?cache=shared&uri=true"
        if _keepalive_conn is None:
            _keepalive_conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000)
    return f"{salt}:{h.hex()}"


def check_password(password: str, stored: str) -> bool:
    # Faille F (revue Opus 04/08) : comparaison à temps constant plutôt que == — un hex compté
    # caractère par caractère par == peut en théorie fuiter un timing exploitable, même si le
    # signal réel est marginal sur un hash SHA256 déjà uniformément distribué. Correction par
    # principe, pas suite à un incident constaté.
    salt, h = stored.split(':', 1)
    computed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000).hex()
    return hmac.compare_digest(computed, h)


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS identities (
                token TEXT PRIMARY KEY,
                identity_hash TEXT UNIQUE NOT NULL,
                nom TEXT NOT NULL,
                adresse TEXT NOT NULL,
                email TEXT,
                password_hash TEXT,
                session_token TEXT,
                reset_token TEXT,
                reset_token_expiry REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                titre TEXT NOT NULL,
                description TEXT,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        # "options" (2026-08-04, faille A signalée par la revue Opus du 04/08) : liste JSON des
        # réponses possibles pour cette question (ex: '["Oui", "Non"]") — avant ce champ, "choix"
        # était un TEXT libre côté /vote, sans aucune validation serveur ("Oui"/"oui"/"Oui "
        # devenaient 3 buckets distincts dans le tally). ALTER TABLE idempotent, même pattern que
        # les autres migrations de ce fichier (voir threads.category plus haut).
        try:
            conn.execute("ALTER TABLE questions ADD COLUMN options TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            """CREATE TABLE IF NOT EXISTS votes (
                vote_token TEXT NOT NULL,
                question_id INTEGER NOT NULL,
                choix TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (vote_token, question_id)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS referral_invites (
                invite_token TEXT PRIMARY KEY,
                referrer_token TEXT NOT NULL,
                invitee_email TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        # Minimisation par défaut (voir wiki.jouyvote.fr/themes:chatbot-fonctionnalites) : le
        # chatbot ne garde RIEN d'une conversation tant que l'utilisateur ne le demande pas
        # explicitement — cette table ne contient QUE des résumés validés par leur auteur, jamais
        # le verbatim d'un échange. owner_token = identities.token en clair (pas dérivé/peppé
        # comme vote_token) : contrairement au vote, un résumé n'est JAMAIS publié ni listé
        # publiquement, uniquement accessible à son auteur via son propre session_token — le
        # modèle de menace (désanonymisation par recoupement public) ne s'applique pas ici.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chat_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_token TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        # Indexée par debate_token (sha256(identity_token+JOUY_PSEUDO_PEPPER), voir
        # chatbot_actions.compute_debate_token), JAMAIS par identities.token en clair — au
        # contraire de chat_summaries.owner_token ci-dessus. Contrainte explicite du wiki
        # (architecture-technique) : "aucune table ne doit permettre de relier vote_token et
        # pseudo entre eux, ni l'un ou l'autre à l'identité déclarée" — un FK en clair vers
        # identities permettrait un JOIN trivial identité→pseudo pour quiconque a accès à la DB,
        # ce que ce pseudonyme est censé empêcher structurellement (contrairement aux résumés,
        # privés par design mais pas soumis à cette contrainte).
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pseudos (
                debate_token TEXT PRIMARY KEY,
                word TEXT NOT NULL,
                color TEXT NOT NULL,
                assigned_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        # Chaque pseudo (mot+couleur) unique tous utilisateurs confondus — ferme un trou latent
        # qui existait déjà pour les candidats déterministes (rien n'empêchait en théorie 2
        # personnes différentes de confirmer par coïncidence le même pseudo), et sert de filet de
        # sécurité contre une race entre 2 confirmations quasi simultanées (voir confirm_pseudo).
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_pseudos_word_color ON pseudos(word, color)"
        )
        # Forum (2026-07-25, spec wiki.jouyvote.fr/themes:chatbot-fonctionnalites, section
        # "Page Forum" — schéma revu par Opus puis corrigé par angelobot le même soir). Comme pour
        # pseudos ci-dessus : tout ce qui est attribué à un utilisateur est indexé par
        # debate_token peppé, jamais identities.token en clair — même contrainte
        # architecture-technique ("aucune table ne doit permettre de relier vote_token et pseudo
        # entre eux, ni l'un ou l'autre à l'identité déclarée").
        conn.execute(
            """CREATE TABLE IF NOT EXISTS threads (
                thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                summary TEXT,
                creator_debate_token TEXT,
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'published', 'archived')),
                embedding_ref TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                published_at TEXT
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status)")
        # Catégories fixes (2026-07-29) — ADD COLUMN idempotent, même pattern que la boucle
        # column_types plus bas : CREATE TABLE IF NOT EXISTS ne touche pas une table déjà
        # existante en prod, donc cette colonne doit être ajoutée séparément à chaque démarrage.
        # Pas de CHECK ici (ALTER TABLE ADD COLUMN ne supporte pas les contraintes en SQLite) —
        # la validité de la valeur est garantie en amont par propose_opinion/FORUM_CATEGORIES,
        # jamais par la DB elle-même.
        try:
            conn.execute("ALTER TABLE threads ADD COLUMN category TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_threads_category ON threads(category)")
        # Filet anti-race (2026-07-25, création de fil couplée au 1er post d'opinion, demande
        # développeur) : titre unique tous fils confondus — si 2 personnes proposent quasi
        # simultanément un fil au titre EXACTEMENT identique, la 2e INSERT échoue proprement
        # (voir create_thread_with_opinion) plutôt que de dupliquer silencieusement le sujet.
        # Ne résout QUE la collision de titre identique — la vraie déduplication sémantique
        # (titres différents mais même sujet) reste la phase 4 (RAG/Qdrant), pas ce filet.
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_threads_title ON threads(title)")
        # PAS de table opinion_versions séparée (1re version d'Opus, abandonnée le soir même) :
        # les réactions pointaient vers l'opinion en général, pas une version précise — modifier le
        # texte aurait fait porter silencieusement d'anciennes réactions sur un texte jamais vu.
        # body/argumentaire vivent directement sur la table, FIGÉS DÈS LA 1re RÉACTION (voir
        # update_opinion_draft ci-dessous, garde applicative — rien en SQL pur ne peut exprimer
        # cette contrainte). Changer d'avis après coup = nouvelle ligne opinion +
        # superseded_by_opinion_id sur l'ancienne, jamais de mutation de ce qui a déjà été réagi.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS opinions (
                opinion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL REFERENCES threads(thread_id),
                author_debate_token TEXT NOT NULL,
                body TEXT NOT NULL,
                argumentaire TEXT,
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'published', 'disavowed')),
                superseded_by_opinion_id INTEGER REFERENCES opinions(opinion_id),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                published_at TEXT
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinions_thread ON opinions(thread_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinions_author ON opinions(author_debate_token)")
        # PAS de UNIQUE(opinion_id, reactor_debate_token) — une même personne peut réagir plusieurs
        # fois dans le temps à la même opinion (changer d'avis) ; chaque réaction est une NOUVELLE
        # ligne, jamais un écrasement. Seule la plus récente (created_at max) compte dans le
        # décompte courant, les précédentes restent visibles comme historique.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS opinion_reactions (
                reaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                opinion_id INTEGER NOT NULL REFERENCES opinions(opinion_id),
                reactor_debate_token TEXT NOT NULL,
                stance TEXT NOT NULL CHECK (stance IN ('adherer', 'opposer', 'neutre')),
                argumentaire TEXT,
                status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'published')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_opinion_reactions_opinion ON opinion_reactions(opinion_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_opinion_reactions_reactor ON opinion_reactions(reactor_debate_token)"
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS thread_remarques (
                remarque_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL REFERENCES threads(thread_id),
                author_debate_token TEXT NOT NULL,
                body TEXT NOT NULL,
                reply_to_remarque_id INTEGER REFERENCES thread_remarques(remarque_id),
                reply_to_opinion_id INTEGER REFERENCES opinions(opinion_id),
                status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'published')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thread_remarques_thread ON thread_remarques(thread_id)"
        )
        # reply_to_reaction_id (2026-07-31) : jusqu'ici aucune façon de rebondir sur une réaction
        # (adhérer/opposer/neutre) elle-même — seulement sur une opinion ou une autre remarque.
        # Même logique que reply_to_remarque_id/reply_to_opinion_id : nullable, mutuellement
        # exclusif en pratique avec les deux autres (une remarque répond à au plus une chose).
        try:
            conn.execute(
                "ALTER TABLE thread_remarques ADD COLUMN reply_to_reaction_id "
                "INTEGER REFERENCES opinion_reactions(reaction_id)"
            )
        except sqlite3.OperationalError:
            pass
        # "voice" (2026-08-03, voix Admin/Mairie, étape 4/6) : NULL = pseudo citoyen normal
        # (comportement inchangé, l'immense majorité des lignes), 'admin'/'mairie' = publié avec
        # cette voix — author_debate_token reste TOUJOURS rempli même dans ce cas (traçabilité
        # interne, voir wiki themes:admin-mairie), seul le RENDU change (badge fixe au lieu du
        # pseudo mot+couleur). Même ALTER TABLE idempotent sur les 3 tables concernées.
        for _voice_table in ("opinions", "opinion_reactions", "thread_remarques"):
            try:
                conn.execute(f"ALTER TABLE {_voice_table} ADD COLUMN voice TEXT")
            except sqlite3.OperationalError:
                pass
        # pseudo_word_logos (2026-07-31) : un logo silhouette généré par MOT (pas par mot+couleur
        # — la couleur est appliquée dynamiquement côté client via CSS mask-image, voir
        # pseudo_logo_gen.py). "word" est la clé, un seul logo partagé par tous les pseudos qui
        # utilisent ce mot quelle que soit leur couleur. Validé par le premier utilisateur qui
        # confirme un pseudo avec ce mot (voir /pseudo/logo/confirm) — jamais généré à l'insu de
        # quiconque, jamais avant qu'un vrai pseudo ne le déclenche.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pseudo_word_logos (
                word TEXT PRIMARY KEY,
                file TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        # Seule exception au principe "chatbot = passage obligé" : l'administration s'adresse
        # directement à un citoyen, jamais via le chatbot comme intermédiaire. admin_identity est
        # une identité FIXE et publique par construction (jamais un pseudo, jamais peppée) —
        # recipient_debate_token reste peppé comme partout ailleurs, même pour ce canal privilégié.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS admin_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipient_debate_token TEXT NOT NULL,
                admin_identity TEXT NOT NULL DEFAULT 'administration',
                body TEXT NOT NULL,
                read_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_messages_recipient ON admin_messages(recipient_debate_token)"
        )
        # Signalement de bug / demande d'intervention admin (2026-07-25, demande développeur) —
        # 2 tables séparées bien que même mécanique (email direct, pas de confirmation utilisateur,
        # rate-limité) : sémantiquement différentes (bug logiciel général vs demande personnelle
        # sur son propre compte, ex. anonymat compromis). debate_token sert UNIQUEMENT au
        # rate-limiting côté serveur — jamais inclus dans le corps de l'email envoyé (voir
        # send_bug_report_email/send_admin_intervention_email).
        conn.execute(
            """CREATE TABLE IF NOT EXISTS bug_reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                debate_token TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bug_reports_debate_token ON bug_reports(debate_token)")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS admin_intervention_requests (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                debate_token TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_intervention_debate_token ON admin_intervention_requests(debate_token)"
        )
        # rate_limit_events (2026-08-04, faille E revue Opus) : brute-force/spam anti-abus sur
        # /login, /register, /vote, /forgot-password — endpoints AVANT authentification, donc pas
        # de debate_token/identité utilisable comme clé (contrairement à bug_reports/admin_
        # intervention_requests ci-dessus, déjà rate-limités par identité) — clé par IP à la place.
        conn.execute(
            """CREATE TABLE IF NOT EXISTS rate_limit_events (
                ip TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_limit_ip_endpoint ON rate_limit_events(ip, endpoint)")
    with db() as conn:
        # Type explicite par colonne (pas juste TEXT pour tout) : ADD COLUMN ne s'applique que
        # si la colonne n'existe pas encore, donc ceci ne corrige que les tables qui n'ont
        # jamais eu cette colonne — les tables déjà migrées avec le mauvais type sont traitées
        # séparément ci-dessous.
        column_types = {
            "password_hash": "TEXT",
            "session_token": "TEXT",
            "reset_token": "TEXT",
            "reset_token_expiry": "REAL",
            "referred_by_token": "TEXT",
            # Voix Admin/Mairie (2026-08-03, spec wiki themes:admin-mairie) : "role" par défaut
            # NULL traité comme "citoyen" partout dans le code (jamais de backfill en masse sur
            # une table qui grossit), pas une colonne booléenne séparée par rôle — un seul champ
            # texte, valeurs attendues "admin"/"mairie"/NULL. "telephone" nullable, prévu pour
            # n'être renseigné QUE sur les comptes Admin (annuaire Admin↔Admin), jamais collecté
            # pour un compte citoyen normal.
            "role": "TEXT",
            "telephone": "TEXT",
        }
        for col, col_type in column_types.items():
            try:
                conn.execute(f"ALTER TABLE identities ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass
    _fix_reset_token_expiry_type()
    _fix_opinion_reactions_status_check()


def _fix_opinion_reactions_status_check():
    """Ajoute 'retracted' aux valeurs autorisées pour opinion_reactions.status (2026-07-30,
    bouton direct "annuler ma réaction" — voir retract_reaction). ALTER TABLE ADD COLUMN ne
    permet pas de modifier une contrainte CHECK existante en SQLite — même technique de
    reconstruction que _fix_reset_token_expiry_type, idempotente (détecte via le SQL de création
    déjà stocké dans sqlite_master si la migration a déjà eu lieu)."""
    with db() as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='opinion_reactions'"
        ).fetchone()
        if row is None or "retracted" in row["sql"]:
            return
        conn.execute("ALTER TABLE opinion_reactions RENAME TO opinion_reactions_old_migration")
        conn.execute(
            """CREATE TABLE opinion_reactions (
                reaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                opinion_id INTEGER NOT NULL REFERENCES opinions(opinion_id),
                reactor_debate_token TEXT NOT NULL,
                stance TEXT NOT NULL CHECK (stance IN ('adherer', 'opposer', 'neutre')),
                argumentaire TEXT,
                status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'retracted')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """INSERT INTO opinion_reactions
               SELECT reaction_id, opinion_id, reactor_debate_token, stance, argumentaire, status, created_at
               FROM opinion_reactions_old_migration"""
        )
        conn.execute("DROP TABLE opinion_reactions_old_migration")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinion_reactions_opinion ON opinion_reactions(opinion_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_opinion_reactions_reactor ON opinion_reactions(reactor_debate_token)"
        )


def _fix_reset_token_expiry_type():
    """Corrige le typage de reset_token_expiry sur une base migrée avant ce fix (2026-07-23) :
    l'ancienne boucle d'ALTER TABLE ajoutait toutes les colonnes en TEXT, y compris celle-ci qui
    doit être REAL pour être comparée à time.time() dans reset_password(). Idempotent (ne fait
    rien si déjà REAL) et sûr à exécuter à chaque démarrage : reconstruit la table avec le bon
    type en conservant toutes les données existantes (CAST gère les valeurs NULL/vides).
    """
    with db() as conn:
        col_type = next(
            (row[2] for row in conn.execute("PRAGMA table_info(identities)") if row[1] == "reset_token_expiry"),
            None,
        )
        if col_type != "TEXT":
            return
        conn.execute("ALTER TABLE identities RENAME TO identities_old_migration")
        conn.execute(
            """CREATE TABLE identities (
                token TEXT PRIMARY KEY,
                identity_hash TEXT UNIQUE NOT NULL,
                nom TEXT NOT NULL,
                adresse TEXT NOT NULL,
                email TEXT,
                password_hash TEXT,
                session_token TEXT,
                reset_token TEXT,
                reset_token_expiry REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """INSERT INTO identities
               SELECT token, identity_hash, nom, adresse, email, password_hash, session_token,
                      reset_token, CAST(reset_token_expiry AS REAL), created_at
               FROM identities_old_migration"""
        )
        conn.execute("DROP TABLE identities_old_migration")


init_db()


class Registration(BaseModel):
    nom: str
    adresse: str
    email: str
    password: str
    invite_token: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


class UnsubscribeRequest(BaseModel):
    session_token: str
    password: str


class ChangePasswordRequest(BaseModel):
    session_token: str
    current_password: str
    new_password: str


class ReferralInviteRequest(BaseModel):
    session_token: str
    invitee_email: str
    confirms_residency_and_age: bool


class ReferralStatusRequest(BaseModel):
    session_token: str


class MyVoteTokenRequest(BaseModel):
    session_token: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_token: str
    message: str
    history: list[ChatMessage] = []
    # Mode debug/traçage (demande développeur 2026-07-25, via angelobot) : réservé aux bots
    # internes, jamais documenté côté citoyen — nécessite ADMIN_KEY, jamais activé par défaut.
    admin_key: str | None = None
    # Chat inline scopé (2026-07-29, bouton "Réagir" sur Forum/Mon activité) : mutuellement
    # exclusifs, voir _element_context_block. L'historique envoyé (req.history) reste TOUJOURS le
    # même que d'habitude (aucune fragmentation côté serveur — le filtrage par élément est une
    # pure question d'affichage frontend, voir static_files/index.html) ; ces 2 champs ne servent
    # QU'à injecter dans le prompt le contenu de l'élément visé, pour que l'utilisateur n'ait pas
    # à le réexpliquer.
    context_opinion_id: int | None = None
    context_remarque_id: int | None = None
    context_reaction_id: int | None = None


class ChatSummarizeRequest(BaseModel):
    session_token: str
    history: list[ChatMessage]


class ChatSaveSummaryRequest(BaseModel):
    session_token: str
    summary: str


class PseudoConfirmRequest(BaseModel):
    session_token: str
    word: str
    color: str


class PseudoLogoPreviewRequest(BaseModel):
    session_token: str
    word: str


class PseudoLogoConfirmRequest(BaseModel):
    session_token: str
    word: str
    image_base64: str


class PseudoLogosRequest(BaseModel):
    session_token: str


class PseudoMineRequest(BaseModel):
    session_token: str


class OpinionConfirmRequest(BaseModel):
    session_token: str
    # Soit thread_id (fil EXISTANT déjà publié), soit new_thread_title (création couplée d'un
    # nouveau fil, 2026-07-25, décision développeur) — jamais les deux, voir opinion_confirm.
    thread_id: int | None = None
    new_thread_title: str | None = None
    new_thread_summary: str | None = None
    new_thread_category: str | None = None
    body: str
    argumentaire: str | None = None
    # "voice" (2026-08-03, voix Admin/Mairie) : None = chemin normal (chatbot). "admin"/"mairie"
    # revérifié en base par _validate_voice, jamais fait confiance à ce champ seul.
    voice: str | None = None


class ReactionConfirmRequest(BaseModel):
    session_token: str
    opinion_id: int
    stance: str
    argumentaire: str | None = None
    voice: str | None = None


class ReactionToggleRequest(BaseModel):
    session_token: str
    opinion_id: int
    stance: str
    voice: str | None = None


class ReactionMineRequest(BaseModel):
    session_token: str
    opinion_id: int


class RemarqueConfirmRequest(BaseModel):
    session_token: str
    thread_id: int
    body: str
    reply_to_remarque_id: int | None = None
    reply_to_opinion_id: int | None = None
    reply_to_reaction_id: int | None = None
    voice: str | None = None


class ActivityMineRequest(BaseModel):
    session_token: str


class ChatSummariesRequest(BaseModel):
    session_token: str


class ChatDeleteSummaryRequest(BaseModel):
    session_token: str


class LogoutRequest(BaseModel):
    session_token: str


class AdminRolesListRequest(BaseModel):
    session_token: str


class AdminRolesSetRequest(BaseModel):
    session_token: str
    target_email: str
    role: str | None = None


class MairieDirectoryRequest(BaseModel):
    session_token: str


class Vote(BaseModel):
    token: str
    question_id: int
    choix: str


class NewQuestion(BaseModel):
    admin_key: str
    titre: str
    description: str = ""
    # Faille A (revue Opus 04/08) : jeu de réponses fermé, obligatoire — plus de "choix" texte
    # libre côté /vote. Validé dans create_question (>=2 options, non vides, distinctes).
    options: list[str]


class QuestionUpdate(BaseModel):
    admin_key: str
    active: bool


def _normalize_identity_field(s: str) -> str:
    """Faille D (revue Opus 04/08) : "Jean Dupont" / "Jean  Dupont" (double espace) / "Jean
    Dupont " (espace de fin) produisaient 3 hashes DIFFÉRENTS avec le simple .strip().lower()
    d'origine — contournement trivial de la contrainte UNIQUE sur identity_hash. Normalisation
    renforcée : décomposition Unicode NFKD + retrait des accents (homoglyphes/variantes
    diacritiques ne créent plus 2 identités), retrait de la ponctuation ("Jean-Dupont" ==
    "Jean Dupont"), espaces multiples réduits à un seul. Correctif PARTIEL et assumé comme tel
    (Opus) : ne règle pas le Sybil sur des noms réellement différents — seul le parrainage social
    (voir themes:representation) fait rempart contre ça, cette fonction ne fait que fermer les
    contournements purement typographiques."""
    s = unicodedata.normalize("NFKD", s.strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Ponctuation remplacée par un ESPACE, pas retirée (2026-08-04, trouvé en écrivant le test
    # dédié) : un retrait pur collait les mots ("Jean-Dupont" -> "jeandupont"), ne matchant plus
    # "Jean Dupont" -> "jean dupont" — la ponctuation doit se comporter comme un séparateur de
    # mots, pas disparaître silencieusement.
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_identity_hash(nom: str, adresse: str) -> str:
    raw = f"{_normalize_identity_field(nom)}|{_normalize_identity_field(adresse)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_vote_token(token: str) -> str:
    raw = f"{token}:{JOUY_VOTE_PEPPER}"
    return hashlib.sha256(raw.encode()).hexdigest()


def get_existing_pseudo(identity_token: str) -> dict | None:
    """Lecture pure, aucun write — None si l'utilisateur n'a pas encore confirmé de pseudo (voir
    confirm_pseudo pour le seul point d'écriture). Remplace l'ancien ensure_pseudo() qui assignait
    automatiquement et silencieusement : depuis le passage à l'option (c) — choix collaboratif
    (mandat angelobot 2026-07-25) — un pseudo n'existe QUE si l'utilisateur en a explicitement
    confirmé un parmi les propositions."""
    debate_token = compute_debate_token(identity_token)
    with db() as conn:
        row = conn.execute(
            "SELECT word, color FROM pseudos WHERE debate_token=?", (debate_token,)
        ).fetchone()
    return {"word": row["word"], "color": row["color"]} if row else None


def confirm_pseudo(identity_token: str, word: str, color: str) -> dict:
    """SEUL point d'écriture de la table pseudos, déclenché uniquement par un clic utilisateur
    explicite (endpoint /pseudo/confirm) — jamais par le LLM.

    Simplifié (2026-07-25, tâtonnement conversationnel plutôt qu'une liste figée de candidats) :
    2 SEULES règles dures, identiques que le pseudo vienne d'une suggestion générée
    (propose_pseudo_candidates) ou d'une proposition libre de l'utilisateur
    (propose_custom_pseudo) — pas de vérification "appartient à la séquence déterministe", le mot
    lui-même n'est technique-ment pas restreint : (1) couleur dans PSEUDO_COLORS, (2) pas déjà
    pris par quelqu'un d'autre. L'UPSERT est protégé par la contrainte UNIQUE(word, color) en DB
    (idx_pseudos_word_color, voir init_db) — filet de sécurité contre une race entre 2
    confirmations quasi simultanées du même pseudo, au-delà de la vérification applicative
    ci-dessous.

    Rechoix libre (2026-07-25, demande développeur via angelobot) : reconfirmer REMPLACE le
    pseudo existant au lieu d'être bloqué — plus de vérification "un pseudo existe déjà pour
    cette identité". TODO IMPORTANT : la condition prévue par le développeur était "rechoix libre
    SI RIEN N'A ÉTÉ PUBLIÉ" — au moment de ce changement, aucune table opinions/témoignages/
    argumentaires n'existe encore (seuls votes, chat_summaries, pseudos existent, et votes est
    délibérément non-reliable au pseudo pour préserver l'anonymat, voir wiki
    architecture-technique) donc "rien n'est publiable" est vrai pour tout le monde aujourd'hui —
    rechoix inconditionnel décidé en connaissance de cause (confirmé par angelobot avant
    implémentation). LE JOUR OÙ une table de publication liée au pseudo existe, ce rechoix doit
    être regaté sur "aucune publication associée à ce debate_token", sans quoi changer de pseudo
    après publication casserait la cohérence des publications déjà attribuées."""
    word = word.strip()
    color = color.strip().lower()
    if not word:
        raise ValueError("mot manquant")
    if color not in PSEUDO_COLORS:
        raise ValueError("couleur non valide")
    debate_token = compute_debate_token(identity_token)
    with db() as conn:
        try:
            conn.execute(
                "INSERT INTO pseudos (debate_token, word, color) VALUES (?, ?, ?) "
                "ON CONFLICT(debate_token) DO UPDATE SET "
                "word=excluded.word, color=excluded.color, assigned_at=CURRENT_TIMESTAMP",
                (debate_token, word, color),
            )
        except sqlite3.IntegrityError:
            raise ValueError("ce mot+couleur est déjà pris par quelqu'un d'autre")
    return {"word": word, "color": color}


# Volontairement DANS le volume /data (même montage persistant que vote.db), PAS sous
# static_files/ (2026-07-31, bug réel trouvé au déploiement) : (1) static_files/ est COPIÉ dans
# l'image au build, appartient à root, illisible en écriture par l'utilisateur "app" du
# conteneur — un logo généré au runtime plantait sur PermissionError ; (2) même corrigé côté
# permissions, un fichier écrit dans la couche writable de l'image serait PERDU au prochain
# rebuild (le piège déjà documenté pour vote.db). Servi séparément via /pseudo_logos_data (voir
# app.mount plus bas), pas /static_files — chat_gris.png (logo unique généré manuellement avant
# ce pipeline) reste sous static_files/pseudo_logos/ tel quel, fichier commité au dépôt, sans
# rapport avec ce dossier runtime.
_PSEUDO_LOGO_DIR = os.path.join(os.path.dirname(DB_PATH), "pseudo_logos")
os.makedirs(_PSEUDO_LOGO_DIR, exist_ok=True)


def _slugify_word(word: str) -> str:
    """Nom de fichier sûr pour un mot de pseudo — mots accentués (ex: "Écureuil") possibles,
    jamais réutilisés tels quels comme nom de fichier. Simple, pas de lib externe : translittère
    les caractères ASCII usuels, retire le reste."""
    import unicodedata
    normalized = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "", normalized).lower() or "mot"


def get_word_logo_file(word: str) -> str | None:
    """Lecture pure — None si aucun logo n'a encore été validé pour ce mot (voir
    pseudo_logo_gen.generate_word_silhouette pour la génération, save_word_logo pour l'écriture)."""
    with db() as conn:
        row = conn.execute("SELECT file FROM pseudo_word_logos WHERE word=?", (word,)).fetchone()
    return row["file"] if row else None


def save_word_logo(word: str, png_bytes: bytes) -> str:
    """SEUL point d'écriture de pseudo_word_logos, déclenché par /pseudo/logo/confirm (clic
    utilisateur explicite, même moment que la confirmation du pseudo lui-même — jamais le LLM).

    Idempotent/race-safe (2026-07-31) : si 2 personnes confirment quasi simultanément un pseudo
    avec le MÊME mot nouveau (2 générations indépendantes en parallèle, chacune valide à ses yeux),
    la table pseudo_word_logos a "word" en clé primaire — la 2e écriture échoue proprement
    (IntegrityError, silencieuse), et on relit le fichier déjà écrit par la 1re plutôt que
    d'écraser ou de planter. Le fichier sur disque de la 2e génération reste orphelin mais inoffensif
    (juste un peu d'espace disque, jamais référencé)."""
    filename = f"{_slugify_word(word)}.png"
    path = os.path.join(_PSEUDO_LOGO_DIR, filename)
    os.makedirs(_PSEUDO_LOGO_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)
    with db() as conn:
        try:
            conn.execute(
                "INSERT INTO pseudo_word_logos (word, file) VALUES (?, ?)", (word, filename)
            )
        except sqlite3.IntegrityError:
            row = conn.execute(
                "SELECT file FROM pseudo_word_logos WHERE word=?", (word,)
            ).fetchone()
            return row["file"]
    return filename


def get_all_word_logos() -> dict[str, str]:
    """Lecture pure — {word: filename} pour tous les logos déjà validés, utilisé par le frontend
    pour peupler sa table pseudo->logo au chargement (voir GET /pseudo/logos)."""
    with db() as conn:
        rows = conn.execute("SELECT word, file FROM pseudo_word_logos").fetchall()
    return {r["word"]: r["file"] for r in rows}


# ===== Forum (2026-07-25, phase 1 : schéma + fonctions d'écriture + tests, ZÉRO branchement =====
# ===== chatbot — voir wiki.jouyvote.fr/themes:chatbot-fonctionnalites, section "Page Forum" =====
#
# Même principe brouillon→confirmation que pseudos/résumés : chaque table publiable porte un
# statut "draft"/"published" (et "disavowed" pour les opinions). Ces fonctions sont les SEULS
# points d'écriture — futur point de vigilance pour la phase chatbot (à venir) : comme pour
# save_summary/confirm_pseudo, aucune action LLM-callable ne doit jamais pouvoir faire passer un
# statut à "published" directement ; seul un endpoint déclenché par un clic utilisateur explicite
# doit appeler *_publish ci-dessous.


def create_thread(
    title: str, summary: str | None = None, creator_identity_token: str | None = None,
    category: str | None = None,
) -> dict:
    """Un fil peut être créé par le chatbot seul (regroupement thématique automatique, voir wiki)
    — creator_debate_token reste NULL dans ce cas plutôt que de forcer une attribution factice.

    category (2026-07-29) : ignorée silencieusement si absente de ALL_CATEGORIES plutôt que de
    faire échouer la création du fil — la catégorisation reste secondaire au contenu lui-même
    (même philosophie de tolérance que le reste des validations LLM-facing de ce module).
    ALL_CATEGORIES (2026-08-03) inclut aussi "admin"/"mairie" (voir chatbot_actions) — cette
    fonction générique reste utilisable par la future écriture directe Admin/Mairie, la
    restriction "un citoyen ne peut pas créer de fil dans ces 2 catégories" vit dans
    propose_opinion (validation contre FORUM_CATEGORIES seul), pas ici."""
    creator_debate_token = compute_debate_token(creator_identity_token) if creator_identity_token else None
    if category not in ALL_CATEGORIES:
        category = None
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO threads (title, summary, creator_debate_token, category) VALUES (?, ?, ?, ?)",
            (title, summary, creator_debate_token, category),
        )
        thread_id = cur.lastrowid
    return {"thread_id": thread_id, "title": title, "summary": summary, "status": "draft", "category": category}


def publish_thread(thread_id: int) -> dict:
    with db() as conn:
        conn.execute(
            "UPDATE threads SET status='published', published_at=CURRENT_TIMESTAMP WHERE thread_id=?",
            (thread_id,),
        )
    return {"thread_id": thread_id, "status": "published"}


def delete_thread_if_empty(thread_id: int) -> dict:
    """Filet de sécurité (2026-07-25, décision développeur, "retire la pression de bien choisir/
    regrouper parfaitement du premier coup") : supprime un fil qui n'a AUCUNE opinion ni remarque
    dedans, quel que soit leur statut (même un brouillon compte comme "pas vide" — mieux vaut un
    faux négatif ici qu'un vrai risque de supprimer un fil qui contient quelque chose). Utilisé en
    interne par create_thread_with_opinion si la publication de l'opinion échoue juste après la
    création du fil (échec partiel), mais reste appelable indépendamment (ex: nettoyage manuel)."""
    with db() as conn:
        thread_row = conn.execute("SELECT thread_id FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
        if thread_row is None:
            raise ValueError("fil introuvable")
        if conn.execute("SELECT 1 FROM opinions WHERE thread_id=? LIMIT 1", (thread_id,)).fetchone():
            raise ValueError("ce fil contient au moins une opinion, il ne peut pas être supprimé")
        if conn.execute("SELECT 1 FROM thread_remarques WHERE thread_id=? LIMIT 1", (thread_id,)).fetchone():
            raise ValueError("ce fil contient au moins une remarque, il ne peut pas être supprimé")
        conn.execute("DELETE FROM threads WHERE thread_id=?", (thread_id,))
    return {"thread_id": thread_id, "deleted": True}


def create_thread_with_opinion(
    title: str, author_identity_token: str, body: str,
    argumentaire: str | None = None, summary: str | None = None, category: str | None = None,
    voice: str | None = None,
) -> dict:
    """Création de fil COUPLÉE au 1er post d'opinion (2026-07-25, décision développeur, en
    réponse directe à la question posée plus tôt sur le wiki) : pas de création spéculative d'un
    fil vide — le fil et sa 1re opinion naissent dans le MÊME geste. 2 garde-fous :

    1. Filet anti-race : UNIQUE(title) en DB (voir init_db). Si 2 personnes proposent quasi
       simultanément un fil au titre EXACTEMENT identique, la 2e ne subit pas une erreur brute —
       son opinion est automatiquement rattachée au fil qui vient de gagner la course, comme si
       elle avait choisi ce fil existant depuis le début (elle n'y est pour rien dans la
       collision technique, ce serait injuste de la lui faire subir). Ne couvre QUE la collision
       de titre identique — la vraie déduplication sémantique reste la phase 4 (RAG).
    2. Auto-nettoyage : si la création de l'opinion échoue APRÈS que le fil a été créé (body vide,
       validation qui échoue...), le fil tout juste créé — encore vide à ce stade — est supprimé
       avant de relayer l'erreur, plutôt que de laisser une trace orpheline en base."""
    title = title.strip()
    if not title:
        raise ValueError("titre de fil manquant")
    if category not in ALL_CATEGORIES:
        category = None
    with db() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO threads (title, summary, status, published_at, category) "
                "VALUES (?, ?, 'published', CURRENT_TIMESTAMP, ?)",
                (title, summary, category),
            )
            thread_id = cur.lastrowid
        except sqlite3.IntegrityError:
            row = conn.execute("SELECT thread_id FROM threads WHERE title=?", (title,)).fetchone()
            if row is None:
                raise
            thread_id = row["thread_id"]

    try:
        opinion = create_opinion(thread_id, author_identity_token, body, argumentaire, voice)
        published = publish_opinion(opinion["opinion_id"])
    except ValueError:
        try:
            delete_thread_if_empty(thread_id)
        except ValueError:
            pass  # le fil n'était finalement pas vide (ex: quelqu'un d'autre y a posté entre-temps via la course perdue) — rien à nettoyer
        raise

    return {"thread_id": thread_id, "opinion_id": opinion["opinion_id"], "status": published["status"]}


def _opinion_has_reactions(conn: sqlite3.Connection, opinion_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM opinion_reactions WHERE opinion_id=? LIMIT 1", (opinion_id,)
    ).fetchone()
    return row is not None


def create_opinion(
    thread_id: int, author_identity_token: str, body: str, argumentaire: str | None = None,
    voice: str | None = None,
) -> dict:
    """Brouillon initial — body/argumentaire librement modifiables tant qu'AUCUNE réaction n'a
    encore été publiée dessus (voir update_opinion_draft). Le fil doit exister et être PUBLIÉ —
    on n'attache jamais une opinion à un fil encore en brouillon ou inexistant (durcissement
    2026-07-25, phase 3 : ce garde-fou manquait en phase 1, jamais exploitable tant que seul
    khadasbot appelait ces fonctions directement, mais devient un vrai risque dès que thread_id
    vient d'un paramètre LLM/utilisateur, voir propose_opinion/chatbot_actions.py). "body" non vide
    validé ICI aussi (pas seulement côté propose_opinion, lecture/validation LLM) — même principe
    de défense en profondeur que le reste du module : ne jamais faire reposer une contrainte sur
    une seule couche.

    "voice" (2026-08-03, voix Admin/Mairie, étape 4/6) : None = pseudo citoyen normal. Sinon,
    _validate_voice vérifie que le rôle réel de l'identité correspond, ET
    _check_category_posting_permission vérifie que la catégorie DU FIL (relue ici, jamais
    supposée) autorise cette voix — un citoyen ne peut donc pas poster dans un fil déjà catégorisé
    "admin"/"mairie" même s'il connaît son thread_id (ex: fil trouvé via list_threads)."""
    if not body.strip():
        raise ValueError("le corps de l'opinion est vide")
    _validate_voice(author_identity_token, voice)
    with db() as conn:
        thread_row = conn.execute("SELECT status, category FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
        if thread_row is None:
            raise ValueError("fil introuvable")
        if thread_row["status"] != "published":
            raise ValueError("ce fil n'est pas encore publié")
    _check_category_posting_permission(thread_row["category"], voice)
    author_debate_token = compute_debate_token(author_identity_token)
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO opinions (thread_id, author_debate_token, body, argumentaire, voice) VALUES (?, ?, ?, ?, ?)",
            (thread_id, author_debate_token, body, argumentaire, voice),
        )
        opinion_id = cur.lastrowid
    return {
        "opinion_id": opinion_id, "thread_id": thread_id, "body": body, "argumentaire": argumentaire,
        "voice": voice, "status": "draft",
    }


def update_opinion_draft(opinion_id: int, body: str | None = None, argumentaire: str | None = None) -> dict:
    """Règle d'intégrité centrale de ce schéma (2026-07-25, root cause de l'abandon de la 1re
    version d'Opus avec table opinion_versions séparée) : dès qu'une réaction existe sur cette
    opinion, body/argumentaire sont GELÉS — les modifier silencieusement ferait porter d'anciennes
    réactions sur un texte qu'elles n'ont jamais vu. Pour changer d'avis après une réaction, créer
    une NOUVELLE opinion et appeler supersede_opinion sur l'ancienne, jamais muter celle-ci."""
    with db() as conn:
        if _opinion_has_reactions(conn, opinion_id):
            raise ValueError("cette opinion a déjà des réactions, elle ne peut plus être modifiée — crée une nouvelle opinion à la place")
        row = conn.execute("SELECT status FROM opinions WHERE opinion_id=?", (opinion_id,)).fetchone()
        if row is None:
            raise ValueError("opinion introuvable")
        if row["status"] != "draft":
            raise ValueError("cette opinion n'est plus un brouillon")
        updates, params = [], []
        if body is not None:
            updates.append("body=?")
            params.append(body)
        if argumentaire is not None:
            updates.append("argumentaire=?")
            params.append(argumentaire)
        if updates:
            params.append(opinion_id)
            conn.execute(f"UPDATE opinions SET {', '.join(updates)} WHERE opinion_id=?", params)
    return {"opinion_id": opinion_id}


def publish_opinion(opinion_id: int) -> dict:
    with db() as conn:
        conn.execute(
            "UPDATE opinions SET status='published', published_at=CURRENT_TIMESTAMP "
            "WHERE opinion_id=? AND status='draft'",
            (opinion_id,),
        )
    return {"opinion_id": opinion_id, "status": "published"}


def disavow_opinion(opinion_id: int, author_identity_token: str) -> dict:
    """"Désavouer" (statut à part, distinct de superseded_by_opinion_id) : l'auteur retire son
    soutien à une opinion SANS la remplacer par une nouvelle formulation — cas explicitement
    évoqué par le développeur ("tant que l'auteur ne désavoue pas son opinion")."""
    author_debate_token = compute_debate_token(author_identity_token)
    with db() as conn:
        row = conn.execute(
            "SELECT author_debate_token FROM opinions WHERE opinion_id=?", (opinion_id,)
        ).fetchone()
        if row is None:
            raise ValueError("opinion introuvable")
        if row["author_debate_token"] != author_debate_token:
            raise ValueError("seul l'auteur peut désavouer sa propre opinion")
        conn.execute("UPDATE opinions SET status='disavowed' WHERE opinion_id=?", (opinion_id,))
    return {"opinion_id": opinion_id, "status": "disavowed"}


def supersede_opinion(old_opinion_id: int, new_opinion_id: int, author_identity_token: str) -> dict:
    """Changement d'avis APRÈS que l'ancienne opinion a déjà des réactions (sinon, autant utiliser
    update_opinion_draft) : la nouvelle opinion est une ligne à part entière (même thread, même
    auteur), l'ancienne pointe vers elle via superseded_by_opinion_id. Les réactions déjà données
    sur l'ancienne restent des faits historiques vrais — pas de désaveu automatique, pas de
    suppression, l'affichage doit juste les recontextualiser comme "réactions à une version
    antérieure" (logique d'affichage, hors scope de cette fonction DB pure)."""
    author_debate_token = compute_debate_token(author_identity_token)
    with db() as conn:
        old_row = conn.execute(
            "SELECT author_debate_token, thread_id FROM opinions WHERE opinion_id=?", (old_opinion_id,)
        ).fetchone()
        new_row = conn.execute(
            "SELECT author_debate_token, thread_id FROM opinions WHERE opinion_id=?", (new_opinion_id,)
        ).fetchone()
        if old_row is None or new_row is None:
            raise ValueError("opinion introuvable")
        if old_row["author_debate_token"] != author_debate_token or new_row["author_debate_token"] != author_debate_token:
            raise ValueError("seul l'auteur peut remplacer sa propre opinion")
        if old_row["thread_id"] != new_row["thread_id"]:
            raise ValueError("la nouvelle opinion doit appartenir au même fil de discussion")
        conn.execute(
            "UPDATE opinions SET superseded_by_opinion_id=? WHERE opinion_id=?",
            (new_opinion_id, old_opinion_id),
        )
    return {"opinion_id": old_opinion_id, "superseded_by_opinion_id": new_opinion_id}


def add_reaction(
    opinion_id: int, reactor_identity_token: str, stance: str, argumentaire: str | None = None,
    voice: str | None = None,
) -> dict:
    """TOUJOURS un INSERT, jamais un UPDATE (pas de contrainte UNIQUE sur (opinion_id,
    reactor_debate_token), voir init_db) — changer d'avis (adhérer→opposer par ex.) crée une
    NOUVELLE ligne, l'ancienne réaction reste visible comme historique, jamais écrasée ni
    supprimée. L'opinion doit exister et être PUBLIÉE (durcissement 2026-07-25, phase 3 — même
    raison que create_opinion : opinion_id devient exploitable dès que la valeur vient d'un
    paramètre LLM/utilisateur).

    "voice" (2026-08-03, voix Admin/Mairie) : PAS de restriction de catégorie ici — "réagir" reste
    libre partout pour tous les rôles (voir _check_category_posting_permission, jamais appelée
    dans cette fonction). Seul _validate_voice s'applique : le rôle réel doit correspondre si une
    voix est demandée."""
    if stance not in ("adherer", "opposer", "neutre"):
        raise ValueError("stance invalide")
    _validate_voice(reactor_identity_token, voice)
    with db() as conn:
        opinion_row = conn.execute(
            "SELECT status, author_debate_token FROM opinions WHERE opinion_id=?", (opinion_id,)
        ).fetchone()
        if opinion_row is None:
            raise ValueError("opinion introuvable")
        if opinion_row["status"] != "published":
            raise ValueError("cette opinion n'est pas publiée")
    reactor_debate_token = compute_debate_token(reactor_identity_token)
    # Auto-réaction (2026-07-30, décision développeur, affinée le même soir) : adhérer/neutre sur
    # sa propre opinion restent normaux (pas de sens à bloquer — l'auteur qui confirme sa position
    # ou reste neutre sur son propre texte n'est pas absurde). Seul "opposer" sur soi-même est
    # bloqué ICI (seul point d'écriture réel des réactions, voir docstring — valable pour tous les
    # chemins : /reaction/confirm ET le bouton direct set_reaction_stance) : s'opposer à sa propre
    # opinion signale en réalité un changement d'avis, pas une vraie réaction — le message invite
    # à reformuler via le chat plutôt que d'enregistrer une opposition à soi-même. Le déclenchement
    # automatique du mécanisme de remplacement (supersede_opinion) n'est PAS câblé côté LLM à ce
    # jour (vérifié : aucune action propose_supersede, aucun endpoint) — hors scope de ce fix,
    # backlog distinct plutôt qu'improvisé ce soir.
    if stance == "opposer" and reactor_debate_token == opinion_row["author_debate_token"]:
        raise ValueError(
            "tu ne peux pas t'opposer à ta propre opinion — si tu as changé d'avis, "
            "reformule-la plutôt via l'assistant"
        )
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO opinion_reactions (opinion_id, reactor_debate_token, stance, argumentaire, voice) "
            "VALUES (?, ?, ?, ?, ?)",
            (opinion_id, reactor_debate_token, stance, argumentaire, voice),
        )
        reaction_id = cur.lastrowid
    return {"reaction_id": reaction_id, "opinion_id": opinion_id, "stance": stance, "voice": voice, "status": "draft"}


def publish_reaction(reaction_id: int) -> dict:
    with db() as conn:
        conn.execute("UPDATE opinion_reactions SET status='published' WHERE reaction_id=? AND status='draft'", (reaction_id,))
    return {"reaction_id": reaction_id, "status": "published"}


def get_current_reaction(opinion_id: int, reactor_identity_token: str) -> dict | None:
    """La réaction la plus RÉCENTE (created_at max) compte seule dans le décompte courant — les
    précédentes restent en base comme historique mais ne sont jamais retournées ici."""
    reactor_debate_token = compute_debate_token(reactor_identity_token)
    with db() as conn:
        row = conn.execute(
            "SELECT reaction_id, stance, argumentaire, status, created_at FROM opinion_reactions "
            "WHERE opinion_id=? AND reactor_debate_token=? AND status='published' "
            "ORDER BY created_at DESC, reaction_id DESC LIMIT 1",
            (opinion_id, reactor_debate_token),
        ).fetchone()
    return dict(row) if row else None


def retract_reaction(opinion_id: int, reactor_identity_token: str) -> dict:
    """Annule la réaction courante de l'utilisateur sur cette opinion (2026-07-30, bouton direct
    "reclic sur le même stance = annule"). Jamais de suppression (voir add_reaction : chaque
    réaction reste un fait historique) — juste un changement de statut, même principe que
    disavow_opinion pour les opinions. Marque TOUTES les réactions encore 'published' de ce
    réacteur sur cette opinion (pas seulement la plus récente) : si on ne marquait que la
    dernière, une réaction PLUS ANCIENNE encore au statut 'published' réapparaîtrait comme
    "courante" dans get_current_reaction/_get_opinion_reaction_summary (qui ne filtrent que sur
    status='published', sans notion de récence au niveau SQL) — en pratique il ne devrait y avoir
    qu'une réaction 'published' à la fois par réacteur, mais ce garde-fou coûte rien. Idempotent :
    ne fait rien si aucune réaction active."""
    reactor_debate_token = compute_debate_token(reactor_identity_token)
    with db() as conn:
        conn.execute(
            "UPDATE opinion_reactions SET status='retracted' "
            "WHERE opinion_id=? AND reactor_debate_token=? AND status='published'",
            (opinion_id, reactor_debate_token),
        )
    return {"opinion_id": opinion_id, "stance": None}


def set_reaction_stance(
    opinion_id: int, reactor_identity_token: str, stance: str, voice: str | None = None,
) -> dict:
    """Bouton direct "Réagir" (2026-07-30, décision développeur) : écrit IMMÉDIATEMENT, sans
    passer par le chatbot ni par la double confirmation habituelle (opinion/remarque) — justifié
    par le choix fermé parmi 3 options, sans texte libre, donc sans besoin de modération ni de
    risque de publication accidentelle d'un contenu non voulu. Élimine aussi structurellement le
    bug d'ID trouvé le 2026-07-30 sur le chat inline : le bouton connaît déjà opinion_id, le LLM
    n'a plus besoin de le deviner pour CE choix précis.

    Toggle : reclic sur le MÊME stance que la réaction courante retire la réaction (voir
    retract_reaction) ; un stance différent (ou l'absence de réaction courante) en crée une
    nouvelle, publiée directement (même mécanisme "changer d'avis" que add_reaction — l'ancienne
    réaction reste un fait historique, jamais écrasée). Aucun argumentaire ici : le texte libre
    accompagnant la réaction continue de passer par propose_reaction (chat), voir
    _element_context_block qui rappelle au modèle le stance déjà choisi par bouton."""
    if stance not in ("adherer", "opposer", "neutre"):
        raise ValueError("stance invalide")
    current = get_current_reaction(opinion_id, reactor_identity_token)
    if current is not None and current["stance"] == stance:
        return retract_reaction(opinion_id, reactor_identity_token)
    reaction = add_reaction(opinion_id, reactor_identity_token, stance, voice=voice)
    publish_reaction(reaction["reaction_id"])
    return {"opinion_id": opinion_id, "stance": stance, "voice": voice}


def create_remarque(
    thread_id: int, author_identity_token: str, body: str,
    reply_to_remarque_id: int | None = None, reply_to_opinion_id: int | None = None,
    reply_to_reaction_id: int | None = None, voice: str | None = None,
) -> dict:
    """Couche informelle en plus du formalisme opinion/réaction — pas de statut "figé", pas de
    versions : une remarque publiée reste telle quelle (pas de règle de mutation particulière au-
    delà du cycle générique brouillon→publication). Le fil doit exister et être PUBLIÉ (même
    durcissement 2026-07-25, phase 3, que create_opinion/add_reaction). reply_to_reaction_id
    (2026-07-31) permet de rebondir sur une réaction (adhérer/opposer/neutre) elle-même, seul
    moyen actuel puisque les réactions n'ont pas de sous-réactions formelles.

    "voice" (2026-08-03, voix Admin/Mairie) : comme add_reaction, aucune restriction de catégorie
    — une remarque est une forme de "réagir", libre partout pour tous les rôles."""
    targets = [x for x in (reply_to_remarque_id, reply_to_opinion_id, reply_to_reaction_id) if x is not None]
    if len(targets) > 1:
        raise ValueError("une remarque répond à au plus une chose : une remarque, une opinion, ou une réaction, jamais plusieurs")
    _validate_voice(author_identity_token, voice)
    with db() as conn:
        thread_row = conn.execute("SELECT status FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
        if thread_row is None:
            raise ValueError("fil introuvable")
        if thread_row["status"] != "published":
            raise ValueError("ce fil n'est pas encore publié")
    author_debate_token = compute_debate_token(author_identity_token)
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO thread_remarques "
            "(thread_id, author_debate_token, body, reply_to_remarque_id, reply_to_opinion_id, reply_to_reaction_id, voice) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, author_debate_token, body, reply_to_remarque_id, reply_to_opinion_id, reply_to_reaction_id, voice),
        )
        remarque_id = cur.lastrowid
    return {"remarque_id": remarque_id, "thread_id": thread_id, "body": body, "voice": voice, "status": "draft"}


def publish_remarque(remarque_id: int) -> dict:
    with db() as conn:
        conn.execute("UPDATE thread_remarques SET status='published' WHERE remarque_id=? AND status='draft'", (remarque_id,))
    return {"remarque_id": remarque_id, "status": "published"}


def send_admin_message(recipient_identity_token: str, body: str) -> dict:
    """SEULE exception au principe "chatbot = passage obligé" — message direct administration→
    citoyen, jamais via le chatbot comme intermédiaire (voir init_db pour la justification
    anonymat : admin_identity est fixe et publique, recipient_debate_token reste peppé)."""
    recipient_debate_token = compute_debate_token(recipient_identity_token)
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO admin_messages (recipient_debate_token, body) VALUES (?, ?)",
            (recipient_debate_token, body),
        )
        message_id = cur.lastrowid
    return {"message_id": message_id, "recipient_debate_token": recipient_debate_token, "body": body}


def get_public_forum_snapshot() -> list[dict]:
    """Forum, phase 2 (2026-07-25) : lecture PUBLIQUE pour les actions LLM list_threads/get_thread
    (chatbot_actions.py reste sans accès DB — voir ctx["threads"], même pattern que
    ctx["summaries"]/ctx["taken_pseudos"]). Uniquement les fils et opinions au statut 'published' —
    jamais un brouillon (le sien ou celui d'un autre), cohérent avec la règle "jamais de
    corrélation privé↔privé" déjà tranchée sur le wiki : on ne compare/expose ici que du contenu
    déjà public, déjà consenti à être vu."""
    with db() as conn:
        threads = conn.execute(
            "SELECT thread_id, title, summary FROM threads WHERE status='published' ORDER BY thread_id"
        ).fetchall()
        snapshot = []
        for t in threads:
            opinion_rows = conn.execute(
                """SELECT o.opinion_id, o.body, o.argumentaire, o.superseded_by_opinion_id,
                          p.word, p.color
                   FROM opinions o
                   LEFT JOIN pseudos p ON p.debate_token = o.author_debate_token
                   WHERE o.thread_id=? AND o.status='published'
                   ORDER BY o.opinion_id""",
                (t["thread_id"],),
            ).fetchall()
            snapshot.append({
                "thread_id": t["thread_id"],
                "title": t["title"],
                "summary": t["summary"],
                "opinions": [
                    {
                        "opinion_id": o["opinion_id"],
                        "auteur": _agree_pseudo_display(o["word"], o["color"]) if o["word"] else "auteur inconnu",
                        "body": o["body"],
                        "argumentaire": o["argumentaire"],
                        "superseded_by_opinion_id": o["superseded_by_opinion_id"],
                    }
                    for o in opinion_rows
                ],
            })
    return snapshot


def _get_opinion_reaction_summary(conn: sqlite3.Connection, opinion_id: int) -> dict:
    """Décompte adhérer/opposer/neutre + détail attribué par pseudo, pour UNE opinion — factorisé
    car utilisé à la fois par get_forum_page_snapshot (2026-07-26, correction d'un oubli de spec :
    la page Forum n'affichait aucune réaction, alors que Mon activité les affichait déjà) et
    get_my_activity. Même règle que get_current_reaction : seule la réaction la plus RÉCENTE par
    réacteur compte, jamais un doublon si quelqu'un a changé d'avis."""
    reaction_rows = conn.execute(
        """SELECT reaction_id, reactor_debate_token, stance, argumentaire, voice, created_at
           FROM opinion_reactions
           WHERE opinion_id=? AND status='published'
           ORDER BY created_at DESC, reaction_id DESC""",
        (opinion_id,),
    ).fetchall()
    latest_by_reactor: dict[str, sqlite3.Row] = {}
    for r in reaction_rows:
        latest_by_reactor.setdefault(r["reactor_debate_token"], r)
    counts = {"adherer": 0, "opposer": 0, "neutre": 0}
    reactions = []
    for reactor_token, r in latest_by_reactor.items():
        counts[r["stance"]] += 1
        pseudo_row = conn.execute(
            "SELECT word, color FROM pseudos WHERE debate_token=?", (reactor_token,)
        ).fetchone()
        word = pseudo_row["word"] if pseudo_row else None
        color = pseudo_row["color"] if pseudo_row else None
        reactions.append({
            "reaction_id": r["reaction_id"],
            "auteur": _voice_or_pseudo_display(r["voice"], word, color),
            "auteur_word": word,
            "auteur_color": color,
            "voice": r["voice"],
            "stance": r["stance"],
            "argumentaire": r["argumentaire"],
        })
    return {"reaction_counts": counts, "reactions": reactions}


def get_forum_page_snapshot() -> list[dict]:
    """Page "Forum" (lecture seule, 2026-07-26, priorité 1 du développeur via angelobot) :
    parcourir les fils/opinions/remarques SANS passer par une conversation avec le chatbot — le
    chatbot reste le SEUL moyen de PUBLIER, cette fonction ne fait que lire ce qui existe déjà.
    Volontairement SÉPARÉE de get_public_forum_snapshot (qui alimente ctx["threads"] pour
    list_threads/get_thread côté LLM) plutôt que de l'étendre : le ctx du chatbot reste allégé par
    conception (pas de remarques, jamais eu besoin d'y toucher), alors que cette page humaine
    affiche la vue complète. Même filtre status='published' partout — jamais un brouillon, le
    sien ou celui d'un autre.

    "category" (2026-07-29) : la clé fixe assignée à la création du fil (voir FORUM_CATEGORIES),
    ou None pour un fil créé avant l'existence des catégories — le frontend (/forum) affiche ces
    fils dans l'onglet "Tous" uniquement, jamais dans un onglet catégorie précis."""
    with db() as conn:
        threads = conn.execute(
            "SELECT thread_id, title, summary, category FROM threads WHERE status='published' ORDER BY thread_id"
        ).fetchall()
        snapshot = []
        for t in threads:
            opinion_rows = conn.execute(
                """SELECT o.opinion_id, o.body, o.argumentaire, o.superseded_by_opinion_id, o.voice,
                          p.word, p.color
                   FROM opinions o
                   LEFT JOIN pseudos p ON p.debate_token = o.author_debate_token
                   WHERE o.thread_id=? AND o.status='published'
                   ORDER BY o.opinion_id""",
                (t["thread_id"],),
            ).fetchall()
            remarque_rows = conn.execute(
                """SELECT r.remarque_id, r.body, r.reply_to_remarque_id, r.reply_to_opinion_id,
                          r.reply_to_reaction_id, r.voice, p.word, p.color
                   FROM thread_remarques r
                   LEFT JOIN pseudos p ON p.debate_token = r.author_debate_token
                   WHERE r.thread_id=? AND r.status='published'
                   ORDER BY r.remarque_id""",
                (t["thread_id"],),
            ).fetchall()
            snapshot.append({
                "thread_id": t["thread_id"],
                "title": t["title"],
                "summary": t["summary"],
                "category": t["category"],
                "opinions": [
                    {
                        "opinion_id": o["opinion_id"],
                        "auteur": _voice_or_pseudo_display(o["voice"], o["word"], o["color"]),
                        "auteur_word": o["word"],
                        "auteur_color": o["color"],
                        "voice": o["voice"],
                        "body": o["body"],
                        "argumentaire": o["argumentaire"],
                        "superseded_by_opinion_id": o["superseded_by_opinion_id"],
                        **_get_opinion_reaction_summary(conn, o["opinion_id"]),
                    }
                    for o in opinion_rows
                ],
                "remarques": [
                    {
                        "remarque_id": r["remarque_id"],
                        "auteur": _voice_or_pseudo_display(r["voice"], r["word"], r["color"]),
                        "auteur_word": r["word"],
                        "auteur_color": r["color"],
                        "voice": r["voice"],
                        "body": r["body"],
                        "reply_to_remarque_id": r["reply_to_remarque_id"],
                        "reply_to_opinion_id": r["reply_to_opinion_id"],
                        "reply_to_reaction_id": r["reply_to_reaction_id"],
                    }
                    for r in remarque_rows
                ],
            })
    return snapshot


def get_my_activity(identity_token: str) -> dict:
    """Page "Mon activité" (lecture seule, 2026-07-26, priorité 1) : les opinions publiées ou
    désavouées de l'utilisateur connecté, chacune avec le décompte des réactions reçues
    (adhérer/opposer/neutre) et le détail des réactions elles-mêmes (voir
    _get_opinion_reaction_summary). Attribution par pseudo confirmée par le développeur (via
    angelobot, 2026-07-26) : une réaction avec argumentaire est essentiellement une mini-opinion,
    même modèle d'identité publique stable que le reste du forum — pas une nouvelle frontière
    d'anonymat à inventer."""
    debate_token = compute_debate_token(identity_token)
    with db() as conn:
        opinion_rows = conn.execute(
            """SELECT o.opinion_id, o.thread_id, o.body, o.argumentaire, o.status, o.voice,
                      o.superseded_by_opinion_id, o.created_at, t.title AS thread_title
               FROM opinions o
               JOIN threads t ON t.thread_id = o.thread_id
               WHERE o.author_debate_token=? AND o.status IN ('published', 'disavowed')
               ORDER BY o.created_at DESC, o.opinion_id DESC""",
            (debate_token,),
        ).fetchall()
        opinions = [
            {
                "opinion_id": o["opinion_id"],
                "thread_id": o["thread_id"],
                "thread_title": o["thread_title"],
                "body": o["body"],
                "argumentaire": o["argumentaire"],
                "status": o["status"],
                "voice": o["voice"],
                "superseded_by_opinion_id": o["superseded_by_opinion_id"],
                **_get_opinion_reaction_summary(conn, o["opinion_id"]),
            }
            for o in opinion_rows
        ]
    return {"opinions": opinions}


def send_reset_email(to_email: str, reset_token: str) -> bool:
    """Envoie le lien de réinitialisation par email via l'API Brevo.

    Ne lève jamais : retourne False en cas d'échec (clé absente, erreur réseau/API), pour
    que /forgot-password ne révèle jamais si l'envoi a réussi (même comportement visible
    de l'extérieur que l'email existe ou non).
    """
    if not BREVO_API_KEY:
        return False
    reset_link = f"{SITE_URL}/reset-password?token={reset_token}"
    body = json.dumps(
        {
            "sender": {"email": BREVO_SENDER_EMAIL, "name": "Jouy Vote Citoyen"},
            "to": [{"email": to_email}],
            "subject": "Réinitialisation de votre mot de passe - Jouy Vote Citoyen",
            "htmlContent": (
                f"<p>Une réinitialisation de mot de passe a été demandée pour ce compte.</p>"
                f'<p><a href="{reset_link}">Cliquez ici pour choisir un nouveau mot de passe</a></p>'
                f"<p>Ce lien expire dans 1 heure. Si vous n'êtes pas à l'origine de cette "
                f"demande, ignorez cet email.</p>"
            ),
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=body,
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError:
        return False


def _send_admin_email(subject: str, description: str) -> bool:
    """Envoi générique vers ADMIN_BUG_EMAIL (même pattern Brevo que send_reset_email) — utilisé
    par report_bug ET request_admin_intervention (2026-07-25, demande développeur : même
    mécanique pour les deux). Le corps ne contient QUE la description — jamais d'email, de
    session_token ou d'identity_token, même si l'appelant les avait sous la main."""
    if not BREVO_API_KEY or not ADMIN_BUG_EMAIL:
        return False
    body = json.dumps(
        {
            "sender": {"email": BREVO_SENDER_EMAIL, "name": "Jouy Vote Citoyen"},
            "to": [{"email": ADMIN_BUG_EMAIL}],
            "subject": subject,
            "htmlContent": f"<p>{description}</p>",
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=body,
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError:
        return False


def _recent_reports_count(table: str, debate_token: str, window_minutes: int = 60) -> int:
    """Rate-limit anti-abus (2026-07-25) : compte les signalements/demandes récents de CETTE
    identité (debate_token peppé, jamais le token brut) — protège contre un LLM manipulé qui
    spammerait l'envoi d'emails. "table" toujours un littéral interne (jamais dérivé d'une entrée
    utilisateur), pas de risque d'injection SQL malgré l'absence de paramétrage sur ce nom."""
    with db() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM {table} WHERE debate_token=? AND created_at >= datetime('now', ?)",
            (debate_token, f"-{window_minutes} minutes"),
        ).fetchone()
    return row["cnt"]


BUG_REPORT_RATE_LIMIT = 3
ADMIN_INTERVENTION_RATE_LIMIT = 3


def _client_ip(request: Request) -> str:
    """Faille E (revue Opus 04/08) : IP réelle du visiteur, à utiliser comme clé de rate-limit sur
    les endpoints AVANT authentification (pas de debate_token disponible à ce stade). Le service
    tourne derrière un tunnel Cloudflare (voir infra) — request.client.host serait l'IP interne du
    tunnel, pas celle du visiteur, d'où la préférence pour l'en-tête que Cloudflare y injecte.
    Repli sur X-Forwarded-For (1er maillon) puis request.client.host si l'un ou l'autre manque
    (ex: appel direct en local/tests, sans passer par le tunnel)."""
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "inconnu"


def _enforce_rate_limit(request: Request, endpoint: str, limit: int, window_minutes: int = 15) -> None:
    """Faille E (revue Opus 04/08) : aucun rate-limit n'existait sur /login, /register, /vote,
    /forgot-password — brute-force de mot de passe et énumération non freinés côté appli. Même
    principe que _recent_reports_count (compte glissant en base, pas un compteur en mémoire — ce
    dernier ne survivrait pas à un redémarrage et se comporterait mal avec plusieurs workers),
    mais clé par (IP, endpoint) plutôt que par identité, puisque ces 4 endpoints s'exécutent AVANT
    toute authentification. Enregistre la tentative EN MÊME TEMPS que la vérification (pas après)
    pour que même une tentative qui échoue plus loin dans l'endpoint compte dans la fenêtre —
    sinon un mot de passe systématiquement faux ne serait jamais compté."""
    ip = _client_ip(request)
    with db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM rate_limit_events WHERE ip=? AND endpoint=? AND created_at >= datetime('now', ?)",
            (ip, endpoint, f"-{window_minutes} minutes"),
        ).fetchone()
        if row["cnt"] >= limit:
            raise HTTPException(429, "Trop de tentatives récentes, réessaie dans quelques minutes.")
        conn.execute("INSERT INTO rate_limit_events (ip, endpoint) VALUES (?, ?)", (ip, endpoint))


def submit_bug_report(identity_token: str, description: str) -> dict:
    """SEUL point d'écriture + d'envoi réel pour un signalement de bug — appelé DIRECTEMENT par
    l'action LLM report_bug (chatbot_actions.py), pas via un clic de confirmation utilisateur.
    Exception délibérée au principe "jamais d'écriture directe par le LLM" appliqué partout
    ailleurs ce soir (opinions, réactions, remarques...) : un signalement de bug est PRIVÉ (visible
    seulement par angelobot/le développeur, jamais public), à faible enjeu, et rate-limité — profil
    de risque très différent d'une opinion publique et permanente."""
    description = description.strip()
    if not description:
        raise ValueError("description vide")
    debate_token = compute_debate_token(identity_token)
    if _recent_reports_count("bug_reports", debate_token) >= BUG_REPORT_RATE_LIMIT:
        raise ValueError("trop de signalements récents, réessaie dans un moment")
    with db() as conn:
        conn.execute("INSERT INTO bug_reports (debate_token, description) VALUES (?, ?)", (debate_token, description))
    sent = _send_admin_email("Signalement de bug — jouyvote.fr", description)
    return {"sent": sent}


def submit_admin_intervention_request(identity_token: str, description: str) -> dict:
    """Même mécanique que submit_bug_report (2026-07-25, demande développeur explicite : même
    mécanique pour les 2) — table séparée car sémantiquement différent : une demande personnelle
    sur son propre compte (ex: exigence d'anonymat compromise), pas un bug logiciel général."""
    description = description.strip()
    if not description:
        raise ValueError("description vide")
    debate_token = compute_debate_token(identity_token)
    if _recent_reports_count("admin_intervention_requests", debate_token) >= ADMIN_INTERVENTION_RATE_LIMIT:
        raise ValueError("trop de demandes récentes, réessaie dans un moment")
    with db() as conn:
        conn.execute(
            "INSERT INTO admin_intervention_requests (debate_token, description) VALUES (?, ?)",
            (debate_token, description),
        )
    sent = _send_admin_email("Demande d'intervention admin — jouyvote.fr", description)
    return {"sent": sent}


try:
    from qdrant_client import QdrantClient
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

from rag_conseil_municipal.embeddings import embed as _embed_conseil_municipal, is_available as _embeddings_available

# Instance Qdrant DÉDIÉE jouyvote (2026-07-26, RAG conseil municipal) — voir docker-compose.yml
# pour le point de vigilance sécurité (jamais l'instance partagée du projet RPG, sans
# authentification, qui exposerait un chemin réseau vers des données sans rapport).
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
CONSEIL_MUNICIPAL_COLLECTION = "conseil_municipal_pv"

_qdrant_client: "QdrantClient | None" = None


def _get_qdrant_client() -> "QdrantClient":
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def search_conseil_municipal_pv(query: str, top_k: int = 5) -> list[dict]:
    """Recherche sémantique (RAG, 2026-07-26) dans les PV de conseil municipal indexés — voir
    rag_conseil_municipal/ (pipeline d'indexation séparé, tourne dans le conteneur opencode,
    jamais ici). Ne lève JAMAIS : dégrade en liste vide si Qdrant ou le modèle d'embedding est
    indisponible — chatbot_actions.search_conseil_municipal traite une liste vide comme "rien
    trouvé sur ce sujet", jamais comme une erreur qui casserait la conversation."""
    if not _QDRANT_AVAILABLE or not _embeddings_available():
        return []
    try:
        client = _get_qdrant_client()
        if not client.collection_exists(CONSEIL_MUNICIPAL_COLLECTION):
            return []
        vector = _embed_conseil_municipal(query)
        results = client.query_points(
            collection_name=CONSEIL_MUNICIPAL_COLLECTION,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": r.payload.get("text"),
                "source_url": r.payload.get("source_url"),
                "meeting_date": r.payload.get("meeting_date"),
            }
            for r in results.points
        ]
    except Exception:
        return []


# Taille max du texte concaténé renvoyé par get_conseil_municipal_document_pv — garde-fou
# défensif contre un PV anormalement long qui exploserait le contexte du prompt, pas une limite
# fonctionnelle attendue en usage réel (le plus gros PV connu à ce jour fait ~22k caractères).
_DOCUMENT_TEXT_MAX_CHARS = 40000


def get_conseil_municipal_document_pv(source_url: str) -> dict | None:
    """Contenu COMPLET d'UN document déjà identifié par son source_url (RAG, 2026-07-26, bug réel
    #16 : une question vague comme "de quoi ça parlait" fait dériver search_conseil_municipal_pv
    — la similarité sémantique accroche sur des formulations administratives génériques (dates,
    "PROCES-VERBAL", "CONSEIL MUNICIPAL") plutôt que sur le contenu, mélange des chunks d'AUTRES
    documents avec celui demandé, et le modèle invente parfois par-dessus plutôt que d'admettre ne
    pas avoir de contenu fiable). Ici : aucune recherche sémantique, tous les chunks de CE document
    précis (scroll filtré par source_url, triés par chunk_index, concaténés) — fiable par
    construction puisqu'on ne mélange jamais avec un autre document. Ne lève JAMAIS : dégrade en
    None si Qdrant/embeddings indisponible, document introuvable, ou toute exception."""
    if not _QDRANT_AVAILABLE:
        return None
    try:
        client = _get_qdrant_client()
        if not client.collection_exists(CONSEIL_MUNICIPAL_COLLECTION):
            return None
        points, offset = client.scroll(
            collection_name=CONSEIL_MUNICIPAL_COLLECTION,
            scroll_filter={"must": [{"key": "source_url", "match": {"value": source_url}}]},
            limit=200,
            with_payload=True,
        )
        if not points:
            return None
        points.sort(key=lambda p: p.payload.get("chunk_index", 0))
        text = "\n\n".join(p.payload.get("text", "") for p in points)
        truncated = len(text) > _DOCUMENT_TEXT_MAX_CHARS
        if truncated:
            text = text[:_DOCUMENT_TEXT_MAX_CHARS]
        meeting_date = points[0].payload.get("meeting_date")
        return {"source_url": source_url, "meeting_date": meeting_date, "text": text, "truncated": truncated}
    except Exception:
        return None


_MOIS_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}


def _parse_meeting_date(date_str: str | None) -> date | None:
    """Parse les 3 formats produits par rag_conseil_municipal/ : "05 juin 2026" (date de séance
    trouvée dans le texte du PV, index.py), "05/03/2025" (repli depuis le nom de fichier, voir
    _parse_date_label) ou "2026-06-05" (frontmatter date_seance ISO des transcriptions manuelles,
    index_transcriptions.py — bug réel #15 bis, 2026-07-26 : ce 3e format n'était pas reconnu,
    faisait retomber la séance sur date.min et l'envoyait en DERNIÈRE position du tri décroissant
    alors que c'était justement la plus récente indexée). Renvoie None si non parsable — jamais
    d'exception."""
    if not date_str:
        return None
    date_str = date_str.strip()
    match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})$", date_str)
    if match:
        year, month, day = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            return None
    match = re.match(r"(\d{1,2})(?:er)?\s+(\S+)\s+(\d{4})", date_str, re.IGNORECASE)
    if match:
        day, month_name, year = match.groups()
        month = _MOIS_FR.get(month_name.lower())
        if month:
            try:
                return date(int(year), month, int(day))
            except ValueError:
                return None
    match = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", date_str)
    if match:
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            return None
    return None


def list_conseil_municipal_meetings() -> list[dict]:
    """Liste des séances de conseil municipal indexées, triées par date décroissante (RAG,
    2026-07-26, bug réel #15 trouvé par Angelo : "c'est le dernier conseil ça ?" recevait une
    réponse fondée sur search_conseil_municipal, qui trouve par PERTINENCE SÉMANTIQUE au sujet, pas
    par RÉCENCE — deux notions différentes, jamais confondre "le contenu le plus proche de la
    question" et "la séance la plus récente". Cette fonction ne fait AUCUNE recherche sémantique,
    juste un tri chronologique — ne lève jamais, dégrade en liste vide."""
    if not _QDRANT_AVAILABLE:
        return []
    try:
        client = _get_qdrant_client()
        if not client.collection_exists(CONSEIL_MUNICIPAL_COLLECTION):
            return []
        seen: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=CONSEIL_MUNICIPAL_COLLECTION, limit=200, offset=offset, with_payload=True
            )
            for p in points:
                url = p.payload.get("source_url")
                if url and url not in seen:
                    seen[url] = {"source_url": url, "meeting_date": p.payload.get("meeting_date")}
            if offset is None:
                break
        meetings = list(seen.values())
        meetings.sort(key=lambda m: _parse_meeting_date(m["meeting_date"]) or date.min, reverse=True)
        return meetings
    except Exception:
        return []


def send_referral_invite_email(to_email: str, referrer_nom: str, invite_token: str) -> bool:
    """Envoie le lien d'inscription pré-approuvé au filleul. Le lien EST le déclencheur
    d'inscription (pas une validation a posteriori d'un compte déjà créé) : le filleul clique,
    complète lui-même son inscription, ce qui prouve qu'il contrôle l'email et consent
    réellement. Ne lève jamais, retourne False en cas d'échec (mêmes garanties que
    send_reset_email)."""
    if not BREVO_API_KEY:
        return False
    invite_link = f"{SITE_URL}/register?token={invite_token}"
    body = json.dumps(
        {
            "sender": {"email": BREVO_SENDER_EMAIL, "name": "Jouy Vote Citoyen"},
            "to": [{"email": to_email}],
            "subject": f"{referrer_nom} vous invite à rejoindre Jouy Vote Citoyen",
            "htmlContent": (
                f"<p>{referrer_nom} vous invite à rejoindre Jouy Vote Citoyen, l'outil de "
                f"démocratie participative locale des habitants de Jouy.</p>"
                f'<p><a href="{invite_link}">Cliquez ici pour compléter votre inscription</a></p>'
                f"<p>Ce lien est personnel, ne le partagez pas.</p>"
            ),
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=body,
        headers={
            "api-key": BREVO_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError:
        return False


def call_chat_llm(messages: list[dict]) -> str | None:
    """Appelle le modèle de chat via l'API OpenRouter (même fournisseur que le conteneur dev).
    Ne lève jamais : retourne None en cas d'échec (clé absente, erreur réseau, réponse
    inattendue), à charge de l'appelant de répondre proprement à l'utilisateur."""
    if not OPENROUTER_API_KEY:
        return None
    body = json.dumps({"model": CHAT_MODEL, "messages": messages}).encode()
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
        return data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return None


def fetch_wiki_home_content() -> str:
    """Va chercher le rendu de la page 'start' du wiki pour l'afficher sur la page d'accueil.

    Retourne une chaîne vide en cas d'échec (jamais d'exception) : l'accueil doit rester
    utilisable même si le wiki est indisponible. Le HTML renvoyé par do=export_xhtml est un
    DOCUMENT complet (head/body) — on n'en garde que le fragment de contenu utile, et les liens
    relatifs (ex: href="/genese") sont réécrits vers le domaine public du wiki, sinon ils
    pointeraient vers des routes inexistantes sur jouyvote.fr.
    """
    url = f"{WIKI_INTERNAL_URL}/doku.php?id=start&do=export_xhtml"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError:
        return ""
    start = html.find('<div class="dokuwiki export">')
    end = html.find("</body>")
    if start == -1 or end == -1:
        return ""
    fragment = html[start:end]
    return fragment.replace('href="/', f'href="{WIKI_PUBLIC_URL}/')


# Accès en lecture au wiki citoyen pour le chatbot (2026-07-26, demande développeur, validée par
# un test réel : le chatbot ne savait pas répondre correctement sur le fonctionnement du SITE
# lui-même — anonymat, pseudo... — en dehors de ce qui est écrit en dur dans son prompt statique,
# alors que ce wiki documente déjà tout ça publiquement).
#
# ALLOWLIST EXPLICITE plutôt qu'un accès à tout le wiki (nécessité de sécurité, pas un choix
# produit) : plusieurs pages du même wiki sont des documents INTERNES à l'équipe technique, jamais
# destinés à un citoyen — architecture-technique (détails d'implémentation de l'anonymat,
# sensible), bugs-jouyvote, developpement, spec, themes:chatbot-fonctionnalites (journal de bord
# technique de ce module), themes:prompt-chatbot (le prompt système lui-même — l'exposer casserait
# sa confidentialité et faciliterait une injection de prompt). Liste par défaut-refus : une
# nouvelle page ajoutée au wiki plus tard n'est PAS automatiquement exposée, il faut l'ajouter ici
# explicitement.
WIKI_CITIZEN_PAGES = {
    "charte-anonymat": "La charte de l'anonymat — règles complètes sur la protection de l'identité.",
    "genese": "Origine et philosophie du projet Jouy Vote Citoyen.",
    "start": "Page d'accueil du wiki citoyen.",
    "themes:anonymat": "L'anonymat sur jouyvote.fr — principes et garanties.",
    "themes:chatbot": "Pourquoi un chatbot citoyen, comment il aide à formuler une opinion.",
    "themes:consensus": "Identification des pôles d'opinion et recherche de consensus.",
    "themes:pseudonyme": "Fonctionnement du pseudonyme (mot+couleur) et pourquoi il est stable.",
    "themes:representants": "Les représentants et leur rôle sur la plateforme.",
    "themes:representation": "Système de parrainage et vérification de résidence.",
    "themes:ressources": "Ressources informatives disponibles sur le site.",
}


def fetch_wiki_page_raw(page_id: str) -> str | None:
    """Lecture seule — syntaxe DokuWiki BRUTE (do=export_raw, pas export_xhtml qui rendrait du
    HTML à nettoyer pour rien : seul le contenu compte pour un LLM, pas le rendu visuel). Limité
    aux pages de WIKI_CITIZEN_PAGES — retourne None (jamais d'exception) si la page n'est pas dans
    l'allowlist ou si le wiki est indisponible."""
    if page_id not in WIKI_CITIZEN_PAGES:
        return None
    url = f"{WIKI_INTERNAL_URL}/doku.php?id={page_id}&do=export_raw"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError:
        return None


@app.get("/wiki-home-content")
def wiki_home_content():
    return {"html": fetch_wiki_home_content()}


@app.post("/register")
def register(r: Registration, request: Request):
    # limit=15 (pas 5) : un parrain qui invite jusqu'à REFERRAL_MAX=5 filleuls peut voir tous ses
    # filleuls s'inscrire en rafale depuis le MÊME réseau (ex: session d'inscription en personne,
    # tout le monde chez la même personne) — marge confortable au-delà des 5+1 (soi-même) minimum
    # légitime, tout en restant loin d'un volume d'abus automatisé.
    _enforce_rate_limit(request, "register", limit=15)
    # Le lien de parrainage EST le déclencheur d'inscription (pas une validation a posteriori
    # d'un compte déjà créé) : un invite_token valide, non consommé, dont l'email correspond
    # exactement, est requis tant que les inscriptions ne sont pas rouvertes globalement
    # (REGISTRATIONS_OPEN, réservé au bootstrap/à une réouverture d'urgence).
    referred_by_token = None
    if not REGISTRATIONS_OPEN:
        if not r.invite_token:
            raise HTTPException(403, "Inscription accessible uniquement via un lien de parrainage.")
        with db() as conn:
            invite = conn.execute(
                "SELECT referrer_token, invitee_email FROM referral_invites WHERE invite_token=? AND used=0",
                (r.invite_token,),
            ).fetchone()
        if not invite:
            raise HTTPException(403, "Invitation invalide ou déjà utilisée.")
        if invite["invitee_email"].strip().lower() != r.email.strip().lower():
            raise HTTPException(403, "Cet email ne correspond pas à l'invitation reçue.")
        with db() as conn:
            current_count = conn.execute(
                "SELECT COUNT(*) as n FROM identities WHERE referred_by_token=?",
                (invite["referrer_token"],),
            ).fetchone()["n"]
        if current_count >= REFERRAL_MAX:
            raise HTTPException(403, "Ce parrain a atteint son quota de filleuls.")
        referred_by_token = invite["referrer_token"]

    identity_hash = compute_identity_hash(r.nom, r.adresse)
    token = secrets.token_urlsafe(32)
    session_token = secrets.token_urlsafe(32)
    password_hash = hash_password(r.password)
    with db() as conn:
        try:
            conn.execute(
                "INSERT INTO identities (token, identity_hash, nom, adresse, email, password_hash, session_token, referred_by_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (token, identity_hash, r.nom.strip(), r.adresse.strip(), r.email, password_hash, session_token, referred_by_token),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Cette identité (nom + adresse) est déjà inscrite.")
        if referred_by_token:
            conn.execute("UPDATE referral_invites SET used=1 WHERE invite_token=?", (r.invite_token,))
    return {"token": token, "session_token": session_token, "message": "Inscription réussie."}


@app.post("/login")
def login(req: LoginRequest, request: Request):
    _enforce_rate_limit(request, "login", limit=10)
    with db() as conn:
        row = conn.execute(
            "SELECT token, nom, password_hash, role FROM identities WHERE email=?",
            (req.email,),
        ).fetchone()
    if not row or not row["password_hash"]:
        raise HTTPException(401, "Email ou mot de passe incorrect.")
    if not check_password(req.password, row["password_hash"]):
        raise HTTPException(401, "Email ou mot de passe incorrect.")
    session_token = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute("UPDATE identities SET session_token=? WHERE token=?", (session_token, row["token"]))
    # token renvoyé en plus de session_token : c'est le jeton d'identité nécessaire pour voter
    # (POST /vote), déjà renvoyé une fois à l'inscription mais jamais récupérable après si
    # l'utilisateur se reconnecte sur un autre appareil ou après avoir vidé son navigateur —
    # sans ça la page Voter serait cassée pour quasi tout le monde. Aucun pouvoir nouveau accordé
    # : l'utilisateur est déjà propriétaire légitime de ce token en s'étant authentifié par
    # mot de passe (c'est la clé primaire de sa propre ligne dans identities).
    # "role" (2026-08-03, voix Admin/Mairie) : None pour l'immense majorité des comptes (citoyen
    # implicite) — sert au frontend UNIQUEMENT à savoir s'il faut afficher le lien "Rôles" dans la
    # nav, jamais utilisé côté serveur comme preuve d'autorisation (chaque route Admin revérifie
    # le rôle en base via _require_admin, le localStorage n'est qu'un affichage).
    return {
        "session_token": session_token, "token": row["token"], "nom": row["nom"],
        "email": req.email, "role": row["role"],
    }


@app.post("/logout")
def logout(req: LogoutRequest):
    with db() as conn:
        conn.execute("UPDATE identities SET session_token=NULL WHERE session_token=?", (req.session_token,))
    return {"ok": True}


@app.post("/admin/roles/list")
def admin_roles_list(req: AdminRolesListRequest):
    """Voix Admin/Mairie (2026-08-03, étape 2/6) : réservé aux comptes Admin — annuaire complet
    (email, nom, téléphone, role) de tous les comptes Admin/Mairie, base de la page de gestion
    des rôles. La restriction "un Mairie ne voit que la liste des Mairie" (étape 3, annuaires)
    utilisera list_privileged_accounts() avec un filtrage différent, pas cette route (réservée
    Admin seul)."""
    _require_admin(req.session_token)
    return {"accounts": list_privileged_accounts()}


@app.post("/admin/roles/set")
def admin_roles_set(req: AdminRolesSetRequest):
    """SEUL endpoint qui modifie le rôle d'un compte — réservé Admin (voir _require_admin).
    role=null retire tout privilège."""
    _require_admin(req.session_token)
    try:
        return set_account_role(req.target_email, req.role)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/mairie/directory")
def mairie_directory(req: MairieDirectoryRequest):
    """Voix Admin/Mairie (2026-08-03, étape 3/6) : annuaire des comptes Mairie, accessible aux
    rôles admin ET mairie (voir _require_admin_or_mairie) — noms uniquement, jamais l'email ni le
    téléphone (voir list_mairie_directory pour la justification, spec wiki
    themes:admin-mairie)."""
    _require_admin_or_mairie(req.session_token)
    return {"accounts": list_mairie_directory()}


@app.post("/change-password")
def change_password(req: ChangePasswordRequest):
    with db() as conn:
        row = conn.execute(
            "SELECT token, password_hash FROM identities WHERE session_token=?",
            (req.session_token,),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Session invalide.")
    if not check_password(req.current_password, row["password_hash"]):
        raise HTTPException(401, "Mot de passe actuel incorrect.")
    password_hash = hash_password(req.new_password)
    with db() as conn:
        conn.execute("UPDATE identities SET password_hash=? WHERE token=?", (password_hash, row["token"]))
    return {"ok": True, "message": "Mot de passe modifié."}


@app.post("/referral/invite")
def referral_invite(req: ReferralInviteRequest):
    if not req.confirms_residency_and_age:
        raise HTTPException(400, "Vous devez confirmer que cette personne habite Jouy et est majeure.")
    with db() as conn:
        referrer = conn.execute(
            "SELECT token, nom FROM identities WHERE session_token=?", (req.session_token,)
        ).fetchone()
    if not referrer:
        raise HTTPException(401, "Session invalide.")
    with db() as conn:
        current_count = conn.execute(
            "SELECT COUNT(*) as n FROM identities WHERE referred_by_token=?", (referrer["token"],)
        ).fetchone()["n"]
    if current_count >= REFERRAL_MAX:
        raise HTTPException(403, f"Quota de {REFERRAL_MAX} filleuls déjà atteint.")
    invite_token = secrets.token_urlsafe(32)
    invitee_email = req.invitee_email.strip().lower()
    with db() as conn:
        conn.execute(
            "INSERT INTO referral_invites (invite_token, referrer_token, invitee_email) VALUES (?, ?, ?)",
            (invite_token, referrer["token"], invitee_email),
        )
    send_referral_invite_email(invitee_email, referrer["nom"], invite_token)
    return {"ok": True, "message": "Invitation envoyée."}


@app.get("/referral/invite/{invite_token}")
def referral_invite_info(invite_token: str):
    with db() as conn:
        invite = conn.execute(
            "SELECT referrer_token, invitee_email, used FROM referral_invites WHERE invite_token=?",
            (invite_token,),
        ).fetchone()
    if not invite:
        raise HTTPException(404, "Invitation introuvable.")
    if invite["used"]:
        raise HTTPException(410, "Cette invitation a déjà été utilisée.")
    with db() as conn:
        referrer = conn.execute("SELECT nom FROM identities WHERE token=?", (invite["referrer_token"],)).fetchone()
    return {
        "invitee_email": invite["invitee_email"],
        "referrer_nom": referrer["nom"] if referrer else "quelqu'un",
    }


@app.post("/referral/status")
def referral_status(req: ReferralStatusRequest):
    with db() as conn:
        referrer = conn.execute(
            "SELECT token FROM identities WHERE session_token=?", (req.session_token,)
        ).fetchone()
    if not referrer:
        raise HTTPException(401, "Session invalide.")
    with db() as conn:
        used = conn.execute(
            "SELECT COUNT(*) as n FROM identities WHERE referred_by_token=?", (referrer["token"],)
        ).fetchone()["n"]
        invites = conn.execute(
            "SELECT invitee_email, used FROM referral_invites WHERE referrer_token=? ORDER BY created_at DESC",
            (referrer["token"],),
        ).fetchall()
    return {
        "used": used,
        "remaining": max(0, REFERRAL_MAX - used),
        "max": REFERRAL_MAX,
        "invites": [{"email": i["invitee_email"], "used": bool(i["used"])} for i in invites],
    }


@app.post("/my-vote-token")
def my_vote_token(req: MyVoteTokenRequest):
    # vote_token est le MÊME pour toutes les questions d'un utilisateur (dérivé uniquement de
    # son token d'identité + le pepper, jamais du question_id) — c'est ce qu'il retrouve déjà
    # publiquement dans /results à côté de son choix. Calcul nécessairement côté serveur : le
    # pepper est un secret serveur, jamais exposé au client.
    identity_token = _require_identity(req.session_token)
    return {"vote_token": compute_vote_token(identity_token)}


def _require_identity(session_token: str) -> str:
    """Résout un session_token en token d'identité, ou lève 401. Factorisé car réutilisé par
    toutes les routes /chat/*, contrairement au reste de l'API qui a chacune sa propre requête
    (gardé identique ici pour ne pas dupliquer 5 fois la même vérification)."""
    with db() as conn:
        row = conn.execute("SELECT token FROM identities WHERE session_token=?", (session_token,)).fetchone()
    if not row:
        raise HTTPException(401, "Session invalide.")
    return row["token"]


def _require_admin(session_token: str) -> str:
    """Voix Admin/Mairie (2026-08-03) : comme _require_identity, mais exige en plus role='admin'
    sur le compte — 403 (pas 401, la session EST valide) si un compte citoyen ou mairie tente
    d'appeler une route réservée à l'Admin. Utilisé par toutes les routes /admin/roles/*."""
    with db() as conn:
        row = conn.execute(
            "SELECT token, role FROM identities WHERE session_token=?", (session_token,)
        ).fetchone()
    if not row:
        raise HTTPException(401, "Session invalide.")
    if row["role"] != "admin":
        raise HTTPException(403, "Réservé aux comptes Admin.")
    return row["token"]


def _require_admin_or_mairie(session_token: str) -> str:
    """Voix Admin/Mairie (2026-08-03, étape 3/6) : comme _require_admin, mais accepte aussi
    role='mairie' — utilisé par /mairie/directory (l'annuaire Mairie que la spec ouvre aux deux
    rôles, pas seulement à l'Admin)."""
    with db() as conn:
        row = conn.execute(
            "SELECT token, role FROM identities WHERE session_token=?", (session_token,)
        ).fetchone()
    if not row:
        raise HTTPException(401, "Session invalide.")
    if row["role"] not in ("admin", "mairie"):
        raise HTTPException(403, "Réservé aux comptes Admin ou Mairie.")
    return row["token"]


def list_privileged_accounts() -> list[dict]:
    """Lecture pure : tous les comptes ayant un rôle Admin ou Mairie (email, nom, role) —
    l'annuaire complet vu par un Admin (voir wiki themes:admin-mairie). Le filtrage "un compte
    Mairie ne voit que les comptes Mairie, pas les coordonnées Admin" se fait à l'appelant
    (endpoint), pas ici — cette fonction reste la source de vérité brute, réutilisée par
    l'étape 3 (annuaires)."""
    with db() as conn:
        rows = conn.execute(
            "SELECT email, nom, telephone, role FROM identities WHERE role IN ('admin', 'mairie') "
            "ORDER BY role, nom"
        ).fetchall()
    return [dict(r) for r in rows]


def list_mairie_directory() -> list[dict]:
    """Lecture pure : uniquement le NOM des comptes Mairie (JAMAIS l'email ni le téléphone —
    spec wiki themes:admin-mairie : "les comptes Mairie voient la liste des comptes Mairie",
    contrairement à l'Admin qui voit en plus les coordonnées des autres Admin, coordonnées
    jamais partagées avec un compte Mairie). Utilisée par /mairie/directory, accessible aux
    rôles admin ET mairie (un Admin n'en a normalement pas besoin, /admin/roles/list lui donne
    déjà une vue plus riche, mais rien ne l'exclut ici)."""
    with db() as conn:
        rows = conn.execute("SELECT nom FROM identities WHERE role='mairie' ORDER BY nom").fetchall()
    return [dict(r) for r in rows]


def set_account_role(target_email: str, role: str | None) -> dict:
    """SEUL point d'écriture du rôle d'un compte (2026-08-03) — appelé uniquement par un Admin
    (voir _require_admin dans l'endpoint). role=None retire tout privilège (retour au citoyen
    implicite). Pas de garde-fou "un Admin ne peut pas se retirer lui-même son propre rôle" —
    volontairement simple pour l'instant (le bootstrap reste toujours possible à la main en base
    si jamais plus aucun Admin n'existe, cas limite qui n'est pas apparu en usage réel)."""
    if role not in (None, "admin", "mairie"):
        raise ValueError("rôle invalide, doit être 'admin', 'mairie' ou absent")
    target_email = target_email.strip()
    with db() as conn:
        row = conn.execute("SELECT token FROM identities WHERE email=?", (target_email,)).fetchone()
        if row is None:
            raise ValueError(f"aucun compte avec l'email {target_email}")
        conn.execute("UPDATE identities SET role=? WHERE token=?", (role, row["token"]))
        updated = conn.execute(
            "SELECT email, nom, role FROM identities WHERE token=?", (row["token"],)
        ).fetchone()
    return dict(updated)


def _voice_or_pseudo_display(voice: str | None, word: str | None, color: str | None) -> str:
    """Nom d'affichage d'un auteur (2026-08-03, voix Admin/Mairie) : "Admin"/"Mairie" fixe si
    voice est renseigné (jamais le pseudo mot+couleur, même si l'identité en a un — la voix
    prime), sinon le pseudo habituel via _agree_pseudo_display, ou "auteur inconnu" si aucun des
    deux (compte de test sans pseudo confirmé, cas déjà géré avant ce champ)."""
    if voice == "admin":
        return "Admin"
    if voice == "mairie":
        return "Mairie"
    return _agree_pseudo_display(word, color) if word else "auteur inconnu"


def _validate_voice(identity_token: str, voice: str | None) -> None:
    """Voix Admin/Mairie (2026-08-03, étape 4/6) : voice=None signifie "publication normale sous
    pseudo citoyen", aucune vérification. voice='admin'/'mairie' n'est accepté QUE si le rôle RÉEL
    de l'identité (relu en base ici, jamais la valeur envoyée par le client) correspond exactement
    — un client ne peut jamais s'auto-déclarer Admin/Mairie en mentant sur ce paramètre, même si
    le reste de la requête est par ailleurs légitime."""
    if voice is None:
        return
    if voice not in ("admin", "mairie"):
        raise ValueError("voix invalide, doit être 'admin', 'mairie' ou absente")
    with db() as conn:
        row = conn.execute("SELECT role FROM identities WHERE token=?", (identity_token,)).fetchone()
    if row is None or row["role"] != voice:
        raise ValueError(f"ce compte n'a pas la voix {voice}")


def _check_category_posting_permission(category: str | None, voice: str | None) -> None:
    """"Poster une nouvelle opinion/fil = restreint à sa propre catégorie" (spec wiki
    themes:admin-mairie) : une catégorie RÉSERVÉE (admin/mairie) ne peut recevoir une NOUVELLE
    opinion que si elle est postée avec la voix correspondante. Volontairement absent des
    fonctions de réaction/remarque (add_reaction, create_remarque) : "réagir" reste libre partout,
    quelle que soit la catégorie du fil, pour tous les rôles — seul "poster une opinion" est
    concerné par cette restriction.

    Fix structurel (2026-08-03) : avant l'ajout de ce garde-fou, un citoyen quelconque pouvait
    déjà, en appelant /opinion/confirm directement (hors chatbot, l'endpoint accepte
    new_thread_category en texte libre sans jamais vérifier qui a le droit d'y poster), créer un
    fil dans la catégorie "admin" ou "mairie" — la validation LLM (propose_opinion, restreinte à
    FORUM_CATEGORIES) ne protège que le chemin chatbot, jamais l'écriture elle-même. Ce trou est
    apparu dès l'élargissement de create_thread_with_opinion/create_thread à ALL_CATEGORIES
    (étape 1/6, préparation de l'écriture directe) — corrigé ici avant qu'aucune vraie donnée
    n'ait pu l'exploiter."""
    if category in RESERVED_CATEGORIES and voice != category:
        raise ValueError(
            f"seul un compte utilisant la voix {RESERVED_CATEGORIES[category]} peut poster "
            f"une opinion dans la catégorie {RESERVED_CATEGORIES[category]}"
        )


@app.post("/chat")
def chat(req: ChatRequest):
    _require_identity(req.session_token)
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for m in req.history[-20:]:  # borne défensive, pas de limite fonctionnelle attendue en usage réel
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})
    reply = call_chat_llm(messages)
    if reply is None:
        raise HTTPException(503, "Le chatbot est momentanément indisponible.")
    return {"reply": reply}


_JOURS_NOMS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MOIS_NOMS = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def _element_context_block(
    context_opinion_id: int | None, context_remarque_id: int | None,
    identity_token: str | None = None, context_reaction_id: int | None = None,
) -> str:
    """Chat inline scopé (2026-07-29, bouton "Réagir" sur Forum/Mon activité) : décrit l'opinion,
    la remarque ou (2026-07-31) la réaction précise sur laquelle l'utilisateur vient de cliquer
    "Réagir", pour qu'il n'ait jamais à la réexpliquer lui-même. Dégrade en chaîne vide si l'id est
    absent, invalide ou pointe vers un contenu non publié — un contexte manquant ne doit jamais
    faire échouer la conversation, juste priver le modèle de ce rappel (il peut toujours demander
    lui-même si besoin).

    identity_token (2026-07-30, boutons directs adherer/opposer/neutre) : optionnel — permet de
    rappeler au modèle le stance déjà posé par bouton pour cette opinion (voir
    set_reaction_stance), pour qu'un message ultérieur de simple argumentaire ("parce que...")
    réutilise ce stance au lieu d'en redemander un ou d'en inventer un par défaut."""
    if len([x for x in (context_opinion_id, context_remarque_id, context_reaction_id) if x is not None]) > 1:
        return ""
    with db() as conn:
        if context_opinion_id is not None:
            row = conn.execute(
                """SELECT o.thread_id, o.body, o.argumentaire, t.title AS thread_title,
                          p.word, p.color
                   FROM opinions o JOIN threads t ON t.thread_id = o.thread_id
                   LEFT JOIN pseudos p ON p.debate_token = o.author_debate_token
                   WHERE o.opinion_id=? AND o.status='published'""",
                (context_opinion_id,),
            ).fetchone()
            if row is None:
                return ""
            # Pseudo de l'auteur (2026-07-30, bug réel signalé par Angelo : "qui a publié cette
            # opinion ?" donnait parfois une réponse garbled ou un placeholder {{résultat}} non
            # résolu) — le modèle devait auparavant appeler get_thread pour le trouver, et se
            # plantait parfois en route. Donné ici directement, comme opinion_id plus tôt le même
            # soir : même leçon, ne jamais laisser le modèle deviner/chercher ce qu'on peut lui
            # fournir directement.
            auteur = _agree_pseudo_display(row["word"], row["color"]) if row["word"] else "auteur inconnu"
            # "C'est toi" (2026-07-30, bug réel connexe signalé par Angelo) : le pseudo de
            # l'utilisateur COURANT n'est jamais mis en texte lisible ailleurs dans le prompt
            # (ctx["pseudo"] n'existe que pour des comparaisons internes côté action, jamais
            # exposé au modèle) — sans ce rappel, le modèle répondait toujours à la 3e personne,
            # même sur la propre opinion de l'utilisateur. Comparaison faite ICI en Python (sur
            # word+color canoniques, pas la forme accordée) plutôt que de faire deviner au modèle
            # si deux pseudos affichés désignent la même personne — même logique que le blocage
            # auto-réaction plus tôt dans la soirée.
            if identity_token is not None and row["word"]:
                own_pseudo = get_existing_pseudo(identity_token)
                if own_pseudo and own_pseudo["word"] == row["word"] and own_pseudo["color"] == row["color"]:
                    auteur += " (c'est TOI, l'utilisateur avec qui tu parles en ce moment)"
            block = (
                f"CONTEXTE : l'utilisateur vient de cliquer sur \"Réagir\" à propos de l'opinion "
                f"opinion_id={context_opinion_id} (thread_id={row['thread_id']}), fil "
                f"\"{row['thread_title']}\", publiée par {auteur} : \"{row['body']}\""
            )
            if row["argumentaire"]:
                block += f" (argumentaire : \"{row['argumentaire']}\")"
            if identity_token is not None:
                current = get_current_reaction(context_opinion_id, identity_token)
                if current is not None:
                    block += (
                        f". L'utilisateur a déjà choisi le stance \"{current['stance']}\" via un "
                        f"bouton direct pour cette opinion — si son message n'est qu'un "
                        f"argumentaire (ex: \"parce que...\"), réutilise CE stance, ne lui redemande "
                        f"pas et n'en invente pas un autre sauf s'il l'exprime clairement"
                    )
            return (
                block + f". Si l'utilisateur veut réagir à CETTE opinion précise, appelle "
                f"directement propose_reaction avec opinion_id={context_opinion_id} — n'appelle "
                f"JAMAIS list_threads/get_thread pour la retrouver, tu as déjà tout ce qu'il faut "
                f"ci-dessus, y compris l'identifiant exact. Concentre-toi sur CET élément précis, "
                f"sans élargir au reste du forum sauf si l'utilisateur le demande explicitement."
            )
        if context_remarque_id is not None:
            row = conn.execute(
                """SELECT r.thread_id, r.body, t.title AS thread_title, p.word, p.color
                   FROM thread_remarques r JOIN threads t ON t.thread_id = r.thread_id
                   LEFT JOIN pseudos p ON p.debate_token = r.author_debate_token
                   WHERE r.remarque_id=? AND r.status='published'""",
                (context_remarque_id,),
            ).fetchone()
            if row is None:
                return ""
            auteur = _agree_pseudo_display(row["word"], row["color"]) if row["word"] else "auteur inconnu"
            if identity_token is not None and row["word"]:
                own_pseudo = get_existing_pseudo(identity_token)
                if own_pseudo and own_pseudo["word"] == row["word"] and own_pseudo["color"] == row["color"]:
                    auteur += " (c'est TOI, l'utilisateur avec qui tu parles en ce moment)"
            return (
                f"CONTEXTE : l'utilisateur vient de cliquer sur \"Réagir\" à propos de la remarque "
                f"remarque_id={context_remarque_id} (thread_id={row['thread_id']}), fil "
                f"\"{row['thread_title']}\", écrite par {auteur} : \"{row['body']}\". Si "
                f"l'utilisateur veut répondre à "
                f"CETTE remarque précise, appelle directement propose_remarque avec "
                f"thread_id={row['thread_id']} et reply_to_remarque_id={context_remarque_id} — "
                f"n'appelle JAMAIS list_threads/get_thread pour la retrouver, tu as déjà tout ce "
                f"qu'il faut ci-dessus, y compris les identifiants exacts. Concentre-toi sur CET "
                f"élément précis, sans élargir au reste du forum sauf si l'utilisateur le demande "
                f"explicitement."
            )
        if context_reaction_id is not None:
            row = conn.execute(
                """SELECT re.opinion_id, re.stance, re.argumentaire, o.thread_id, t.title AS thread_title,
                          p.word, p.color
                   FROM opinion_reactions re
                   JOIN opinions o ON o.opinion_id = re.opinion_id
                   JOIN threads t ON t.thread_id = o.thread_id
                   LEFT JOIN pseudos p ON p.debate_token = re.reactor_debate_token
                   WHERE re.reaction_id=? AND re.status='published'""",
                (context_reaction_id,),
            ).fetchone()
            if row is None:
                return ""
            auteur = _agree_pseudo_display(row["word"], row["color"]) if row["word"] else "auteur inconnu"
            if identity_token is not None and row["word"]:
                own_pseudo = get_existing_pseudo(identity_token)
                if own_pseudo and own_pseudo["word"] == row["word"] and own_pseudo["color"] == row["color"]:
                    auteur += " (c'est TOI, l'utilisateur avec qui tu parles en ce moment)"
            block = (
                f"CONTEXTE : l'utilisateur vient de cliquer sur \"Réagir\" à propos d'une réaction "
                f"reaction_id={context_reaction_id} (opinion_id={row['opinion_id']}, "
                f"thread_id={row['thread_id']}, fil \"{row['thread_title']}\") — {auteur} a choisi le "
                f"stance \"{row['stance']}\" sur cette opinion"
            )
            if row["argumentaire"]:
                block += f", avec l'argumentaire : \"{row['argumentaire']}\""
            return (
                block + f". Une réaction n'a pas de sous-réaction formelle : si l'utilisateur veut "
                f"rebondir sur CETTE réaction précise, appelle directement propose_remarque avec "
                f"thread_id={row['thread_id']} et reply_to_reaction_id={context_reaction_id} — "
                f"n'appelle JAMAIS list_threads/get_thread pour la retrouver, tu as déjà tout ce "
                f"qu'il faut ci-dessus, y compris les identifiants exacts. Concentre-toi sur CET "
                f"élément précis, sans élargir au reste du forum sauf si l'utilisateur le demande "
                f"explicitement."
            )
    return ""


def current_date_block() -> str:
    """Bloc de contexte "date du jour" (2026-07-26, manque trouvé par Angelo en réel : "tu sais
    quel jour on est ?" -> "je n'ai pas accès à l'heure actuelle"). Toujours injecté dans le
    system prompt de /chat/v2, jamais laissé au modèle à déduire. Heure de Paris (pas UTC brut,
    l'utilisateur est en France) — fonction pure, testable indépendamment de la requête HTTP."""
    now = datetime.now(ZoneInfo("Europe/Paris"))
    jour_semaine = _JOURS_NOMS[now.weekday()]
    mois = _MOIS_NOMS[now.month - 1]
    return f"Nous sommes le {jour_semaine} {now.day} {mois} {now.year} ({now.strftime('%Y-%m-%d')})."


@app.post("/chat/v2")
def chat_v2(req: ChatRequest):
    """POC tool-calling (mandat angelobot 2026-07-25, voir wiki.jouyvote.fr/themes:prompt-chatbot).

    Isolé de /chat (production, inchangé) tant que ce POC n'est pas validé — même contrat
    d'entrée (ChatRequest) pour rester simple à tester en parallèle, mais boucle d'exécution
    entièrement différente (liste d'actions typées + response_format json_schema strict, voir
    chatbot_executor.py/chatbot_actions.py)."""
    identity_token = _require_identity(req.session_token)
    existing_pseudo = get_existing_pseudo(identity_token)
    with db() as conn:
        summary_rows = conn.execute(
            "SELECT id, summary, created_at FROM chat_summaries WHERE owner_token=? ORDER BY created_at DESC",
            (identity_token,),
        ).fetchall()
        taken_rows = conn.execute("SELECT word, color FROM pseudos").fetchall()
    taken_pseudos = {(r["word"], r["color"]) for r in taken_rows}
    # Bloc actif tant qu'aucun pseudo n'est confirmé — sur autant de tours que nécessaire (pas de
    # education_state pour cette passe, donc pas de suivi plus fin que ce signal binaire). Bug réel
    # #24 (2026-08-04/05, Angelo) : 3 exemples tirés au hasard RECALCULÉS à chaque appel (pas une
    # constante figée) — voir chatbot_actions.random_available_pseudo_candidates/
    # build_onboarding_context_block, le modèle n'a plus besoin de suivre un état lui-même.
    # Bug réel connexe (2026-08-06, Angelo) : ce fix n'était branché QUE sur l'onboarding — un
    # rechoix explicite (pseudo déjà confirmé) retombait sur l'ancien mécanisme un-par-un. Même
    # principe étendu ici : 3 alternatives déjà calculées, fournies même quand un pseudo existe
    # déjà, à ne mentionner QUE si l'utilisateur demande explicitement à changer (voir
    # build_pseudo_rechoice_context_block).
    pseudo_suggestions = random_available_pseudo_candidates(taken_pseudos)
    if existing_pseudo:
        current_display = _agree_pseudo_display(existing_pseudo["word"], existing_pseudo["color"])
        context_block = build_pseudo_rechoice_context_block(current_display, pseudo_suggestions)
    else:
        context_block = build_onboarding_context_block(pseudo_suggestions)
    # Date du jour (2026-07-26, manque trouvé par Angelo en réel : "tu sais quel jour on est ?" ->
    # "je n'ai pas accès à l'heure actuelle") — toujours injectée, indépendamment de l'onboarding,
    # utile aussi pour raisonner sur list_conseil_municipal_seances ("y a-t-il eu un conseil
    # depuis juin ?").
    context_block = f"{current_date_block()}\n\n{context_block}".strip() if context_block else current_date_block()
    element_block = _element_context_block(
        req.context_opinion_id, req.context_remarque_id, identity_token, req.context_reaction_id
    )
    if element_block:
        context_block = f"{context_block}\n\n{element_block}".strip()
    # Scope des outils par contexte (2026-08-04, revue Opus indépendante, voir chatbot_actions.
    # build_tools_description) : 2 situations déduites de l'ÉTAT, pas de l'intention devinée — le
    # widget "Réagir" (élément forum ciblé) prime sur l'onboarding si les deux signaux sont
    # présents en même temps (cas marginal : naviguer le Forum avant d'avoir choisi de pseudo),
    # car c'est le signal le plus précis des deux. "assistance générale" (aucun des deux signaux)
    # garde le jeu complet, inchangé.
    if element_block:
        action_scope = FORUM_REACTION_ACTIONS
    elif not existing_pseudo:
        action_scope = ONBOARDING_ACTIONS
    else:
        action_scope = None
    system_prompt = build_system_prompt(CHAT_SYSTEM_PROMPT, context_block=context_block, action_names=action_scope)
    conversation_messages = [{"role": m.role, "content": m.content} for m in req.history[-20:]]
    conversation_messages.append({"role": "user", "content": req.message})
    ctx = {
        "identity_token": identity_token,
        "history": conversation_messages,
        "summaries": [dict(r) for r in summary_rows],
        "pseudo": existing_pseudo,
        "taken_pseudos": taken_pseudos,
        # Forum phase 2 (2026-07-25) : snapshot des fils/opinions PUBLIÉS pour list_threads/
        # get_thread — voir get_public_forum_snapshot pour la justification anonymat (jamais un
        # brouillon, jamais de corrélation privé↔privé).
        "threads": get_public_forum_snapshot(),
        # report_bug/request_admin_intervention (2026-07-25) : callables plutôt que données brutes
        # — chatbot_actions.py reste sans accès DB/réseau direct, l'écriture + l'envoi d'email
        # réels restent entièrement dans main.py (voir submit_bug_report/submit_admin_intervention_
        # request pour la justification de l'exception "écriture directe par le LLM").
        "report_bug_fn": lambda description: submit_bug_report(identity_token, description),
        "request_admin_intervention_fn": lambda description: submit_admin_intervention_request(identity_token, description),
        # Accès lecture au wiki citoyen (2026-07-26) : index (donnée statique, pas de coût réseau)
        # + callable pour la lecture d'une page précise à la demande (network call, jamais fait
        # d'office pour toutes les pages sur chaque tour — voir fetch_wiki_page_raw).
        "wiki_pages_index": WIKI_CITIZEN_PAGES,
        "get_wiki_page_fn": fetch_wiki_page_raw,
        # RAG conseil municipal (2026-07-26) : recherche sémantique à la demande, jamais fait
        # d'office (contrairement à ctx["threads"], pas une simple lecture DB — appel réseau vers
        # Qdrant + calcul d'embedding, voir search_conseil_municipal_pv).
        "search_conseil_municipal_fn": search_conseil_municipal_pv,
        # list_conseil_municipal_fn (2026-07-26, bug réel #15) : tri chronologique pur, distinct
        # de la recherche sémantique — voir list_conseil_municipal_meetings.
        "list_conseil_municipal_fn": list_conseil_municipal_meetings,
        # get_conseil_municipal_document_fn (2026-07-26, bug réel #16) : accès direct et complet à
        # UN document déjà identifié, sans recherche sémantique — voir get_conseil_municipal_document_pv.
        "get_conseil_municipal_document_fn": get_conseil_municipal_document_pv,
    }
    trace_requested = bool(req.admin_key) and req.admin_key == ADMIN_KEY
    result = run_turn(
        system_prompt, conversation_messages, ctx,
        trace=trace_requested, response_format=build_response_format(action_scope),
    )
    if result["error"] == "llm_indisponible":
        raise HTTPException(503, "Le chatbot est momentanément indisponible.")
    return result


@app.post("/pseudo/confirm")
def pseudo_confirm(req: PseudoConfirmRequest):
    """SEUL endpoint qui écrit un pseudo — jamais le LLM (voir chatbot_actions.
    propose_pseudo_candidates/get_or_assign_pseudo, tous deux lecture seule). Appelé uniquement
    par un clic utilisateur explicite sur une des propositions affichées."""
    identity_token = _require_identity(req.session_token)
    try:
        return confirm_pseudo(identity_token, req.word, req.color)
    except ValueError as e:
        status = 409 if "déjà pris" in str(e) else 400
        raise HTTPException(status, str(e))


@app.post("/pseudo/mine")
def pseudo_mine(req: PseudoMineRequest):
    """Lecture seule — le pseudo déjà confirmé de l'utilisateur connecté, ou None. Manquait
    jusqu'ici en dehors du contexte chatbot (2026-07-31, besoin réel : afficher/proposer un logo
    sur "Mon compte" pour un pseudo confirmé AVANT l'existence du pipeline de logos, ou pour un
    compte de test qui n'est jamais passé par le flux de confirmation avec preview)."""
    identity_token = _require_identity(req.session_token)
    pseudo = get_existing_pseudo(identity_token)
    if pseudo is None:
        return {"word": None, "color": None, "display": None}
    return {
        "word": pseudo["word"], "color": pseudo["color"],
        "display": _agree_pseudo_display(pseudo["word"], pseudo["color"]),
    }


@app.post("/pseudo/logo/preview")
def pseudo_logo_preview(req: PseudoLogoPreviewRequest):
    """Étape "valider ce logo" (2026-07-31, décision développeur via angelobot) : appelé quand
    l'utilisateur entre dans l'étape de double-confirmation d'un pseudo, POUR PRÉVISUALISER un
    logo silhouette avant qu'il soit permanent — jamais écrit en base/disque ici (voir
    /pseudo/logo/confirm pour le seul point d'écriture). Si le mot a déjà un logo validé par
    quelqu'un d'autre (mot déjà utilisé), le relit directement sans regénérer (gratuit, instantané).
    Sinon, génère synchrone (5-8s, ~0,014$) — jamais bloquant plus que ça, jamais en arrière-plan
    (décision développeur : difficile de valider quelque chose encore en cours de génération)."""
    _require_identity(req.session_token)
    word = req.word.strip()
    if not word:
        raise HTTPException(400, "mot manquant")
    existing_file = get_word_logo_file(word)
    if existing_file:
        with open(os.path.join(_PSEUDO_LOGO_DIR, existing_file), "rb") as f:
            png_bytes = f.read()
        return {
            "image_base64": base64.b64encode(png_bytes).decode(),
            "is_new": False,
        }
    png_bytes = pseudo_logo_gen.generate_word_silhouette(word)
    if png_bytes is None:
        # Dégradation propre (2026-07-31) : la génération d'image est un agrément, jamais un
        # blocage du parcours pseudo — is_new/image_base64 absents, le frontend retombe sur
        # l'affichage texte habituel pour ce mot, sans erreur bruyante pour l'utilisateur.
        return {"image_base64": None, "is_new": True}
    return {"image_base64": base64.b64encode(png_bytes).decode(), "is_new": True}


@app.post("/pseudo/logo/confirm")
def pseudo_logo_confirm(req: PseudoLogoConfirmRequest):
    """SEUL endpoint qui écrit un logo de mot — jamais le LLM, jamais automatique : appelé par le
    frontend juste après un /pseudo/confirm réussi, SEULEMENT si /pseudo/logo/preview avait
    renvoyé une image nouvelle pour ce mot (is_new=true). Idempotent (voir save_word_logo) : si le
    mot a déjà un logo au moment de l'appel (race entre 2 confirmations quasi simultanées), la
    2e écriture est silencieusement ignorée, le fichier déjà en place fait foi."""
    _require_identity(req.session_token)
    word = req.word.strip()
    if not word:
        raise HTTPException(400, "mot manquant")
    try:
        png_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(400, "image invalide")
    filename = save_word_logo(word, png_bytes)
    return {"word": word, "file": filename}


@app.post("/pseudo/logos")
def pseudo_logos(req: PseudoLogosRequest):
    """Table complète mot->fichier logo, lue une fois par le frontend au chargement (voir
    PSEUDO_WORD_LOGOS côté JS) — authentifiée comme le reste (session_token requis) mais sans lien
    avec l'identité de l'appelant, cette donnée est publique par nature (les logos sont affichés à
    tout le monde sur le Forum)."""
    _require_identity(req.session_token)
    return get_all_word_logos()


@app.post("/opinion/confirm")
def opinion_confirm(req: OpinionConfirmRequest):
    """SEUL endpoint qui écrit une opinion — jamais le LLM (voir chatbot_actions.propose_opinion,
    lecture/validation pure). Crée le brouillon ET le publie en un seul geste (contrairement au
    pseudo, une opinion n'a pas de phase "brouillon persistant consultable plus tard" côté produit
    — le brouillon décrit dans le wiki se passe entièrement EN CONVERSATION, pas en DB), déclenché
    uniquement par un clic utilisateur explicite.

    2 chemins exclusifs (2026-07-25, création de fil couplée) : soit thread_id (fil EXISTANT déjà
    publié, chemin historique), soit new_thread_title (nouveau fil créé dans le MÊME geste que
    cette opinion — voir create_thread_with_opinion pour le filet anti-race et l'auto-nettoyage)."""
    identity_token = _require_identity(req.session_token)
    if req.thread_id is not None and req.new_thread_title is not None:
        raise HTTPException(400, "précise soit thread_id, soit new_thread_title, jamais les deux")
    try:
        if req.new_thread_title is not None:
            return create_thread_with_opinion(
                req.new_thread_title, identity_token, req.body, req.argumentaire,
                req.new_thread_summary, req.new_thread_category, req.voice,
            )
        if req.thread_id is None:
            raise ValueError("il faut soit thread_id (fil existant), soit new_thread_title (nouveau fil)")
        opinion = create_opinion(req.thread_id, identity_token, req.body, req.argumentaire, req.voice)
        return publish_opinion(opinion["opinion_id"])
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/reaction/confirm")
def reaction_confirm(req: ReactionConfirmRequest):
    """Chemin LLM (voir chatbot_actions.propose_reaction, lecture/validation pure) — jamais
    d'écriture directe par le modèle. Même logique que /opinion/confirm : création + publication
    en un seul geste, déclenché uniquement par un clic utilisateur explicite. Utilisé quand la
    réaction porte un argumentaire en texte libre (voir /reaction/toggle pour le choix de stance
    seul, sans passer par le chat, 2026-07-30)."""
    identity_token = _require_identity(req.session_token)
    try:
        reaction = add_reaction(req.opinion_id, identity_token, req.stance, req.argumentaire, req.voice)
        return publish_reaction(reaction["reaction_id"])
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/reaction/toggle")
def reaction_toggle(req: ReactionToggleRequest):
    """Bouton direct "Réagir" (2026-07-30, décision développeur, voir set_reaction_stance) :
    ÉCRITURE DIRECTE sans passer par le chatbot ni par une double confirmation — justifié par le
    choix fermé parmi 3 options (adherer/opposer/neutre), sans texte libre, donc sans risque de
    contenu non désiré à modérer. Aucun argumentaire ici (voir /reaction/confirm pour l'ajouter
    via le chat, sur la MÊME opinion, une fois le stance déjà posé par bouton). Renvoie le stance
    résultant (None si la réaction vient d'être annulée) + les décomptes à jour de l'opinion, pour
    que le frontend puisse rafraîchir l'affichage sans requête supplémentaire."""
    identity_token = _require_identity(req.session_token)
    if req.stance not in ("adherer", "opposer", "neutre"):
        raise HTTPException(400, "stance invalide")
    try:
        result = set_reaction_stance(req.opinion_id, identity_token, req.stance, req.voice)
    except ValueError as e:
        raise HTTPException(400, str(e))
    with db() as conn:
        counts = _get_opinion_reaction_summary(conn, req.opinion_id)
    return {**result, **counts}


@app.post("/reaction/mine")
def reaction_mine(req: ReactionMineRequest):
    """Bouton direct "Réagir" (2026-07-30) : renvoie le stance actuel de l'utilisateur connecté
    sur cette opinion (ou None), pour pré-sélectionner/surligner le bon bouton à l'ouverture du
    panneau — jamais l'inverse (get_forum_page_snapshot/forum/snapshot restent identiques pour
    tout le monde, sans filtre par identité, voir leur docstring)."""
    identity_token = _require_identity(req.session_token)
    current = get_current_reaction(req.opinion_id, identity_token)
    return {"stance": current["stance"] if current else None}


@app.post("/remarque/confirm")
def remarque_confirm(req: RemarqueConfirmRequest):
    """SEUL endpoint qui écrit une remarque — jamais le LLM (voir chatbot_actions.
    propose_remarque, lecture/validation pure). Même logique : création + publication en un seul
    geste, déclenché uniquement par un clic utilisateur explicite."""
    identity_token = _require_identity(req.session_token)
    try:
        remarque = create_remarque(
            req.thread_id, identity_token, req.body, req.reply_to_remarque_id, req.reply_to_opinion_id,
            req.reply_to_reaction_id, req.voice,
        )
        return publish_remarque(remarque["remarque_id"])
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/forum/snapshot")
def forum_snapshot(req: ActivityMineRequest):
    """Page "Forum" (2026-07-26, priorité 1 du développeur) : lecture du contenu déjà public par
    construction (voir get_forum_page_snapshot), MAIS restreinte aux utilisateurs connectés
    (2026-07-26, demande directe d'Angelo, décision temporaire tant que les tests avec des comptes
    bidon continuent — objectif : éviter qu'un vrai visiteur public tombe dessus pendant la phase
    de test, réouverture prévue plus tard une fois la montée en charge stabilisée). POST +
    session_token (pas GET public), même pattern que /activity/mine — _require_identity ne sert
    qu'à vérifier la session, le contenu renvoyé est identique pour tout utilisateur connecté
    (pas de filtre par identité, contrairement à get_my_activity)."""
    _require_identity(req.session_token)
    return {"threads": get_forum_page_snapshot()}


@app.post("/activity/mine")
def activity_mine(req: ActivityMineRequest):
    """Page "Mon activité" (2026-07-26, priorité 1) : lecture authentifiée, réservée à
    l'utilisateur connecté sur SES propres opinions (voir get_my_activity — filtre déjà sur son
    propre debate_token, jamais un paramètre arbitraire côté client)."""
    identity_token = _require_identity(req.session_token)
    return get_my_activity(identity_token)


@app.post("/chat/summarize")
def chat_summarize(req: ChatSummarizeRequest):
    _require_identity(req.session_token)
    if not req.history:
        raise HTTPException(400, "Rien à résumer.")
    convo_text = "\n".join(f"{m.role}: {m.content}" for m in req.history)
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
    summary = call_chat_llm(messages)
    if summary is None:
        raise HTTPException(503, "Résumé momentanément indisponible.")
    # Le résumé n'est PAS sauvegardé ici : c'est une proposition, l'utilisateur doit encore la
    # valider (ou la modifier) avant tout appel à /chat/save-summary — voir principe de
    # minimisation par défaut sur le wiki.
    return {"summary": summary}


@app.post("/chat/save-summary")
def chat_save_summary(req: ChatSaveSummaryRequest):
    owner_token = _require_identity(req.session_token)
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO chat_summaries (owner_token, summary) VALUES (?, ?)",
            (owner_token, req.summary),
        )
    return {"ok": True, "id": cur.lastrowid}


@app.post("/chat/summaries")
def chat_list_summaries(req: ChatSummariesRequest):
    owner_token = _require_identity(req.session_token)
    with db() as conn:
        rows = conn.execute(
            "SELECT id, summary, created_at FROM chat_summaries WHERE owner_token=? ORDER BY created_at DESC",
            (owner_token,),
        ).fetchall()
    return {"summaries": [dict(r) for r in rows]}


@app.delete("/chat/summaries/{summary_id}")
def chat_delete_summary(summary_id: int, req: ChatDeleteSummaryRequest):
    owner_token = _require_identity(req.session_token)
    with db() as conn:
        cur = conn.execute(
            "DELETE FROM chat_summaries WHERE id=? AND owner_token=?", (summary_id, owner_token)
        )
    if cur.rowcount == 0:
        raise HTTPException(404, "Résumé introuvable.")
    return {"ok": True}


@app.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, request: Request):
    _enforce_rate_limit(request, "forgot-password", limit=5)
    # Réponse strictement identique que l'email existe ou non, et le token n'apparaît JAMAIS
    # dans la réponse HTTP (contrairement à l'ancienne version) : seul un envoi par email au
    # titulaire du compte donne accès au lien de réinitialisation.
    generic = {"message": "Si cet email existe, un lien de réinitialisation a été envoyé."}
    with db() as conn:
        row = conn.execute("SELECT token FROM identities WHERE email=?", (req.email,)).fetchone()
    if not row:
        return generic
    reset_token = secrets.token_urlsafe(32)
    expiry = time.time() + 3600
    with db() as conn:
        conn.execute("UPDATE identities SET reset_token=?, reset_token_expiry=? WHERE token=?", (reset_token, expiry, row["token"]))
    send_reset_email(req.email, reset_token)
    return generic


@app.post("/reset-password")
def reset_password(req: ResetPasswordRequest):
    with db() as conn:
        row = conn.execute(
            "SELECT token, reset_token_expiry FROM identities WHERE reset_token=?",
            (req.token,),
        ).fetchone()
    if not row:
        raise HTTPException(400, "Token invalide.")
    # reset_token_expiry est REAL sur une base fraîchement créée, mais TEXT sur une base migrée
    # via l'ALTER TABLE générique de init_db() (toutes les nouvelles colonnes y sont ajoutées en
    # TEXT) — cast explicite pour supporter les deux cas plutôt que de supposer un type.
    if time.time() > float(row["reset_token_expiry"] or 0):
        raise HTTPException(400, "Token expiré.")
    password_hash = hash_password(req.password)
    with db() as conn:
        conn.execute(
            "UPDATE identities SET password_hash=?, reset_token=NULL, reset_token_expiry=NULL WHERE token=?",
            (password_hash, row["token"]),
        )
    return {"ok": True}


@app.delete("/unsubscribe")
def unsubscribe(req: UnsubscribeRequest):
    with db() as conn:
        row = conn.execute(
            "SELECT token, password_hash FROM identities WHERE session_token=?",
            (req.session_token,),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Session invalide.")
    if not check_password(req.password, row["password_hash"]):
        raise HTTPException(401, "Mot de passe incorrect.")
    # Les votes ne sont volontairement PAS supprimés : vote_token n'a jamais été lié à
    # l'identité (voir compute_vote_token), et permettre de voter puis d'effacer son vote après
    # coup si le résultat déplaît casserait la fiabilité du décompte pour tout le monde.
    with db() as conn:
        conn.execute("DELETE FROM identities WHERE token=?", (row["token"],))
    return {"ok": True, "message": "Compte supprimé."}


@app.get("/questions")
def list_questions():
    with db() as conn:
        rows = conn.execute(
            "SELECT id, titre, description, options FROM questions WHERE active=1 ORDER BY id DESC"
        ).fetchall()
    return [
        {"id": r["id"], "titre": r["titre"], "description": r["description"], "options": json.loads(r["options"])}
        for r in rows
    ]


@app.post("/questions")
def create_question(q: NewQuestion):
    if q.admin_key != ADMIN_KEY:
        raise HTTPException(403, "clé admin invalide")
    # Faille A (revue Opus 04/08) : jeu de réponses fermé obligatoire — au moins 2 options non
    # vides et distinctes une fois nettoyées, sinon /vote n'aurait rien de solide à valider.
    cleaned_options = [o.strip() for o in q.options if o.strip()]
    if len(cleaned_options) < 2:
        raise HTTPException(400, "il faut au moins 2 options de réponse non vides")
    if len(set(cleaned_options)) != len(cleaned_options):
        raise HTTPException(400, "les options de réponse doivent être distinctes")
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO questions (titre, description, options) VALUES (?, ?, ?)",
            (q.titre, q.description, json.dumps(cleaned_options)),
        )
        return {"id": cur.lastrowid}


@app.patch("/questions/{question_id}")
def update_question(question_id: int, q: QuestionUpdate):
    if q.admin_key != ADMIN_KEY:
        raise HTTPException(403, "clé admin invalide")
    with db() as conn:
        cur = conn.execute(
            "UPDATE questions SET active=? WHERE id=?",
            (1 if q.active else 0, question_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "question introuvable")
    return {"ok": True}


@app.post("/vote")
def vote(v: Vote, request: Request):
    _enforce_rate_limit(request, "vote", limit=20)
    vote_token = compute_vote_token(v.token)
    with db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM identities WHERE token=?", (v.token,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, "jeton inconnu")
        # Failles A/B (revue Opus 04/08) : /vote ne vérifiait ni l'existence de la question (pas
        # de FK — foreign_keys off par défaut en SQLite) ni son statut actif, et acceptait
        # n'importe quel "choix" en texte libre. Un `question_id` inexistant ou une question
        # fermée était accepté silencieusement, restitué ensuite par /results.
        question = conn.execute(
            "SELECT active, options FROM questions WHERE id=?", (v.question_id,)
        ).fetchone()
        if not question:
            raise HTTPException(404, "question introuvable")
        if not question["active"]:
            raise HTTPException(400, "cette question n'est plus ouverte au vote")
        if v.choix not in json.loads(question["options"]):
            raise HTTPException(400, "choix invalide pour cette question")
        already = conn.execute(
            "SELECT 1 FROM votes WHERE vote_token=? AND question_id=?",
            (vote_token, v.question_id),
        ).fetchone()
        if already:
            raise HTTPException(409, "déjà voté sur cette question")
        conn.execute(
            "INSERT INTO votes (vote_token, question_id, choix) VALUES (?, ?, ?)",
            (vote_token, v.question_id, v.choix),
        )
    # vote_token sert de "reçu" : l'électeur peut revenir plus tard sur /results/{question_id}
    # et vérifier lui-même que la ligne portant ce jeton correspond bien à son choix. Le jeton
    # est dérivé (sha256(token+pepper)) et ne permet pas de remonter à l'identité — c'est le seul
    # moyen pour l'utilisateur de le connaître, il ne peut pas le recalculer côté client sans le
    # pepper (secret serveur).
    return {"ok": True, "vote_token": vote_token}


@app.get("/results/{question_id}")
def results(question_id: int):
    with db() as conn:
        rows = conn.execute(
            "SELECT choix, COUNT(*) as n FROM votes WHERE question_id=? GROUP BY choix",
            (question_id,),
        ).fetchall()
        # Liste complète (vote_token, choix), pas juste le total agrégé : vérifiabilité
        # individuelle (principe Helios) — n'importe qui peut recompter et un électeur peut
        # retrouver sa propre ligne via le vote_token reçu à l'issue de /vote. Aucun risque de
        # remonter à l'identité (le jeton ne permet pas ça, voir compute_vote_token) — MAIS
        # (revue Opus 04/08, faille C) vote_token est STABLE par identité pour TOUTES les
        # questions, donc quiconque recoupe plusieurs /results peut reconstruire le profil de
        # vote complet d'un électeur (même jeton d'une question à l'autre). Décision d'Angelo
        # (04/08) : garder le jeton global en connaissance de ce risque plutôt que dériver un
        # jeton distinct par question (ce qui casserait le "reçu unique" par personne) — pas un
        # oubli, un arbitrage assumé. Voir [[themes:anonymat]] pour la documentation complète.
        detail = conn.execute(
            "SELECT vote_token, choix FROM votes WHERE question_id=? ORDER BY vote_token",
            (question_id,),
        ).fetchall()
    return {
        "tally": {r["choix"]: r["n"] for r in rows},
        "votes": [{"vote_token": r["vote_token"], "choix": r["choix"]} for r in detail],
        "total": sum(r["n"] for r in rows),
    }


# Routes de "fallback SPA" : le routage entre pages se fait côté client (JS, history.pushState),
# mais une navigation directe ou un rafraîchissement sur /vote, /assistant, etc. doit quand même
# renvoyer l'appli (index.html), pas un 404 — StaticFiles seul ne sait servir que des fichiers
# qui existent réellement sur disque. /register et /reset-password ne sont jamais dans le menu
# (accessibles seulement via un lien reçu par email) mais ont besoin du même traitement.
_SPA_ROUTES = [
    "/", "/login", "/vote", "/assistant", "/parrainer", "/compte", "/register", "/reset-password",
    "/forum", "/mon-activite", "/admin/roles", "/publier",
]
for _route in _SPA_ROUTES:
    app.add_api_route(
        _route,
        lambda: FileResponse("static_files/index.html"),
        methods=["GET"],
        include_in_schema=False,
    )

app.mount("/static_files", StaticFiles(directory="static_files"), name="static")
app.mount("/pseudo_logos_data", StaticFiles(directory=_PSEUDO_LOGO_DIR), name="pseudo_logos_data")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)