"""Prototype de vote citoyen local pour Jouy (28).

Sépare identité (nom/adresse/email) et vote (jeton/choix) pour garantir
l'anonymat du vote tout en gardant une vérification de résidence déclarative.
"""
import hashlib
import json
import os
import secrets
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DB_PATH = os.environ.get("DB_PATH") or os.path.join(os.path.dirname(__file__), "vote.db")
ADMIN_KEY = os.environ.get("JOUY_ADMIN_KEY")
if not ADMIN_KEY:
    raise RuntimeError("La variable d'environnement JOUY_ADMIN_KEY doit être définie.")

JOUY_VOTE_PEPPER = os.environ.get("JOUY_VOTE_PEPPER")
if not JOUY_VOTE_PEPPER:
    raise RuntimeError("La variable d'environnement JOUY_VOTE_PEPPER doit être définie.")

# Envoi d'email (lien de réinitialisation de mot de passe) via l'API transactionnelle Brevo.
# BREVO_API_KEY absente = fonctionnalité désactivée proprement (pas de crash au démarrage,
# contrairement à ADMIN_KEY/PEPPER) : /forgot-password répond alors sans jamais rien envoyer
# ni révéler le token, cf. forgot_password() plus bas.
BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL", "noreply@jouyvote.fr")
SITE_URL = os.environ.get("SITE_URL", "https://jouyvote.fr")

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
CHAT_SYSTEM_PROMPT = (
    "Tu es l'assistant citoyen de Jouy Vote Citoyen, un outil de démocratie participative locale "
    "pour les habitants de Jouy (28). Tu aides les joviens à formuler clairement une opinion ou "
    "une doléance, sans jamais trahir le sens de ce qu'ils veulent dire — tu proposes une "
    "reformulation, tu ne publies jamais rien toi-même, c'est toujours la personne qui décide. "
    "Pour toute question factuelle sur les décisions ou comptes-rendus du conseil municipal : tu "
    "n'as PAS ENCORE accès à ces documents dans cette première version du site — dis-le "
    "clairement plutôt que d'inventer une réponse. Reste bref, concret, et dans le sujet de la "
    "vie municipale de Jouy."
)

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
    salt, h = stored.split(':', 1)
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100_000).hex() == h


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
        }
        for col, col_type in column_types.items():
            try:
                conn.execute(f"ALTER TABLE identities ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass
    _fix_reset_token_expiry_type()


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


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_token: str
    message: str
    history: list[ChatMessage] = []


class ChatSummarizeRequest(BaseModel):
    session_token: str
    history: list[ChatMessage]


class ChatSaveSummaryRequest(BaseModel):
    session_token: str
    summary: str


class ChatSummariesRequest(BaseModel):
    session_token: str


class ChatDeleteSummaryRequest(BaseModel):
    session_token: str


class LogoutRequest(BaseModel):
    session_token: str


class Vote(BaseModel):
    token: str
    question_id: int
    choix: str


class NewQuestion(BaseModel):
    admin_key: str
    titre: str
    description: str = ""


class QuestionUpdate(BaseModel):
    admin_key: str
    active: bool


def compute_identity_hash(nom: str, adresse: str) -> str:
    raw = f"{nom.strip().lower()}|{adresse.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def compute_vote_token(token: str) -> str:
    raw = f"{token}:{JOUY_VOTE_PEPPER}"
    return hashlib.sha256(raw.encode()).hexdigest()


def send_reset_email(to_email: str, reset_token: str) -> bool:
    """Envoie le lien de réinitialisation par email via l'API Brevo.

    Ne lève jamais : retourne False en cas d'échec (clé absente, erreur réseau/API), pour
    que /forgot-password ne révèle jamais si l'envoi a réussi (même comportement visible
    de l'extérieur que l'email existe ou non).
    """
    if not BREVO_API_KEY:
        return False
    reset_link = f"{SITE_URL}/?reset_token={reset_token}"
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


def send_referral_invite_email(to_email: str, referrer_nom: str, invite_token: str) -> bool:
    """Envoie le lien d'inscription pré-approuvé au filleul. Le lien EST le déclencheur
    d'inscription (pas une validation a posteriori d'un compte déjà créé) : le filleul clique,
    complète lui-même son inscription, ce qui prouve qu'il contrôle l'email et consent
    réellement. Ne lève jamais, retourne False en cas d'échec (mêmes garanties que
    send_reset_email)."""
    if not BREVO_API_KEY:
        return False
    invite_link = f"{SITE_URL}/?invite_token={invite_token}"
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


@app.get("/wiki-home-content")
def wiki_home_content():
    return {"html": fetch_wiki_home_content()}


@app.post("/register")
def register(r: Registration):
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
def login(req: LoginRequest):
    with db() as conn:
        row = conn.execute(
            "SELECT token, nom, password_hash FROM identities WHERE email=?",
            (req.email,),
        ).fetchone()
    if not row or not row["password_hash"]:
        raise HTTPException(401, "Email ou mot de passe incorrect.")
    if not check_password(req.password, row["password_hash"]):
        raise HTTPException(401, "Email ou mot de passe incorrect.")
    session_token = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute("UPDATE identities SET session_token=? WHERE token=?", (session_token, row["token"]))
    return {"session_token": session_token, "nom": row["nom"], "email": req.email}


@app.post("/logout")
def logout(req: LogoutRequest):
    with db() as conn:
        conn.execute("UPDATE identities SET session_token=NULL WHERE session_token=?", (req.session_token,))
    return {"ok": True}


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


def _require_identity(session_token: str) -> str:
    """Résout un session_token en token d'identité, ou lève 401. Factorisé car réutilisé par
    toutes les routes /chat/*, contrairement au reste de l'API qui a chacune sa propre requête
    (gardé identique ici pour ne pas dupliquer 5 fois la même vérification)."""
    with db() as conn:
        row = conn.execute("SELECT token FROM identities WHERE session_token=?", (session_token,)).fetchone()
    if not row:
        raise HTTPException(401, "Session invalide.")
    return row["token"]


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
def forgot_password(req: ForgotPasswordRequest):
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
            "SELECT id, titre, description FROM questions WHERE active=1 ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/questions")
def create_question(q: NewQuestion):
    if q.admin_key != ADMIN_KEY:
        raise HTTPException(403, "clé admin invalide")
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO questions (titre, description) VALUES (?, ?)",
            (q.titre, q.description),
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
def vote(v: Vote):
    vote_token = compute_vote_token(v.token)
    with db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM identities WHERE token=?", (v.token,)
        ).fetchone()
        if not exists:
            raise HTTPException(404, "jeton inconnu")
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
        # retrouver sa propre ligne via le vote_token reçu à l'issue de /vote. Aucun risque pour
        # l'anonymat : le jeton ne permet pas de remonter à l'identité (voir compute_vote_token).
        detail = conn.execute(
            "SELECT vote_token, choix FROM votes WHERE question_id=? ORDER BY vote_token",
            (question_id,),
        ).fetchall()
    return {
        "tally": {r["choix"]: r["n"] for r in rows},
        "votes": [{"vote_token": r["vote_token"], "choix": r["choix"]} for r in detail],
        "total": sum(r["n"] for r in rows),
    }


app.mount("/", StaticFiles(directory="static_files", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)