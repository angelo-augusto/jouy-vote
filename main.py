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


@app.post("/register")
def register(r: Registration):
    identity_hash = compute_identity_hash(r.nom, r.adresse)
    token = secrets.token_urlsafe(32)
    session_token = secrets.token_urlsafe(32)
    password_hash = hash_password(r.password)
    with db() as conn:
        try:
            conn.execute(
                "INSERT INTO identities (token, identity_hash, nom, adresse, email, password_hash, session_token) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (token, identity_hash, r.nom.strip(), r.adresse.strip(), r.email, password_hash, session_token),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Cette identité (nom + adresse) est déjà inscrite.")
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
    vote_token = compute_vote_token(row["token"])
    with db() as conn:
        conn.execute("DELETE FROM votes WHERE vote_token=?", (vote_token,))
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
    return {"ok": True}


@app.get("/results/{question_id}")
def results(question_id: int):
    with db() as conn:
        rows = conn.execute(
            "SELECT choix, COUNT(*) as n FROM votes WHERE question_id=? GROUP BY choix",
            (question_id,),
        ).fetchall()
        tokens = conn.execute(
            "SELECT vote_token FROM votes WHERE question_id=? ORDER BY vote_token",
            (question_id,),
        ).fetchall()
    return {
        "tally": {r["choix"]: r["n"] for r in rows},
        "voted_tokens": [t["vote_token"] for t in tokens],
        "total": sum(r["n"] for r in rows),
    }


app.mount("/", StaticFiles(directory="static_files", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)