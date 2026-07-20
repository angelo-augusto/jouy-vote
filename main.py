"""Prototype de vote citoyen local pour Jouy (28).

Sépare identité (nom/adresse/email) et vote (jeton/choix) pour garantir
l'anonymat du vote tout en gardant une vérification de résidence déclarative.
"""
import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# DB_PATH surchargeable par variable d'environnement pour la containerisation (le conteneur de
# production monte un volume dédié à /data, séparé du code de l'appli — la base ne doit jamais
# vivre dans l'image elle-même, perdue sinon à chaque redéploiement).
DB_PATH = os.environ.get("DB_PATH") or os.path.join(os.path.dirname(__file__), "vote.db")
ADMIN_KEY = os.environ.get("JOUY_ADMIN_KEY")
if not ADMIN_KEY:
    raise RuntimeError(
        "La variable d'environnement JOUY_ADMIN_KEY doit être définie."
    )

JOUY_VOTE_PEPPER = os.environ.get("JOUY_VOTE_PEPPER")
if not JOUY_VOTE_PEPPER:
    raise RuntimeError(
        "La variable d'environnement JOUY_VOTE_PEPPER doit être définie."
    )

# Persistent connection to keep shared in-memory database alive
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


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS identities (
                token TEXT PRIMARY KEY,
                identity_hash TEXT UNIQUE NOT NULL,
                nom TEXT NOT NULL,
                adresse TEXT NOT NULL,
                email TEXT,
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


init_db()


class Registration(BaseModel):
    nom: str
    adresse: str
    email: str | None = None


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


@app.post("/register")
def register(r: Registration):
    identity_hash = compute_identity_hash(r.nom, r.adresse)
    token = secrets.token_urlsafe(32)
    with db() as conn:
        try:
            conn.execute(
                "INSERT INTO identities (token, identity_hash, nom, adresse, email) VALUES (?, ?, ?, ?, ?)",
                (token, identity_hash, r.nom.strip(), r.adresse.strip(), r.email),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Cette identité (nom + adresse) est déjà inscrite.")
    return {"token": token}


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


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)