import os
import pytest
from httpx import ASGITransport, AsyncClient

os.environ["JOUY_ADMIN_KEY"] = "test-admin-key-42"
os.environ["JOUY_VOTE_PEPPER"] = "test-vote-pepper-42"

import main

main.DB_PATH = ":memory:"
main.init_db()

from main import app, compute_identity_hash, compute_vote_token

ADMIN_KEY = "test-admin-key-42"
PASSWORD = "test-password-42"


@pytest.fixture(autouse=True)
def reset_db():
    with main.db() as conn:
        conn.execute("DELETE FROM votes")
        conn.execute("DELETE FROM questions")
        conn.execute("DELETE FROM identities")


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def admin_question(client):
    resp = await client.post(
        "/questions",
        json={"admin_key": ADMIN_KEY, "titre": "Test Question"},
    )
    data = resp.json()
    return data["id"]


@pytest.fixture
async def registered_user(client):
    body = {"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice@test.fr", "password": PASSWORD}
    resp = await client.post("/register", json=body)
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
async def logged_in_user(client, registered_user):
    resp = await client.post("/login", json={"email": "alice@test.fr", "password": PASSWORD})
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.anyio
async def test_register_success(client):
    body = {"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice@test.fr", "password": PASSWORD}
    resp = await client.post("/register", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert "session_token" in data
    assert "message" in data


@pytest.mark.anyio
async def test_register_double_rejected(client):
    body = {"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice@test.fr", "password": PASSWORD}
    resp1 = await client.post("/register", json=body)
    assert resp1.status_code == 200
    resp2 = await client.post("/register", json=body)
    assert resp2.status_code == 409
    assert "déjà inscrite" in resp2.json().get("detail", "")


@pytest.mark.anyio
async def test_login_success(client, registered_user):
    resp = await client.post("/login", json={"email": "alice@test.fr", "password": PASSWORD})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_token" in data
    assert data["nom"] == "Alice"
    assert data["email"] == "alice@test.fr"


@pytest.mark.anyio
async def test_login_wrong_password(client, registered_user):
    resp = await client.post("/login", json={"email": "alice@test.fr", "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_login_unknown_email(client):
    resp = await client.post("/login", json={"email": "unknown@test.fr", "password": PASSWORD})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_logout(client, logged_in_user):
    session = logged_in_user["session_token"]
    resp = await client.post("/logout", json={"session_token": session})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@pytest.fixture
def captured_reset_email(monkeypatch):
    """Intercepte send_reset_email au lieu d'appeler Brevo : capture (email, token) envoyés."""
    calls = []

    def fake_send(to_email, reset_token):
        calls.append((to_email, reset_token))
        return True

    monkeypatch.setattr(main, "send_reset_email", fake_send)
    return calls


@pytest.mark.anyio
async def test_forgot_password_never_leaks_token_in_response(client, registered_user, captured_reset_email):
    resp = await client.post("/forgot-password", json={"email": "alice@test.fr"})
    assert resp.status_code == 200
    data = resp.json()
    assert "reset_token" not in data
    assert data["message"] is not None
    assert len(captured_reset_email) == 1
    assert captured_reset_email[0][0] == "alice@test.fr"


@pytest.mark.anyio
async def test_forgot_password_unknown_email(client, captured_reset_email):
    resp = await client.post("/forgot-password", json={"email": "unknown@test.fr"})
    assert resp.status_code == 200
    assert "Si cet email existe" in resp.json()["message"]
    assert captured_reset_email == []


@pytest.mark.anyio
async def test_reset_password(client, registered_user, captured_reset_email):
    forgot = await client.post("/forgot-password", json={"email": "alice@test.fr"})
    assert "reset_token" not in forgot.json()
    token = captured_reset_email[0][1]
    resp = await client.post("/reset-password", json={"token": token, "password": "new-password"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    login = await client.post("/login", json={"email": "alice@test.fr", "password": "new-password"})
    assert login.status_code == 200


@pytest.mark.anyio
async def test_reset_password_expired_token(client, registered_user):
    resp = await client.post("/reset-password", json={"token": "fake-token", "password": "new-password"})
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_send_reset_email_returns_false_without_brevo_key(monkeypatch):
    monkeypatch.setattr(main, "BREVO_API_KEY", None)
    assert main.send_reset_email("alice@test.fr", "some-token") is False


@pytest.mark.anyio
async def test_init_db_fixes_reset_token_expiry_type_and_keeps_data(tmp_path, monkeypatch):
    """Régression : sur la vraie base de prod (créée avant password_hash/session_token/
    reset_token/reset_token_expiry), ces colonnes avaient été ajoutées via l'ancienne boucle
    générique ALTER TABLE ADD COLUMN ... TEXT, qui typait TOUT en TEXT — y compris
    reset_token_expiry, qui doit être un nombre pour être comparé à time.time(). Une base
    fraîchement créée par CREATE TABLE (reset_token_expiry REAL) ne reproduisait pas ce bug :
    c'est pour ça que les tests sur schéma neuf passaient alors que la prod plantait avec un
    TypeError. init_db() reconstruit désormais la table si ce mauvais typage est détecté ; ce
    test reproduit l'état AVANT ce fix (colonne TEXT + une vraie ligne de données) et vérifie
    que init_db() corrige le type ET conserve les données existantes.
    """
    import sqlite3

    db_file = tmp_path / "migrated.db"
    conn = sqlite3.connect(str(db_file))
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
            reset_token_expiry TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        "INSERT INTO identities (token, identity_hash, nom, adresse, email, reset_token, reset_token_expiry) "
        "VALUES ('tok-1', 'hash-1', 'Carole', '3 Rue de la Mairie', 'carole@test.fr', 'rt-1', '9999999999.0')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(main, "DB_PATH", str(db_file))
    main.init_db()

    with main.db() as conn:
        col_type = next(
            row[2] for row in conn.execute("PRAGMA table_info(identities)") if row[1] == "reset_token_expiry"
        )
        row = conn.execute("SELECT nom, email, reset_token, reset_token_expiry FROM identities WHERE token='tok-1'").fetchone()
    assert col_type == "REAL"
    assert row["nom"] == "Carole"
    assert row["email"] == "carole@test.fr"
    assert row["reset_token"] == "rt-1"
    assert row["reset_token_expiry"] == 9999999999.0

    # Idempotence : relancer init_db() sur une base déjà corrigée ne doit rien casser.
    main.init_db()
    with main.db() as conn:
        row2 = conn.execute("SELECT reset_token_expiry FROM identities WHERE token='tok-1'").fetchone()
    assert row2["reset_token_expiry"] == 9999999999.0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as migrated_client:
        await migrated_client.post(
            "/register",
            json={"nom": "Bob", "adresse": "2 Rue de la Mairie", "email": "bob@test.fr", "password": PASSWORD},
        )
        calls = []
        monkeypatch.setattr(main, "send_reset_email", lambda email, token: calls.append((email, token)))
        await migrated_client.post("/forgot-password", json={"email": "bob@test.fr"})
        token = calls[0][1]
        resp = await migrated_client.post("/reset-password", json={"token": token, "password": "new-password"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


@pytest.mark.anyio
async def test_change_password_success(client, logged_in_user):
    session = logged_in_user["session_token"]
    resp = await client.post(
        "/change-password",
        json={"session_token": session, "current_password": PASSWORD, "new_password": "new-password-99"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    login = await client.post("/login", json={"email": "alice@test.fr", "password": "new-password-99"})
    assert login.status_code == 200
    old_login = await client.post("/login", json={"email": "alice@test.fr", "password": PASSWORD})
    assert old_login.status_code == 401


@pytest.mark.anyio
async def test_change_password_wrong_current_password(client, logged_in_user):
    session = logged_in_user["session_token"]
    resp = await client.post(
        "/change-password",
        json={"session_token": session, "current_password": "wrong", "new_password": "new-password-99"},
    )
    assert resp.status_code == 401
    login = await client.post("/login", json={"email": "alice@test.fr", "password": PASSWORD})
    assert login.status_code == 200  # mot de passe original toujours valide, rien changé


@pytest.mark.anyio
async def test_change_password_invalid_session(client):
    resp = await client.post(
        "/change-password",
        json={"session_token": "not-a-real-session", "current_password": "x", "new_password": "new-password-99"},
    )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_unsubscribe(client, registered_user, logged_in_user):
    session = logged_in_user["session_token"]
    resp = await client.request("DELETE", "/unsubscribe", json={"session_token": session, "password": PASSWORD})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


@pytest.mark.anyio
async def test_unsubscribe_wrong_password(client, logged_in_user):
    session = logged_in_user["session_token"]
    resp = await client.request("DELETE", "/unsubscribe", json={"session_token": session, "password": "wrong"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_unsubscribe_invalid_session(client, registered_user):
    resp = await client.request("DELETE", "/unsubscribe", json={"session_token": "fake", "password": PASSWORD})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_double_registration_rejected(client):
    body = {"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice@test.fr", "password": PASSWORD}
    body2 = {"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice2@test.fr", "password": PASSWORD}
    resp1 = await client.post("/register", json=body)
    assert resp1.status_code == 200
    resp2 = await client.post("/register", json=body2)
    assert resp2.status_code == 409


@pytest.mark.anyio
async def test_double_registration_case_insensitive(client):
    body1 = {"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice@test.fr", "password": PASSWORD}
    body2 = {"nom": "  alice  ", "adresse": "  1 rue de la mairie  ", "email": "alice2@test.fr", "password": PASSWORD}
    resp1 = await client.post("/register", json=body1)
    assert resp1.status_code == 200
    resp2 = await client.post("/register", json=body2)
    assert resp2.status_code == 409


@pytest.mark.anyio
async def test_unknown_token_rejected_at_vote(client, admin_question):
    qid = admin_question
    resp = await client.post(
        "/vote",
        json={"token": "fake-token", "question_id": qid, "choix": "Oui"},
    )
    assert resp.status_code == 404
    detail = resp.json().get("detail", "")
    assert "jeton inconnu" in detail


@pytest.mark.anyio
async def test_double_vote_rejected(client, admin_question):
    qid = admin_question
    reg = await client.post(
        "/register", json={"nom": "Bob", "adresse": "2 Rue des Lys", "email": "bob@test.fr", "password": PASSWORD}
    )
    token = reg.json()["token"]
    resp1 = await client.post(
        "/vote", json={"token": token, "question_id": qid, "choix": "Oui"}
    )
    assert resp1.status_code == 200
    resp2 = await client.post(
        "/vote", json={"token": token, "question_id": qid, "choix": "Non"}
    )
    assert resp2.status_code == 409
    detail = resp2.json().get("detail", "")
    assert "déjà voté" in detail


@pytest.mark.anyio
async def test_full_workflow(client, admin_question):
    qid = admin_question

    reg = await client.post(
        "/register", json={"nom": "Charlie", "adresse": "3 Place de l'Église", "email": "charlie@test.fr", "password": PASSWORD}
    )
    assert reg.status_code == 200
    token = reg.json()["token"]

    vote = await client.post(
        "/vote", json={"token": token, "question_id": qid, "choix": "Oui"}
    )
    assert vote.status_code == 200

    results = await client.get(f"/results/{qid}")
    assert results.status_code == 200
    data = results.json()
    assert data["tally"] == {"Oui": 1}
    assert data["total"] == 1
    expected_vote_token = compute_vote_token(token)
    assert expected_vote_token in data["voted_tokens"]


@pytest.mark.anyio
async def test_join_does_not_link_identity_to_vote(client, admin_question):
    qid = admin_question
    reg = await client.post(
        "/register", json={"nom": "Denis", "adresse": "4 Rue du Secret", "email": "denis@test.fr", "password": PASSWORD}
    )
    assert reg.status_code == 200
    token = reg.json()["token"]

    vote_resp = await client.post(
        "/vote", json={"token": token, "question_id": qid, "choix": "Non"}
    )
    assert vote_resp.status_code == 200

    with main.db() as conn:
        rows = conn.execute(
            "SELECT * FROM identities JOIN votes ON identities.token=votes.vote_token"
        ).fetchall()
    assert len(rows) == 0, (
        "Un JOIN direct identities.token=votes.vote_token ne devrait jamais "
        "renvoyer de ligne, sinon l'anonymat est cassé."
    )


@pytest.mark.anyio
async def test_create_question_admin_key_accept(client):
    resp = await client.post(
        "/questions",
        json={"admin_key": ADMIN_KEY, "titre": "Admin Question"},
    )
    assert resp.status_code == 200
    assert "id" in resp.json()


@pytest.mark.anyio
async def test_create_question_admin_key_reject(client):
    resp = await client.post(
        "/questions",
        json={"admin_key": "wrong-key", "titre": "Should not appear"},
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_patch_question_deactivate(client):
    resp = await client.post(
        "/questions",
        json={"admin_key": ADMIN_KEY, "titre": "To deactivate"},
    )
    qid = resp.json()["id"]

    patch_resp = await client.patch(
        f"/questions/{qid}",
        json={"admin_key": ADMIN_KEY, "active": False},
    )
    assert patch_resp.status_code == 200

    questions = await client.get("/questions")
    ids = [q["id"] for q in questions.json()]
    assert qid not in ids


@pytest.mark.anyio
async def test_admin_key_not_set_prevents_start():
    saved = os.environ.pop("JOUY_ADMIN_KEY", None)
    try:
        import importlib
        import sys

        with pytest.raises(RuntimeError, match="JOUY_ADMIN_KEY"):
            importlib.reload(main)
    finally:
        if saved:
            os.environ["JOUY_ADMIN_KEY"] = saved