import os
import pytest
from httpx import ASGITransport, AsyncClient

os.environ["JOUY_ADMIN_KEY"] = "test-admin-key-42"
os.environ["JOUY_VOTE_PEPPER"] = "test-vote-pepper-42"
os.environ["JOUY_PSEUDO_PEPPER"] = "test-pseudo-pepper-42"

import main

main.DB_PATH = ":memory:"
main.init_db()
# Les inscriptions sont fermées par défaut en prod (coupe-circuit anti-Sybil, cf REGISTRATIONS_OPEN
# dans main.py) — la plupart des tests ont besoin de /register fonctionnel, donc rouvert ici. Le
# comportement "fermé" a son propre test dédié (test_register_closed_by_default) qui remet le
# flag à False ponctuellement.
main.REGISTRATIONS_OPEN = True

from main import app, compute_identity_hash, compute_vote_token

ADMIN_KEY = "test-admin-key-42"
PASSWORD = "test-password-42"


@pytest.fixture(autouse=True)
def reset_db():
    with main.db() as conn:
        conn.execute("DELETE FROM votes")
        conn.execute("DELETE FROM questions")
        conn.execute("DELETE FROM identities")
        conn.execute("DELETE FROM pseudos")


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
async def test_register_closed_by_default(client, monkeypatch):
    monkeypatch.setattr(main, "REGISTRATIONS_OPEN", False)
    body = {"nom": "Zoe", "adresse": "9 Rue Nouvelle", "email": "zoe@test.fr", "password": PASSWORD}
    resp = await client.post("/register", json=body)
    assert resp.status_code == 403
    assert "parrainage" in resp.json()["detail"].lower()
    # aucun compte créé malgré la tentative
    login = await client.post("/login", json={"email": "zoe@test.fr", "password": PASSWORD})
    assert login.status_code == 401


@pytest.mark.anyio
async def test_login_success(client, registered_user):
    resp = await client.post("/login", json={"email": "alice@test.fr", "password": PASSWORD})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_token" in data
    assert data["nom"] == "Alice"
    assert data["email"] == "alice@test.fr"
    # Nécessaire pour que la page Voter fonctionne après une reconnexion (pas seulement juste
    # après l'inscription dans le même navigateur) : /login doit aussi renvoyer le jeton
    # d'identité, déjà renvoyé une fois à l'inscription.
    assert data["token"] == registered_user["token"]


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


def test_fetch_wiki_home_content_extracts_fragment_and_rewrites_links(monkeypatch):
    fake_doc = (
        "<!DOCTYPE html><html><head><title>start</title></head><body>"
        '<div class="dokuwiki export">\n'
        '<h1>Jouy Vote</h1>\n<a href="/genese">Genèse</a>\n'
        "</div>\n</body></html>"
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return fake_doc.encode()

    monkeypatch.setattr(main.urllib.request, "urlopen", lambda url, timeout=5: FakeResponse())
    html = main.fetch_wiki_home_content()
    assert "<h1>Jouy Vote</h1>" in html
    assert 'href="https://wiki.jouyvote.fr/genese"' in html
    assert "</body>" not in html
    assert "<head>" not in html


def test_fetch_wiki_home_content_returns_empty_on_network_error(monkeypatch):
    def raise_error(url, timeout=5):
        raise main.urllib.error.URLError("boom")

    monkeypatch.setattr(main.urllib.request, "urlopen", raise_error)
    assert main.fetch_wiki_home_content() == ""


@pytest.mark.anyio
async def test_wiki_home_content_endpoint(client, monkeypatch):
    monkeypatch.setattr(main, "fetch_wiki_home_content", lambda: "<p>contenu</p>")
    resp = await client.get("/wiki-home-content")
    assert resp.status_code == 200
    assert resp.json() == {"html": "<p>contenu</p>"}


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
async def test_unsubscribe_keeps_votes(client, admin_question, registered_user, logged_in_user):
    """Les votes déjà exprimés restent comptabilisés après suppression du compte — sinon on
    pourrait voter puis effacer son vote après coup si le résultat déplaît, ce qui casserait la
    fiabilité du décompte pour tout le monde. vote_token n'a jamais été lié à l'identité, donc
    rien n'est perdu en anonymat en les gardant."""
    qid = admin_question
    token = registered_user["token"]
    vote_resp = await client.post("/vote", json={"token": token, "question_id": qid, "choix": "Oui"})
    assert vote_resp.status_code == 200
    vote_token = vote_resp.json()["vote_token"]

    session = logged_in_user["session_token"]
    resp = await client.request("DELETE", "/unsubscribe", json={"session_token": session, "password": PASSWORD})
    assert resp.status_code == 200

    results = await client.get(f"/results/{qid}")
    assert {"vote_token": vote_token, "choix": "Oui"} in results.json()["votes"]


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
    vote_token = vote.json()["vote_token"]
    assert vote_token == compute_vote_token(token)

    results = await client.get(f"/results/{qid}")
    assert results.status_code == 200
    data = results.json()
    assert data["tally"] == {"Oui": 1}
    assert data["total"] == 1
    # Vérifiabilité individuelle (Helios) : l'électeur retrouve sa ligne exacte (jeton + choix)
    # dans les résultats publics grâce au reçu renvoyé par /vote, pas juste un total agrégé.
    assert {"vote_token": vote_token, "choix": "Oui"} in data["votes"]


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


@pytest.fixture
def captured_referral_email(monkeypatch):
    """Intercepte send_referral_invite_email au lieu d'appeler Brevo : capture
    (email, referrer_nom, invite_token) envoyés."""
    calls = []

    def fake_send(to_email, referrer_nom, invite_token):
        calls.append((to_email, referrer_nom, invite_token))
        return True

    monkeypatch.setattr(main, "send_referral_invite_email", fake_send)
    return calls


@pytest.mark.anyio
async def test_referral_invite_requires_confirmation(client, logged_in_user):
    resp = await client.post(
        "/referral/invite",
        json={
            "session_token": logged_in_user["session_token"],
            "invitee_email": "filleul@test.fr",
            "confirms_residency_and_age": False,
        },
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_referral_invite_requires_valid_session(client):
    resp = await client.post(
        "/referral/invite",
        json={
            "session_token": "fake-session",
            "invitee_email": "filleul@test.fr",
            "confirms_residency_and_age": True,
        },
    )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_referral_invite_success_sends_email(client, logged_in_user, captured_referral_email):
    resp = await client.post(
        "/referral/invite",
        json={
            "session_token": logged_in_user["session_token"],
            "invitee_email": "Filleul@Test.fr",
            "confirms_residency_and_age": True,
        },
    )
    assert resp.status_code == 200
    assert len(captured_referral_email) == 1
    email, referrer_nom, invite_token = captured_referral_email[0]
    assert email == "filleul@test.fr"  # normalisé en minuscules
    assert referrer_nom == "Alice"
    assert invite_token


@pytest.mark.anyio
async def test_referral_invite_info_resolves_referrer_name(client, logged_in_user, captured_referral_email):
    await client.post(
        "/referral/invite",
        json={
            "session_token": logged_in_user["session_token"],
            "invitee_email": "filleul@test.fr",
            "confirms_residency_and_age": True,
        },
    )
    invite_token = captured_referral_email[0][2]
    resp = await client.get(f"/referral/invite/{invite_token}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["referrer_nom"] == "Alice"
    assert data["invitee_email"] == "filleul@test.fr"


@pytest.mark.anyio
async def test_referral_invite_info_unknown_token(client):
    resp = await client.get("/referral/invite/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_register_with_valid_invite_succeeds(client, logged_in_user, captured_referral_email, monkeypatch):
    monkeypatch.setattr(main, "REGISTRATIONS_OPEN", False)
    await client.post(
        "/referral/invite",
        json={
            "session_token": logged_in_user["session_token"],
            "invitee_email": "filleul@test.fr",
            "confirms_residency_and_age": True,
        },
    )
    invite_token = captured_referral_email[0][2]

    resp = await client.post(
        "/register",
        json={
            "nom": "Filleul",
            "adresse": "5 Rue du Filleul",
            "email": "filleul@test.fr",
            "password": PASSWORD,
            "invite_token": invite_token,
        },
    )
    assert resp.status_code == 200

    # l'invitation est marquée utilisée, réutilisation refusée
    resp2 = await client.post(
        "/register",
        json={
            "nom": "Filleul Bis",
            "adresse": "6 Rue du Filleul",
            "email": "filleul@test.fr",
            "password": PASSWORD,
            "invite_token": invite_token,
        },
    )
    assert resp2.status_code == 403


@pytest.mark.anyio
async def test_register_with_invite_email_mismatch_rejected(client, logged_in_user, captured_referral_email, monkeypatch):
    monkeypatch.setattr(main, "REGISTRATIONS_OPEN", False)
    await client.post(
        "/referral/invite",
        json={
            "session_token": logged_in_user["session_token"],
            "invitee_email": "filleul@test.fr",
            "confirms_residency_and_age": True,
        },
    )
    invite_token = captured_referral_email[0][2]

    resp = await client.post(
        "/register",
        json={
            "nom": "Imposteur",
            "adresse": "7 Rue Suspecte",
            "email": "autre-email@test.fr",
            "password": PASSWORD,
            "invite_token": invite_token,
        },
    )
    assert resp.status_code == 403
    assert "correspond" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_referral_max_five_enforced(client, logged_in_user, captured_referral_email, monkeypatch):
    monkeypatch.setattr(main, "REGISTRATIONS_OPEN", False)
    session = logged_in_user["session_token"]

    for i in range(main.REFERRAL_MAX):
        await client.post(
            "/referral/invite",
            json={
                "session_token": session,
                "invitee_email": f"filleul{i}@test.fr",
                "confirms_residency_and_age": True,
            },
        )
        invite_token = captured_referral_email[i][2]
        resp = await client.post(
            "/register",
            json={
                "nom": f"Filleul {i}",
                "adresse": f"{i} Rue du Filleul",
                "email": f"filleul{i}@test.fr",
                "password": PASSWORD,
                "invite_token": invite_token,
            },
        )
        assert resp.status_code == 200

    # le quota est atteint : création d'une 6e invitation refusée
    resp = await client.post(
        "/referral/invite",
        json={
            "session_token": session,
            "invitee_email": "filleul-refuse@test.fr",
            "confirms_residency_and_age": True,
        },
    )
    assert resp.status_code == 403
    assert "quota" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_referral_status_reports_quota_and_invites(client, logged_in_user, captured_referral_email):
    session = logged_in_user["session_token"]
    await client.post(
        "/referral/invite",
        json={"session_token": session, "invitee_email": "filleul@test.fr", "confirms_residency_and_age": True},
    )
    resp = await client.post("/referral/status", json={"session_token": session})
    assert resp.status_code == 200
    data = resp.json()
    assert data["used"] == 0  # invitation envoyée mais pas encore inscrite
    assert data["remaining"] == main.REFERRAL_MAX
    assert data["max"] == main.REFERRAL_MAX
    assert data["invites"] == [{"email": "filleul@test.fr", "used": False}]


@pytest.mark.anyio
async def test_referral_status_invalid_session(client):
    resp = await client.post("/referral/status", json={"session_token": "fake"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_my_vote_token_matches_results(client, admin_question, registered_user, logged_in_user):
    qid = admin_question
    token = registered_user["token"]
    vote_resp = await client.post("/vote", json={"token": token, "question_id": qid, "choix": "Oui"})
    expected_vote_token = vote_resp.json()["vote_token"]

    resp = await client.post("/my-vote-token", json={"session_token": logged_in_user["session_token"]})
    assert resp.status_code == 200
    assert resp.json()["vote_token"] == expected_vote_token

    # même valeur retrouvée publiquement dans /results, à côté du choix
    results = await client.get(f"/results/{qid}")
    assert {"vote_token": expected_vote_token, "choix": "Oui"} in results.json()["votes"]


@pytest.mark.anyio
async def test_my_vote_token_invalid_session(client):
    resp = await client.post("/my-vote-token", json={"session_token": "fake"})
    assert resp.status_code == 401


@pytest.fixture
def mocked_chat_llm(monkeypatch):
    """Intercepte call_chat_llm au lieu d'appeler OpenRouter : capture les messages envoyés,
    retourne une réponse fixe."""
    calls = []

    def fake_call(messages):
        calls.append(messages)
        return "réponse simulée du chatbot"

    monkeypatch.setattr(main, "call_chat_llm", fake_call)
    return calls


@pytest.mark.anyio
async def test_chat_requires_valid_session(client):
    resp = await client.post("/chat", json={"session_token": "fake", "message": "bonjour"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_chat_success(client, logged_in_user, mocked_chat_llm):
    resp = await client.post(
        "/chat",
        json={"session_token": logged_in_user["session_token"], "message": "bonjour"},
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "réponse simulée du chatbot"
    sent_messages = mocked_chat_llm[0]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[-1] == {"role": "user", "content": "bonjour"}


@pytest.mark.anyio
async def test_chat_includes_history(client, logged_in_user, mocked_chat_llm):
    history = [{"role": "user", "content": "premier message"}, {"role": "assistant", "content": "première réponse"}]
    await client.post(
        "/chat",
        json={"session_token": logged_in_user["session_token"], "message": "deuxième message", "history": history},
    )
    sent_messages = mocked_chat_llm[0]
    assert {"role": "user", "content": "premier message"} in sent_messages
    assert {"role": "assistant", "content": "première réponse"} in sent_messages


@pytest.mark.anyio
async def test_chat_llm_unavailable_returns_503(client, logged_in_user, monkeypatch):
    monkeypatch.setattr(main, "call_chat_llm", lambda messages: None)
    resp = await client.post(
        "/chat", json={"session_token": logged_in_user["session_token"], "message": "bonjour"}
    )
    assert resp.status_code == 503


def test_chatbot_actions_registry_excludes_commit_actions():
    """Barrière structurelle (revue Opus 2026-07-25, POC tool-calling) : les actions de commit
    ne doivent JAMAIS être dans le registre appelable par le LLM, quel que soit ce qu'un prompt
    pourrait dire — c'est le dict ACTIONS qui fait foi, pas une consigne."""
    import chatbot_actions

    assert set(chatbot_actions.ACTIONS.keys()) == {
        "say_user", "get_vote_token", "propose_summary", "list_summaries",
        "get_or_assign_pseudo", "propose_pseudo_candidates",
    }
    for forbidden in ("save_summary", "delete_summary", "confirm_publication", "confirm_pseudo"):
        assert forbidden not in chatbot_actions.ACTIONS


def test_get_vote_token_action_matches_compute_vote_token():
    import chatbot_actions

    result = chatbot_actions.get_vote_token({}, {"identity_token": "un-token-de-test"})
    assert result["vote_token"] == chatbot_actions.compute_vote_token("un-token-de-test")


def test_debate_token_distinct_from_vote_token_for_same_identity():
    """Pepper dédié (revue de sécurité 2026-07-25) : même identity_token, deux dérivations
    totalement indépendantes — condition nécessaire pour qu'un admin connaissant un pepper ne
    puisse pas en déduire l'autre jeton."""
    import chatbot_actions

    identity_token = "un-token-de-test"
    assert chatbot_actions.compute_debate_token(identity_token) != chatbot_actions.compute_vote_token(identity_token)


def test_debate_token_deterministic():
    import chatbot_actions

    a = chatbot_actions.compute_debate_token("token-x")
    b = chatbot_actions.compute_debate_token("token-x")
    assert a == b
    assert a != chatbot_actions.compute_debate_token("token-y")


def test_derive_pseudo_deterministic_and_valid():
    import chatbot_actions

    debate_token = chatbot_actions.compute_debate_token("token-x")
    pseudo_a = chatbot_actions.derive_pseudo(debate_token)
    pseudo_b = chatbot_actions.derive_pseudo(debate_token)
    assert pseudo_a == pseudo_b
    assert pseudo_a["word"] in chatbot_actions.PSEUDO_WORDS
    assert pseudo_a["color"] in chatbot_actions.PSEUDO_COLORS


def test_get_or_assign_pseudo_action_reads_from_ctx():
    import chatbot_actions

    result = chatbot_actions.get_or_assign_pseudo({}, {"pseudo": {"word": "Renard", "color": "bleu"}})
    assert result == {"word": "Renard", "color": "bleu"}


def test_get_or_assign_pseudo_action_errors_without_ctx():
    import chatbot_actions

    result = chatbot_actions.get_or_assign_pseudo({}, {})
    assert "error" in result


def test_generate_pseudo_candidates_deterministic_and_distinct():
    """Même identity_token → toujours les mêmes propositions (pas de tirage aléatoire à chaque
    appel), et toutes distinctes entre elles au sein d'un même lot."""
    import chatbot_actions

    a = chatbot_actions.generate_pseudo_candidates("identity-x", n=3)
    b = chatbot_actions.generate_pseudo_candidates("identity-x", n=3)
    assert a == b
    assert len(a) == 3
    assert len({(c["word"], c["color"]) for c in a}) == 3
    for c in a:
        assert c["word"] in chatbot_actions.PSEUDO_WORDS
        assert c["color"] in chatbot_actions.PSEUDO_COLORS


def test_propose_pseudo_candidates_action_reads_identity_from_ctx():
    import chatbot_actions

    result = chatbot_actions.propose_pseudo_candidates({}, {"identity_token": "identity-x"})
    assert result["candidates"] == chatbot_actions.generate_pseudo_candidates("identity-x")


def test_propose_pseudo_candidates_action_errors_without_identity():
    import chatbot_actions

    result = chatbot_actions.propose_pseudo_candidates({}, {})
    assert "error" in result


def test_get_existing_pseudo_none_then_set_after_confirm():
    """Aucune écriture automatique (contrairement à l'ancien ensure_pseudo) : None tant que
    confirm_pseudo n'a pas été appelé explicitement."""
    import main as main_module

    identity_token = "identity-pour-pseudo-test"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    assert main_module.get_existing_pseudo(identity_token) is None

    candidate = main_module.generate_pseudo_candidates(identity_token)[0]
    confirmed = main_module.confirm_pseudo(identity_token, candidate["word"], candidate["color"])
    assert confirmed == candidate
    assert main_module.get_existing_pseudo(identity_token) == candidate

    with main.db() as conn:
        rows = conn.execute("SELECT * FROM pseudos WHERE debate_token=?", (debate_token,)).fetchall()
    assert len(rows) == 1
    # La table ne doit JAMAIS contenir le jeton d'identité en clair, seulement le hash dérivé.
    assert rows[0]["debate_token"] != identity_token

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))


def test_confirm_pseudo_rejects_value_outside_candidates():
    import main as main_module

    identity_token = "identity-pour-pseudo-test-2"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    with pytest.raises(ValueError, match="ne fait pas partie"):
        main_module.confirm_pseudo(identity_token, "PseudoInventé", "rose-fluo")


def test_confirm_pseudo_rejects_second_confirmation():
    import main as main_module

    identity_token = "identity-pour-pseudo-test-3"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    candidates = main_module.generate_pseudo_candidates(identity_token)
    main_module.confirm_pseudo(identity_token, candidates[0]["word"], candidates[0]["color"])
    with pytest.raises(ValueError, match="déjà attribué"):
        main_module.confirm_pseudo(identity_token, candidates[1]["word"], candidates[1]["color"])

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))


@pytest.mark.anyio
async def test_pseudo_confirm_endpoint_requires_valid_session(client):
    resp = await client.post("/pseudo/confirm", json={"session_token": "fake", "word": "Renard", "color": "bleu"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_pseudo_confirm_endpoint_rejects_invalid_candidate(client, logged_in_user):
    resp = await client.post(
        "/pseudo/confirm",
        json={"session_token": logged_in_user["session_token"], "word": "Inventé", "color": "rose-fluo"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_pseudo_confirm_endpoint_succeeds_then_rejects_second_attempt(client, logged_in_user):
    import main as main_module

    identity_token = logged_in_user["token"]
    candidates = main_module.generate_pseudo_candidates(identity_token)
    resp1 = await client.post(
        "/pseudo/confirm",
        json={"session_token": logged_in_user["session_token"], "word": candidates[0]["word"], "color": candidates[0]["color"]},
    )
    assert resp1.status_code == 200
    assert resp1.json() == candidates[0]

    resp2 = await client.post(
        "/pseudo/confirm",
        json={"session_token": logged_in_user["session_token"], "word": candidates[1]["word"], "color": candidates[1]["color"]},
    )
    assert resp2.status_code == 409


@pytest.mark.anyio
async def test_chat_v2_injects_onboarding_block_until_pseudo_confirmed(client, logged_in_user, monkeypatch):
    """Signal 'Nouveau, sans pseudo' : le bloc onboarding doit apparaître dans le system_prompt
    tant qu'aucun pseudo n'est confirmé, et disparaître juste après confirmation — sur autant de
    tours que nécessaire avant, pas seulement le tout premier appel."""
    import main as main_module

    captured_prompts = []

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5):
        captured_prompts.append(system_prompt)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)

    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut"})
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "encore avant de choisir"})
    assert "Premier contact" in captured_prompts[0]
    assert "Premier contact" in captured_prompts[1]  # toujours actif au 2e tour, pas juste le 1er

    identity_token = logged_in_user["token"]
    candidates = main_module.generate_pseudo_candidates(identity_token)
    main_module.confirm_pseudo(identity_token, candidates[0]["word"], candidates[0]["color"])

    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut à nouveau"})
    assert "Premier contact" not in captured_prompts[2]


@pytest.fixture
def mocked_openrouter_structured(monkeypatch):
    """Intercepte chatbot_llm.call_openrouter tel qu'importé dans chatbot_executor — capture les
    appels, retourne des réponses JSON pré-scriptées dans l'ordre fourni par le test."""
    import chatbot_executor

    calls = []
    responses = []

    def fake_call(messages, response_format=None, max_tokens=4096, model=None):
        calls.append(messages)
        return responses.pop(0), {}

    monkeypatch.setattr(chatbot_executor, "call_openrouter", fake_call)
    return calls, responses


def test_list_summaries_action_reads_from_ctx():
    import chatbot_actions

    fake_summaries = [{"id": 1, "summary": "test", "created_at": "2026-07-25T00:00:00"}]
    result = chatbot_actions.list_summaries({}, {"summaries": fake_summaries})
    assert result["summaries"] == fake_summaries


def test_list_summaries_action_defaults_to_empty_list():
    import chatbot_actions

    assert chatbot_actions.list_summaries({}, {}) == {"summaries": []}


@pytest.mark.anyio
async def test_run_turn_single_say_user_closes_turn(mocked_openrouter_structured):
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Bonjour !"}]}))

    result = chatbot_executor.run_turn("system", [{"role": "user", "content": "salut"}], {})
    assert result["error"] is None
    assert result["replies"] == ["Bonjour !"]
    assert len(calls) == 1  # un seul aller-retour, pas de relance


@pytest.mark.anyio
async def test_run_turn_chains_action_then_say_user_in_same_batch(mocked_openrouter_structured):
    """Un lot {action non-parole puis say_user} doit se clore en UN SEUL appel LLM (pas de
    relance) — c'est justement l'intérêt du format liste face au function-calling natif."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "get_vote_token"},
            {"action": "say_user", "text": "Ton jeton : {{résultat}}"},
        ]
    }))

    result = chatbot_executor.run_turn("system", [{"role": "user", "content": "mon jeton ?"}], {"identity_token": "tok-abc"})
    assert result["error"] is None
    assert len(calls) == 1
    expected = __import__("chatbot_actions").compute_vote_token("tok-abc")
    assert result["replies"] == [f"Ton jeton : {expected}"]


@pytest.mark.anyio
async def test_run_turn_substitutes_multi_field_result_naturally(mocked_openrouter_structured):
    """Régression (bug réel trouvé en conditions réelles le 2026-07-25) : get_or_assign_pseudo
    renvoie 2 clés (word, color) — la 1re version de _substitute_placeholder dumpait le dict JSON
    brut dans la réponse affichée à l'utilisateur au lieu d'un texte naturel."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "get_or_assign_pseudo"},
            {"action": "say_user", "text": "Tu es {{résultat}}"},
        ]
    }))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "mon pseudo ?"}],
        {"pseudo": {"word": "Clairière", "color": "corail"}},
    )
    assert result["error"] is None
    assert result["replies"] == ["Tu es Clairière corail"]
    assert "{" not in result["replies"][0]  # jamais de JSON brut affiché à l'utilisateur


@pytest.mark.anyio
async def test_run_turn_strips_unresolved_placeholder_instead_of_leaking_it(mocked_openrouter_structured):
    """Régression (bug réel trouvé en conditions réelles le 2026-07-25, tour suivant) : le LLM
    écrit parfois {{résultat}} sans avoir appelé l'action correspondante dans le même lot (il
    croit connaître la valeur depuis le contexte) — jamais laisser fuiter la syntaxe technique
    littérale vers l'utilisateur, même si la cause première (consigne de prompt) doit aussi être
    renforcée séparément."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [{"action": "say_user", "text": "Ton pseudo est {{résultat}}."}]
    }))

    result = chatbot_executor.run_turn("system", [{"role": "user", "content": "rappelle mon pseudo"}], {})
    assert result["error"] is None
    assert "{{résultat}}" not in result["replies"][0]
    assert "{" not in result["replies"][0]


@pytest.mark.anyio
async def test_run_turn_strips_placeholder_for_list_valued_result(mocked_openrouter_structured):
    """Régression (bug réel signalé par Angelo avec capture d'écran, 2026-07-25) :
    propose_pseudo_candidates renvoie {"candidates": [...]} — une clé unique dont la valeur est
    une LISTE de dicts. str() de cette liste produit un repr Python brut avec des guillemets
    simples, affiché tel quel dans le chat ("Voici : [{'word': 'Falaise', 'color': ...}]")."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "propose_pseudo_candidates"},
            {"action": "say_user", "text": "Voici quelques idées : {{résultat}}"},
        ]
    }))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose-moi un pseudo"}], {"identity_token": "tok-abc"}
    )
    assert result["error"] is None
    assert "{{résultat}}" not in result["replies"][0]
    assert "{" not in result["replies"][0]
    assert "[" not in result["replies"][0]  # pas de fuite de la liste non plus


@pytest.mark.anyio
async def test_run_turn_relaunches_llm_when_batch_ends_without_say_user(mocked_openrouter_structured):
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({"actions": [{"action": "get_vote_token"}]}))
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Voilà."}]}))

    result = chatbot_executor.run_turn("system", [{"role": "user", "content": "mon jeton ?"}], {"identity_token": "tok-abc"})
    assert result["error"] is None
    assert len(calls) == 2  # relance car le 1er lot ne se termine pas par say_user
    assert result["replies"] == ["Voilà."]


@pytest.mark.anyio
async def test_run_turn_rejects_unregistered_action_without_executing_it(mocked_openrouter_structured):
    """Même si un JSON malformé/hostile contenait une action de commit, elle est ignorée — le
    dispatch ne connaît que ACTIONS.get(), jamais d'exécution par nom arbitraire."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "save_summary", "text": "je triche"},
            {"action": "say_user", "text": "fin"},
        ]
    }))

    result = chatbot_executor.run_turn("system", [{"role": "user", "content": "x"}], {})
    assert result["replies"] == ["fin"]
    rejected = [a for a in result["actions_log"] if a["action"] == "save_summary"]
    assert rejected and rejected[0]["error"] == "action non autorisée pour le LLM"


@pytest.mark.anyio
async def test_run_turn_max_iterations_reached(mocked_openrouter_structured):
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    for _ in range(3):
        responses.append(json.dumps({"actions": [{"action": "get_vote_token"}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "x"}], {"identity_token": "t"}, max_iterations=3
    )
    assert result["error"] == "max_iterations_atteinte"
    assert len(calls) == 3


@pytest.mark.anyio
async def test_chat_v2_requires_valid_session(client):
    resp = await client.post("/chat/v2", json={"session_token": "fake", "message": "bonjour"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_chat_v2_success(client, logged_in_user, monkeypatch):
    import main as main_module

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5):
        assert ctx["identity_token"]  # résolu depuis la session, pas None/vide
        return {"replies": ["réponse simulée v2"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    resp = await client.post(
        "/chat/v2",
        json={"session_token": logged_in_user["session_token"], "message": "bonjour"},
    )
    assert resp.status_code == 200
    assert resp.json()["replies"] == ["réponse simulée v2"]


@pytest.mark.anyio
async def test_chat_v2_passes_saved_summaries_in_ctx(client, logged_in_user, monkeypatch):
    """L'endpoint doit charger les résumés déjà sauvegardés depuis la DB et les fournir dans
    ctx["summaries"] — c'est ce que list_summaries lit, sans accès DB propre (voir
    chatbot_actions.list_summaries)."""
    import main as main_module

    identity_token = logged_in_user["token"]
    with main.db() as conn:
        conn.execute(
            "INSERT INTO chat_summaries (owner_token, summary) VALUES (?, ?)",
            (identity_token, "un résumé déjà sauvegardé"),
        )

    captured_ctx = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5):
        captured_ctx.update(ctx)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    await client.post(
        "/chat/v2",
        json={"session_token": logged_in_user["session_token"], "message": "bonjour"},
    )
    assert len(captured_ctx["summaries"]) == 1
    assert captured_ctx["summaries"][0]["summary"] == "un résumé déjà sauvegardé"


@pytest.mark.anyio
async def test_chat_v2_llm_unavailable_returns_503(client, logged_in_user, monkeypatch):
    import main as main_module

    monkeypatch.setattr(
        main_module, "run_turn",
        lambda *a, **k: {"replies": [], "actions_log": [], "error": "llm_indisponible"},
    )
    resp = await client.post(
        "/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "bonjour"}
    )
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_chat_summarize_requires_history(client, logged_in_user):
    resp = await client.post(
        "/chat/summarize", json={"session_token": logged_in_user["session_token"], "history": []}
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_chat_summarize_does_not_save(client, logged_in_user, mocked_chat_llm):
    resp = await client.post(
        "/chat/summarize",
        json={
            "session_token": logged_in_user["session_token"],
            "history": [{"role": "user", "content": "je m'inquiète du bruit avenue de la Gare"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["summary"] == "réponse simulée du chatbot"
    # proposé, pas sauvegardé : minimisation par défaut tant que l'utilisateur n'a pas validé
    summaries = await client.post("/chat/summaries", json={"session_token": logged_in_user["session_token"]})
    assert summaries.json()["summaries"] == []


@pytest.mark.anyio
async def test_chat_save_summary_then_list_then_delete(client, logged_in_user):
    session = logged_in_user["session_token"]
    save = await client.post(
        "/chat/save-summary", json={"session_token": session, "summary": "Résumé validé par l'utilisateur."}
    )
    assert save.status_code == 200
    summary_id = save.json()["id"]

    listed = await client.post("/chat/summaries", json={"session_token": session})
    assert listed.status_code == 200
    assert len(listed.json()["summaries"]) == 1
    assert listed.json()["summaries"][0]["summary"] == "Résumé validé par l'utilisateur."

    deleted = await client.request("DELETE", f"/chat/summaries/{summary_id}", json={"session_token": session})
    assert deleted.status_code == 200

    listed_after = await client.post("/chat/summaries", json={"session_token": session})
    assert listed_after.json()["summaries"] == []


@pytest.mark.anyio
async def test_chat_summaries_are_private_per_user(client, admin_question):
    # Alice sauvegarde un résumé
    alice = await client.post(
        "/register", json={"nom": "Alice", "adresse": "1 Rue de la Mairie", "email": "alice@test.fr", "password": PASSWORD}
    )
    alice_session = alice.json()["session_token"]
    await client.post("/chat/save-summary", json={"session_token": alice_session, "summary": "secret d'Alice"})

    # Denis ne doit rien voir dans sa propre liste, ni pouvoir supprimer le résumé d'Alice
    denis = await client.post(
        "/register", json={"nom": "Denis", "adresse": "4 Rue du Secret", "email": "denis@test.fr", "password": PASSWORD}
    )
    denis_session = denis.json()["session_token"]
    denis_list = await client.post("/chat/summaries", json={"session_token": denis_session})
    assert denis_list.json()["summaries"] == []

    alice_list = await client.post("/chat/summaries", json={"session_token": alice_session})
    alice_summary_id = alice_list.json()["summaries"][0]["id"]
    delete_attempt = await client.request(
        "DELETE", f"/chat/summaries/{alice_summary_id}", json={"session_token": denis_session}
    )
    assert delete_attempt.status_code == 404
    # toujours là après la tentative de suppression par quelqu'un d'autre
    alice_list_after = await client.post("/chat/summaries", json={"session_token": alice_session})
    assert len(alice_list_after.json()["summaries"]) == 1


@pytest.mark.anyio
async def test_chat_delete_summary_invalid_session(client):
    resp = await client.request("DELETE", "/chat/summaries/1", json={"session_token": "fake"})
    assert resp.status_code == 401


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