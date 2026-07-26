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
    pourrait dire — c'est le dict ACTIONS qui fait foi, pas une consigne.

    report_bug/request_admin_intervention (2026-07-25 soir) sont la SEULE exception délibérée
    (avec admin_messages, jamais LLM-callable non plus en fait) : décidée avec le développeur,
    documentée comme telle (voir ACTIONS dans chatbot_actions.py) — profil de risque très
    différent (privé, faible enjeu, rate-limité) des vraies actions de commit ci-dessous, qui
    restent interdites sans exception."""
    import chatbot_actions

    assert set(chatbot_actions.ACTIONS.keys()) == {
        "say_user", "get_vote_token", "propose_summary", "list_summaries",
        "get_or_assign_pseudo", "propose_pseudo_candidates", "propose_custom_pseudo",
        "list_threads", "get_thread", "propose_opinion", "propose_reaction", "propose_remarque",
        "report_bug", "request_admin_intervention", "list_wiki_pages", "get_wiki_page",
        "search_conseil_municipal", "get_conseil_municipal_document", "list_conseil_municipal_seances",
    }
    for forbidden in (
        "save_summary", "delete_summary", "confirm_publication", "confirm_pseudo",
        # Forum (2026-07-25, phases 2/3 — lecture/validation pure) : aucune action de publication
        # n'existe encore pour le LLM, voir main.py pour les fonctions d'écriture correspondantes.
        "create_thread", "publish_thread", "create_opinion", "update_opinion_draft",
        "publish_opinion", "disavow_opinion", "supersede_opinion", "add_reaction",
        "publish_reaction", "create_remarque", "publish_remarque", "send_admin_message",
        "create_thread_with_opinion", "delete_thread_if_empty",
        "submit_bug_report", "submit_admin_intervention_request",
    ):
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


def test_agree_pseudo_display_feminine_word_agrees_variable_colors():
    """Régression (bug réel signalé par le développeur, 2026-07-25) : 'Clairière vert' au lieu de
    'Clairière verte' — accord de genre manquant."""
    import chatbot_actions

    assert chatbot_actions.PSEUDO_WORD_GENDER["Clairière"] == "f"
    assert chatbot_actions._agree_pseudo_display("Clairière", "vert") == "Clairière verte"
    assert chatbot_actions._agree_pseudo_display("Clairière", "bleu") == "Clairière bleue"
    assert chatbot_actions._agree_pseudo_display("Clairière", "violet") == "Clairière violette"
    assert chatbot_actions._agree_pseudo_display("Clairière", "blanc") == "Clairière blanche"
    assert chatbot_actions._agree_pseudo_display("Clairière", "noir") == "Clairière noire"
    assert chatbot_actions._agree_pseudo_display("Clairière", "gris") == "Clairière grise"
    # Couleurs invariables en genre : inchangées même pour un mot féminin.
    assert chatbot_actions._agree_pseudo_display("Clairière", "rouge") == "Clairière rouge"
    assert chatbot_actions._agree_pseudo_display("Clairière", "orange") == "Clairière orange"
    assert chatbot_actions._agree_pseudo_display("Clairière", "jaune") == "Clairière jaune"


def test_agree_pseudo_display_masculine_word_unchanged():
    import chatbot_actions

    assert chatbot_actions.PSEUDO_WORD_GENDER["Renard"] == "m"
    assert chatbot_actions._agree_pseudo_display("Renard", "vert") == "Renard vert"
    assert chatbot_actions._agree_pseudo_display("Renard", "blanc") == "Renard blanc"


def test_agree_pseudo_display_unknown_word_defaults_masculine():
    """Mot personnalisé (propose_custom_pseudo) hors de PSEUDO_WORD_GENDER — genre inconnu, pas
    de règle fiable sur un mot libre, donc masculin par défaut (convention française)."""
    import chatbot_actions

    assert "Loup" not in chatbot_actions.PSEUDO_WORD_GENDER
    assert chatbot_actions._agree_pseudo_display("Loup", "vert") == "Loup vert"


def test_check_pseudo_availability_includes_display_field():
    import chatbot_actions

    result = chatbot_actions._check_pseudo_availability("Clairière", "vert", {"taken_pseudos": set()})
    assert result["display"] == "Clairière verte"


def test_get_or_assign_pseudo_action_reads_from_ctx():
    import chatbot_actions

    result = chatbot_actions.get_or_assign_pseudo({}, {"pseudo": {"word": "Renard", "color": "bleu"}})
    assert result["word"] == "Renard"
    assert result["color"] == "bleu"
    assert result["display"] == "Renard bleu"


def test_get_or_assign_pseudo_action_errors_without_ctx():
    import chatbot_actions

    result = chatbot_actions.get_or_assign_pseudo({}, {})
    assert "error" in result


def test_generate_pseudo_candidates_deterministic_and_distinct():
    """Même identity_token → toujours la même séquence (pas de tirage aléatoire à chaque appel),
    et toutes distinctes entre elles au sein d'un même lot."""
    import chatbot_actions

    a = chatbot_actions.generate_pseudo_candidates("identity-x", n=3)
    b = chatbot_actions.generate_pseudo_candidates("identity-x", n=3)
    assert a == b
    assert len(a) == 3
    assert len({(c["word"], c["color"]) for c in a}) == 3
    for c in a:
        assert c["word"] in chatbot_actions.PSEUDO_WORDS
        assert c["color"] in chatbot_actions.PSEUDO_COLORS


def test_check_pseudo_availability_valid_and_free():
    import chatbot_actions

    result = chatbot_actions._check_pseudo_availability("Renard", "bleu", {"taken_pseudos": set()})
    assert result["word"] == "Renard"
    assert result["color"] == "bleu"
    assert result["available"] is True

    # "gris" ajouté à la palette le 2026-07-25 (demande développeur, via angelobot).
    result_gris = chatbot_actions._check_pseudo_availability("Renard", "gris", {"taken_pseudos": set()})
    assert result_gris["available"] is True
    assert result_gris["display"] == "Renard gris"
    assert result["display"] == "Renard bleu"


def test_check_pseudo_availability_rejects_invalid_color():
    import chatbot_actions

    result = chatbot_actions._check_pseudo_availability("Renard", "rose-fluo", {})
    assert result["available"] is False
    assert "error" in result


def test_check_pseudo_availability_rejects_taken():
    import chatbot_actions

    ctx = {"taken_pseudos": {("Renard", "bleu")}}
    result = chatbot_actions._check_pseudo_availability("Renard", "bleu", ctx)
    assert result["available"] is False
    assert "error" in result


def test_propose_pseudo_candidates_action_returns_one_candidate_at_index():
    """Une SEULE proposition par appel, pas une liste — tâtonnement, pas un batch figé."""
    import chatbot_actions

    ctx = {"identity_token": "identity-x", "taken_pseudos": set()}
    result0 = chatbot_actions.propose_pseudo_candidates({"index": 0}, ctx)
    result1 = chatbot_actions.propose_pseudo_candidates({"index": 1}, ctx)
    assert result0["available"] is True
    assert result1["available"] is True
    assert (result0["word"], result0["color"]) != (result1["word"], result1["color"])
    # index=0 redonne toujours la même 1re idée (déterminisme)
    assert chatbot_actions.propose_pseudo_candidates({"index": 0}, ctx) == result0


def test_propose_pseudo_candidates_action_errors_without_identity():
    import chatbot_actions

    result = chatbot_actions.propose_pseudo_candidates({"index": 0}, {})
    assert "error" in result


def test_propose_pseudo_candidates_reports_taken_candidate_unavailable():
    import chatbot_actions

    identity_token = "identity-x"
    candidate0 = chatbot_actions.generate_pseudo_candidates(identity_token, n=1)[0]
    ctx = {"identity_token": identity_token, "taken_pseudos": {(candidate0["word"], candidate0["color"])}}
    result = chatbot_actions.propose_pseudo_candidates({"index": 0}, ctx)
    assert result["available"] is False


def test_propose_custom_pseudo_action_checks_availability():
    import chatbot_actions

    result = chatbot_actions.propose_custom_pseudo(
        {"word": "Loup", "color": "rose", "appropriate": True}, {"taken_pseudos": set()}
    )
    assert result["available"] is False  # "rose" hors palette
    result_ok = chatbot_actions.propose_custom_pseudo(
        {"word": "Loup", "color": "bleu", "appropriate": True}, {"taken_pseudos": set()}
    )
    assert result_ok["word"] == "Loup"
    assert result_ok["color"] == "bleu"
    assert result_ok["available"] is True
    assert result_ok["display"] == "Loup bleu"


def test_propose_custom_pseudo_content_gate_blocks_button_regardless_of_technical_validity():
    """Régression (bug réel signalé par le développeur avec capture d'écran, 2026-07-25,
    'étoile noir') : le texte de l'assistant disait 'je refuse par prudence' mais le bouton de
    confirmation restait cliquable — le jugement de contenu ne vivait que dans le texte libre,
    jamais dans le résultat structuré qui pilote le bouton côté frontend. Fix : 'appropriate'
    devient obligatoire et court-circuite AVANT toute vérification technique."""
    import chatbot_actions

    result = chatbot_actions.propose_custom_pseudo(
        {"word": "Étoile", "color": "noir", "appropriate": False}, {"taken_pseudos": set()}
    )
    assert result["available"] is False
    assert "error" in result
    # Même avec une couleur techniquement valide et un mot non pris : appropriate=False gagne.
    assert result["word"] == "Étoile"
    assert result["color"] == "noir"


def test_propose_pseudo_candidates_content_gate_blocks_button():
    import chatbot_actions

    ctx = {"identity_token": "identity-x", "taken_pseudos": set()}
    result = chatbot_actions.propose_pseudo_candidates({"index": 0, "appropriate": False}, ctx)
    assert result["available"] is False
    assert "error" in result


@pytest.mark.anyio
async def test_run_turn_mentioning_past_candidate_without_recall_produces_no_button_action(
    mocked_openrouter_structured,
):
    """Régression bug réel #9 (2026-07-25, capture développeur) : pour 'Fourmi rouge' proposé lors
    d'un TOUR PRÉCÉDENT (visible dans l'historique de conversation), le say_user du tour suivant a
    affirmé "tu devrais voir apparaître un bouton" alors qu'aucun bouton n'existait réellement.
    Le frontend (renderPseudoCandidates) ne cherche un bouton que dans actions_log DE CE TOUR — une
    simple mention textuelle d'un candidat passé ne peut structurellement pas en faire réapparaître
    un. Ce test verrouille cette garantie : si le LLM ne rappelle pas propose_pseudo_candidates/
    propose_custom_pseudo DANS ce tour, actions_log ne contient aucune action avec
    result["available"] is True, quel que soit le texte du say_user."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "say_user", "text": "Tu devrais voir apparaître un bouton pour Fourmi rouge !"},
        ]
    }))

    conversation_messages = [
        {"role": "user", "content": "Je veux un pseudo animal."},
        {"role": "assistant", "content": "Que penses-tu de Fourmi rouge ?"},
        {"role": "user", "content": "Et pour Fourmi rouge, finalement ?"},
    ]
    result = chatbot_executor.run_turn("system", conversation_messages, {"identity_token": "tok-fourmi"})

    assert result["error"] is None
    button_eligible_actions = [
        a for a in result["actions_log"]
        if a.get("action") in ("propose_pseudo_candidates", "propose_custom_pseudo")
        and a.get("result", {}).get("available") is True
    ]
    assert button_eligible_actions == []


def test_get_existing_pseudo_none_then_set_after_confirm():
    """Aucune écriture automatique (contrairement à l'ancien ensure_pseudo) : None tant que
    confirm_pseudo n'a pas été appelé explicitement."""
    import main as main_module

    identity_token = "identity-pour-pseudo-test"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    assert main_module.get_existing_pseudo(identity_token) is None

    confirmed = main_module.confirm_pseudo(identity_token, "Renard", "bleu")
    assert confirmed == {"word": "Renard", "color": "bleu"}
    assert main_module.get_existing_pseudo(identity_token) == confirmed

    with main.db() as conn:
        rows = conn.execute("SELECT * FROM pseudos WHERE debate_token=?", (debate_token,)).fetchall()
    assert len(rows) == 1
    # La table ne doit JAMAIS contenir le jeton d'identité en clair, seulement le hash dérivé.
    assert rows[0]["debate_token"] != identity_token

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))


def test_confirm_pseudo_accepts_custom_word_not_from_generator():
    """Simplification (2026-07-25) : le mot n'a plus besoin de venir de la séquence déterministe
    — seules 2 règles dures comptent (couleur valide, pas déjà pris), identiques pour un pseudo
    généré ou personnalisé."""
    import main as main_module

    identity_token = "identity-pour-pseudo-test-custom"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    confirmed = main_module.confirm_pseudo(identity_token, "MotTotalementInventé", "violet")
    assert confirmed == {"word": "MotTotalementInventé", "color": "violet"}

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))


def test_confirm_pseudo_rejects_invalid_color():
    import main as main_module

    identity_token = "identity-pour-pseudo-test-badcolor"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    with pytest.raises(ValueError, match="couleur non valide"):
        main_module.confirm_pseudo(identity_token, "Renard", "rose-fluo")


def test_confirm_pseudo_allows_free_rechoice_replacing_previous_pseudo():
    """Rechoix libre (2026-07-25, demande développeur via angelobot) : reconfirmer un pseudo
    REMPLACE le précédent au lieu d'être bloqué. Décision explicite : aucune table de publication
    liée au pseudo n'existe encore (voir TODO dans confirm_pseudo), donc "rien n'est publiable"
    est vrai pour tout le monde aujourd'hui — rechoix inconditionnel assumé, pas un oubli."""
    import main as main_module

    identity_token = "identity-pour-pseudo-test-3"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))

    main_module.confirm_pseudo(identity_token, "Renard", "bleu")
    result = main_module.confirm_pseudo(identity_token, "Hibou", "vert")
    assert result == {"word": "Hibou", "color": "vert"}
    assert main_module.get_existing_pseudo(identity_token) == {"word": "Hibou", "color": "vert"}

    with main.db() as conn:
        rows = conn.execute("SELECT * FROM pseudos WHERE debate_token=?", (debate_token,)).fetchall()
        assert len(rows) == 1  # remplacement (UPDATE), pas une 2e ligne

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (debate_token,))


def test_confirm_pseudo_rejects_word_color_pair_already_taken_by_another_identity():
    """Contrainte UNIQUE(word,color) : deux identités différentes ne peuvent pas confirmer le
    même pseudo — filet de sécurité DB en plus de la vérification applicative en amont
    (propose_*/ctx["taken_pseudos"])."""
    import main as main_module

    id_a, id_b = "identity-pseudo-unique-a", "identity-pseudo-unique-b"
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token IN (?, ?)", (
            main_module.compute_debate_token(id_a), main_module.compute_debate_token(id_b),
        ))

    main_module.confirm_pseudo(id_a, "Renard", "bleu")
    with pytest.raises(ValueError, match="déjà pris"):
        main_module.confirm_pseudo(id_b, "Renard", "bleu")

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token IN (?, ?)", (
            main_module.compute_debate_token(id_a), main_module.compute_debate_token(id_b),
        ))


def test_confirm_pseudo_rechoice_to_pair_taken_by_another_identity_still_rejected():
    """Le rechoix libre (voir test ci-dessus) ne contourne PAS la contrainte d'unicité entre 2
    identités différentes — seul le blocage "j'ai déjà un pseudo, donc je ne peux plus en changer"
    a été retiré, pas la règle "ce mot+couleur est pris par quelqu'un d'autre". Message sans
    ambiguïté (régression bug réel "Chat gris" signalé par le développeur, 2026-07-25 via
    angelobot, avant que la vraie cause — pas de rechoix libre à l'époque — ne soit identifiée)."""
    import main as main_module

    id_a, id_b = "identity-msg-clarity-a", "identity-msg-clarity-b"
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token IN (?, ?)", (
            main_module.compute_debate_token(id_a), main_module.compute_debate_token(id_b),
        ))

    main_module.confirm_pseudo(id_a, "Renard", "bleu")
    main_module.confirm_pseudo(id_b, "Hibou", "vert")

    with pytest.raises(ValueError, match="pris par quelqu'un d'autre"):
        main_module.confirm_pseudo(id_b, "Renard", "bleu")


# ===== Forum (2026-07-25, phase 1 : schéma + fonctions d'écriture, zéro branchement chatbot) =====
# Spec : wiki.jouyvote.fr/themes:chatbot-fonctionnalites, section "Page Forum" (version après
# corrections angelobot du 2026-07-25 soir — plus de table opinion_versions séparée, body/
# argumentaire directement sur opinions, superseded_by_opinion_id, statut disavowed distinct).


def test_create_thread_with_and_without_creator():
    import main as main_module

    with_creator = main_module.create_thread("Sujet A", summary="résumé", creator_identity_token="identity-forum-1")
    assert with_creator["status"] == "draft"
    without_creator = main_module.create_thread("Sujet B (créé par le chatbot seul)")
    assert without_creator["status"] == "draft"

    with main.db() as conn:
        row_with = conn.execute("SELECT creator_debate_token FROM threads WHERE thread_id=?", (with_creator["thread_id"],)).fetchone()
        row_without = conn.execute("SELECT creator_debate_token FROM threads WHERE thread_id=?", (without_creator["thread_id"],)).fetchone()
    assert row_with["creator_debate_token"] == main_module.compute_debate_token("identity-forum-1")
    assert row_without["creator_debate_token"] is None

    published = main_module.publish_thread(with_creator["thread_id"])
    assert published["status"] == "published"


def test_opinion_draft_freely_editable_before_any_reaction():
    import main as main_module

    thread = main_module.create_thread("Sujet opinion")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-2", "Version initiale")
    assert opinion["status"] == "draft"

    main_module.update_opinion_draft(opinion["opinion_id"], body="Version corrigée avant publication")
    with main.db() as conn:
        row = conn.execute("SELECT body FROM opinions WHERE opinion_id=?", (opinion["opinion_id"],)).fetchone()
    assert row["body"] == "Version corrigée avant publication"

    result = main_module.publish_opinion(opinion["opinion_id"])
    assert result["status"] == "published"


def test_opinion_frozen_once_a_reaction_exists():
    """Régression du problème identifié par angelobot le 2026-07-25 (root cause de l'abandon de la
    table opinion_versions séparée) : modifier le texte après une réaction ferait porter
    silencieusement cette réaction sur un texte jamais vu — doit être structurellement impossible."""
    import main as main_module

    thread = main_module.create_thread("Sujet gel")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-3", "Opinion publiée")
    main_module.publish_opinion(opinion["opinion_id"])

    main_module.add_reaction(opinion["opinion_id"], "identity-forum-4", "adherer")

    with pytest.raises(ValueError, match="déjà des réactions"):
        main_module.update_opinion_draft(opinion["opinion_id"], body="Tentative de modification après réaction")

    with main.db() as conn:
        row = conn.execute("SELECT body FROM opinions WHERE opinion_id=?", (opinion["opinion_id"],)).fetchone()
    assert row["body"] == "Opinion publiée"  # inchangé


def test_disavow_opinion_requires_author_and_sets_status():
    import main as main_module

    thread = main_module.create_thread("Sujet désaveu")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-5", "Une opinion")
    main_module.publish_opinion(opinion["opinion_id"])

    with pytest.raises(ValueError, match="seul l'auteur"):
        main_module.disavow_opinion(opinion["opinion_id"], "identity-forum-6")

    result = main_module.disavow_opinion(opinion["opinion_id"], "identity-forum-5")
    assert result["status"] == "disavowed"


def test_supersede_opinion_links_old_to_new_without_deleting_reactions():
    """Changer d'avis après une réaction : nouvelle opinion + superseded_by_opinion_id, jamais de
    suppression — les réactions déjà données sur l'ancienne restent des faits historiques vrais."""
    import main as main_module

    thread = main_module.create_thread("Sujet supersede")
    main_module.publish_thread(thread["thread_id"])
    old_opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-7", "Ancienne position")
    main_module.publish_opinion(old_opinion["opinion_id"])
    reaction = main_module.add_reaction(old_opinion["opinion_id"], "identity-forum-8", "adherer")
    main_module.publish_reaction(reaction["reaction_id"])

    new_opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-7", "Nouvelle position")
    main_module.publish_opinion(new_opinion["opinion_id"])

    result = main_module.supersede_opinion(old_opinion["opinion_id"], new_opinion["opinion_id"], "identity-forum-7")
    assert result["superseded_by_opinion_id"] == new_opinion["opinion_id"]

    with main.db() as conn:
        old_row = conn.execute(
            "SELECT superseded_by_opinion_id, status FROM opinions WHERE opinion_id=?", (old_opinion["opinion_id"],)
        ).fetchone()
        reaction_row = conn.execute(
            "SELECT status FROM opinion_reactions WHERE reaction_id=?", (reaction["reaction_id"],)
        ).fetchone()
    assert old_row["superseded_by_opinion_id"] == new_opinion["opinion_id"]
    assert old_row["status"] == "published"  # pas de désaveu automatique
    assert reaction_row["status"] == "published"  # réaction jamais supprimée/altérée


def test_supersede_opinion_rejects_different_author_or_thread():
    import main as main_module

    thread_a = main_module.create_thread("Fil A")
    main_module.publish_thread(thread_a["thread_id"])
    thread_b = main_module.create_thread("Fil B")
    main_module.publish_thread(thread_b["thread_id"])
    opinion_a = main_module.create_opinion(thread_a["thread_id"], "identity-forum-9", "Opinion A")
    opinion_b_other_author = main_module.create_opinion(thread_a["thread_id"], "identity-forum-10", "Opinion B")
    opinion_c_other_thread = main_module.create_opinion(thread_b["thread_id"], "identity-forum-9", "Opinion C")

    with pytest.raises(ValueError, match="seul l'auteur"):
        main_module.supersede_opinion(opinion_a["opinion_id"], opinion_b_other_author["opinion_id"], "identity-forum-9")

    with pytest.raises(ValueError, match="même fil"):
        main_module.supersede_opinion(opinion_a["opinion_id"], opinion_c_other_thread["opinion_id"], "identity-forum-9")


def test_reactions_allow_multiple_over_time_no_unique_constraint():
    """Régression du design retenu (2026-07-25, corrigé le même soir par angelobot) : plus de
    contrainte UNIQUE(opinion_id, reactor_debate_token) — une personne peut changer d'avis
    plusieurs fois, chaque réaction est une ligne à part, la plus récente PUBLIÉE compte seule."""
    import main as main_module

    thread = main_module.create_thread("Sujet réactions multiples")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-11", "Une opinion")
    main_module.publish_opinion(opinion["opinion_id"])

    r1 = main_module.add_reaction(opinion["opinion_id"], "identity-forum-12", "adherer", argumentaire="Je suis d'accord")
    main_module.publish_reaction(r1["reaction_id"])
    current = main_module.get_current_reaction(opinion["opinion_id"], "identity-forum-12")
    assert current["stance"] == "adherer"

    r2 = main_module.add_reaction(opinion["opinion_id"], "identity-forum-12", "opposer", argumentaire="J'ai changé d'avis")
    main_module.publish_reaction(r2["reaction_id"])
    current = main_module.get_current_reaction(opinion["opinion_id"], "identity-forum-12")
    assert current["stance"] == "opposer"
    assert current["reaction_id"] == r2["reaction_id"]

    with main.db() as conn:
        rows = conn.execute(
            "SELECT reaction_id, stance FROM opinion_reactions WHERE opinion_id=? AND reactor_debate_token=?",
            (opinion["opinion_id"], main_module.compute_debate_token("identity-forum-12")),
        ).fetchall()
    assert len(rows) == 2  # les 2 réactions coexistent, aucune écrasée/supprimée


def test_add_reaction_rejects_invalid_stance():
    import main as main_module

    thread = main_module.create_thread("Sujet stance invalide")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-13", "Une opinion")

    with pytest.raises(ValueError, match="stance invalide"):
        main_module.add_reaction(opinion["opinion_id"], "identity-forum-14", "pour")


def test_create_remarque_reply_to_remarque_or_opinion_but_not_both():
    import main as main_module

    thread = main_module.create_thread("Sujet remarques")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-15", "Une opinion")
    remarque1 = main_module.create_remarque(thread["thread_id"], "identity-forum-16", "Bonjour tout le monde")
    main_module.publish_remarque(remarque1["remarque_id"])

    reply_to_remarque = main_module.create_remarque(
        thread["thread_id"], "identity-forum-17", "Réponse à la remarque", reply_to_remarque_id=remarque1["remarque_id"]
    )
    assert reply_to_remarque["status"] == "draft"

    reply_to_opinion = main_module.create_remarque(
        thread["thread_id"], "identity-forum-18", "Réaction informelle à l'opinion", reply_to_opinion_id=opinion["opinion_id"]
    )
    assert reply_to_opinion["status"] == "draft"

    with pytest.raises(ValueError, match="au plus une chose"):
        main_module.create_remarque(
            thread["thread_id"], "identity-forum-19", "Impossible",
            reply_to_remarque_id=remarque1["remarque_id"], reply_to_opinion_id=opinion["opinion_id"],
        )


def test_send_admin_message_indexes_by_peppered_debate_token_not_raw_identity():
    """Seule exception au chatbot-passage-obligé — mais l'anonymat reste respecté : le
    destinataire est indexé par debate_token peppé, jamais identities.token en clair."""
    import main as main_module

    identity_token = "identity-forum-admin-recipient"
    result = main_module.send_admin_message(identity_token, "Message de l'administration")
    assert result["recipient_debate_token"] == main_module.compute_debate_token(identity_token)
    assert result["recipient_debate_token"] != identity_token

    with main.db() as conn:
        row = conn.execute(
            "SELECT admin_identity, body FROM admin_messages WHERE message_id=?", (result["message_id"],)
        ).fetchone()
    assert row["admin_identity"] == "administration"
    assert row["body"] == "Message de l'administration"


# ===== Création de fil couplée au 1er post d'opinion (2026-07-25 soir, décision développeur) =====


def test_create_thread_with_opinion_happy_path():
    import main as main_module

    with main.db() as conn:
        conn.execute("DELETE FROM threads WHERE title=?", ("Sujet couplé création",))

    result = main_module.create_thread_with_opinion(
        "Sujet couplé création", "identity-forum-coupled-1", "Je pense que...", argumentaire="parce que..."
    )
    assert result["status"] == "published"

    with main.db() as conn:
        thread_row = conn.execute("SELECT status FROM threads WHERE thread_id=?", (result["thread_id"],)).fetchone()
        opinion_row = conn.execute("SELECT status, body FROM opinions WHERE opinion_id=?", (result["opinion_id"],)).fetchone()
    assert thread_row["status"] == "published"
    assert opinion_row["status"] == "published"
    assert opinion_row["body"] == "Je pense que..."


def test_create_thread_with_opinion_race_reattaches_loser_to_winner_thread():
    """Régression du filet anti-race (2026-07-25) : si un fil au titre EXACTEMENT identique existe
    déjà (course perdue), l'opinion est rattachée à CE fil plutôt que de subir une erreur brute —
    la personne n'y est pour rien dans la collision technique."""
    import main as main_module

    with main.db() as conn:
        conn.execute("DELETE FROM threads WHERE title=?", ("Sujet course anti-race",))

    winner = main_module.create_thread_with_opinion(
        "Sujet course anti-race", "identity-forum-race-1", "Première opinion (gagnant de la course)"
    )
    loser = main_module.create_thread_with_opinion(
        "Sujet course anti-race", "identity-forum-race-2", "Deuxième opinion (perdant de la course)"
    )

    assert loser["thread_id"] == winner["thread_id"]  # rattaché au MÊME fil, pas un doublon
    assert loser["status"] == "published"

    with main.db() as conn:
        threads = conn.execute("SELECT thread_id FROM threads WHERE title=?", ("Sujet course anti-race",)).fetchall()
    assert len(threads) == 1  # un seul fil créé, pas 2


def test_create_thread_with_opinion_cleans_up_empty_thread_on_partial_failure():
    """Régression de l'auto-nettoyage (2026-07-25) : si le fil est créé mais que la publication de
    l'opinion échoue juste après (ici : body vide), le fil tout juste créé — encore vide — est
    supprimé plutôt que de laisser une trace orpheline en base."""
    import main as main_module

    with main.db() as conn:
        conn.execute("DELETE FROM threads WHERE title=?", ("Sujet échec partiel",))

    with pytest.raises(ValueError):
        main_module.create_thread_with_opinion("Sujet échec partiel", "identity-forum-partial-1", "   ")

    with main.db() as conn:
        threads = conn.execute("SELECT thread_id FROM threads WHERE title=?", ("Sujet échec partiel",)).fetchall()
    assert threads == []  # aucune trace orpheline


def test_delete_thread_if_empty_rejects_thread_with_content():
    import main as main_module

    thread = main_module.create_thread("Fil non vide pour test suppression")
    main_module.publish_thread(thread["thread_id"])
    main_module.create_opinion(thread["thread_id"], "identity-forum-delete-1", "Une opinion")

    with pytest.raises(ValueError, match="contient au moins une opinion"):
        main_module.delete_thread_if_empty(thread["thread_id"])

    with main.db() as conn:
        conn.execute("DELETE FROM opinions WHERE thread_id=?", (thread["thread_id"],))
        conn.execute("DELETE FROM threads WHERE thread_id=?", (thread["thread_id"],))


def test_delete_thread_if_empty_succeeds_on_truly_empty_thread():
    import main as main_module

    thread = main_module.create_thread("Fil vide pour test suppression")
    result = main_module.delete_thread_if_empty(thread["thread_id"])
    assert result["deleted"] is True

    with main.db() as conn:
        row = conn.execute("SELECT thread_id FROM threads WHERE thread_id=?", (thread["thread_id"],)).fetchone()
    assert row is None


def test_delete_thread_if_empty_errors_on_unknown_thread():
    import main as main_module

    with pytest.raises(ValueError, match="introuvable"):
        main_module.delete_thread_if_empty(999999)


# ===== report_bug / request_admin_intervention (2026-07-25 soir, demande développeur) — envoi =====
# ===== DIRECT sans confirmation utilisateur, exception délibérée, rate-limitée =====


def test_submit_bug_report_writes_row_and_attempts_send(monkeypatch):
    import main as main_module

    sent_calls = []
    monkeypatch.setattr(main_module, "_send_admin_email", lambda subject, description: sent_calls.append((subject, description)) or True)

    result = main_module.submit_bug_report("identity-bugreport-1", "Le bouton de confirmation n'apparaît pas")
    assert result["sent"] is True
    assert len(sent_calls) == 1
    assert "Le bouton de confirmation n'apparaît pas" in sent_calls[0][1]

    debate_token = main_module.compute_debate_token("identity-bugreport-1")
    with main.db() as conn:
        row = conn.execute("SELECT description FROM bug_reports WHERE debate_token=?", (debate_token,)).fetchone()
    assert row["description"] == "Le bouton de confirmation n'apparaît pas"

    with main.db() as conn:
        conn.execute("DELETE FROM bug_reports WHERE debate_token=?", (debate_token,))


def test_submit_bug_report_rejects_empty_description():
    import main as main_module

    with pytest.raises(ValueError, match="vide"):
        main_module.submit_bug_report("identity-bugreport-2", "   ")


def test_submit_bug_report_rate_limited_after_threshold(monkeypatch):
    """Régression anti-abus (2026-07-25) : un LLM manipulé ne doit pas pouvoir spammer l'envoi
    d'emails — au-delà de BUG_REPORT_RATE_LIMIT signalements récents pour la même identité, la
    fonction refuse plutôt que d'envoyer indéfiniment."""
    import main as main_module

    monkeypatch.setattr(main_module, "_send_admin_email", lambda subject, description: True)
    identity_token = "identity-bugreport-ratelimit"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM bug_reports WHERE debate_token=?", (debate_token,))

    for i in range(main_module.BUG_REPORT_RATE_LIMIT):
        main_module.submit_bug_report(identity_token, f"Signalement {i}")

    with pytest.raises(ValueError, match="trop de signalements"):
        main_module.submit_bug_report(identity_token, "Signalement de trop")

    with main.db() as conn:
        conn.execute("DELETE FROM bug_reports WHERE debate_token=?", (debate_token,))


def test_submit_admin_intervention_request_writes_row_and_attempts_send(monkeypatch):
    import main as main_module

    monkeypatch.setattr(main_module, "_send_admin_email", lambda subject, description: True)
    result = main_module.submit_admin_intervention_request("identity-admin-intervention-1", "Je crois qu'on a deviné mon identité")
    assert result["sent"] is True

    debate_token = main_module.compute_debate_token("identity-admin-intervention-1")
    with main.db() as conn:
        row = conn.execute("SELECT description FROM admin_intervention_requests WHERE debate_token=?", (debate_token,)).fetchone()
    assert row["description"] == "Je crois qu'on a deviné mon identité"

    with main.db() as conn:
        conn.execute("DELETE FROM admin_intervention_requests WHERE debate_token=?", (debate_token,))


def test_submit_admin_intervention_request_rate_limited_independently_from_bug_reports(monkeypatch):
    """Les 2 rate-limits sont INDÉPENDANTS (2 tables séparées) — épuiser celui de report_bug ne
    doit pas affecter request_admin_intervention pour la même identité."""
    import main as main_module

    monkeypatch.setattr(main_module, "_send_admin_email", lambda subject, description: True)
    identity_token = "identity-independent-ratelimits"
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM bug_reports WHERE debate_token=?", (debate_token,))
        conn.execute("DELETE FROM admin_intervention_requests WHERE debate_token=?", (debate_token,))

    for i in range(main_module.BUG_REPORT_RATE_LIMIT):
        main_module.submit_bug_report(identity_token, f"Signalement {i}")
    with pytest.raises(ValueError):
        main_module.submit_bug_report(identity_token, "Signalement de trop")

    result = main_module.submit_admin_intervention_request(identity_token, "Une vraie demande distincte")
    assert result["sent"] is True

    with main.db() as conn:
        conn.execute("DELETE FROM bug_reports WHERE debate_token=?", (debate_token,))
        conn.execute("DELETE FROM admin_intervention_requests WHERE debate_token=?", (debate_token,))


def test_report_bug_action_calls_ctx_callable():
    import chatbot_actions

    calls = []
    ctx = {"report_bug_fn": lambda description: calls.append(description) or {"sent": True}}
    result = chatbot_actions.report_bug({"description": "Un vrai bug rencontré"}, ctx)
    assert result == {"sent": True}
    assert calls == ["Un vrai bug rencontré"]


def test_report_bug_action_rejects_empty_description_without_calling_ctx():
    import chatbot_actions

    calls = []
    ctx = {"report_bug_fn": lambda description: calls.append(description) or {"sent": True}}
    result = chatbot_actions.report_bug({"description": "   "}, ctx)
    assert result["sent"] is False
    assert calls == []  # jamais appelé pour une description vide


def test_report_bug_action_surfaces_rate_limit_error_from_ctx_callable():
    import chatbot_actions

    def raising_fn(description):
        raise ValueError("trop de signalements récents, réessaie dans un moment")

    result = chatbot_actions.report_bug({"description": "Un bug"}, {"report_bug_fn": raising_fn})
    assert result["sent"] is False
    assert "trop de signalements" in result["error"]


def test_request_admin_intervention_action_calls_ctx_callable():
    import chatbot_actions

    calls = []
    ctx = {"request_admin_intervention_fn": lambda description: calls.append(description) or {"sent": True}}
    result = chatbot_actions.request_admin_intervention({"description": "J'ai besoin d'aide"}, ctx)
    assert result == {"sent": True}
    assert calls == ["J'ai besoin d'aide"]


@pytest.mark.anyio
async def test_chat_v2_passes_report_bug_and_admin_intervention_callables_in_ctx(client, logged_in_user, monkeypatch):
    import main as main_module

    captured_ctx = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured_ctx.update(ctx)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    monkeypatch.setattr(main_module, "_send_admin_email", lambda subject, description: True)
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut"})

    assert callable(captured_ctx["report_bug_fn"])
    assert callable(captured_ctx["request_admin_intervention_fn"])
    result = captured_ctx["report_bug_fn"]("Test d'intégration ctx")
    assert result["sent"] is True

    identity_token = logged_in_user["token"]
    debate_token = main_module.compute_debate_token(identity_token)
    with main.db() as conn:
        conn.execute("DELETE FROM bug_reports WHERE debate_token=?", (debate_token,))


# ===== Accès en lecture au wiki citoyen (2026-07-26, demande développeur) — allowlist, jamais =====
# ===== d'accès aux pages internes (architecture-technique, prompt système...) =====


def test_fetch_wiki_page_raw_rejects_page_outside_allowlist():
    """Sécurité (2026-07-26) : allowlist explicite, pas un accès à tout le wiki — une page hors
    de WIKI_CITIZEN_PAGES (ex. le prompt système lui-même, ou l'architecture technique sensible)
    ne doit jamais être lisible par le chatbot public, même si son id est deviné."""
    import main as main_module

    assert main_module.fetch_wiki_page_raw("themes:prompt-chatbot") is None
    assert main_module.fetch_wiki_page_raw("architecture-technique") is None
    assert main_module.fetch_wiki_page_raw("bugs-jouyvote") is None
    assert main_module.fetch_wiki_page_raw("page-totalement-inventee") is None


def test_list_wiki_pages_action_reads_from_ctx():
    import chatbot_actions

    fake_index = {"charte-anonymat": "La charte."}
    result = chatbot_actions.list_wiki_pages({}, {"wiki_pages_index": fake_index})
    assert result == {"pages": fake_index}


def test_list_wiki_pages_action_defaults_to_empty_dict():
    import chatbot_actions

    assert chatbot_actions.list_wiki_pages({}, {}) == {"pages": {}}


def test_get_wiki_page_action_calls_ctx_callable():
    import chatbot_actions

    calls = []

    def fake_fn(page_id):
        calls.append(page_id)
        return "===== Contenu brut de la page ====="

    result = chatbot_actions.get_wiki_page({"page_id": "charte-anonymat"}, {"get_wiki_page_fn": fake_fn})
    assert result == {"page_id": "charte-anonymat", "content": "===== Contenu brut de la page ====="}
    assert calls == ["charte-anonymat"]


def test_get_wiki_page_action_errors_when_page_not_found():
    import chatbot_actions

    result = chatbot_actions.get_wiki_page({"page_id": "page-hors-allowlist"}, {"get_wiki_page_fn": lambda page_id: None})
    assert "error" in result


def test_get_wiki_page_action_errors_when_callable_missing():
    import chatbot_actions

    result = chatbot_actions.get_wiki_page({"page_id": "charte-anonymat"}, {})
    assert "error" in result


def test_search_conseil_municipal_action_calls_ctx_callable():
    """RAG conseil municipal (2026-07-26) : même pattern que get_wiki_page — chatbot_actions.py
    reste sans accès réseau/Qdrant direct, tout passe par un callable fourni dans ctx."""
    import chatbot_actions

    calls = []

    def fake_fn(query):
        calls.append(query)
        return [{"text": "Le conseil a voté...", "source_url": "https://jouy28.com/x.pdf", "meeting_date": "05 juin 2026"}]

    result = chatbot_actions.search_conseil_municipal({"query": "budget voirie"}, {"search_conseil_municipal_fn": fake_fn})
    assert result == {"results": [{"text": "Le conseil a voté...", "source_url": "https://jouy28.com/x.pdf", "meeting_date": "05 juin 2026"}]}
    assert calls == ["budget voirie"]


def test_search_conseil_municipal_action_handles_empty_results():
    """Une liste vide doit rester une liste vide (pas une erreur) — c'est le signal "rien trouvé
    sur ce sujet" que le prompt demande au modèle de communiquer honnêtement."""
    import chatbot_actions

    result = chatbot_actions.search_conseil_municipal({"query": "sujet jamais évoqué"}, {"search_conseil_municipal_fn": lambda query: []})
    assert result == {"results": []}


def test_search_conseil_municipal_action_errors_when_callable_missing():
    import chatbot_actions

    result = chatbot_actions.search_conseil_municipal({"query": "budget"}, {})
    assert "error" in result


def test_search_conseil_municipal_action_errors_on_empty_query():
    import chatbot_actions

    result = chatbot_actions.search_conseil_municipal({"query": ""}, {"search_conseil_municipal_fn": lambda query: []})
    assert "error" in result


def test_search_conseil_municipal_action_survives_exception_from_fn():
    """Défense en profondeur (même famille que le bug réel #13, chatbot_executor.py) : même si
    le callable lève (Qdrant down, etc.), l'action ne doit jamais faire planter le tour."""
    import chatbot_actions

    def failing_fn(query):
        raise RuntimeError("Qdrant indisponible")

    result = chatbot_actions.search_conseil_municipal({"query": "budget"}, {"search_conseil_municipal_fn": failing_fn})
    assert "error" in result


def test_search_conseil_municipal_pv_degrades_gracefully_without_qdrant():
    """main.search_conseil_municipal_pv (l'implémentation réelle, réseau/Qdrant) doit renvoyer
    une liste vide plutôt que lever si Qdrant/le modèle d'embedding est indisponible — le venv de
    test n'installe volontairement pas qdrant-client/sentence-transformers (dépendances lourdes,
    réservées au conteneur de prod/opencode), ce qui exerce réellement ce chemin de dégradation."""
    import main as main_module

    assert main_module.search_conseil_municipal_pv("n'importe quelle question") == []


def test_list_conseil_municipal_seances_action_calls_ctx_callable():
    """Même pattern que search_conseil_municipal (2026-07-26, bug réel #15) : l'action délègue
    intégralement à ctx["list_conseil_municipal_fn"], distincte de search_conseil_municipal_fn —
    ne jamais confondre les deux callables."""
    import chatbot_actions

    fake_meetings = [
        {"source_url": "https://jouy28.com/x.pdf", "meeting_date": "05 juin 2026"},
        {"source_url": "https://jouy28.com/y.pdf", "meeting_date": "07 avril 2026"},
    ]
    result = chatbot_actions.list_conseil_municipal_seances({}, {"list_conseil_municipal_fn": lambda: fake_meetings})
    assert result == {"meetings": fake_meetings}


def test_list_conseil_municipal_seances_action_errors_when_callable_missing():
    import chatbot_actions

    result = chatbot_actions.list_conseil_municipal_seances({}, {})
    assert "error" in result


def test_list_conseil_municipal_seances_action_survives_exception_from_fn():
    """Même défense en profondeur que search_conseil_municipal (famille du bug réel #13) : ne
    jamais faire planter le tour si Qdrant est down."""
    import chatbot_actions

    def failing_fn():
        raise RuntimeError("Qdrant indisponible")

    result = chatbot_actions.list_conseil_municipal_seances({}, {"list_conseil_municipal_fn": failing_fn})
    assert "error" in result


def test_tools_description_documents_list_conseil_municipal_seances_with_recency_rule():
    """Le prompt doit expliciter la règle anti-confusion (bug réel #15) : ne jamais utiliser
    search_conseil_municipal pour une question de récence/chronologie."""
    import chatbot_actions

    assert "list_conseil_municipal_seances" in chatbot_actions.TOOLS_DESCRIPTION
    assert "JAMAIS" in chatbot_actions.TOOLS_DESCRIPTION


def test_parse_meeting_date_handles_text_extracted_format():
    """Format produit par _extract_meeting_date dans index.py : "05 juin 2026"."""
    import main as main_module
    from datetime import date

    assert main_module._parse_meeting_date("05 juin 2026") == date(2026, 6, 5)


def test_parse_meeting_date_handles_iso_transcription_format():
    """Bug réel #15 bis (2026-07-26) : le frontmatter date_seance des transcriptions manuelles
    (index_transcriptions.py) est stocké tel quel au format ISO "2026-06-05" — sans ce cas, la
    séance retombait sur date.min et se retrouvait en dernier du tri décroissant alors qu'elle
    était la plus récente indexée (repéré en vérifiant la vraie réponse /chat/v2 en conditions
    réelles : la séance du 5 juin apparaissait en fin de liste malgré le tri "décroissant")."""
    import main as main_module
    from datetime import date

    assert main_module._parse_meeting_date("2026-06-05") == date(2026, 6, 5)


def test_parse_meeting_date_handles_filename_fallback_format():
    """Format de repli produit par _parse_date_label dans index.py : "05/03/2025"."""
    import main as main_module
    from datetime import date

    assert main_module._parse_meeting_date("05/03/2025") == date(2025, 3, 5)


def test_parse_meeting_date_returns_none_for_unparsable_or_missing():
    import main as main_module

    assert main_module._parse_meeting_date(None) is None
    assert main_module._parse_meeting_date("") is None
    assert main_module._parse_meeting_date("n'importe quoi") is None


def test_list_conseil_municipal_meetings_degrades_gracefully_without_qdrant():
    """Même garde-fou que search_conseil_municipal_pv : le venv de test n'installe pas
    qdrant-client volontairement, ce qui exerce réellement ce chemin de dégradation."""
    import main as main_module

    assert main_module.list_conseil_municipal_meetings() == []


def test_get_conseil_municipal_document_action_calls_ctx_callable():
    """Bug réel #16 (2026-07-26) : même pattern que search_conseil_municipal/
    list_conseil_municipal_seances — délègue intégralement à ctx["get_conseil_municipal_document_fn"]."""
    import chatbot_actions

    fake_document = {"source_url": "https://jouy28.com/x.pdf", "meeting_date": "05 juin 2026", "text": "...", "truncated": False}
    calls = []

    def fake_fn(source_url):
        calls.append(source_url)
        return fake_document

    result = chatbot_actions.get_conseil_municipal_document(
        {"source_url": "https://jouy28.com/x.pdf"}, {"get_conseil_municipal_document_fn": fake_fn}
    )
    assert result == {"document": fake_document}
    assert calls == ["https://jouy28.com/x.pdf"]


def test_get_conseil_municipal_document_action_errors_on_empty_source_url():
    import chatbot_actions

    result = chatbot_actions.get_conseil_municipal_document({"source_url": ""}, {"get_conseil_municipal_document_fn": lambda u: None})
    assert "error" in result


def test_get_conseil_municipal_document_action_errors_when_callable_missing():
    import chatbot_actions

    result = chatbot_actions.get_conseil_municipal_document({"source_url": "https://jouy28.com/x.pdf"}, {})
    assert "error" in result


def test_get_conseil_municipal_document_action_errors_when_document_not_found():
    import chatbot_actions

    result = chatbot_actions.get_conseil_municipal_document(
        {"source_url": "https://jouy28.com/inconnu.pdf"}, {"get_conseil_municipal_document_fn": lambda u: None}
    )
    assert "error" in result


def test_get_conseil_municipal_document_action_survives_exception_from_fn():
    import chatbot_actions

    def failing_fn(source_url):
        raise RuntimeError("Qdrant indisponible")

    result = chatbot_actions.get_conseil_municipal_document(
        {"source_url": "https://jouy28.com/x.pdf"}, {"get_conseil_municipal_document_fn": failing_fn}
    )
    assert "error" in result


def test_tools_description_documents_get_conseil_municipal_document_and_bug16_rule():
    import chatbot_actions

    assert "get_conseil_municipal_document" in chatbot_actions.TOOLS_DESCRIPTION
    assert "bug réel #16" in chatbot_actions.TOOLS_DESCRIPTION


def test_get_conseil_municipal_document_pv_degrades_gracefully_without_qdrant():
    """Même garde-fou que search_conseil_municipal_pv/list_conseil_municipal_meetings : le venv de
    test n'installe pas qdrant-client volontairement."""
    import main as main_module

    assert main_module.get_conseil_municipal_document_pv("https://jouy28.com/x.pdf") is None


def test_current_date_block_contains_iso_date():
    """Manque trouvé par Angelo en réel (2026-07-26) : "tu sais quel jour on est ?" ->
    "je n'ai pas accès à l'heure actuelle". Fonction pure, testable sans requête HTTP — vérifie
    juste la présence du format ISO exploitable, pas la formulation exacte en français."""
    import main as main_module
    from datetime import datetime
    from zoneinfo import ZoneInfo

    block = main_module.current_date_block()
    today_iso = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
    assert today_iso in block


@pytest.mark.anyio
async def test_chat_v2_injects_current_date_in_system_prompt(client, logged_in_user, monkeypatch):
    import main as main_module

    captured = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured["system_prompt"] = system_prompt
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "quel jour sommes-nous ?"})

    assert main_module.current_date_block() in captured["system_prompt"]


@pytest.mark.anyio
async def test_chat_v2_passes_wiki_index_and_callable_in_ctx(client, logged_in_user, monkeypatch):
    import main as main_module

    captured_ctx = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured_ctx.update(ctx)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut"})

    assert "charte-anonymat" in captured_ctx["wiki_pages_index"]
    assert callable(captured_ctx["get_wiki_page_fn"])


def test_propose_opinion_action_new_thread_valid():
    import chatbot_actions

    result = chatbot_actions.propose_opinion(
        {"new_thread_title": "Un tout nouveau sujet", "body": "Je pense que..."}, {"threads": []}
    )
    assert result["available"] is True
    assert result["new_thread_title"] == "Un tout nouveau sujet"


def test_propose_opinion_action_new_thread_rejects_exact_title_collision():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 5, "title": "Sujet déjà existant", "summary": None, "opinions": []}]}
    result = chatbot_actions.propose_opinion(
        {"new_thread_title": "Sujet déjà existant", "body": "Je pense que..."}, ctx
    )
    assert result["available"] is False
    assert "thread_id=5" in result["error"]


def test_propose_opinion_action_rejects_both_thread_id_and_new_thread_title():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 1, "title": "Sujet", "summary": None, "opinions": []}]}
    result = chatbot_actions.propose_opinion(
        {"thread_id": 1, "new_thread_title": "Autre titre", "body": "Je pense que..."}, ctx
    )
    assert result["available"] is False


@pytest.mark.anyio
async def test_opinion_confirm_endpoint_creates_new_thread_coupled_with_opinion(client, logged_in_user):
    import main as main_module

    with main.db() as conn:
        conn.execute("DELETE FROM threads WHERE title=?", ("Fil créé via endpoint couplé",))

    resp = await client.post("/opinion/confirm", json={
        "session_token": logged_in_user["session_token"],
        "new_thread_title": "Fil créé via endpoint couplé",
        "body": "Je pense que...",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "published"

    with main.db() as conn:
        row = conn.execute("SELECT status FROM threads WHERE title=?", ("Fil créé via endpoint couplé",)).fetchone()
    assert row["status"] == "published"


@pytest.mark.anyio
async def test_opinion_confirm_endpoint_rejects_both_thread_id_and_new_thread_title(client, logged_in_user):
    resp = await client.post("/opinion/confirm", json={
        "session_token": logged_in_user["session_token"],
        "thread_id": 1, "new_thread_title": "Autre titre", "body": "Je pense que...",
    })
    assert resp.status_code == 400


def test_mcp_chatbot_executor_shares_same_system_prompt_as_prod():
    """Régression (2026-07-26, signalé par angelobot) : mcp_chatbot_executor.py (outil de test
    utilisé par angelobot) avait son propre BASE_PROMPT dupliqué à la main, jamais mis à jour avec
    les règles anonymat/modération/iconifiable ajoutées la veille — comportement de test différent
    de la vraie prod sans que personne ne s'en aperçoive avant qu'angelobot ne le remarque. Fix :
    import partagé (chatbot_actions.CHAT_SYSTEM_PROMPT), plus de copie possible. Ce test verrouille
    l'IDENTITÉ (pas juste la ressemblance) pour qu'une future dérive soit détectée immédiatement."""
    import chatbot_actions
    import main as main_module
    import mcp_chatbot_executor

    assert mcp_chatbot_executor.BASE_PROMPT is chatbot_actions.CHAT_SYSTEM_PROMPT
    assert main_module.CHAT_SYSTEM_PROMPT is chatbot_actions.CHAT_SYSTEM_PROMPT


def test_mcp_chatbot_executor_ctx_includes_forum_threads_fixture():
    """Régression connexe (2026-07-26) : ctx ne contenait pas la clé "threads" du tout — list_threads/
    get_thread/propose_opinion voyaient toujours une liste vide dans les tests d'angelobot, un
    comportement structurellement différent de la vraie prod (qui a toujours un snapshot réel, même
    vide) plutôt qu'une clé absente."""
    import mcp_chatbot_executor

    assert isinstance(mcp_chatbot_executor._TEST_THREADS, list)
    assert len(mcp_chatbot_executor._TEST_THREADS) > 0
    assert "thread_id" in mcp_chatbot_executor._TEST_THREADS[0]


def test_mcp_chatbot_executor_report_bug_never_sends_real_email():
    """Régression connexe (2026-07-26) : les callables report_bug_fn/request_admin_intervention_fn
    du mode test MCP ne doivent JAMAIS déclencher un vrai envoi Brevo, pour ne pas spammer
    ADMIN_BUG_EMAIL à chaque essai d'angelobot."""
    import mcp_chatbot_executor

    result = mcp_chatbot_executor._mock_report_bug_fn("test")
    assert result["sent"] is False
    result2 = mcp_chatbot_executor._mock_request_admin_intervention_fn("test")
    assert result2["sent"] is False


def test_tools_description_lists_actual_palette_no_stale_count():
    """Régression (2026-07-25, via angelobot) : TOOLS_DESCRIPTION disait "parmi les 8" en dur,
    devenu faux dès l'ajout de "gris" (9e couleur) — pire, le modèle n'avait jamais la liste
    réelle des couleurs nulle part dans le prompt, seulement un compte. La palette est maintenant
    injectée dynamiquement depuis PSEUDO_COLORS, donc toujours synchronisée."""
    import chatbot_actions

    assert "__PALETTE_COULEURS__" not in chatbot_actions.TOOLS_DESCRIPTION
    for color in chatbot_actions.PSEUDO_COLORS:
        assert color in chatbot_actions.TOOLS_DESCRIPTION
    assert "parmi les 8" not in chatbot_actions.TOOLS_DESCRIPTION


def test_pseudo_words_excludes_non_iconifiable_words():
    """Régression (2026-07-25, critère 'iconifiable' demandé par Angelo via angelobot) : Aurore,
    Clairière, Frimas, Brume jugés trop abstraits/atmosphériques pour un logo simple — retirés du
    générateur déterministe (PSEUDO_WORDS) pour cohérence avec le nouveau critère de rejet
    (voir CHAT_SYSTEM_PROMPT). Restent dans PSEUDO_WORD_GENDER : ce dict sert aussi à accorder
    correctement un mot LIBREMENT proposé par un utilisateur (propose_custom_pseudo), pas
    seulement les mots que le système génère lui-même."""
    import chatbot_actions

    for word in ("Aurore", "Clairière", "Frimas", "Brume"):
        assert word not in chatbot_actions.PSEUDO_WORDS
        assert word in chatbot_actions.PSEUDO_WORD_GENDER


def test_chat_system_prompt_includes_iconifiable_rejection_criterion():
    """Régression (2026-07-25) : le critère iconifiable doit être un motif de refus DUR (même
    mécanique que appropriate=false pour la connotation), pas une simple nuance de ton — avec les
    exemples de contraste exacts donnés par le développeur."""
    import main as main_module

    prompt = main_module.CHAT_SYSTEM_PROMPT
    assert "ICONIFIABLE" in prompt
    for good_example in ("Renard", "Hibou", "Chêne", "Faucon", "Comète", "Phare", "Écureuil", "Corail"):
        assert good_example in prompt
    for bad_example in ("Aurore", "Clairière", "Frimas", "Brume"):
        assert bad_example in prompt
    assert "appropriate=false" in prompt


@pytest.mark.anyio
async def test_pseudo_confirm_endpoint_requires_valid_session(client):
    resp = await client.post("/pseudo/confirm", json={"session_token": "fake", "word": "Renard", "color": "bleu"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_pseudo_confirm_endpoint_rejects_invalid_color(client, logged_in_user):
    resp = await client.post(
        "/pseudo/confirm",
        json={"session_token": logged_in_user["session_token"], "word": "Inventé", "color": "rose-fluo"},
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_pseudo_confirm_endpoint_succeeds_then_allows_free_rechoice(client, logged_in_user):
    """Rechoix libre (2026-07-25) : une 2e confirmation REMPLACE la précédente, ne renvoie plus
    409 — voir main.confirm_pseudo pour le TODO sur le futur garde-fou "rien publié"."""
    resp1 = await client.post(
        "/pseudo/confirm",
        json={"session_token": logged_in_user["session_token"], "word": "Renard", "color": "bleu"},
    )
    assert resp1.status_code == 200
    assert resp1.json() == {"word": "Renard", "color": "bleu"}

    resp2 = await client.post(
        "/pseudo/confirm",
        json={"session_token": logged_in_user["session_token"], "word": "Hibou", "color": "vert"},
    )
    assert resp2.status_code == 200
    assert resp2.json() == {"word": "Hibou", "color": "vert"}


@pytest.mark.anyio
async def test_chat_v2_injects_onboarding_block_until_pseudo_confirmed(client, logged_in_user, monkeypatch):
    """Signal 'Nouveau, sans pseudo' : le bloc onboarding doit apparaître dans le system_prompt
    tant qu'aucun pseudo n'est confirmé, et disparaître juste après confirmation — sur autant de
    tours que nécessaire avant, pas seulement le tout premier appel."""
    import main as main_module

    captured_prompts = []

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured_prompts.append(system_prompt)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)

    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut"})
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "encore avant de choisir"})
    assert "Premier contact" in captured_prompts[0]
    assert "Premier contact" in captured_prompts[1]  # toujours actif au 2e tour, pas juste le 1er


@pytest.mark.anyio
async def test_chat_v2_passes_taken_pseudos_in_ctx(client, logged_in_user, monkeypatch):
    import main as main_module

    other_identity = "identity-pseudo-taken-ctx-test"
    other_debate_token = main_module.compute_debate_token(other_identity)
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (other_debate_token,))
    main_module.confirm_pseudo(other_identity, "Genêt", "noir")

    captured_ctx = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured_ctx.update(ctx)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut"})
    assert ("Genêt", "noir") in captured_ctx["taken_pseudos"]

    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (other_debate_token,))


def test_get_public_forum_snapshot_excludes_drafts_and_shows_author_pseudo():
    """Forum phase 2 (2026-07-25) : le snapshot public ne contient QUE des fils/opinions
    'published' — jamais un brouillon (le sien ou celui d'un autre), cohérent avec la règle
    'jamais de corrélation privé↔privé' déjà tranchée sur le wiki. L'auteur d'une opinion est
    affiché sous son pseudo accordé grammaticalement, jamais son identité réelle."""
    import main as main_module

    author_identity = "identity-forum-snapshot-author"
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (main_module.compute_debate_token(author_identity),))
    main_module.confirm_pseudo(author_identity, "Clairière", "vert")

    published_thread = main_module.create_thread("Fil publié pour snapshot")
    main_module.publish_thread(published_thread["thread_id"])
    draft_thread = main_module.create_thread("Fil resté en brouillon")

    published_opinion = main_module.create_opinion(published_thread["thread_id"], author_identity, "Opinion publiée")
    main_module.publish_opinion(published_opinion["opinion_id"])
    draft_opinion = main_module.create_opinion(published_thread["thread_id"], author_identity, "Opinion restée brouillon")

    snapshot = main_module.get_public_forum_snapshot()
    thread_ids = {t["thread_id"] for t in snapshot}
    assert published_thread["thread_id"] in thread_ids
    assert draft_thread["thread_id"] not in thread_ids  # brouillon jamais exposé

    found = next(t for t in snapshot if t["thread_id"] == published_thread["thread_id"])
    opinion_ids = {o["opinion_id"] for o in found["opinions"]}
    assert published_opinion["opinion_id"] in opinion_ids
    assert draft_opinion["opinion_id"] not in opinion_ids  # brouillon jamais exposé

    published_row = next(o for o in found["opinions"] if o["opinion_id"] == published_opinion["opinion_id"])
    assert published_row["auteur"] == "Clairière verte"  # pseudo accordé, jamais l'identité réelle


def test_get_forum_page_snapshot_includes_remarques_excludes_drafts():
    """Page "Forum" (2026-07-26) : contrairement à get_public_forum_snapshot (ctx du chatbot,
    volontairement allégé), la page humaine inclut aussi les remarques publiées de chaque fil,
    avec le même filtre 'jamais un brouillon' et la même attribution par pseudo que les
    opinions."""
    import main as main_module

    author_identity = "identity-forum-page-author"
    with main.db() as conn:
        conn.execute("DELETE FROM pseudos WHERE debate_token=?", (main_module.compute_debate_token(author_identity),))
    main_module.confirm_pseudo(author_identity, "Renard", "orange")

    thread = main_module.create_thread("Fil pour page Forum")
    main_module.publish_thread(thread["thread_id"])

    published_remarque = main_module.create_remarque(thread["thread_id"], author_identity, "Une remarque publiée")
    main_module.publish_remarque(published_remarque["remarque_id"])
    draft_remarque = main_module.create_remarque(thread["thread_id"], author_identity, "Une remarque restée brouillon")

    snapshot = main_module.get_forum_page_snapshot()
    found = next(t for t in snapshot if t["thread_id"] == thread["thread_id"])
    remarque_ids = {r["remarque_id"] for r in found["remarques"]}
    assert published_remarque["remarque_id"] in remarque_ids
    assert draft_remarque["remarque_id"] not in remarque_ids  # brouillon jamais exposé

    published_row = next(r for r in found["remarques"] if r["remarque_id"] == published_remarque["remarque_id"])
    assert published_row["auteur"] == "Renard orange"


def test_get_forum_page_snapshot_includes_reactions_on_opinions():
    """Régression : oubli de spec initial signalé par Angelo (2026-07-26) — la page Forum
    n'affichait aucune réaction sur les opinions (contrairement à Mon activité, qui les affichait
    déjà). Chaque opinion du snapshot Forum doit maintenant porter le même reaction_counts/
    reactions que get_my_activity (voir _get_opinion_reaction_summary, factorisé entre les 2)."""
    import main as main_module

    author_identity = "identity-forum-page-reactions-author"
    reactor_identity = "identity-forum-page-reactions-reactor"
    with main.db() as conn:
        conn.execute(
            "DELETE FROM pseudos WHERE debate_token IN (?, ?)",
            (main_module.compute_debate_token(author_identity), main_module.compute_debate_token(reactor_identity)),
        )
    main_module.confirm_pseudo(reactor_identity, "Sittelle", "jaune")

    thread = main_module.create_thread("Fil pour réactions page Forum")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], author_identity, "Une opinion à réactions")
    main_module.publish_opinion(opinion["opinion_id"])
    reaction = main_module.add_reaction(opinion["opinion_id"], reactor_identity, "opposer", "Pas convaincu")
    main_module.publish_reaction(reaction["reaction_id"])

    snapshot = main_module.get_forum_page_snapshot()
    found_thread = next(t for t in snapshot if t["thread_id"] == thread["thread_id"])
    found_opinion = next(o for o in found_thread["opinions"] if o["opinion_id"] == opinion["opinion_id"])
    assert found_opinion["reaction_counts"] == {"adherer": 0, "opposer": 1, "neutre": 0}
    assert found_opinion["reactions"] == [{"auteur": "Sittelle jaune", "stance": "opposer", "argumentaire": "Pas convaincu"}]


@pytest.mark.anyio
async def test_forum_snapshot_endpoint_public_no_auth(client):
    """Lecture publique, aucun session_token requis — cohérent avec le fait que le contenu est
    déjà public par construction (opinions/remarques publiées, visibles en conversation avec le
    chatbot par n'importe qui)."""
    import main as main_module

    thread = main_module.create_thread("Fil pour endpoint /forum/snapshot")
    main_module.publish_thread(thread["thread_id"])

    resp = await client.get("/forum/snapshot")
    assert resp.status_code == 200
    thread_ids = {t["thread_id"] for t in resp.json()["threads"]}
    assert thread["thread_id"] in thread_ids


@pytest.mark.anyio
async def test_spa_fallback_routes_serve_app_for_forum_and_activite(client):
    """Régression trouvée en vérifiant en conditions réelles (Playwright) : /forum et
    /mon-activite avaient été ajoutées au routeur client (ROUTES en JS) mais pas à _SPA_ROUTES
    côté serveur — une navigation directe ou un rafraîchissement sur ces pages renvoyait un 404
    au lieu de l'app (StaticFiles seul ne sait servir que des fichiers existants sur disque)."""
    for path in ("/forum", "/mon-activite"):
        resp = await client.get(path)
        assert resp.status_code == 200, f"{path} devrait servir l'app, pas un 404"


def test_get_my_activity_counts_only_latest_reaction_per_reactor():
    """Page "Mon activité" (2026-07-26) : le décompte adhérer/opposer/neutre ne compte QUE la
    réaction la plus récente par réacteur (même règle que get_current_reaction) — si quelqu'un
    change d'avis (opposer puis adherer), une seule réaction doit compter au final, jamais les
    deux. Attribution par pseudo confirmée par le développeur (2026-07-26, via angelobot)."""
    import main as main_module

    author_identity = "identity-activity-author"
    reactor_identity = "identity-activity-reactor-flipflop"
    with main.db() as conn:
        conn.execute(
            "DELETE FROM pseudos WHERE debate_token IN (?, ?)",
            (main_module.compute_debate_token(author_identity), main_module.compute_debate_token(reactor_identity)),
        )
    main_module.confirm_pseudo(reactor_identity, "Hibou", "bleu")

    thread = main_module.create_thread("Fil pour Mon activité")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], author_identity, "Mon opinion", "Mon argumentaire")
    main_module.publish_opinion(opinion["opinion_id"])

    r1 = main_module.add_reaction(opinion["opinion_id"], reactor_identity, "opposer", "D'abord contre")
    main_module.publish_reaction(r1["reaction_id"])
    r2 = main_module.add_reaction(opinion["opinion_id"], reactor_identity, "adherer", "Finalement pour")
    main_module.publish_reaction(r2["reaction_id"])

    activity = main_module.get_my_activity(author_identity)
    found = next(o for o in activity["opinions"] if o["opinion_id"] == opinion["opinion_id"])
    assert found["reaction_counts"] == {"adherer": 1, "opposer": 0, "neutre": 0}
    assert len(found["reactions"]) == 1
    assert found["reactions"][0]["auteur"] == "Hibou bleu"
    assert found["reactions"][0]["argumentaire"] == "Finalement pour"


def test_get_my_activity_excludes_other_users_opinions():
    """Page "Mon activité" ne renvoie que les opinions de l'utilisateur connecté, jamais celles
    d'un autre — filtrées sur son propre debate_token, jamais un paramètre arbitraire."""
    import main as main_module

    me = "identity-activity-me"
    someone_else = "identity-activity-someone-else"

    thread = main_module.create_thread("Fil pour exclusion Mon activité")
    main_module.publish_thread(thread["thread_id"])
    my_opinion = main_module.create_opinion(thread["thread_id"], me, "Mon avis à moi")
    main_module.publish_opinion(my_opinion["opinion_id"])
    other_opinion = main_module.create_opinion(thread["thread_id"], someone_else, "L'avis de quelqu'un d'autre")
    main_module.publish_opinion(other_opinion["opinion_id"])

    activity = main_module.get_my_activity(me)
    opinion_ids = {o["opinion_id"] for o in activity["opinions"]}
    assert my_opinion["opinion_id"] in opinion_ids
    assert other_opinion["opinion_id"] not in opinion_ids


@pytest.mark.anyio
async def test_activity_mine_endpoint_requires_valid_session(client):
    resp = await client.post("/activity/mine", json={"session_token": "session-invalide"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_activity_mine_endpoint_returns_own_opinions(client, logged_in_user):
    import main as main_module

    thread = main_module.create_thread("Fil pour endpoint /activity/mine")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], logged_in_user["token"], "Mon opinion via endpoint")
    main_module.publish_opinion(opinion["opinion_id"])

    resp = await client.post("/activity/mine", json={"session_token": logged_in_user["session_token"]})
    assert resp.status_code == 200
    opinion_ids = {o["opinion_id"] for o in resp.json()["opinions"]}
    assert opinion["opinion_id"] in opinion_ids


@pytest.mark.anyio
async def test_chat_v2_passes_public_threads_snapshot_in_ctx(client, logged_in_user, monkeypatch):
    import main as main_module

    thread = main_module.create_thread("Fil visible depuis chat/v2")
    main_module.publish_thread(thread["thread_id"])

    captured_ctx = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured_ctx.update(ctx)
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)
    await client.post("/chat/v2", json={"session_token": logged_in_user["session_token"], "message": "salut"})
    assert any(t["thread_id"] == thread["thread_id"] for t in captured_ctx["threads"])


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


@pytest.mark.anyio
async def test_run_turn_omits_trace_key_by_default(mocked_openrouter_structured):
    """Mode debug/traçage (demande développeur 2026-07-25, via angelobot, pour ne plus déboguer à
    l'aveugle via captures d'écran) : par défaut (trace=False), aucune clé "trace" dans le retour —
    ne coûte rien et ne change pas la forme de la réponse pour un usage normal."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Bonjour !"}]}))

    result = chatbot_executor.run_turn("system", [{"role": "user", "content": "salut"}], {})
    assert "trace" not in result


@pytest.mark.anyio
async def test_run_turn_trace_captures_raw_completion_and_carried_result(mocked_openrouter_structured):
    """Avec trace=True : une entrée par itération LLM, avec la complétion brute, les actions
    parsées, et l'état de carried_result en entrée/sortie — exactement ce qui manquait pour
    diagnostiquer les bugs #7/#8/#9 sans capture d'écran."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [{"action": "propose_pseudo_candidates", "index": 0, "appropriate": True}]
    }))
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "{{résultat}} ?"}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "un pseudo"}], {"identity_token": "tok-trace"}, trace=True,
    )
    assert result["error"] is None
    assert len(result["trace"]) == 2

    first = result["trace"][0]
    assert first["iteration"] == 0
    assert "propose_pseudo_candidates" in first["raw_completion"]
    assert first["parsed_actions"][0]["action"] == "propose_pseudo_candidates"
    assert first["carried_result_in"] is None
    assert first["carried_result_out"]["display"] is not None

    second = result["trace"][1]
    assert second["carried_result_in"] == first["carried_result_out"]
    assert second["carried_result_out"] is None


def test_list_summaries_action_reads_from_ctx():
    import chatbot_actions

    fake_summaries = [{"id": 1, "summary": "test", "created_at": "2026-07-25T00:00:00"}]
    result = chatbot_actions.list_summaries({}, {"summaries": fake_summaries})
    assert result["summaries"] == fake_summaries


def test_list_summaries_action_defaults_to_empty_list():
    import chatbot_actions

    assert chatbot_actions.list_summaries({}, {}) == {"summaries": []}


def test_tools_description_instructs_spontaneous_list_summaries_at_conversation_start():
    """Régression (2026-07-26, demande développeur) : le modèle n'appelait pas spontanément
    list_summaries en début de conversation, répondant parfois "je n'ai rien en cours" alors que
    des résumés pertinents existaient déjà — une nouvelle conversation signifie que LA MÉMOIRE DU
    MODÈLE est vide, pas que l'utilisateur n'a rien en cours."""
    import chatbot_actions

    assert "EN DÉBUT DE CONVERSATION" in chatbot_actions.TOOLS_DESCRIPTION
    assert "je n'ai rien en cours" in chatbot_actions.TOOLS_DESCRIPTION.lower()


def test_tools_description_generalizes_no_promise_without_action_to_forum_drafts():
    """Régression bug réel #14 (2026-07-26, capture développeur via angelobot) : le modèle a
    rédigé un brouillon d'opinion complet en prose libre dans un say_user, sans jamais appeler
    propose_opinion, tout en promettant un bouton de confirmation à venir — la règle "n'affirme
    jamais qu'un bouton va apparaître sans avoir appelé l'action dans ce même lot" n'était écrite
    que pour les 2 actions pseudo, jamais généralisée aux actions forum."""
    import chatbot_actions

    assert "RÈGLE GÉNÉRALISÉE" in chatbot_actions.TOOLS_DESCRIPTION
    assert "jamais un brouillon uniquement" in chatbot_actions.TOOLS_DESCRIPTION.lower()


def test_tools_description_documents_search_conseil_municipal_with_citation_rule():
    """RAG conseil municipal (2026-07-26) : la consigne doit exiger de citer la source précise et
    d'admettre l'absence de résultat plutôt que d'inventer — jamais répondre "de mémoire" sur ce
    sujet (le prompt statique interdisait auparavant explicitement cette capacité, remplacé)."""
    import chatbot_actions

    assert "search_conseil_municipal" in chatbot_actions.TOOLS_DESCRIPTION
    assert "n'as PAS ENCORE accès" not in chatbot_actions.CHAT_SYSTEM_PROMPT
    assert "search_conseil_municipal" in chatbot_actions.CHAT_SYSTEM_PROMPT


def test_tools_description_disambiguates_summary_vs_opinion_draft():
    """Même régression bug #14 : au tour suivant, le modèle a appelé propose_summary (brouillon de
    résumé PRIVÉ) alors que le contexte demandait un brouillon d'opinion pour le Forum — les 2
    actions partagent le mot "brouillon" dans leur description sans distinction explicite."""
    import chatbot_actions

    assert "DÉSAMBIGUÏSATION" in chatbot_actions.TOOLS_DESCRIPTION


def test_tools_description_forbids_inventing_details_absent_from_summary():
    """Même régression bug #14 : le modèle a inventé un nom de rue, un carrefour et un type de
    panneau absents du résumé privé source, en rédigeant un brouillon d'opinion trop "concret"."""
    import chatbot_actions

    assert "n'invente JAMAIS un nom de rue" in chatbot_actions.TOOLS_DESCRIPTION


def test_list_threads_action_strips_opinions_from_ctx():
    """Forum phase 2 (2026-07-25) : list_threads ne renvoie que titre/résumé, jamais les opinions
    à l'intérieur — évite de charger tout le contenu du forum juste pour une vue d'ensemble."""
    import chatbot_actions

    fake_threads = [
        {"thread_id": 1, "title": "Sujet A", "summary": "résumé A", "opinions": [{"opinion_id": 1, "body": "..."}]},
        {"thread_id": 2, "title": "Sujet B", "summary": None, "opinions": []},
    ]
    result = chatbot_actions.list_threads({}, {"threads": fake_threads})
    assert result["threads"] == [
        {"thread_id": 1, "title": "Sujet A", "summary": "résumé A"},
        {"thread_id": 2, "title": "Sujet B", "summary": None},
    ]


def test_list_threads_action_defaults_to_empty_list():
    import chatbot_actions

    assert chatbot_actions.list_threads({}, {}) == {"threads": []}


def test_get_thread_action_returns_full_detail_including_opinions():
    import chatbot_actions

    fake_threads = [
        {"thread_id": 1, "title": "Sujet A", "summary": "résumé A", "opinions": [{"opinion_id": 1, "body": "Une opinion"}]},
    ]
    result = chatbot_actions.get_thread({"thread_id": 1}, {"threads": fake_threads})
    assert result == fake_threads[0]


def test_get_thread_action_errors_on_unknown_thread_id():
    import chatbot_actions

    result = chatbot_actions.get_thread({"thread_id": 999}, {"threads": []})
    assert "error" in result


# ===== Forum phase 3 (2026-07-25) : propose_opinion/propose_reaction/propose_remarque — =====
# ===== lecture/validation pure, aucun write, même principe que propose_pseudo_candidates =====


def test_propose_opinion_action_valid_thread():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 1, "title": "Sujet", "summary": None, "opinions": []}]}
    result = chatbot_actions.propose_opinion({"thread_id": 1, "body": "Je pense que...", "argumentaire": "parce que..."}, ctx)
    assert result["available"] is True
    assert result["thread_title"] == "Sujet"
    assert result["body"] == "Je pense que..."


def test_propose_opinion_action_rejects_unknown_or_unpublished_thread():
    import chatbot_actions

    result = chatbot_actions.propose_opinion({"thread_id": 999, "body": "Je pense que..."}, {"threads": []})
    assert result["available"] is False
    assert "error" in result


def test_propose_opinion_action_rejects_empty_body():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 1, "title": "Sujet", "summary": None, "opinions": []}]}
    result = chatbot_actions.propose_opinion({"thread_id": 1, "body": "   "}, ctx)
    assert result["available"] is False


def test_propose_reaction_action_valid_opinion():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 1, "title": "Sujet", "summary": None, "opinions": [
        {"opinion_id": 42, "auteur": "Renard bleu", "body": "Une opinion", "argumentaire": None, "superseded_by_opinion_id": None},
    ]}]}
    result = chatbot_actions.propose_reaction({"opinion_id": 42, "stance": "adherer"}, ctx)
    assert result["available"] is True
    assert result["opinion_body"] == "Une opinion"


def test_propose_reaction_action_rejects_invalid_stance():
    import chatbot_actions

    result = chatbot_actions.propose_reaction({"opinion_id": 42, "stance": "pour"}, {"threads": []})
    assert result["available"] is False
    assert result["error"] == "stance invalide"


def test_propose_reaction_action_rejects_unknown_opinion():
    import chatbot_actions

    result = chatbot_actions.propose_reaction({"opinion_id": 999, "stance": "neutre"}, {"threads": []})
    assert result["available"] is False


def test_propose_remarque_action_valid():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 1, "title": "Sujet", "summary": None, "opinions": []}]}
    result = chatbot_actions.propose_remarque({"thread_id": 1, "body": "Bonjour !"}, ctx)
    assert result["available"] is True


def test_propose_remarque_action_rejects_both_reply_targets():
    import chatbot_actions

    ctx = {"threads": [{"thread_id": 1, "title": "Sujet", "summary": None, "opinions": []}]}
    result = chatbot_actions.propose_remarque(
        {"thread_id": 1, "body": "Impossible", "reply_to_remarque_id": 1, "reply_to_opinion_id": 2}, ctx
    )
    assert result["available"] is False


def test_propose_remarque_action_rejects_unknown_thread():
    import chatbot_actions

    result = chatbot_actions.propose_remarque({"thread_id": 999, "body": "Bonjour"}, {"threads": []})
    assert result["available"] is False


def test_create_opinion_rejects_unpublished_or_unknown_thread():
    """Durcissement 2026-07-25 (phase 3) : on n'attache jamais une opinion à un fil en brouillon
    ou inexistant, même si l'appel vient directement de Python (pas seulement via propose_opinion,
    qui filtre déjà via ctx["threads"] — filet de sécurité côté fonction elle-même)."""
    import main as main_module

    with pytest.raises(ValueError, match="introuvable"):
        main_module.create_opinion(999999, "identity-forum-hardening-1", "Une opinion")

    draft_thread = main_module.create_thread("Fil non publié pour test durcissement")
    with pytest.raises(ValueError, match="pas encore publié"):
        main_module.create_opinion(draft_thread["thread_id"], "identity-forum-hardening-2", "Une opinion")


def test_add_reaction_rejects_unpublished_or_unknown_opinion():
    import main as main_module

    with pytest.raises(ValueError, match="introuvable"):
        main_module.add_reaction(999999, "identity-forum-hardening-3", "adherer")

    thread = main_module.create_thread("Fil pour test durcissement réaction")
    main_module.publish_thread(thread["thread_id"])
    draft_opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-hardening-4", "Opinion brouillon")
    with pytest.raises(ValueError, match="n'est pas publiée"):
        main_module.add_reaction(draft_opinion["opinion_id"], "identity-forum-hardening-5", "adherer")


def test_create_remarque_rejects_unpublished_or_unknown_thread():
    import main as main_module

    with pytest.raises(ValueError, match="introuvable"):
        main_module.create_remarque(999999, "identity-forum-hardening-6", "Bonjour")

    draft_thread = main_module.create_thread("Fil non publié pour test remarque")
    with pytest.raises(ValueError, match="pas encore publié"):
        main_module.create_remarque(draft_thread["thread_id"], "identity-forum-hardening-7", "Bonjour")


@pytest.mark.anyio
async def test_opinion_confirm_endpoint_creates_and_publishes_in_one_call(client, logged_in_user):
    import main as main_module

    thread = main_module.create_thread("Fil pour test endpoint opinion")
    main_module.publish_thread(thread["thread_id"])

    resp = await client.post("/opinion/confirm", json={
        "session_token": logged_in_user["session_token"], "thread_id": thread["thread_id"],
        "body": "Je pense que...", "argumentaire": "parce que...",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "published"


@pytest.mark.anyio
async def test_opinion_confirm_endpoint_rejects_unpublished_thread(client, logged_in_user):
    import main as main_module

    draft_thread = main_module.create_thread("Fil brouillon pour test endpoint opinion rejet")

    resp = await client.post("/opinion/confirm", json={
        "session_token": logged_in_user["session_token"], "thread_id": draft_thread["thread_id"],
        "body": "Je pense que...",
    })
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_reaction_confirm_endpoint_creates_and_publishes_in_one_call(client, logged_in_user):
    import main as main_module

    thread = main_module.create_thread("Fil pour test endpoint réaction")
    main_module.publish_thread(thread["thread_id"])
    opinion = main_module.create_opinion(thread["thread_id"], "identity-forum-endpoint-reaction", "Une opinion")
    main_module.publish_opinion(opinion["opinion_id"])

    resp = await client.post("/reaction/confirm", json={
        "session_token": logged_in_user["session_token"], "opinion_id": opinion["opinion_id"], "stance": "adherer",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "published"


@pytest.mark.anyio
async def test_remarque_confirm_endpoint_creates_and_publishes_in_one_call(client, logged_in_user):
    import main as main_module

    thread = main_module.create_thread("Fil pour test endpoint remarque")
    main_module.publish_thread(thread["thread_id"])

    resp = await client.post("/remarque/confirm", json={
        "session_token": logged_in_user["session_token"], "thread_id": thread["thread_id"], "body": "Bonjour !",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "published"


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


def test_substitute_placeholder_excludes_boolean_flags():
    """Régression (bug réel trouvé en conditions réelles le 2026-07-25, 'Roseau orange True') :
    un résultat contenant un champ booléen de statut (ex: "available") ne doit jamais apparaître
    littéralement dans le texte affiché — seules les valeurs de CONTENU (mot, couleur...) sont
    jointes, jamais les indicateurs internes True/False."""
    import chatbot_executor

    result = chatbot_executor._substitute_placeholder(
        "Voici : {{résultat}}",
        {"word": "Roseau", "color": "orange", "available": True},
    )
    assert result == "Voici : Roseau orange"
    assert "True" not in result
    assert "False" not in result


def test_substitute_placeholder_prefers_display_field():
    """"display" (forme accordée grammaticalement) doit primer sur la jointure brute word+color,
    pour que {{résultat}} rende "Clairière verte" et non "Clairière vert"."""
    import chatbot_executor

    result = chatbot_executor._substitute_placeholder(
        "Que penses-tu de {{résultat}} ?",
        {"word": "Clairière", "color": "vert", "display": "Clairière verte", "available": True},
    )
    assert result == "Que penses-tu de Clairière verte ?"


def test_substitute_placeholder_strips_markdown_wrapping_when_unresolved():
    """Régression bug réel #8 (2026-07-25, capture développeur : "Que dirais-tu de **** ?" revenu
    après le fix carried_result) : malgré la consigne TOOLS_DESCRIPTION interdisant d'entourer
    "{{résultat}}" de "**", le LLM l'a quand même fait. Sans previous_result à substituer, un
    simple retrait du token laissait "**" + "**" adjacents qui collent visuellement en "****". Le
    wrapping doit être retiré EN MÊME TEMPS que le token, pas seulement le token seul."""
    import chatbot_executor

    result = chatbot_executor._substitute_placeholder("Que dirais-tu de **{{résultat}}** ?", None)
    assert "****" not in result
    assert "**" not in result

    result_quotes = chatbot_executor._substitute_placeholder("Que dirais-tu de «{{résultat}}» ?", None)
    assert "«»" not in result_quotes


@pytest.mark.anyio
async def test_run_turn_fills_empty_say_user_after_proposal_with_fallback_text(mocked_openrouter_structured):
    """Régression (root cause d'une répétition signalée par le développeur, 2026-07-25,
    "Clairière vert" reproposé 5 fois) : le LLM laisse parfois say_user vide juste après une
    action de proposition — sans texte, le filtre frontend anti-bulle-vide masque la bulle,
    effaçant toute trace du candidat dans l'historique renvoyé au modèle au tour suivant, qui
    "oublie" alors ce qu'il a déjà proposé et repart d'index=0. Le filet de sécurité doit
    remplir automatiquement un texte de repli plutôt que de laisser un vide silencieux."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "propose_pseudo_candidates", "index": 0, "appropriate": True},
            {"action": "say_user", "text": ""},
        ]
    }))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}], {"identity_token": "tok-abc", "taken_pseudos": set()}
    )
    assert result["error"] is None
    assert result["replies"] != [""]
    assert result["replies"][0].strip() != ""


@pytest.mark.anyio
async def test_run_turn_replaces_text_when_result_content_missing_despite_nonempty_text(mocked_openrouter_structured):
    """Régression (bug réel #6, mesuré à ~2/5 en conditions réelles malgré une clarification de
    prompt, 2026-07-25) : le LLM écrit parfois un texte NON VIDE mais avec un "trou" ("Que
    penses-tu de **** ?") — sans jamais avoir inclus {{résultat}}, donc rien à substituer, et le
    texte n'est pas vide donc le filet du bug #5 ne se déclenche pas. Fix : si le contenu réel du
    résultat (word/color accordé) n'apparaît nulle part dans le texte final, remplacer
    entièrement par le rendu garanti plutôt que de laisser fuiter un message à moitié vide."""
    import json
    import chatbot_actions
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "propose_pseudo_candidates", "index": 0, "appropriate": True},
            {"action": "say_user", "text": "Que penses-tu de **** ? Clique pour confirmer."},
        ]
    }))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}], {"identity_token": "tok-abc", "taken_pseudos": set()}
    )
    assert result["error"] is None
    assert "****" not in result["replies"][0]
    # Le texte est remplacé par le rendu garanti (display) plutôt que le texte troué du modèle.
    candidate = chatbot_actions.generate_pseudo_candidates("tok-abc", n=1)[0]
    expected_display = chatbot_actions._agree_pseudo_display(candidate["word"], candidate["color"])
    assert result["replies"][0] == expected_display


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
def test_substitute_placeholder_ignores_list_valued_entries():
    """Régression (bug réel signalé par Angelo avec capture d'écran, 2026-07-25) : un résultat
    dont une valeur est une LISTE (ex: l'ancien format {"candidates": [...]} de
    propose_pseudo_candidates, plus produit aujourd'hui mais le filet de sécurité doit rester
    généraliste pour toute future action qui renverrait une structure composite) produisait un
    repr Python brut avec des guillemets simples, affiché tel quel dans le chat. Test unitaire
    direct de la fonction plutôt que via une action réelle, puisque plus aucune action du registre
    actuel ne renvoie ce type de forme — la défense doit néanmoins rester active."""
    import chatbot_executor

    result = chatbot_executor._substitute_placeholder(
        "Voici quelques idées : {{résultat}}",
        {"candidates": [{"word": "Falaise", "color": "rouge"}, {"word": "Genêt", "color": "vert"}]},
    )
    assert "{{résultat}}" not in result
    assert "{" not in result
    assert "[" not in result


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
async def test_run_turn_carries_result_across_relaunch_for_say_user_fallback(mocked_openrouter_structured):
    """Régression (bug réel #7, root cause profonde des bugs #5/#6, mesurée à ~2 échecs sur 5 en
    conditions réelles même après leurs fixes) : quand un lot se termine par une action non-parole
    (relance LLM), le résultat de cette action doit rester disponible pour le say_user qui arrive
    dans la COMPLÉTION SUIVANTE — avant ce fix, "previous_result" était réinitialisé à None au
    début de chaque nouvelle itération, donc le say_user "troué" du 2e appel n'avait plus aucune
    trace du résultat à substituer/vérifier."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    # 1er appel : le lot se termine par l'action de proposition SEULE, sans say_user → relance.
    responses.append(json.dumps({
        "actions": [{"action": "propose_pseudo_candidates", "index": 0, "appropriate": True}]
    }))
    # 2e appel (après relance) : say_user "troué", sans avoir jamais inclus {{résultat}}.
    responses.append(json.dumps({
        "actions": [{"action": "say_user", "text": "Que penses-tu de **** ?"}]
    }))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}], {"identity_token": "tok-abc", "taken_pseudos": set()}
    )
    assert result["error"] is None
    assert len(calls) == 2
    assert "****" not in result["replies"][0]
    assert result["replies"][0].strip() != ""


@pytest.mark.anyio
async def test_run_turn_resolves_say_user_placed_before_propose_in_same_batch(mocked_openrouter_structured):
    """Régression bug réel #10 (2026-07-25, trouvé via le mode debug/traçage tout juste déployé,
    différent du bug #7 qui était INTER-itérations) : dans un même lot, le modèle a placé
    say_user (avec "{{résultat}}") AVANT propose_pseudo_candidates au lieu d'après — previous_result
    encore vide à ce moment précis de la boucle, donc rien à substituer ("Que penses-tu de ?").
    Fix : si le say_user référence un résultat vide, on exécute en avance la prochaine action
    propose_* du même lot (lecture pure/déterministe, sans effet de bord) pour résoudre le texte."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    # Ordre INVERSÉ par rapport à l'usage attendu : say_user avant propose_pseudo_candidates. Le
    # lot ne se termine pas par say_user (propose est en dernier) → relance normale (bug #7),
    # d'où une 2e réponse pour clore le tour — pas ce qu'on teste ici, juste réaliste.
    responses.append(json.dumps({
        "actions": [
            {"action": "say_user", "text": "Que penses-tu de {{résultat}} ?"},
            {"action": "propose_pseudo_candidates", "index": 0, "appropriate": True},
        ]
    }))
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Dis-moi ce que tu en penses !"}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}], {"identity_token": "tok-ordre-inverse", "taken_pseudos": set()}
    )
    assert result["error"] is None
    assert len(calls) == 2
    first_reply = result["replies"][0]
    assert "{{résultat}}" not in first_reply
    assert first_reply.strip() != ""
    assert "?" in first_reply
    # Le mot+couleur réellement proposé (déterministe pour cette identité) doit apparaître.
    propose_result = [a for a in result["actions_log"] if a["action"] == "propose_pseudo_candidates"][0]["result"]
    assert propose_result["display"] in first_reply


@pytest.mark.anyio
async def test_run_turn_resolves_say_user_when_previous_result_has_no_usable_value(mocked_openrouter_structured):
    """Régression bug réel #10 (variante trouvée en réel sur jouyvote.fr juste après le 1er fix,
    2026-07-25) : previous_result n'est pas TOUJOURS vide/None quand le say_user précède le
    propose_* dans le même lot — il peut être un résultat NON-VIDE mais SANS AUCUNE valeur
    exploitable (ex. {"threads": [...]} d'un list_threads dans l'itération précédente). La
    condition initiale du fix bug #10 ("if not previous_result") ne détectait pas ce cas — corrigé
    en vérifiant que previous_result rend une valeur utilisable (_render_result_value), pas
    seulement sa vérité booléenne."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    # 1er appel : une action sans rapport (list_threads) qui ne se termine pas par say_user.
    responses.append(json.dumps({"actions": [{"action": "list_threads"}]}))
    # 2e appel : previous_result = résultat de list_threads (non-None, mais sans valeur utilisable)
    # — say_user AVANT propose_pseudo_candidates dans le même lot, comme le bug #10 original.
    responses.append(json.dumps({
        "actions": [
            {"action": "say_user", "text": "Que penses-tu de {{résultat}} ?"},
            {"action": "propose_pseudo_candidates", "index": 0, "appropriate": True},
        ]
    }))
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Dis-moi ce que tu en penses !"}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}],
        {"identity_token": "tok-previous-result-inutilisable", "taken_pseudos": set()},
    )
    assert result["error"] is None
    # list_threads (iteration 0) ne produit pas de say_user — le 1er reply vient de l'itération 1.
    first_reply = result["replies"][0]
    assert "{{résultat}}" not in first_reply
    propose_result = [a for a in result["actions_log"] if a["action"] == "propose_pseudo_candidates"][0]["result"]
    assert propose_result["display"] in first_reply


@pytest.mark.anyio
async def test_run_turn_does_not_force_repeat_citation_of_already_mentioned_pseudo(mocked_openrouter_structured):
    """Régression bug réel #11 (2026-07-25, trouvé EN RÉEL sur jouyvote.fr juste après le fix du
    bug #10, dans le même échange) : une fois qu'un pseudo a été cité dans un say_user, un
    say_user de SUIVI plus tard dans le même tour ("si ça te plaît, clique sur le bouton...") se
    faisait remplacer À TORT par le simple nom du pseudo — le check bug #6 ("display absent du
    texte") se redéclenchait pour CHAQUE say_user suivant tant que carried_result restait un
    résultat pseudo, sans mémoire qu'il avait déjà été cité une fois."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({"actions": [{"action": "list_threads"}]}))
    responses.append(json.dumps({
        "actions": [
            {"action": "say_user", "text": "Que penses-tu de {{résultat}} ?"},
            {"action": "propose_pseudo_candidates", "index": 0, "appropriate": True},
        ]
    }))
    follow_up_text = "Si ça te plaît, tu peux cliquer sur le bouton de confirmation qui apparaît à côté."
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": follow_up_text}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}],
        {"identity_token": "tok-pas-de-repetition", "taken_pseudos": set()},
    )
    assert result["error"] is None
    # list_threads (iteration 0) ne produit pas de say_user — le suivi est le 2e reply (index 1).
    assert result["replies"][1] == follow_up_text


@pytest.mark.anyio
async def test_run_turn_forces_error_message_when_say_user_ignores_available_false(mocked_openrouter_structured):
    """Régression bug réel #12 (2026-07-25, trouvé EN RÉEL en testant propose_opinion, phase 3 du
    forum) : propose_opinion a renvoyé available=false + error="fil introuvable (ou pas encore
    publié)" (mauvais thread_id), mais le say_user a quand même affirmé "j'ai préparé le
    brouillon... clique pour confirmer" — ignorant complètement l'échec structuré. Généralise le
    principe des bugs #5/#6 (jusque-là réservé aux actions pseudo via "display") à toute action
    qui expose un booléen "available"."""
    import json
    import chatbot_executor

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [{"action": "propose_opinion", "thread_id": 999, "body": "Une opinion"}]
    }))
    false_success_text = (
        "J'ai préparé le brouillon de ton opinion. Clique sur le bouton de confirmation pour valider."
    )
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": false_success_text}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "publie mon opinion"}],
        {"identity_token": "tok-opinion-echec", "threads": []},
    )
    assert result["error"] is None
    assert result["replies"][0] != false_success_text
    assert "introuvable" in result["replies"][0]


@pytest.mark.anyio
async def test_run_turn_survives_exception_raised_by_action(mocked_openrouter_structured, monkeypatch):
    """Régression bug réel #13 (2026-07-26, capture développeur : "L'assistant n'a pas pu répondre
    correctement, réessaie." sur le flux résumé privé + confirmation de sujet) : la docstring de
    run_turn promet de "n'échoue jamais par exception non capturée", mais aucun try/except
    n'entourait l'appel direct à une fonction d'action — une exception (ex. get_wiki_page en
    panne réseau) remontait telle quelle jusqu'à /chat/v2, provoquant un 500 brut côté client au
    lieu d'un message structuré. Fix : chaque appel passe par _run_action, qui capture toute
    exception et la transforme en résultat structuré {"available": False, "error": ...}."""
    import json
    import chatbot_executor

    def boom(params, ctx):
        raise RuntimeError("panne réseau simulée")

    monkeypatch.setitem(chatbot_executor.ACTIONS, "list_threads", boom)

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({"actions": [{"action": "list_threads"}]}))
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Voici les fils : {{résultat}}"}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "quels fils existent ?"}], {},
    )
    assert result["error"] is None  # ne plante jamais tout le tour, même sur une exception
    failed = [a for a in result["actions_log"] if a["action"] == "list_threads"][0]
    assert failed["result"]["available"] is False
    assert "problème technique" in failed["result"]["error"]
    # le say_user suivant reflète l'échec réel, jamais une fausse promesse de succès
    assert "problème technique" in result["replies"][0]


@pytest.mark.anyio
async def test_run_turn_peek_ahead_survives_exception_raised_by_action(mocked_openrouter_structured, monkeypatch):
    """Même filet que le test précédent, mais sur le chemin de PEEK-AHEAD du bug #10 (exécution
    anticipée de propose_pseudo_candidates/propose_custom_pseudo quand say_user les précède dans
    le même lot) — ce 2e point d'appel avait le même trou avant le fix du bug #13."""
    import json
    import chatbot_executor

    def boom(params, ctx):
        raise RuntimeError("panne simulée")

    monkeypatch.setitem(chatbot_executor.ACTIONS, "propose_pseudo_candidates", boom)

    calls, responses = mocked_openrouter_structured
    responses.append(json.dumps({
        "actions": [
            {"action": "say_user", "text": "Que penses-tu de {{résultat}} ?"},
            {"action": "propose_pseudo_candidates", "index": 0, "appropriate": True},
        ]
    }))
    responses.append(json.dumps({"actions": [{"action": "say_user", "text": "Ok, notée."}]}))

    result = chatbot_executor.run_turn(
        "system", [{"role": "user", "content": "propose"}], {"taken_pseudos": set()},
    )
    assert result["error"] is None
    assert "{{résultat}}" not in result["replies"][0]


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

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
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
async def test_chat_v2_trace_requires_valid_admin_key(client, logged_in_user, monkeypatch):
    """Mode debug/traçage réservé aux bots internes (demande développeur 2026-07-25) : sans
    admin_key correct, run_turn est appelé avec trace=False — jamais activé par erreur ou par un
    citoyen qui devinerait le paramètre."""
    import main as main_module

    captured = {}

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
        captured["trace"] = trace
        return {"replies": ["ok"], "actions_log": [], "error": None}

    monkeypatch.setattr(main_module, "run_turn", fake_run_turn)

    await client.post(
        "/chat/v2",
        json={"session_token": logged_in_user["session_token"], "message": "bonjour"},
    )
    assert captured["trace"] is False

    await client.post(
        "/chat/v2",
        json={"session_token": logged_in_user["session_token"], "message": "bonjour", "admin_key": "mauvaise-clé"},
    )
    assert captured["trace"] is False

    await client.post(
        "/chat/v2",
        json={"session_token": logged_in_user["session_token"], "message": "bonjour", "admin_key": "test-admin-key-42"},
    )
    assert captured["trace"] is True


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

    def fake_run_turn(system_prompt, conversation_messages, ctx, model=None, max_iterations=5, trace=False):
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