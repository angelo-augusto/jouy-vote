#!/usr/bin/env python3
"""Sauvegarde chiffrée hors répertoire de données de vote.db, avec test de restauration
automatique (2026-08-11, demande Angelo relayée par angelobot — la base contient des données
citoyennes réelles nom/adresse/email/password_hash, jamais sur GitHub, toujours chiffrée dès
qu'elle quitte /mnt/stockage/jouyvote/data).

Étapes : copie cohérente (sqlite3.Connection.backup, jamais un cp brut sur une DB potentiellement
en écriture) -> gzip -> chiffrement GPG symétrique -> test de restauration LOCAL complet
(déchiffre, décompresse, PRAGMA integrity_check, compte les identités) -> envoi vers Cloudflare
R2 (2026-08-12, hors machine, décision Angelo) -> test de restauration DEPUIS R2 (télécharge
l'objet qu'on vient d'envoyer, mêmes vérifications — prouve que la copie hors machine est
vraiment exploitable, pas seulement "l'upload n'a pas levé d'erreur") -> rotation locale ET sur
R2. Alerte Matrix (Salon KhadasBot) si une étape échoue, pas seulement un code de sortie non-zéro
dans un log que personne ne relit.

Limite de sécurité assumée (à documenter honnêtement, pas à cacher) : la passphrase GPG vit sur
CETTE machine (fichier restreint 600) — ce chiffrement protège la donnée une fois sortie du
Khadas (ex: destination hors machine mal configurée), pas contre une compromission complète du
Khadas lui-même, qui aurait accès à la passphrase en même temps que la base. Pour une vraie
protection même en cas de compromission du Khadas, il faudrait chiffrer avec la clé PUBLIQUE
d'Angelo (lui seul détient la clé privée, ailleurs) — pas fait ici par simplicité, à revisiter si
Angelo le souhaite. Filet posé le 2026-08-12 : une copie de la passphrase vit chez Angelo (hors
machine), pour ne pas perdre la clé en même temps qu'une panne du disque qui contient tout le
reste.

Dépendance boto3 : PAS installée dans le python système (Ubuntu 24.04, environnement
externally-managed) — tourne via le venv dédié .venv-backup/ (voir cron), jamais de
--break-system-packages."""
import gzip
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("/mnt/stockage/jouyvote/data")
DB_PATH = DATA_DIR / "vote.db"
BACKUP_DIR = Path("/mnt/stockage/jouyvote/backups")
PASSPHRASE_FILE = Path("/mnt/stockage/jouyvote/.backup_passphrase")
R2_CREDENTIALS_FILE = Path("/mnt/stockage/jouyvote/.r2_credentials.env")
KEEP_DAYS = 30

MATRIX_HOMESERVER = "http://localhost:6167"
MATRIX_ACCESS_TOKEN = "1rSRzsRY85SCjG4pfq7xuPNDNJeppb6X"
MATRIX_ROOM = "!vJ87OTnQC1EARY86xvfWzXzHhAN3WHWYrlAQ82mknEo"


def alert(message: str) -> None:
    """Best-effort — une alerte ratée ne doit jamais faire planter le reste du script."""
    try:
        subprocess.run(
            [
                "/home/angelo/codage/mcp-messenger/matrix-cli", "send", MATRIX_ROOM,
                f"[BACKUP jouyvote] {message}",
            ],
            env={**os.environ, "MATRIX_HOMESERVER": MATRIX_HOMESERVER, "MATRIX_ACCESS_TOKEN": MATRIX_ACCESS_TOKEN},
            cwd="/home/angelo/codage/mcp-messenger",
            timeout=30,
        )
    except Exception:
        pass


def fail(message: str) -> None:
    alert(f"ÉCHEC : {message}")
    print(f"ÉCHEC : {message}", file=sys.stderr)
    sys.exit(1)


def load_r2_credentials() -> dict[str, str] | None:
    """Parsing volontairement minimal (KEY=VALUE, une par ligne) — pas de dépendance
    supplémentaire pour un fichier aussi simple, jamais chargé depuis le repo (voir
    R2_CREDENTIALS_FILE, hors de /home/angelo/codage/jouy-vote)."""
    if not R2_CREDENTIALS_FILE.exists():
        return None
    creds: dict[str, str] = {}
    for line in R2_CREDENTIALS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        creds[key.strip()] = value.strip()
    required = {"R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"}
    if not required.issubset(creds):
        return None
    return creds


def r2_client(creds: dict[str, str]):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=creds["R2_ENDPOINT"],
        aws_access_key_id=creds["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def decrypt_and_check(gpg_bytes: bytes, passphrase_file: Path, tmp_path: Path) -> tuple[str, int]:
    """Factorisé (2026-08-12) entre le test de restauration LOCAL et celui DEPUIS R2 — même
    vérification exacte des deux côtés, pour ne jamais avoir 2 définitions de "restaurable" qui
    pourraient diverger silencieusement."""
    decrypted = subprocess.run(
        ["gpg", "--batch", "--yes", "--passphrase-file", str(passphrase_file), "-d"],
        input=gpg_bytes, check=True, capture_output=True,
    ).stdout
    tmp_path.write_bytes(gzip.decompress(decrypted))
    conn = sqlite3.connect(str(tmp_path))
    integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
    identities_count = conn.execute("SELECT COUNT(*) FROM identities;").fetchone()[0]
    conn.close()
    return integrity, identities_count


def main() -> None:
    # umask restrictif (2026-08-11) : cron tourne souvent avec umask 022 par défaut — sans ça,
    # les fichiers temporaires en clair (plaintext DB, y compris password_hash/nom/adresse/email
    # réels) seraient world-readable le temps du traitement, même brièvement.
    os.umask(0o077)
    if not DB_PATH.exists():
        fail(f"base introuvable ({DB_PATH})")
    if not PASSPHRASE_FILE.exists():
        fail(f"fichier de passphrase absent ({PASSPHRASE_FILE}) — voir README-backup.md pour la génération initiale")

    BACKUP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(BACKUP_DIR, 0o700)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plain_tmp = BACKUP_DIR / f".tmp-{stamp}.db"
    gz_tmp = BACKUP_DIR / f".tmp-{stamp}.db.gz"
    out_path = BACKUP_DIR / f"vote-{stamp}.db.gz.gpg"

    try:
        # Copie cohérente (2026-08-11) : Connection.backup() gère un verrou de lecture propre,
        # contrairement à un simple cp qui pourrait copier un fichier en cours d'écriture (WAL).
        src = sqlite3.connect(str(DB_PATH))
        dst = sqlite3.connect(str(plain_tmp))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()

        with open(plain_tmp, "rb") as f_in, gzip.open(gz_tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        subprocess.run(
            [
                "gpg", "--batch", "--yes", "--passphrase-file", str(PASSPHRASE_FILE),
                "--symmetric", "--cipher-algo", "AES256", "-o", str(out_path), str(gz_tmp),
            ],
            check=True, capture_output=True,
        )
        os.chmod(out_path, 0o600)
    finally:
        plain_tmp.unlink(missing_ok=True)
        gz_tmp.unlink(missing_ok=True)

    # Test de restauration LOCAL automatique (2026-08-11, demandé en bonus par Angelo) : une
    # sauvegarde jamais restaurée n'est qu'une hypothèse — on vérifie ICI, à chaque exécution,
    # que le fichier produit est réellement exploitable, pas seulement "écrit sans erreur".
    restore_tmp = BACKUP_DIR / f".restore-test-{stamp}.db"
    try:
        integrity, identities_count = decrypt_and_check(out_path.read_bytes(), PASSPHRASE_FILE, restore_tmp)
    except Exception as e:
        out_path.unlink(missing_ok=True)
        fail(f"test de restauration local impossible ({e}) — archive {out_path.name} supprimée, pas gardée si non restaurable")
    finally:
        restore_tmp.unlink(missing_ok=True)

    if integrity != "ok":
        out_path.unlink(missing_ok=True)
        fail(f"intégrité SQLite KO ({integrity}) sur la sauvegarde locale restaurée — archive {out_path.name} supprimée")

    # Rotation locale (2026-08-11) : garde KEEP_DAYS jours, jamais moins d'une sauvegarde même si
    # toutes sont plus vieilles que le seuil (filet contre un cron resté en échec longtemps).
    cutoff = datetime.now() - timedelta(days=KEEP_DAYS)
    archives = sorted(BACKUP_DIR.glob("vote-*.db.gz.gpg"))
    for archive in archives[:-1]:
        if datetime.fromtimestamp(archive.stat().st_mtime) < cutoff:
            archive.unlink()

    print(f"OK local - {out_path.name} ({identities_count} identités, intégrité: {integrity})")

    # Envoi hors machine vers Cloudflare R2 (2026-08-12, décision Angelo — voir README-backup.md).
    # Dégradation propre si les identifiants ne sont pas là (jamais un crash du reste du script,
    # la sauvegarde locale reste utile même sans ça) — mais alerte quand même, ce n'est pas un
    # état normal une fois la 2e étape branchée.
    creds = load_r2_credentials()
    if creds is None:
        alert(f"sauvegarde locale OK ({out_path.name}) mais identifiants R2 absents ({R2_CREDENTIALS_FILE}) — pas d'envoi hors machine cette fois")
        return

    client = r2_client(creds)
    try:
        client.upload_file(str(out_path), creds["R2_BUCKET"], out_path.name)
    except Exception as e:
        fail(f"envoi vers R2 échoué ({e}) — sauvegarde locale {out_path.name} reste disponible")

    # Test de restauration DEPUIS R2 (2026-08-12) : télécharge ce qu'on vient d'envoyer et
    # applique EXACTEMENT la même vérification que le test local — prouve que la copie hors
    # machine est vraiment exploitable, pas seulement que l'upload n'a pas levé d'erreur réseau.
    r2_restore_tmp = BACKUP_DIR / f".r2-restore-test-{stamp}.db"
    try:
        obj = client.get_object(Bucket=creds["R2_BUCKET"], Key=out_path.name)
        r2_integrity, r2_identities_count = decrypt_and_check(obj["Body"].read(), PASSPHRASE_FILE, r2_restore_tmp)
    except Exception as e:
        fail(f"test de restauration depuis R2 impossible ({e}) — objet {out_path.name} présent sur R2 mais non vérifié")
    finally:
        r2_restore_tmp.unlink(missing_ok=True)

    if r2_integrity != "ok" or r2_identities_count != identities_count:
        fail(f"la copie R2 de {out_path.name} diverge de l'original (intégrité={r2_integrity}, identités={r2_identities_count} vs {identities_count})")

    # Rotation sur R2 (2026-08-12) : même logique que la rotation locale, appliquée séparément
    # (les 2 rétentions n'ont pas besoin d'être identiques dans le temps, mais le sont ici par
    # simplicité — KEEP_DAYS commun).
    objects = client.list_objects_v2(Bucket=creds["R2_BUCKET"], Prefix="vote-").get("Contents", [])
    objects.sort(key=lambda o: o["LastModified"])
    for obj_meta in objects[:-1]:
        if obj_meta["LastModified"].replace(tzinfo=None) < cutoff:
            client.delete_object(Bucket=creds["R2_BUCKET"], Key=obj_meta["Key"])

    print(f"OK R2 - {out_path.name} ({r2_identities_count} identités, intégrité: {r2_integrity})")


if __name__ == "__main__":
    main()
