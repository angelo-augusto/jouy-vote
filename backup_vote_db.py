#!/usr/bin/env python3
"""Sauvegarde chiffrée hors répertoire de données de vote.db, avec test de restauration
automatique (2026-08-11, demande Angelo relayée par angelobot — la base contient des données
citoyennes réelles nom/adresse/email/password_hash, jamais sur GitHub, toujours chiffrée dès
qu'elle quitte /mnt/stockage/jouyvote/data).

Étapes : copie cohérente (sqlite3.Connection.backup, jamais un cp brut sur une DB potentiellement
en écriture) -> gzip -> chiffrement GPG symétrique -> test de restauration complet (déchiffre,
décompresse, PRAGMA integrity_check, compte les identités) -> rotation des archives trop
anciennes. Alerte Matrix (Salon KhadasBot) si une étape échoue, pas seulement un code de sortie
non-zéro dans un log que personne ne relit.

Limite de sécurité assumée (à documenter honnêtement, pas à cacher) : la passphrase GPG vit sur
CETTE machine (fichier restreint 600) — ce chiffrement protège la donnée une fois sortie du
Khadas (ex: destination hors machine mal configurée), pas contre une compromission complète du
Khadas lui-même, qui aurait accès à la passphrase en même temps que la base. Pour une vraie
protection même en cas de compromission du Khadas, il faudrait chiffrer avec la clé PUBLIQUE
d'Angelo (lui seul détient la clé privée, ailleurs) — pas fait ici par simplicité, à revisiter si
Angelo le souhaite."""
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

    # Test de restauration automatique (2026-08-11, demandé en bonus par Angelo) : une sauvegarde
    # jamais restaurée n'est qu'une hypothèse — on vérifie ICI, à chaque exécution, que le
    # fichier produit est réellement exploitable, pas seulement "écrit sans erreur".
    restore_tmp = BACKUP_DIR / f".restore-test-{stamp}.db"
    try:
        decrypted = subprocess.run(
            ["gpg", "--batch", "--yes", "--passphrase-file", str(PASSPHRASE_FILE), "-d", str(out_path)],
            check=True, capture_output=True,
        ).stdout
        with open(restore_tmp, "wb") as f:
            f.write(gzip.decompress(decrypted))
        conn = sqlite3.connect(str(restore_tmp))
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        identities_count = conn.execute("SELECT COUNT(*) FROM identities;").fetchone()[0]
        conn.close()
    except Exception as e:
        out_path.unlink(missing_ok=True)
        fail(f"test de restauration impossible ({e}) — archive {out_path.name} supprimée, pas gardée si non restaurable")
    finally:
        restore_tmp.unlink(missing_ok=True)

    if integrity != "ok":
        out_path.unlink(missing_ok=True)
        fail(f"intégrité SQLite KO ({integrity}) sur la sauvegarde restaurée — archive {out_path.name} supprimée")

    # Rotation (2026-08-11) : garde KEEP_DAYS jours, jamais moins d'une sauvegarde même si toutes
    # sont plus vieilles que le seuil (filet contre un cron resté en échec longtemps).
    cutoff = datetime.now() - timedelta(days=KEEP_DAYS)
    archives = sorted(BACKUP_DIR.glob("vote-*.db.gz.gpg"))
    for archive in archives[:-1]:
        if datetime.fromtimestamp(archive.stat().st_mtime) < cutoff:
            archive.unlink()

    print(f"OK - {out_path.name} ({identities_count} identités, intégrité: {integrity})")


if __name__ == "__main__":
    main()
