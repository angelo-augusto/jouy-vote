# Sauvegarde de vote.db

Mis en place le 2026-08-11 à la demande d'Angelo (relayée par angelobot) : la base contient des
données citoyennes réelles (nom, adresse, email, password_hash) — jamais sur GitHub en clair,
toujours chiffrée dès qu'elle sort de `/mnt/stockage/jouyvote/data`.

## Ce qui est en place (fait, testé, automatisé)

`backup_vote_db.py` tourne chaque nuit à 3h15 (cron) :

1. Copie cohérente de `vote.db` via `sqlite3.Connection.backup()` (jamais un `cp` brut, qui
   pourrait copier un fichier en cours d'écriture).
2. Compression gzip.
3. Chiffrement GPG symétrique (AES256), passphrase dans `/mnt/stockage/.backup_passphrase`
   (permissions 600, générée aléatoirement, jamais dans le repo).
4. **Test de restauration automatique à chaque exécution** (demandé en bonus) : déchiffre,
   décompresse, `PRAGMA integrity_check`, compte les lignes de `identities` — si quoi que ce soit
   échoue, l'archive produite est supprimée (jamais gardée si non restaurable) et une alerte part
   sur le Salon KhadasBot (Matrix).
5. **Envoi vers Cloudflare R2** (2026-08-12, décision Angelo — voir section ci-dessous) : bucket
   `jouy-vote`, identifiants dans `/mnt/stockage/jouyvote/.r2_credentials.env` (permissions 600,
   jamais dans le repo).
6. **Test de restauration DEPUIS R2** (même exigence que le test local, pas une confiance
   aveugle en un "upload sans erreur") : télécharge l'objet qu'on vient d'envoyer, déchiffre,
   décompresse, `PRAGMA integrity_check`, compare le comptage d'`identities` à l'original. Si ça
   diverge, alerte immédiate — l'objet reste sur R2 pour inspection plutôt que d'être supprimé en
   silence (contrairement à l'archive locale, où une restauration ratée entraîne la suppression).
7. Rotation : garde 30 jours d'archives, jamais moins d'une même si toutes sont plus vieilles —
   appliquée séparément en local ET sur R2.

Sorties locales dans `/mnt/stockage/jouyvote/backups/` (permissions 700/600), log dans
`logs/backup.log`. Dépendance `boto3` (client S3, R2 est compatible) installée dans un venv
dédié `.venv-backup/` (jamais dans le python système, Ubuntu 24.04 bloque `pip install` hors
venv sans `--break-system-packages`) — le cron pointe directement vers
`.venv-backup/bin/python3`.

**Limite de sécurité assumée, à connaître** : la passphrase vit sur la même machine que la base.
Ce chiffrement protège la donnée une fois SORTIE du Khadas (ex: destination mal configurée,
accès réseau non autorisé) — il ne protège PAS contre une compromission complète du Khadas
lui-même, qui aurait accès à la passphrase en même temps qu'à la base. Une vraie protection même
dans ce cas nécessiterait de chiffrer avec la clé PUBLIQUE d'Angelo (lui seul détient la clé
privée, ailleurs) — pas fait pour l'instant, à revisiter si souhaité.

**Trou corrigé le 2026-08-12 (repéré par Angelo en relisant ce document, transmis par
angelobot)** : la passphrase vivait UNIQUEMENT sur `/mnt/stockage/.backup_passphrase` —
même disque physique que la base elle-même. En cas de panne de ce disque précis (exactement le
scénario que la sauvegarde hors machine est censée couvrir), les archives chiffrées sur R2
survivraient physiquement mais deviendraient indéchiffrables pour toujours : la seule clé aurait
disparu avec le reste. Corrigé en donnant une copie de la passphrase à Angelo (canal Matrix
direct, 2026-08-12), à conserver dans son gestionnaire de mots de passe personnel — hors de cette
machine, donc hors de portée d'une panne disque locale. Le fichier local reste la source utilisée
par le script (pas de changement de pipeline), la copie chez Angelo n'est qu'un filet de secours
pour la restauration en cas de perte totale du Khadas.

## Destination hors machine : Cloudflare R2 (branché le 2026-08-12)

Angelo a tranché : Cloudflare R2 (10 Go/mois gratuits, zéro coût de sortie, compte Cloudflare
déjà existant pour le tunnel jouyvote — pas de nouveau service à gérer), plutôt que rsync vers
`dell-papa` (hors ligne 22 jours au moment de la décision, pas fiable) ou Backblaze B2.

R2 est compatible S3 — branché avec `boto3` (pas de `rclone`/`awscli` système, qui aurait
nécessité `sudo` sur cette machine en environnement Python "externally-managed"). Identifiants
R2 (Account ID, endpoint, bucket, Access Key ID, Secret Access Key) reçus d'Angelo via Matrix le
2026-08-12, stockés dans `/mnt/stockage/jouyvote/.r2_credentials.env` (600, hors repo).

Testé de bout en bout à plusieurs reprises, y compris en environnement cron minimal (`env -i`) :
upload réel, téléchargement de vérification, intégrité + comptage confirmés identiques à
l'original des deux côtés (local et R2).
