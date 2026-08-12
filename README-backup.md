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
5. Rotation : garde 30 jours d'archives, jamais moins d'une même si toutes sont plus vieilles.

Sorties dans `/mnt/stockage/jouyvote/backups/` (permissions 700/600), log dans
`logs/backup.log`.

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

## Ce qui manque encore : une vraie destination HORS MACHINE

À ce stade, les archives chiffrées restent sur `/mnt/stockage` — **même disque physique que la
base elle-même** (`/dev/sda5`). Ça protège contre une erreur applicative (mauvaise migration,
suppression accidentelle) mais PAS contre une panne de ce disque, qui emporterait base ET
sauvegardes ensemble. Pour la vraie redondance demandée, il faut une 2e destination sur un autre
support physique.

Vérifié le 2026-08-11 : pas de solution déjà en place à réutiliser sur cette machine (pas de
`rclone`, pas de compte cloud configuré). Sur le tailnet existant, `dell-papa` (le Dell) est
hors ligne depuis 22 jours au moment de cette vérification — pas une destination fiable pour un
push automatisé tant qu'il n'est pas remis en ligne durablement.

**Décision à prendre par Angelo avant de coder cette 2e étape** (documenté plutôt que tranché
seul, comme convenu pour les décisions structurantes) :
- Option A — rsync opportuniste vers `dell-papa` via Tailscale, dès qu'il est en ligne (gratuit,
  pas de nouveau compte, mais fiabilité dépendante de la disponibilité du Dell).
- Option B — stockage cloud privé chiffré (ex. Backblaze B2, palier gratuit ~10 Go, largement
  suffisant pour cette base) — nécessite qu'Angelo crée le compte et partage une clé API (canal
  Matrix, même pratique que pour le token GitHub du 10/08).

Les deux sont compatibles avec le pipeline déjà en place : il suffirait d'ajouter un envoi de
l'archive déjà chiffrée (`rsync`/`rclone`) juste après l'étape 4 ci-dessus, rien à reconstruire.
