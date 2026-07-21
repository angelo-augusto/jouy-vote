FROM node:22-slim

# python3/pip/venv ajoutés après coup : bug réel trouvé en test — le projet jouy-vote est du
# FastAPI/Python, mais l'image de base ne fournissait que Node.js (pour OpenCode lui-même).
# OpenCode a correctement écrit le code + les tests mais ne pouvait ensuite RIEN exécuter
# ("pip: command not found", pas de python3 du tout dans l'image).
# openssh-server ajouté à la demande d'Angelo (2026-07-20) : accès portable au conteneur si le
# Docker déménage un jour sur un autre serveur (pas juste docker exec local).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates python3 python3-pip python3-venv openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

# Installation d'OpenCode (script officiel, https://opencode.ai) en root, puis copie du binaire
# vers un chemin système accessible à tous — le script installe dans $HOME, qui appartiendrait à
# root et serait inaccessible une fois basculé sur l'utilisateur non-root ci-dessous.
RUN curl -fsSL https://opencode.ai/install | bash \
    && find /root -iname "opencode" -type f -executable -exec cp {} /usr/local/bin/opencode \; \
    && chmod +x /usr/local/bin/opencode

# Utilisateur non-root dédié pour le TRAVAIL (OpenCode, sessions SSH une fois authentifiées) —
# isolation demandée par Angelo pour un modèle externe (OpenRouter) auquel il ne fait pas une
# confiance totale : pas de root à l'intérieur des sessions de travail, même si le PROCESS sshd
# lui-même doit démarrer en root (architecture standard : le master sshd gère les clés d'hôte et
# la séparation de privilèges, chaque connexion authentifiée est ensuite exécutée sous "coder" —
# ce root reste confiné au namespace du conteneur, jamais exposé à l'hôte, compromis accepté
# explicitement par Angelo le 2026-07-20 pour la portabilité).
# UID/GID 1002 forcé pour correspondre à l'utilisateur hôte (angelo) propriétaire du répertoire
# monté en bind-mount — sans ça, useradd assigne le premier UID libre (1001, "node" existant déjà
# en 1000 dans l'image de base), qui ne correspond à AUCUN propriétaire des fichiers montés :
# OpenCode ne peut alors RIEN écrire dans /workspace ("Permission denied" trouvé en test réel).
RUN groupadd -g 1002 coder && useradd -u 1002 -g 1002 -ms /bin/bash coder

# Clés publiques d'Angelo autorisées à se connecter en SSH : "angelo@Dell-Papa" (déjà utilisée
# ailleurs sur ce Khadas) et "angelo-termius" (client mobile/Termius). Authentification par clé
# UNIQUEMENT : mot de passe désactivé, root désactivé. Ce fichier n'est que le contenu INITIAL —
# le répertoire /home/coder/.ssh est monté sur un volume Docker nommé (docker-compose.yml) qui
# persiste indépendamment des recréations/rebuilds du conteneur ; toute clé ajoutée à chaud après
# le premier démarrage survit désormais aux `docker compose up` suivants.
RUN mkdir -p /home/coder/.ssh \
    && printf '%s\n%s\n' \
         "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKbFlLzqqj3USqYAjrD9YrW/oVOTcZtM3Hvc0fP7aqm3 angelo@Dell-Papa" \
         "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPrIO+2O7W+qvvHT2n2yaDZRsqy+/B1jXB1j0VJcUXvt angelo-termius" \
         > /home/coder/.ssh/authorized_keys \
    && chown -R coder:coder /home/coder/.ssh \
    && chmod 700 /home/coder/.ssh \
    && chmod 600 /home/coder/.ssh/authorized_keys \
    && ssh-keygen -A \
    && sed -i \
         -e 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' \
         -e 's/^#\?PermitRootLogin.*/PermitRootLogin no/' \
         -e 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' \
         /etc/ssh/sshd_config \
    && echo "AllowUsers coder" >> /etc/ssh/sshd_config

WORKDIR /workspace

# Venv Python pré-créé et mis en tête de PATH pour l'utilisateur coder : Debian bookworm marque
# son python3 système "externally-managed" (PEP 668), donc un simple `pip install` échouerait de
# toute façon même avec pip installé — un venv transparent évite d'avoir à l'expliquer à chaque
# session OpenCode.
RUN su coder -c "python3 -m venv /home/coder/.venv"
ENV PATH="/home/coder/.venv/bin:${PATH}"

# sshd doit démarrer en root (host keys + séparation de privilèges) — le conteneur est désormais
# un service persistant (docker compose up -d), plus un "one-shot" comme avant. OpenCode reste
# invocable via `docker compose exec opencode opencode run '...'` OU via une vraie session SSH.
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
