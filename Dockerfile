FROM node:22-slim

# python3/pip/venv ajoutés après coup : bug réel trouvé en test — le projet jouy-vote est du
# FastAPI/Python, mais l'image de base ne fournissait que Node.js (pour OpenCode lui-même).
# OpenCode a correctement écrit le code + les tests mais ne pouvait ensuite RIEN exécuter
# ("pip: command not found", pas de python3 du tout dans l'image).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Installation d'OpenCode (script officiel, https://opencode.ai) en root, puis copie du binaire
# vers un chemin système accessible à tous — le script installe dans $HOME, qui appartiendrait à
# root et serait inaccessible une fois basculé sur l'utilisateur non-root ci-dessous.
RUN curl -fsSL https://opencode.ai/install | bash \
    && find /root -iname "opencode" -type f -executable -exec cp {} /usr/local/bin/opencode \; \
    && chmod +x /usr/local/bin/opencode

# Utilisateur non-root dédié — isolation demandée par Angelo pour un modèle externe (OpenRouter)
# auquel il ne fait pas une confiance totale : pas de root à l'intérieur du conteneur.
# UID/GID 1002 forcé pour correspondre à l'utilisateur hôte (angelo) propriétaire du répertoire
# monté en bind-mount — sans ça, useradd assigne le premier UID libre (1001, "node" existant déjà
# en 1000 dans l'image de base), qui ne correspond à AUCUN propriétaire des fichiers montés :
# OpenCode ne peut alors RIEN écrire dans /workspace ("Permission denied" trouvé en test réel).
RUN groupadd -g 1002 coder && useradd -u 1002 -g 1002 -ms /bin/bash coder
USER coder
WORKDIR /workspace

# Venv Python pré-créé et mis en tête de PATH : Debian bookworm marque son python3 système
# "externally-managed" (PEP 668), donc un simple `pip install` échouerait de toute façon même
# avec pip installé — un venv transparent évite d'avoir à l'expliquer à chaque session OpenCode.
RUN python3 -m venv /home/coder/.venv
ENV PATH="/home/coder/.venv/bin:${PATH}"

ENTRYPOINT ["opencode"]
