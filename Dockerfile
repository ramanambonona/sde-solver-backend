FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements d'abord (pour meilleur caching)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Exposer le port
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
