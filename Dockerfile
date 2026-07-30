# Image de base légère avec Python 3.11
FROM python:3.11-slim

# Dépendances système nécessaires à librosa/soundfile pour lire les fichiers
# audio (.wav/.mp3) : libsndfile pour soundfile, ffmpeg pour le décodage mp3
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python d'abord (permet de profiter du cache Docker
# tant que requirements.txt ne change pas, même si le code source change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source et les fichiers audio de démo
COPY src/ src/
COPY test_audio/ test_audio/

# Le port sur lequel l'API FastAPI écoute
EXPOSE 8000

# Au démarrage du conteneur : lancer le serveur API
# (Les modèles Hugging Face seront téléchargés au premier lancement, dans le
# cache par défaut de transformers à l'intérieur du conteneur)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
