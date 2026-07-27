"""
API REST pour le pipeline de détection de sentiment vocal.
Endpoint principal : POST /predict (fichier audio -> JSON).
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.pipeline import VoiceSentimentPipeline

app = FastAPI(
    title="Voice Sentiment Pipeline API",
    description="Transcrit un appel vocal (Wav2Vec 2.0) et analyse son sentiment (DistilCamemBERT).",
    version="1.0.0",
)

SUPPORTED_EXTENSIONS = (".wav", ".mp3")

# Chargé une seule fois au démarrage du serveur (pas à chaque requête)
print("Initialisation du pipeline (peut prendre 1-2 minutes)...")
pipeline = VoiceSentimentPipeline()
print("Pipeline prêt, API disponible.")


@app.get("/")
def root():
    return {"message": "Voice Sentiment Pipeline API. Voir /docs pour la documentation interactive."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio (.wav ou .mp3), le transcrit, analyse son sentiment,
    et retourne {transcription, sentiment, confidence}.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        return JSONResponse(
            status_code=400,
            content={"error": f"Format non supporté. Formats acceptés : {SUPPORTED_EXTENSIONS}"},
        )

    # Sauvegarde temporaire du fichier reçu (le pipeline travaille à partir d'un chemin disque)
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = pipeline.predict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # nettoyage du fichier temporaire

    if "error" in result:
        return JSONResponse(status_code=422, content=result)

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
