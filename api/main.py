"""API scaffold. Prediction endpoints will be added with trained artifacts."""
from fastapi import FastAPI

app = FastAPI(title="BrainScan API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_ready": False}
