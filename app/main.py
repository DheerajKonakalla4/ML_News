# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI(title="Fake News Detector", version="1.0")

# ---- load model artifacts (relative to this file) ----
ROOT = Path(__file__).resolve().parents[1]     # NEWS/
MODELS = ROOT / "models"
TFIDF_PATH = MODELS / "tfidf.joblib"
CLF_PATH   = MODELS / "logreg_calibrated.joblib"

if not TFIDF_PATH.exists() or not CLF_PATH.exists():
    raise RuntimeError(
        f"Missing model files. Expected:\n- {TFIDF_PATH}\n- {CLF_PATH}\n"
        "Run: python scripts/train_baseline.py"
    )

tfidf = joblib.load(TFIDF_PATH)
clf   = joblib.load(CLF_PATH)

class Item(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: Item):
    text = (item.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    X = tfidf.transform([text])
    prob_true = float(clf.predict_proba(X)[:, 1])
    label = "True" if prob_true >= 0.5 else "Fake"
    return {"label": label, "prob": round(prob_true, 4)}
