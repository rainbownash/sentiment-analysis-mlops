from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "artifacts/bow/bow_lr.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "artifacts/bow/bow_vectorizer.pkl"
LE_PATH = PROJECT_ROOT / "artifacts/le.pkl"

# artefactos
model = None
vectorizer = None
le = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LE_PATH)

    print("Models loaded")

    yield

    print("Shut down")

# app
app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# input
class TextInput(BaseModel):
    text: str

# endpoint
@app.get("/")
def root():
    return {"message": "Sentiment Analysis API running"}

@app.post("/predict")
def predict(input_data: TextInput):

    X = vectorizer.transform([input_data.text])

    pred = model.predict(X)[0]
    label = le.inverse_transform([pred])[0]
    probs = model.predict_proba(X)[0]

    return {
        "text": input_data.text,
        "prediction": label,
        "confidence": float(max(probs))
    }