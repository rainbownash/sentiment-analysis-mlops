from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models/bow/bow_lr.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models/bow/bow_vectorizer.pkl"

# artefactos
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# app
app = FastAPI(title="Sentiment Analysis API")

# input
class TextInput(BaseModel):
    text: str

# endpoint
@app.post("/predict")
def predict(input_data: TextInput):

    X = vectorizer.transform([input_data.text])

    pred = model.predict(X)[0]

    return {
        "text": input_data.text,
        "prediction": int(pred)
    }