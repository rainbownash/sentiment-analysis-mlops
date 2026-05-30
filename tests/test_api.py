from fastapi.testclient import TestClient
from src.api.main import app
import src.api.main as main

client = TestClient(app)

# test del endpoint raíz
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

# test de predict
def test_predict(monkeypatch):

    class DummyModel:
        def predict(self, X):
            return [1]
        
        def predict_proba(self, X):
            return [[0.9, 0.1]]

    class DummyVectorizer:
        def transform(self, text):
            return text

    class DummyLabelEncoder:
        def inverse_transform(self, x):
            return ["POSITIVE"]

    monkeypatch.setattr(main, "model", DummyModel())
    monkeypatch.setattr(main, "vectorizer", DummyVectorizer())
    monkeypatch.setattr(main, "le", DummyLabelEncoder())

    response = client.post(
        "/predict",
        json={"text": "I love this product"}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "POSITIVE"