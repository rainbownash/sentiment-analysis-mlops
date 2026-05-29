from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from gensim.models import Word2Vec

from src.features.preprocessing import clean_base
from src.features.text_features import build_bow_matrix, prepare_w2v_tokens, preprocess_bow
from src.features.embeddings import review_vector
from src.models.evaluation import get_metrics

from src.pipeline import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CARGAR DATOS

DATA_PATH = PROJECT_ROOT / "data" / "dataset_practica.csv"

df = pd.read_csv(DATA_PATH)

# LIMPIEZA Y PREPROCESADO
df['text_clean'] = df.apply(lambda x: clean_base(x['review_body'], x['language']), axis=1)

df['tokens_bow'] = df.apply(lambda x: preprocess_bow(x['text_clean'], x['language']), axis=1)

X_bow, vectorizer = build_bow_matrix(df)

df['tokens_w2v'] = prepare_w2v_tokens(df)

# ENCODER
le = LabelEncoder()
y = le.fit_transform(df['sentiment_label'])

# SPLIT 
X_train_idx, X_test_idx, y_train, y_test = train_test_split(
    np.arange(len(y)),
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_bow = X_bow[X_train_idx]
X_test_bow = X_bow[X_test_idx]

X_train_tokens = df['tokens_w2v'].iloc[X_train_idx]
X_test_tokens = df['tokens_w2v'].iloc[X_test_idx]

# BOW
bow_lr, metrics_bow_lr, y_pred_bow_lr = run_experiment(
    LogisticRegression(
        max_iter=1000, 
        random_state=42
    ),
    X_train_bow,
    X_test_bow,
    y_train,
    y_test,
    "BoW",
    "LogisticRegression"
)

bow_rf, metrics_bow_rf, y_pred_bow_rf = run_experiment(
    RandomForestClassifier(
        n_estimators=200, # número de árboles
        max_depth=None, # profundidad máxima, None = crecer hasta hojas puras
        random_state=42,
        n_jobs=-1 # usa todos los núcleos disponibles
        ),
    X_train_bow,
    X_test_bow,
    y_train,
    y_test,
    "BoW",
    "RandomForest"
)

bow_svm, metrics_bow_svm, y_pred_bow_svm = run_experiment(
    SVC(
        kernel='linear', 
        C=1, 
        random_state=42
        ),
    X_train_bow,
    X_test_bow,
    y_train,
    y_test,
    "BoW",
    "SVM"
)

# W2V
w2v_model = Word2Vec(
    sentences=X_train_tokens.tolist(), # solo train para evitar data leakage
    vector_size=100, # tamaño del embedding, suele ser entre 100 y 300
    window=5, # contexto (5 palabras de cada lado)
    min_count=2, # ignorar palabras muy raras
    workers=4, # núcleos para entrenamiento
    seed=42
)

X_train_w2v = np.vstack([review_vector(t, w2v_model) for t in X_train_tokens])
X_test_w2v = np.vstack([review_vector(t, w2v_model) for t in X_test_tokens])

w2v_lr, metrics_w2v_lr, y_pred_w2v_lr = run_experiment(
    LogisticRegression(
        max_iter=1000, 
        random_state=42
        ),
    X_train_w2v,
    X_test_w2v,
    y_train,
    y_test,
    "Word2Vec",
    "LogisticRegression"
)

w2v_rf, metrics_w2v_rf, y_pred_w2v_rf = run_experiment(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    X_train_w2v,
    X_test_w2v,
    y_train,
    y_test,
    "Word2Vec",
    "RandomForest"
)

w2v_svm, metrics_w2v_svm, y_pred_w2v_svm = run_experiment(
    SVC(
        kernel='linear', 
        C=1, 
        random_state=42
        ),
    X_train_w2v,
    X_test_w2v,
    y_train,
    y_test,
    "Word2Vec",
    "SVM"
)

# RESULTADOS
results = pd.concat([
    get_metrics(y_test, y_pred_bow_lr, "BoW", "LogisticRegression"),
    get_metrics(y_test, y_pred_bow_rf, "BoW", "RandomForest"),
    get_metrics(y_test, y_pred_bow_svm, "BoW", "SVM"),
    get_metrics(y_test, y_pred_w2v_lr, "Word2Vec", "LogisticRegression"),
    get_metrics(y_test, y_pred_w2v_rf, "Word2Vec", "RandomForest"),
    get_metrics(y_test, y_pred_w2v_svm, "Word2Vec", "SVM")
], ignore_index=True)

results_sorted = results.sort_values(by="F1 macro", ascending=False)

# GUARDADO
Path(PROJECT_ROOT / "models/bow").mkdir(parents=True, exist_ok=True)
Path(PROJECT_ROOT / "models/w2v").mkdir(parents=True, exist_ok=True)
Path(PROJECT_ROOT / "results").mkdir(parents=True, exist_ok=True)

joblib.dump(bow_lr, PROJECT_ROOT / "models/bow/bow_lr.pkl")
joblib.dump(bow_rf, PROJECT_ROOT / "models/bow/bow_rf.pkl")
joblib.dump(bow_svm, PROJECT_ROOT / "models/bow/bow_svm.pkl")

joblib.dump(vectorizer, PROJECT_ROOT / "models/bow/bow_vectorizer.pkl")

joblib.dump(w2v_lr, PROJECT_ROOT / "models/w2v/w2v_lr.pkl")
joblib.dump(w2v_rf, PROJECT_ROOT / "models/w2v/w2v_rf.pkl")
joblib.dump(w2v_svm, PROJECT_ROOT / "models/w2v/w2v_svm.pkl")

w2v_model.save(str(PROJECT_ROOT / "models/w2v/word2vec.model"))

results_sorted.to_csv(PROJECT_ROOT / "results/model_comparison.csv", index=False)

joblib.dump(le, PROJECT_ROOT / "models/le.pkl")