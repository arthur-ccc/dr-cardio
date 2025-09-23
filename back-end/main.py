import sys
import os

# adiciona a raiz do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.tree import DecisionTreeClassifier
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
from classifiers.generic_classifier import GenericClassifier

app = FastAPI(title="Dr Derma API", version="1.0")

# Caminhos absolutos para o modelo e mapa de doenças
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "decision_tree.pkl")
MAP_PATH = os.path.join(BASE_DIR, "models", "disease_map.pkl")

# Carrega o modelo e mapa
model = joblib.load(MODEL_PATH)
disease_map = joblib.load(MAP_PATH)
clf = GenericClassifier(model)

origins = [
    "http://localhost:3000",  # React padrão
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema de entrada da API
class PredictRequest(BaseModel):
    features: list[float]  # vetor de features (mesmo tamanho que X_train)

@app.get("/")
def home():
    return {"message": "Dr Derma API is running 🚀"}

@app.post("/predict")
def predict(request: PredictRequest):
    # Faz a predição
    prediction = clf.predict([request.features])[0]
    disease_name = disease_map.get(prediction + 1, "Desconhecida")

    # Retorna apenas o nome da doença
    return {"disease_name": disease_name}

