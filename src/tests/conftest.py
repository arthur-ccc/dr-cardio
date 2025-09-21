import os
import sys
import pathlib
import pandas as pd
import numpy as np
import pytest

# === Ajuste de paths ===
THIS_FILE = pathlib.Path(__file__).resolve()      # .../src/tests/conftest.py
SRC_DIR   = THIS_FILE.parents[1]                  # .../src
ROOT_DIR  = SRC_DIR.parent                        # .../

# Garante que /src está no sys.path (para importar preprocessing, classifiers, etc.)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# === Imports do projeto (batem com o seu main.py) ===
from preprocessing.preprocess import load_data, preprocess
from classifiers.generic_classifier import GenericClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# === Resolução do dataset ===
# Você pode sobrescrever com a variável de ambiente DATASET_PATH se quiser.
DATASET_ENV = os.getenv("DATASET_PATH")
DATA_CANDIDATES = [
    DATASET_ENV,
    str(SRC_DIR / "data" / "demartology.csv"),  # seu layout atual
    str(ROOT_DIR / "data" / "demartology.csv"), # caso mova para a raiz um dia
    "data/demartology.csv",                     # fallback relativo ao CWD
]

def _resolve_dataset_path():
    for p in DATA_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Arquivo de dados não encontrado. Tente definir a variável de ambiente "
        "DATASET_PATH ou verifique se existe 'src/data/demartology.csv'.\n"
        f"Tentativas: {DATA_CANDIDATES}"
    )

# === Fixtures ===
@pytest.fixture(scope="session")
def df():
    path = _resolve_dataset_path()
    return load_data(path)

@pytest.fixture(scope="session")
def splits(df):
    # preprocess retorna (X_train, X_test, y_train, y_test, DISEASE_MAP)
    return preprocess(df, target_col="class")

@pytest.fixture(scope="session")
def Xy_full(df):
    # espelha o preprocess: y = df['class'] - 1
    X = df.drop("class", axis=1).copy()
    y = df["class"].astype(int) - 1
    return X, y

@pytest.fixture(scope="session")
def classifiers():
    return {
        "Decision Tree": GenericClassifier(DecisionTreeClassifier(random_state=42)),
        "Random Forest": GenericClassifier(RandomForestClassifier(n_estimators=200, random_state=42)),
    }
