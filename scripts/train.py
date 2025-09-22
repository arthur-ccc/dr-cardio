# scripts/train.py
import joblib
from classifiers.generic_classifier import GenericClassifier
from preprocessing.preprocess import load_data, preprocess
from sklearn.tree import DecisionTreeClassifier
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# carrega dataset
data = load_data("data/demartology.csv")
X_train, X_test, y_train, y_test, disease_map = preprocess(data)

# treina
clf = GenericClassifier(DecisionTreeClassifier(random_state=42))
clf.train(X_train, y_train)

# 🔥 salva só o modelo puro (sklearn)
joblib.dump(clf.model, os.path.join(MODEL_DIR, "decision_tree.pkl"))

# opcional: salvar também o mapa de doenças separado
joblib.dump(disease_map, os.path.join(MODEL_DIR, "disease_map.pkl"))
