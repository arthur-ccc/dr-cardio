import joblib
import os

# Caminho correto
model_path = os.path.join(os.path.dirname(__file__), "../models/decision_tree.pkl")

# Carrega apenas o modelo
clf = joblib.load(model_path)

# Lista de teste (mesma ordem de X_train)
features = [
    2, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 55
]

# Predição
pred = clf.predict([features])
print("Predição:", pred)

# Probabilidades, se suportado
if hasattr(clf, "predict_proba"):
    probs = clf.predict_proba([features])
    print("Probabilidades:", probs)
