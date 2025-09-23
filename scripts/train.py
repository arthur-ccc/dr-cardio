import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


# scripts/train.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.classifiers.generic_classifier import GenericClassifier
from src.preprocessing.preprocess import load_data, preprocess
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "src", "data", "demartology.csv")
data = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, disease_map = preprocess(data)




classifiers = {
    "Decision Tree": GenericClassifier(DecisionTreeClassifier(random_state=42)),
    "Random Forest": GenericClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
}

for classifier in classifiers.values():
    classifier.train(X_train, y_train)

# treina
clf_dt = classifiers["Decision Tree"]
clf_dt.train(X_train, y_train)

joblib.dump(clf_dt.model, os.path.join(MODEL_DIR, "decision_tree.pkl"))


# randomforest
clf_rf = classifiers["Random Forest"]
clf_rf.train(X_train, y_train)

joblib.dump(clf_rf.model, os.path.join(MODEL_DIR, "random_forest.pkl"))
