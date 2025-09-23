from preprocessing.preprocess import load_data, preprocess
from classifiers.generic_classifier import GenericClassifier
from accuracy.evaluate import evaluate_model
from data.mock import generate_mock_dataset, export #MOCKADASSO

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "demartology.csv")
path = "data/demartology.csv"

df = load_data(file_path)
X_train, X_test, y_train, y_test, le = preprocess(df, target_col="class")


classifiers = {
    "Decision Tree": GenericClassifier(DecisionTreeClassifier(random_state=42)),
    "Random Forest": GenericClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
}

for classifier in classifiers.values():
    classifier.train(X_train, y_train)

# paths = classifiers["Decision Tree"].decision_path(X_test)

# for i, p in enumerate(paths[:3]):
#     print(f"Amostra {i}:")
#     for step in p:
#         print("  ", step)

for nome, classificador in classifiers.items():
    print(f"=> Avaliando {nome}")
    evaluate_model(classificador, X_test, y_test)
    print("=" * 70)
