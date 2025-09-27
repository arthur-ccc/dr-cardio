from preprocessing.preprocess import load_data, preprocess
from classifiers.generic_classifier import GenericClassifier
from accuracy.evaluate import evaluate_model
from data.mock import generate_mock_dataset, export #MOCKADASSO

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = load_data("data/demartology.csv")
X_train, X_test, y_train, y_test, le = preprocess(df, target_col="class")


classifiers = {
    "Decision Tree": GenericClassifier(DecisionTreeClassifier(random_state=42)),
    "Random Forest": GenericClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
}

for nome, classificador in classifiers.items():
    print(f"=> Avaliando {nome}")
    classificador.train(X_train, y_train)
    evaluate_model(classificador, X_test, y_test)
    print("=" * 70)
