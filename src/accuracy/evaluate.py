from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

def evaluate_model(model, X_test, y_test, disease_map=None):
    """
    Avalia um modelo de classificação multiclasse.

    Parâmetros:
    - model: classificador treinado (GenericClassifier)
    - X_test: features de teste
    - y_test: labels de teste (numéricos começando em 0)
    - disease_map: dicionário {int: nome_da_doenca}
    """

    # Predição
    y_pred = model.predict(X_test)

    # Accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Definir nomes das classes (se disease_map for fornecido)
    if disease_map:
        target_names = [disease_map[i+1] for i in sorted(set(y_test))]
    else:
        target_names = None

    # Classification report
    print("\nClassification Report:\n", classification_report(
        y_test,
        y_pred,
        target_names=target_names
    ))

    # ROC AUC (multiclasse ou binário)
    try:
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] > 2:
            # Multiclasse: one-vs-rest
            y_test_bin = label_binarize(y_test, classes=range(y_proba.shape[1]))
            roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
        else:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        print("ROC AUC:", roc_auc)
    except (NotImplementedError, ValueError):
        print("ROC AUC não disponível para este classificador")
