from sklearn.base import BaseEstimator


class GenericClassifier:
    
    def __init__(self, model: BaseEstimator):
        self.model = model
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            raise NotImplementedError("Este classificador não suporta probabilidade.")
        
    def decision_path(self, X_test):
        """
        Retorna o caminho percorrido na árvore para cada amostra.
        Cada caminho é uma lista de condições (feature, limiar, direção).
        """
        if not hasattr(self.model, "decision_path"):
            raise NotImplementedError("Este modelo não suporta decision_path.")
        
        node_indicator = self.model.decision_path(X_test)
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        paths = []
        for sample_id in range(X_test.shape[0]):
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]

            sample_path = []
            for node_id in node_index:
                if feature[node_id] != -2:  # -2 significa nó folha
                    if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                        threshold_sign = "<="
                    else:
                        threshold_sign = ">"
                    sample_path.append(
                        f"X[{feature[node_id]}] (valor={X_test[sample_id, feature[node_id]]}) "
                        f"{threshold_sign} {threshold[node_id]:.2f}"
                    )
                else:
                    sample_path.append(f"Nó folha (classe = {self.model.classes_[np.argmax(self.model.tree_.value[node_id])]})")
            paths.append(sample_path)
        
        return paths