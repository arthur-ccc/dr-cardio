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