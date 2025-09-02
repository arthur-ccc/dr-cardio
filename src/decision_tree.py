# src/decision_tree.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import Dict, List, Any, Union
import warnings

class CardiovascularDiagnosisModel:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, **kwargs):
        """
        Inicializa o modelo de árvore de decisão para diagnóstico cardiovascular.
        
        Args:
            max_depth: Profundidade máxima da árvore
            min_samples_split: Número mínimo de amostras para dividir um nó
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            **kwargs
        )
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.classes_ = None
        self.is_trained = False
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, list, np.ndarray]) -> None:
        """
        Treina o modelo de árvore de decisão.
        
        Args:
            X: DataFrame com features ou array numpy
            y: Series com labels, lista ou array numpy
        """
        # Converter X para DataFrame se não for
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Converter y para Series se não for
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Verificar e tratar valores missing em X
        if X.isnull().any().any():
            warnings.warn("Dados contêm valores missing. Preenchendo com a moda.")
            X = X.fillna(X.mode().iloc[0])
        
        # Verificar se y tem valores missing
        if y.isnull().any():
            raise ValueError("Labels contêm valores missing. Remova ou preencha os valores.")
        
        # Salvar nomes das features para referência futura
        self.feature_names = X.columns.tolist()
        
        # Codificar labels se forem strings
        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
        else:
            y_encoded = y.values
            self.classes_ = np.unique(y)
        
        # Treinar o modelo
        self.model.fit(X, y_encoded)
        self.is_trained = True
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[str, np.str_]:
        """
        Faz predições usando o modelo treinado.
        
        Args:
            X: DataFrame com features para predição
            
        Returns:
            Predição do modelo
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Chame o método train primeiro.")
        
        # Converter X para DataFrame se não for
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Verificar se todas as features de treino estão presentes
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features ausentes: {missing_features}. "
                           f"Features esperadas: {self.feature_names}")
        
        # Verificar features extras que não foram usadas no treino
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            warnings.warn(f"Features extras serão ignoradas: {extra_features}")
            X = X[self.feature_names]
        
        # Preencher valores missing se necessário
        if X.isnull().any().any():
            warnings.warn("Dados de teste contêm valores missing. Preenchendo com a moda.")
            X = X.fillna(pd.Series({col: X[col].mode()[0] if not X[col].mode().empty else 0 
                                  for col in X.columns}))
        
        # Fazer predição
        prediction_encoded = self.model.predict(X)
        
        # Decodificar se usamos label encoder
        if hasattr(self, 'label_encoder') and self.label_encoder.classes_.size > 0:
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
        else:
            prediction = prediction_encoded
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def explain_prediction(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Explica a predição fornecendo o caminho de decisão e importância das features.
        
        Args:
            X: DataFrame com features para explicação
            
        Returns:
            Dicionário com informações de explicação
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Chame o método train primeiro.")
        
        # Converter X para DataFrame se não for
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Preparar dados (mesma lógica do predict)
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features ausentes: {missing_features}")
        
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            X = X[self.feature_names]
        
        if X.isnull().any().any():
            X = X.fillna(pd.Series({col: X[col].mode()[0] if not X[col].mode().empty else 0 
                                  for col in X.columns}))
        
        # Obter caminho de decisão
        decision_path = self.model.decision_path(X)
        node_indicator = decision_path.indices
        
        # Obter importância das features
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Obter regras de decisão (simplificado)
        decision_rules = []
        for node_id in node_indicator:
            if node_id == 0:  # Nó raiz
                continue
            
            if self.model.tree_.children_left[node_id] != self.model.tree_.children_right[node_id]:
                feature = self.feature_names[self.model.tree_.feature[node_id]]
                threshold = self.model.tree_.threshold[node_id]
                decision_rules.append(f"{feature} <= {threshold:.2f}")
        
        return {
            "decision_path": node_indicator.tolist(),
            "feature_importance": feature_importance,
            "decision_rules": decision_rules,
            "predicted_class": self.predict(X)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna a importância de cada feature.
        
        Returns:
            Dicionário com importância das features
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Chame o método train primeiro.")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def save_model(self, filepath: str) -> None:
        """
        Salva o modelo treinado em um arquivo.
        
        Args:
            filepath: Caminho do arquivo para salvar
        """
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado. Não é possível salvar.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'classes': self.classes_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Carrega um modelo treinado de um arquivo.
        
        Args:
            filepath: Caminho do arquivo para carregar
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.classes_ = model_data['classes']
            self.is_trained = True
            
        except FileNotFoundError:
            raise ValueError(f"Arquivo não encontrado: {filepath}")
        except Exception as e:
            raise ValueError(f"Erro ao carregar modelo: {str(e)}")