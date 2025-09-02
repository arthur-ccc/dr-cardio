# src/decision_tree.py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import Dict, Any, Union
import warnings

class CardiovascularDiagnosisModel:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, **kwargs):
        """
        Initialize the decision tree model for cardiovascular diagnosis.
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
        Train the decision tree model.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Handle missing values without showing warnings
        if X.isnull().any().any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = X.fillna(X.mode().iloc[0])

        if y.isnull().any():
            raise ValueError("Labels contain missing values. Remove or fill them.")

        self.feature_names = X.columns.tolist()

        if y.dtype == 'object':
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
        else:
            y_encoded = y.values
            self.classes_ = np.unique(y)

        self.model.fit(X, y_encoded)
        self.is_trained = True

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[str, np.str_]:
        """
        Make predictions using the trained model.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            # Suppress warning about extra features
            X = X[self.feature_names]

        if X.isnull().any().any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = X.fillna(pd.Series({col: X[col].mode()[0] if not X[col].mode().empty else 0
                                        for col in X.columns}))

        prediction_encoded = self.model.predict(X)

        if hasattr(self, 'label_encoder') and self.label_encoder.classes_.size > 0:
            prediction = self.label_encoder.inverse_transform(prediction_encoded)
        else:
            prediction = prediction_encoded

        return prediction[0] if len(prediction) == 1 else prediction

    def explain_prediction(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Explain prediction with decision path and feature importance.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            X = X[self.feature_names]

        if X.isnull().any().any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = X.fillna(pd.Series({col: X[col].mode()[0] if not X[col].mode().empty else 0
                                        for col in X.columns}))

        decision_path = self.model.decision_path(X)
        node_indicator = decision_path.indices

        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        decision_rules = []
        for node_id in node_indicator:
            if node_id == 0:
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
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save_model(self, filepath: str) -> None:
        if not self.is_trained:
            raise ValueError("Model is not trained. Cannot save.")

        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'classes': self.classes_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.classes_ = model_data['classes']
            self.is_trained = True

        except FileNotFoundError:
            raise ValueError(f"File not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")