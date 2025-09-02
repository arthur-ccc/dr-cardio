import pytest
import pandas as pd
import numpy as np
from src.decision_tree import CardiovascularDiagnosisModel

@pytest.fixture
def diagnosis_model():
    return CardiovascularDiagnosisModel()

@pytest.fixture
def synthetic_training_data():
    """Synthetic data for training the decision tree"""
    n_samples = 100
    return pd.DataFrame({
        "chest_pain": np.random.choice([0, 1], n_samples),
        "shortness_of_breath": np.random.choice([0, 1], n_samples),
        "palpitations": np.random.choice([0, 1], n_samples),
        "high_blood_pressure": np.random.choice([0, 1], n_samples),
        "diagnosis": np.random.choice(["Angina", "MI", "Arrhythmia", "Normal"], n_samples)
    })

def test_model_training(diagnosis_model, synthetic_training_data):
    """Tests the training of the decision tree model"""
    X = synthetic_training_data.drop("diagnosis", axis=1)
    y = synthetic_training_data["diagnosis"]
    
    diagnosis_model.train(X, y)
    
    assert hasattr(diagnosis_model, 'model')
    assert diagnosis_model.model is not None

def test_model_prediction(diagnosis_model, synthetic_training_data):
    """Tests the model's prediction capability"""
    X = synthetic_training_data.drop("diagnosis", axis=1)
    y = synthetic_training_data["diagnosis"]
    
    diagnosis_model.train(X, y)
    
    test_case = pd.DataFrame({
        "chest_pain": [1],
        "shortness_of_breath": [1],
        "palpitations": [0],
        "high_blood_pressure": [1]
    })
    
    prediction = diagnosis_model.predict(test_case)
    
    assert prediction is not None
    assert isinstance(prediction, (str, np.str_))

def test_prediction_explainability(diagnosis_model, synthetic_training_data):
    """Tests the explainability of predictions"""
    X = synthetic_training_data.drop("diagnosis", axis=1)
    y = synthetic_training_data["diagnosis"]
    
    diagnosis_model.train(X, y)
    
    test_case = pd.DataFrame({
        "chest_pain": [1],
        "shortness_of_breath": [1],
        "palpitations": [0],
        "high_blood_pressure": [1]
    })
    
    explanation = diagnosis_model.explain_prediction(test_case)
    
    assert explanation is not None
    assert isinstance(explanation, dict)
    assert "decision_path" in explanation
    assert "feature_importance" in explanation

def test_model_persistence(diagnosis_model, synthetic_training_data, tmp_path):
    """Tests saving and loading the model"""
    X = synthetic_training_data.drop("diagnosis", axis=1)
    y = synthetic_training_data["diagnosis"]
    
    diagnosis_model.train(X, y)
    
    # Save model
    model_path = tmp_path / "model.pkl"
    diagnosis_model.save_model(model_path)
    
    # Load model
    loaded_model = CardiovascularDiagnosisModel()
    loaded_model.load_model(model_path)
    
    # Test prediction with loaded model
    test_case = pd.DataFrame({
        "chest_pain": [1],
        "shortness_of_breath": [1],
        "palpitations": [0],
        "high_blood_pressure": [1]
    })
    
    original_prediction = diagnosis_model.predict(test_case)
    loaded_prediction = loaded_model.predict(test_case)
    
    assert original_prediction == loaded_prediction

def test_handle_unknown_symptoms(diagnosis_model):
    """Tests how the model handles symptoms not seen during training"""
    # Train with a limited set of symptoms
    training_data = pd.DataFrame({
        "chest_pain": [1, 0, 1, 0],
        "shortness_of_breath": [1, 1, 0, 0],
        "diagnosis": ["Angina", "Normal", "MI", "Normal"]
    })
    
    diagnosis_model.train(
        training_data.drop("diagnosis", axis=1),
        training_data["diagnosis"]
    )
    
    # Test with a symptom not seen during training
    test_case = pd.DataFrame({
        "chest_pain": [1],
        "shortness_of_breath": [1],
        "unseen_symptom": [1]  # Symptom not present in training data
    })
    
    # Should handle unseen features gracefully
    try:
        prediction = diagnosis_model.predict(test_case)
        assert prediction is not None
    except Exception as e:
        # If it raises an exception, it should be informative
        assert "feature" in str(e).lower() or "column" in str(e).lower()

def test_decision_tree_with_imbalanced_data():
    """Tests decision tree behavior with imbalanced data"""
    model = CardiovascularDiagnosisModel()
    
    # Create imbalanced data (many more normal cases than diseased)
    n_normal = 1000
    n_disease = 100
    
    X = pd.DataFrame({
        "chest_pain": [1] * n_disease + [0] * n_normal,
        "shortness_of_breath": [1] * n_disease + [0] * n_normal,
        "age": np.random.randint(20, 80, n_disease + n_normal),
        "systolic_bp": np.random.randint(90, 180, n_disease + n_normal)
    })
    
    y = ["Disease"] * n_disease + ["Normal"] * n_normal
    
    model.train(X, y)
    
    # Check if the model handles imbalance correctly
    test_case = pd.DataFrame({
        "chest_pain": [1],
        "shortness_of_breath": [1],
        "age": [65],
        "systolic_bp": [160]
    })
    
    prediction = model.predict(test_case)
    explanation = model.explain_prediction(test_case)
    
    assert prediction in model.classes_
    assert "feature_importance" in explanation
    assert "decision_path" in explanation

def test_decision_tree_with_missing_values():
    """Tests decision tree behavior with missing values"""
    model = CardiovascularDiagnosisModel()
    
    # Create data with missing values
    X = pd.DataFrame({
        "chest_pain": [1, 0, 1, np.nan, 0],
        "shortness_of_breath": [1, np.nan, 0, 1, 0],
        "age": [45, 60, np.nan, 55, 70],
        "systolic_bp": [140, np.nan, 120, 160, 130]
    })
    
    y = ["MI", "Angina", "Normal", "MI", "Normal"]
    
    # Should handle missing values gracefully during training
    try:
        model.train(X, y)
        trained_successfully = True
    except Exception as e:
        trained_successfully = False
        assert "missing" in str(e).lower() or "values" in str(e).lower()
    
    if trained_successfully:
        # Test prediction with missing values
        test_case = pd.DataFrame({
            "chest_pain": [1],
            "shortness_of_breath": [np.nan],
            "age": [50],
            "systolic_bp": [np.nan]
        })
        
        prediction = model.predict(test_case)
        assert prediction in model.classes_

def test_decision_tree_feature_importance_calculation():
    """Tests feature importance calculation in the decision tree"""
    model = CardiovascularDiagnosisModel()
    
    n_samples = 500
    X = pd.DataFrame({
        "chest_pain": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),  # Important feature
        "shortness_of_breath": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),  # Less important
        "age": np.random.randint(20, 80, n_samples),  # Continuous feature
        "noise_factor": np.random.randn(n_samples)    # Noise (less important)
    })
    
    # Target mainly depends on chest_pain
    y = np.where(X["chest_pain"] > 0.5, "Disease", "Normal")
    
    model.train(X, y)
    feature_importance = model.get_feature_importance()
    
    assert "chest_pain" in feature_importance
    assert feature_importance["chest_pain"] > feature_importance["noise_factor"]
    assert feature_importance["chest_pain"] > feature_importance["shortness_of_breath"]

def test_decision_tree_with_different_hyperparameters():
    """Tests decision tree performance with different hyperparameters"""
    n_samples = 1000
    X = pd.DataFrame({
        "chest_pain": np.random.choice([0, 1], n_samples),
        "shortness_of_breath": np.random.choice([0, 1], n_samples),
        "age": np.random.randint(20, 80, n_samples),
        "systolic_bp": np.random.randint(90, 180, n_samples)
    })
    
    y = np.where(
        (X["chest_pain"] == 1) & (X["age"] > 50) & (X["systolic_bp"] > 140),
        "High_Risk",
        "Low_Risk"
    )
    
    configurations = [
        {"max_depth": 3, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 10},
        {"max_depth": 10, "min_samples_split": 20},
        {"max_depth": None, "min_samples_split": 2}
    ]
    
    for config in configurations:
        model = CardiovascularDiagnosisModel(**config)
        model.train(X, y)
        
        assert model.model is not None
        
        test_case = pd.DataFrame({
            "chest_pain": [1],
            "shortness_of_breath": [0],
            "age": [65],
            "systolic_bp": [160]
        })
        
        prediction = model.predict(test_case)
        assert prediction in model.classes_
