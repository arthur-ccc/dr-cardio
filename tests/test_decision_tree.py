import pytest
import pandas as pd
import numpy as np
from src.decision_tree import CardiovascularDiagnosisModel

@pytest.fixture
def diagnosis_model():
    return CardiovascularDiagnosisModel()

@pytest.fixture
def synthetic_training_data():
    """Dados sintéticos para treinamento da árvore de decisão"""
    n_samples = 100
    return pd.DataFrame({
        "dor_no_peito": np.random.choice([0, 1], n_samples),
        "falta_de_ar": np.random.choice([0, 1], n_samples),
        "palpitacoes": np.random.choice([0, 1], n_samples),
        "pressao_alta": np.random.choice([0, 1], n_samples),
        "diagnostico": np.random.choice(["Angina", "IAM", "Aritmia", "Normal"], n_samples)
    })

def test_model_training(diagnosis_model, synthetic_training_data):
    """Testa o treinamento do modelo de árvore de decisão"""
    X = synthetic_training_data.drop("diagnostico", axis=1)
    y = synthetic_training_data["diagnostico"]
    
    diagnosis_model.train(X, y)
    
    assert hasattr(diagnosis_model, 'model')
    assert diagnosis_model.model is not None

def test_model_prediction(diagnosis_model, synthetic_training_data):
    """Testa a predição do modelo"""
    X = synthetic_training_data.drop("diagnostico", axis=1)
    y = synthetic_training_data["diagnostico"]
    
    diagnosis_model.train(X, y)
    
    test_case = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [1],
        "palpitacoes": [0],
        "pressao_alta": [1]
    })
    
    prediction = diagnosis_model.predict(test_case)
    
    assert prediction is not None
    assert isinstance(prediction, (str, np.str_))

def test_prediction_explainability(diagnosis_model, synthetic_training_data):
    """Testa a explicabilidade das predições"""
    X = synthetic_training_data.drop("diagnostico", axis=1)
    y = synthetic_training_data["diagnostico"]
    
    diagnosis_model.train(X, y)
    
    test_case = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [1],
        "palpitacoes": [0],
        "pressao_alta": [1]
    })
    
    explanation = diagnosis_model.explain_prediction(test_case)
    
    assert explanation is not None
    assert isinstance(explanation, dict)
    assert "decision_path" in explanation
    assert "feature_importance" in explanation

def test_model_persistence(diagnosis_model, synthetic_training_data, tmp_path):
    """Testa o salvamento e carregamento do modelo"""
    X = synthetic_training_data.drop("diagnostico", axis=1)
    y = synthetic_training_data["diagnostico"]
    
    diagnosis_model.train(X, y)
    
    # Salvar modelo
    model_path = tmp_path / "model.pkl"
    diagnosis_model.save_model(model_path)
    
    # Carregar modelo
    loaded_model = CardiovascularDiagnosisModel()
    loaded_model.load_model(model_path)
    
    # Testar predição com modelo carregado
    test_case = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [1],
        "palpitacoes": [0],
        "pressao_alta": [1]
    })
    
    original_prediction = diagnosis_model.predict(test_case)
    loaded_prediction = loaded_model.predict(test_case)
    
    assert original_prediction == loaded_prediction

def test_handle_unknown_symptoms(diagnosis_model):
    """Testa como o modelo lida com sintomas não vistos durante o treinamento"""
    # Treinar com um conjunto limitado de sintomas
    training_data = pd.DataFrame({
        "dor_no_peito": [1, 0, 1, 0],
        "falta_de_ar": [1, 1, 0, 0],
        "diagnostico": ["Angina", "Normal", "IAM", "Normal"]
    })
    
    diagnosis_model.train(
        training_data.drop("diagnostico", axis=1),
        training_data["diagnostico"]
    )
    
    # Testar com sintoma não visto durante o treinamento
    test_case = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [1],
        "sintoma_nao_treinado": [1]  # Sintoma não presente nos dados de treino
    })
    
    # Deve lidar graciosamente com características não vistas
    try:
        prediction = diagnosis_model.predict(test_case)
        assert prediction is not None
    except Exception as e:
        # Se lançar exceção, deve ser uma exceção esperada e informativa
        assert "feature" in str(e).lower() or "coluna" in str(e).lower()
def test_decision_tree_with_imbalanced_data():
    """
    Testa o comportamento da árvore de decisão com dados desbalanceados
    """
    model = CardiovascularDiagnosisModel()
    
    # Criar dados desbalanceados (muito mais casos normais que patológicos)
    n_normal = 1000
    n_disease = 100
    
    X = pd.DataFrame({
        "dor_no_peito": [1] * n_disease + [0] * n_normal,
        "falta_de_ar": [1] * n_disease + [0] * n_normal,
        "idade": np.random.randint(20, 80, n_disease + n_normal),
        "pressao_sistolica": np.random.randint(90, 180, n_disease + n_normal)
    })
    
    y = ["Doença"] * n_disease + ["Normal"] * n_normal
    
    model.train(X, y)
    
    # Verificar se o modelo lida adequadamente com o desbalanceamento
    test_case = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [1],
        "idade": [65],
        "pressao_sistolica": [160]
    })
    
    prediction = model.predict(test_case)
    explanation = model.explain_prediction(test_case)
    
    assert prediction in model.classes_
    assert "feature_importance" in explanation
    assert "decision_path" in explanation

def test_decision_tree_with_missing_values():
    """
    Testa o comportamento da árvore de decisão com valores missing
    """
    model = CardiovascularDiagnosisModel()
    
    # Criar dados com valores missing
    X = pd.DataFrame({
        "dor_no_peito": [1, 0, 1, np.nan, 0],
        "falta_de_ar": [1, np.nan, 0, 1, 0],
        "idade": [45, 60, np.nan, 55, 70],
        "pressao_sistolica": [140, np.nan, 120, 160, 130]
    })
    
    y = ["IAM", "Angina", "Normal", "IAM", "Normal"]
    
    # Deve lidar graciosamente com valores missing durante o treinamento
    try:
        model.train(X, y)
        trained_successfully = True
    except Exception as e:
        trained_successfully = False
        # Se falhar, deve ser com uma mensagem de erro apropriada
        assert "missing" in str(e).lower() or "valores" in str(e).lower()
    
    if trained_successfully:
        # Testar predição com valores missing
        test_case = pd.DataFrame({
            "dor_no_peito": [1],
            "falta_de_ar": [np.nan],
            "idade": [50],
            "pressao_sistolica": [np.nan]
        })
        
        prediction = model.predict(test_case)
        assert prediction in model.classes_

def test_decision_tree_feature_importance_calculation():
    """
    Testa o cálculo de importância de características na árvore de decisão
    """
    model = CardiovascularDiagnosisModel()
    
    # Criar dados onde uma característica é muito mais importante
    n_samples = 500
    X = pd.DataFrame({
        "dor_no_peito": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),  # Característica importante
        "falta_de_ar": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),   # Característica menos importante
        "idade": np.random.randint(20, 80, n_samples),                       # Característica contínua
        "fator_ruido": np.random.randn(n_samples)                            # Ruído (pouca importância)
    })
    
    # Criar alvo que depende principalmente da dor_no_peito
    y = np.where(X["dor_no_peito"] > 0.5, "Doença", "Normal")
    
    model.train(X, y)
    feature_importance = model.get_feature_importance()
    
    # A dor_no_peito deve ser a característica mais importante
    assert "dor_no_peito" in feature_importance
    assert feature_importance["dor_no_peito"] > feature_importance["fator_ruido"]
    assert feature_importance["dor_no_peito"] > feature_importance["falta_de_ar"]

def test_decision_tree_with_different_hyperparameters():
    """
    Testa o desempenho da árvore de decisão com diferentes hiperparâmetros
    """
    # Criar dados sintéticos
    n_samples = 1000
    X = pd.DataFrame({
        "dor_no_peito": np.random.choice([0, 1], n_samples),
        "falta_de_ar": np.random.choice([0, 1], n_samples),
        "idade": np.random.randint(20, 80, n_samples),
        "pressao_sistolica": np.random.randint(90, 180, n_samples)
    })
    
    # Criar alvo com relação não linear complexa
    y = np.where(
        (X["dor_no_peito"] == 1) & (X["idade"] > 50) & (X["pressao_sistolica"] > 140),
        "Alto_Risco",
        "Baixo_Risco"
    )
    
    # Testar com diferentes configurações de hiperparâmetros
    configurations = [
        {"max_depth": 3, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 10},
        {"max_depth": 10, "min_samples_split": 20},
        {"max_depth": None, "min_samples_split": 2}
    ]
    
    for config in configurations:
        model = CardiovascularDiagnosisModel(**config)
        model.train(X, y)
        
        # Verificar se o modelo treina sem erro
        assert model.model is not None
        
        # Fazer previsões de exemplo
        test_case = pd.DataFrame({
            "dor_no_peito": [1],
            "falta_de_ar": [0],
            "idade": [65],
            "pressao_sistolica": [160]
        })
        
        prediction = model.predict(test_case)
        assert prediction in model.classes_