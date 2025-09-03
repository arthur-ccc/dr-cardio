import pandas as pd
import pytest

CSV_PATH = "CardioSymptomsDataset.csv"

@pytest.fixture
def dataset():
    return pd.read_csv(CSV_PATH)

def test_file_loads(dataset):
    """Verifica se o CSV pode ser carregado sem erros"""
    assert not dataset.empty or len(dataset.columns) > 0

def test_required_columns(dataset):
    """Confere se todas as colunas obrigatórias estão no CSV"""
    expected_columns = [
        "chest_pain","chest_tightness","angina","shortness_of_breath","dyspnea",
        "orthopnea","paroxysmal_nocturnal_dyspnea","palpitations","tachycardia","arrhythmia",
        "fatigue","weakness","dizziness","syncope","lightheadedness","sweating",    
        "edema","ankle_swelling","nausea","vomiting","diagnostic"
    ]
    assert list(dataset.columns) == expected_columns

def test_symptom_columns_are_boolean(dataset):
    """Verifica se todas as colunas de sintomas são booleanas ou 0/1"""
    symptom_columns = dataset.columns.drop("diagnostic")
    for col in symptom_columns:
        assert dataset[col].dropna().isin([0, 1, True, False]).all()

def test_diagnostic_is_string(dataset):
    """Checa se a coluna de diagnóstico é do tipo string"""
    assert dataset["diagnostic"].dtype == object

def test_no_missing_values(dataset):
    """Garante que não existem valores nulos no dataset"""
    assert not dataset.isnull().values.any()
