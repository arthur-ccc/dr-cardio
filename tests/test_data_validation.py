import pytest
# esse arquivo ainda não existe
from src.data_validation import DataValidator

@pytest.fixture
def data_validator():
    return DataValidator()

def test_validate_medical_terms(data_validator):
    """Testa a validação de termos médicos contra um vocabulário controlado"""
    valid_terms = ["dor no peito", "infarto agudo do miocárdio", "eletrocardiograma"]
    invalid_terms = ["dor na barriga", "gripe", "raio-x"]
    
    for term in valid_terms:
        assert data_validator.validate_medical_term(term) == True
    
    for term in invalid_terms:
        assert data_validator.validate_medical_term(term) == False

def test_detect_contradictions(data_validator):
    """Testa a detecção de contradições entre diferentes fontes"""
    claims_from_source_a = [("aspirina", "REDUZ_RISCO", "infarto")]
    claims_from_source_b = [("aspirina", "AUMENTA_RISCO", "sangramento")]
    
    # Estas não são diretamente contraditórias
    contradiction_score_1 = data_validator.detect_contradictions(
        claims_from_source_a, claims_from_source_b
    )
    assert contradiction_score_1 < 0.5  # Baixo score de contradição
    
    claims_from_source_c = [("aspirina", "REDUZ_RISCO", "infarto")]
    claims_from_source_d = [("aspirina", "NÃO_REDUZ_RISCO", "infarto")]
    
    # Estas são diretamente contraditórias
    contradiction_score_2 = data_validator.detect_contradictions(
        claims_from_source_c, claims_from_source_d
    )
    assert contradiction_score_2 > 0.8  # Alto score de contradição

def test_validate_source_reliability(data_validator):
    """Testa a avaliação da confiabilidade de fontes"""
    high_quality_source = "Journal of the American College of Cardiology"
    low_quality_source = "Blog de saúde não verificada"
    
    assert data_validator.validate_source_reliability(high_quality_source) > 0.8
    assert data_validator.validate_source_reliability(low_quality_source) < 0.4