import pytest
import time
from src.nlp import NLPProcessor
from src.knowledge_structure import KnowledgeOrganizer

@pytest.mark.performance
def test_nlp_processing_performance(nlp_processor):
    """Testa o desempenho do processamento NLP com texto longo"""
    # Gerar texto longo (~50KB)
    long_text = "Paciente com " + "dor no peito, " * 2000
    
    start_time = time.time()
    entities = nlp_processor.extract_entities(long_text)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    assert processing_time < 10.0  # Não deve levar mais de 10 segundos
    assert len(entities) > 0

@pytest.mark.performance
def test_knowledge_organization_performance():
    """Testa o desempenho da organização de conhecimento com muitos dados"""
    organizer = KnowledgeOrganizer()
    
    # Gerar muitas entidades e relações
    n_entities = 10000
    entities = [(f"sintoma_{i}", "SINTOMA", f"artigo_{i % 100}") for i in range(n_entities)]
    relations = [(f"sintoma_{i}", "RELACIONADO_A", f"doenca_{i % 50}", f"artigo_{i % 100}") 
                for i in range(n_entities // 2)]
    
    start_time = time.time()
    symptom_df, disease_df, exam_df, relation_df = organizer.create_tables(entities, relations)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    assert processing_time < 30.0  # Não deve levar mais de 30 segundos
    assert len(symptom_df) == n_entities  # Todas as entidades devem ser processadas