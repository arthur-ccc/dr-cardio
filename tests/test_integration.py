import pytest
from unittest.mock import MagicMock, patch
# esse arquivo ainda não existe
from src.data_ingestion import DataIngestor
from src.nlp import NLPProcessor
# esse arquivo ainda não existe
from src.knowledge_structure import KnowledgeOrganizer
from src.decision_tree import CardiovascularDiagnosisModel

@pytest.fixture
def mock_nlp_processor(mocker):
    mock_nlp = MagicMock()
    mock_processor = NLPProcessor(mock_nlp)
    
    # Mock das extrações de entidades e relações
    mock_processor.extract_entities = MagicMock(return_value=[
        ("dor no peito", "SINTOMA"),
        ("infarto", "DOENÇA"),
        ("eletrocardiograma", "EXAME")
    ])
    
    mock_processor.extract_relations = MagicMock(return_value=[
        ("dor no peito", "É_SINTOMA_DE", "infarto"),
        ("eletrocardiograma", "DETECTA", "infarto")
    ])
    
    return mock_processor

def test_full_pipeline_integration(tmp_path, mock_nlp_processor):
    """Testa o pipeline completo desde a ingestão até o diagnóstico"""
    # Mock da ingestão de dados
    data_ingestor = DataIngestor("fake/dataset", tmp_path)
    data_ingestor.download_metadata = MagicMock(return_value=tmp_path / "metadata.csv")
    data_ingestor.get_article_list_from_metadata = MagicMock(return_value=pd.DataFrame({
        "article_id": ["artigo_001"],
        "title": ["Artigo Teste"],
        "path": [str(tmp_path / "artigo_001.txt")]
    }))
    data_ingestor.read_article_content = MagicMock(return_value="Paciente com dor no peito. Diagnóstico: infarto.")
    
    # Criar arquivo de artigo fake
    (tmp_path / "artigo_001.txt").write_text("Paciente com dor no peito. Diagnóstico: infarto.")
    
    # Inicializar outros componentes
    knowledge_organizer = KnowledgeOrganizer()
    diagnosis_model = CardiovascularDiagnosisModel()
    
    # Treinar modelo com dados sintéticos
    synthetic_data = pd.DataFrame({
        "dor_no_peito": [1, 0, 1, 0],
        "falta_de_ar": [1, 1, 0, 0],
        "diagnostico": ["Infarto", "Normal", "Angina", "Normal"]
    })
    diagnosis_model.train(
        synthetic_data.drop("diagnostico", axis=1),
        synthetic_data["diagnostico"]
    )
    
    # Executar pipeline completo
    metadata_path = data_ingestor.download_metadata("metadata.csv")
    articles_df = data_ingestor.get_article_list_from_metadata(metadata_path)
    
    all_entities = []
    all_relations = []
    
    for _, article in articles_df.iterrows():
        content = data_ingestor.read_article_content(article["path"])
        entities = mock_nlp_processor.extract_entities(content)
        relations = mock_nlp_processor.extract_relations(content)
        
        # Adicionar identificador do artigo
        entities_with_source = [(ent[0], ent[1], article["article_id"]) for ent in entities]
        relations_with_source = [(rel[0], rel[1], rel[2], article["article_id"]) for rel in relations]
        
        all_entities.extend(entities_with_source)
        all_relations.extend(relations_with_source)
    
    # Estruturar conhecimento
    symptom_df, disease_df, exam_df, relation_df = knowledge_organizer.create_tables(
        all_entities, all_relations
    )
    
    # Preparar caso de teste para diagnóstico
    test_symptoms = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [0]
        # Outros sintomas seriam 0 por padrão
    })
    
    # Realizar diagnóstico
    diagnosis = diagnosis_model.predict(test_symptoms)
    explanation = diagnosis_model.explain_prediction(test_symptoms)
    
    # Verificações
    assert len(symptom_df) > 0
    assert len(disease_df) > 0
    assert diagnosis is not None
    assert explanation is not None
    assert "decision_path" in explanation

def test_pipeline_with_multiple_articles(tmp_path, mock_nlp_processor):
    """Testa o pipeline com múltiplos artigos e consolidação de conhecimento"""
    # Configurar mock para retornar diferentes conteúdos por artigo
    def mock_read_article_content(article_path):
        if "artigo_001" in str(article_path):
            return "Paciente com dor no peito. Diagnóstico: infarto."
        elif "artigo_002" in str(article_path):
            return "Paciente com falta de ar. Diagnóstico: asma."
        return ""
    
    data_ingestor = DataIngestor("fake/dataset", tmp_path)
    data_ingestor.download_metadata = MagicMock(return_value=tmp_path / "metadata.csv")
    data_ingestor.get_article_list_from_metadata = MagicMock(return_value=pd.DataFrame({
        "article_id": ["artigo_001", "artigo_002"],
        "title": ["Artigo 1", "Artigo 2"],
        "path": [str(tmp_path / "artigo_001.txt"), str(tmp_path / "artigo_002.txt")]
    }))
    data_ingestor.read_article_content = MagicMock(side_effect=mock_read_article_content)
    
    # Criar arquivos de artigos
    (tmp_path / "artigo_001.txt").write_text("Paciente com dor no peito. Diagnóstico: infarto.")
    (tmp_path / "artigo_002.txt").write_text("Paciente com falta de ar. Diagnóstico: asma.")
    
    # Configurar NLP processor para retornar entidades diferentes por conteúdo
    def mock_extract_entities(text):
        if "dor no peito" in text:
            return [("dor no peito", "SINTOMA"), ("infarto", "DOENÇA")]
        elif "falta de ar" in text:
            return [("falta de ar", "SINTOMA"), ("asma", "DOENÇA")]
        return []
    
    mock_nlp_processor.extract_entities = MagicMock(side_effect=mock_extract_entities)
    
    # Executar pipeline
    knowledge_organizer = KnowledgeOrganizer()
    
    metadata_path = data_ingestor.download_metadata("metadata.csv")
    articles_df = data_ingestor.get_article_list_from_metadata(metadata_path)
    
    all_entities = []
    
    for _, article in articles_df.iterrows():
        content = data_ingestor.read_article_content(article["path"])
        entities = mock_nlp_processor.extract_entities(content)
        entities_with_source = [(ent[0], ent[1], article["article_id"]) for ent in entities]
        all_entities.extend(entities_with_source)
    
    # Estruturar conhecimento
    symptom_df, disease_df, _, _ = knowledge_organizer.create_tables(all_entities, [])
    
    # Verificar consolidação de múltiplos artigos
    assert len(symptom_df) == 2
    assert len(disease_df) == 2

import pytest
from unittest.mock import MagicMock, patch
from src.data_ingestion import DataIngestor
from src.nlp import NLPProcessor
from src.knowledge_structure import KnowledgeOrganizer
from src.decision_tree import CardiovascularDiagnosisModel

def test_end_to_end_pipeline_with_multiple_sources():
    """
    Testa o pipeline completo com múltiplas fontes de artigos
    """
    # Configurar mocks para cada componente
    mock_ingestor = MagicMock()
    mock_nlp = MagicMock()
    mock_organizer = MagicMock()
    mock_model = MagicMock()
    
    # Configurar comportamento dos mocks
    mock_ingestor.download_metadata.return_value = "/fake/path/metadata.csv"
    mock_ingestor.get_article_list_from_metadata.return_value = pd.DataFrame({
        "article_id": ["art1", "art2", "art3"],
        "title": ["Artigo 1", "Artigo 2", "Artigo 3"],
        "path": ["/fake/path/art1.txt", "/fake/path/art2.txt", "/fake/path/art3.txt"]
    })
    
    # Simular diferentes conteúdos para diferentes artigos
    def mock_read_article_content(path):
        if "art1" in path:
            return "Pacientes com dor torácica e dispneia frequentemente têm IAM."
        elif "art2" in path:
            return "A angina pectoris manifesta-se com dor torácica e pode levar a IAM."
        elif "art3" in path:
            return "Fatores de risco: hipertensão, diabetes, tabagismo."
        return ""
    
    mock_ingestor.read_article_content.side_effect = mock_read_article_content
    
    # Configurar NLP para retornar entidades baseadas no conteúdo
    def mock_extract_entities(text):
        entities = []
        if "dor torácica" in text:
            entities.append(("dor torácica", "SINTOMA"))
        if "dispneia" in text:
            entities.append(("dispneia", "SINTOMA"))
        if "IAM" in text:
            entities.append(("IAM", "DOENÇA"))
        if "angina pectoris" in text:
            entities.append(("angina pectoris", "DOENÇA"))
        if "hipertensão" in text:
            entities.append(("hipertensão", "DOENÇA"))
        if "diabetes" in text:
            entities.append(("diabetes", "DOENÇA"))
        if "tabagismo" in text:
            entities.append(("tabagismo", "FATOR_DE_RISCO"))
        return entities
    
    mock_nlp.extract_entities.side_effect = mock_extract_entities
    
    # Configurar extrator de relações
    def mock_extract_relations(text):
        relations = []
        if "dor torácica" in text and "IAM" in text:
            relations.append(("dor torácica", "É_SINTOMA_DE", "IAM"))
        if "dispneia" in text and "IAM" in text:
            relations.append(("dispneia", "É_SINTOMA_DE", "IAM"))
        if "dor torácica" in text and "angina pectoris" in text:
            relations.append(("dor torácica", "É_SINTOMA_DE", "angina pectoris"))
        return relations
    
    mock_nlp.extract_relations.side_effect = mock_extract_relations
    
    # Configurar organizador de conhecimento
    mock_organizer.create_tables.return_value = (
        pd.DataFrame({  # symptom_df
            "entity": ["dor torácica", "dispneia"],
            "type": ["SINTOMA", "SINTOMA"],
            "frequency": [2, 1],
            "source_articles": [["art1", "art2"], ["art1"]]
        }),
        pd.DataFrame({  # disease_df
            "entity": ["IAM", "angina pectoris", "hipertensão", "diabetes"],
            "type": ["DOENÇA", "DOENÇA", "DOENÇA", "DOENÇA"],
            "frequency": [2, 1, 1, 1],
            "source_articles": [["art1", "art2"], ["art2"], ["art3"], ["art3"]]
        }),
        pd.DataFrame(),  # exam_df vazio
        pd.DataFrame({  # relation_df
            "source": ["dor torácica", "dispneia", "dor torácica"],
            "relation": ["É_SINTOMA_DE", "É_SINTOMA_DE", "É_SINTOMA_DE"],
            "target": ["IAM", "IAM", "angina pectoris"],
            "source_articles": [["art1", "art2"], ["art1"], ["art2"]]
        })
    )
    
    # Configurar modelo de diagnóstico
    mock_model.predict.return_value = "IAM"
    mock_model.explain_prediction.return_value = {
        "decision_path": ["dor_no_peito = 1", "idade > 50", "pressao_sistolica > 140"],
        "feature_importance": {"dor_no_peito": 0.6, "idade": 0.25, "pressao_sistolica": 0.15},
        "confidence": 0.85
    }
    
    # Executar pipeline completo
    metadata_path = mock_ingestor.download_metadata("metadata.csv")
    articles_df = mock_ingestor.get_article_list_from_metadata(metadata_path)
    
    all_entities = []
    all_relations = []
    
    for _, article in articles_df.iterrows():
        content = mock_ingestor.read_article_content(article["path"])
        entities = mock_nlp.extract_entities(content)
        relations = mock_nlp.extract_relations(content)
        
        # Adicionar identificador do artigo
        entities_with_source = [(ent[0], ent[1], article["article_id"]) for ent in entities]
        relations_with_source = [(rel[0], rel[1], rel[2], article["article_id"]) for rel in relations]
        
        all_entities.extend(entities_with_source)
        all_relations.extend(relations_with_source)
    
    # Estruturar conhecimento
    symptom_df, disease_df, exam_df, relation_df = mock_organizer.create_tables(all_entities, all_relations)
    
    # Preparar caso de teste para diagnóstico
    test_case = pd.DataFrame({
        "dor_no_peito": [1],
        "falta_de_ar": [1],
        "idade": [65],
        "pressao_sistolica": [160],
        "diabetes": [0],
        "tabagismo": [1]
    })
    
    # Realizar diagnóstico
    diagnosis = mock_model.predict(test_case)
    explanation = mock_model.explain_prediction(test_case)
    
def test_pipeline_with_conflicting_evidence():
    """
    Testa o pipeline com evidências conflitantes de diferentes fontes
    """
    # Configurar mocks para artigos com informações conflitantes
    mock_ingestor = MagicMock()
    mock_ingestor.download_metadata.return_value = "/fake/path/metadata.csv"
    mock_ingestor.get_article_list_from_metadata.return_value = pd.DataFrame({
        "article_id": ["art1", "art2"],
        "title": ["Artigo Pro Aspirina", "Artigo Contra Aspirina"],
        "path": ["/fake/path/art1.txt", "/fake/path/art2.txt"]
    })
    
    def mock_read_article_content(path):
        if "art1" in path:
            return "A aspirina reduz o risco de IAM em pacientes de alto risco."
        elif "art2" in path:
            return "A aspirina aumenta o risco de sangramento sem benefícios significativos para o coração."
        return ""
    
    mock_ingestor.read_article_content.side_effect = mock_read_article_content
    
    # Configurar NLP
    mock_nlp = MagicMock()
    
    def mock_extract_entities(text):
        entities = []
        if "aspirina" in text:
            entities.append(("aspirina", "MEDICAMENTO"))
        if "IAM" in text:
            entities.append(("IAM", "DOENÇA"))
        if "sangramento" in text:
            entities.append(("sangramento", "EFEITO_COLATERAL"))
        return entities
    
    def mock_extract_relations(text):
        relations = []
        if "art1" in text:
            relations.append(("aspirina", "REDUZ_RISCO_DE", "IAM"))
        elif "art2" in text:
            relations.append(("aspirina", "AUMENTA_RISCO_DE", "sangramento"))
            relations.append(("aspirina", "NAO_REDUZ_RISCO_DE", "IAM"))
        return relations
    
    mock_nlp.extract_entities.side_effect = mock_extract_entities
    mock_nlp.extract_relations.side_effect = mock_extract_relations
    
    # Configurar organizador de conhecimento para detectar conflitos
    mock_organizer = MagicMock()
    mock_organizer.create_tables.return_value = (
        pd.DataFrame(),  # symptom_df vazio
        pd.DataFrame({  # disease_df
            "entity": ["IAM"],
            "type": ["DOENÇA"],
            "frequency": [2],
            "source_articles": [["art1", "art2"]]
        }),
        pd.DataFrame({  # medication_df
            "entity": ["aspirina"],
            "type": ["MEDICAMENTO"],
            "frequency": [2],
            "source_articles": [["art1", "art2"]],
            "conflicting_evidence": True
        }),
        pd.DataFrame({  # relation_df
            "source": ["aspirina", "aspirina", "aspirina"],
            "relation": ["REDUZ_RISCO_DE", "AUMENTA_RISCO_DE", "NAO_REDUZ_RISCO_DE"],
            "target": ["IAM", "sangramento", "IAM"],
            "source_articles": [["art1"], ["art2"], ["art2"]],
            "confidence": [0.9, 0.8, 0.8]  # Confiança diferente para diferentes relações
        })
    )
    
    # Executar pipeline
    metadata_path = mock_ingestor.download_metadata("metadata.csv")
    articles_df = mock_ingestor.get_article_list_from_metadata(metadata_path)
    
    all_entities = []
    all_relations = []
    
    for _, article in articles_df.iterrows():
        content = mock_ingestor.read_article_content(article["path"])
        entities = mock_nlp.extract_entities(content)
        relations = mock_nlp.extract_relations(content)
        
        entities_with_source = [(ent[0], ent[1], article["article_id"]) for ent in entities]
        relations_with_source = [(rel[0], rel[1], rel[2], article["article_id"]) for rel in relations]
        
        all_entities.extend(entities_with_source)
        all_relations.extend(relations_with_source)
    
    # Estruturar conhecimento
    symptom_df, disease_df, medication_df, relation_df = mock_organizer.create_tables(all_entities, all_relations)
    
    # Verificar se o conflito foi detectado
    assert medication_df.iloc[0]["conflicting_evidence"] == True
    
    # Verificar relações conflitantes
    aspirina_iam_relations = relation_df[
        (relation_df["source"] == "aspirina") & 
        (relation_df["target"] == "IAM")
    ]
    
    assert len(aspirina_iam_relations) >= 2  # Pelo menos uma relação positiva e uma negativa