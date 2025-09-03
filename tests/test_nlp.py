import pytest
import spacy
import pandas as pd
from pathlib import Path
from src.nlp import NLPProcessor


# Esse teste diferente do anterior faz um mock de um arquivo CSV que visa ser mais condizente com a base que temos em projeto, o "CardioSymptomsDataset.csv"
# O último como tá em hardcode não se faz necessário carregar, então não se usa da base, é necessário refatorar ele para essa lista de testes.
@pytest.fixture(scope="module")
def nlp_model():
    # Carrega o modelo small em inglês
    return spacy.load("en_core_web_sm")

@pytest.fixture
def mock_csv_path(tmp_path: Path) -> str:
    """
    Cria um arquivo CSV FALSO em um diretório temporário para o teste.
    """
    csv_content = """symptom,disease,frequency
chest pain,myocardial infarction,High
shortness of breath,heart failure,High
dizziness,arrhythmia,Low
fatigue,anemia,Medium
"""
    # tmp_path é uma fixture mágica do pytest que cria um diretório temporário
    p = tmp_path / "mock_dataset.csv"
    p.write_text(csv_content)
    return str(p)

@pytest.fixture
def nlp_processor(nlp_model, mock_csv_path: str) -> NLPProcessor:
    """
    ESTA É A MUDANÇA PRINCIPAL: A fixture agora cria o NLPProcessor
    usando o caminho para o nosso CSV de teste.
    Isso forçará a mudança no construtor da sua classe.
    """
    # O teste vai falhar aqui até que você altere NLPProcessor.__init__
    return NLPProcessor(nlp_model, csv_path=mock_csv_path)


# --- Testes para Extração de Entidades (Orientada a Dados) ---

class TestExtractEntities:

    def test_extract_entities_from_csv_data(self, nlp_processor):
        """
        Verifica se as entidades do nosso CSV mock são extraídas corretamente.
        """
        text = "The patient shows signs of chest pain and dizziness."
        result = nlp_processor.extract_entities(text)
        result_set = {(ent[0].lower(), ent[1]) for ent in result}
        
        # Estas entidades existem no nosso mock_dataset.csv
        expected_set = {
            ("chest pain", "SYMPTOM"),
            ("dizziness", "SYMPTOM")
        }
        assert result_set == expected_set

    def test_does_not_extract_entities_not_in_csv(self, nlp_processor):
        """
        Garante que termos que não estão no CSV não sejam extraídos.
        """
        # "Headache" não está no nosso mock_dataset.csv
        text = "The patient complained of a headache."
        result = nlp_processor.extract_entities(text)
        assert len(result) == 0

    def test_process_empty_string(self, nlp_processor):
        """Testa se uma string vazia retorna uma lista vazia."""
        result = nlp_processor.extract_entities("")
        assert len(result) == 0


# --- Testes para Extração de Relações (Nova Lógica) ---

class TestExtractRelations:

    def test_extract_relation_on_cooccurrence_in_csv(self, nlp_processor):
        """
        TESTE PRINCIPAL DA NOVA LÓGICA:
        Verifica se uma relação é criada quando um sintoma e uma doença
        que estão ligados no CSV aparecem na MESMA frase.
        """
        # "chest pain" e "myocardial infarction" estão relacionados no CSV
        text = "Given the symptoms, chest pain is a strong indicator of myocardial infarction."
        relations = nlp_processor.extract_relations(text)
        
        assert len(relations) == 1
        expected_relation = ("chest pain", "RELATED_TO", "myocardial infarction")
        
        # Compara ignorando maiúsculas/minúsculas
        actual_relation = relations[0]
        assert actual_relation[0].lower() == expected_relation[0].lower()
        assert actual_relation[1] == expected_relation[1]
        assert actual_relation[2].lower() == expected_relation[2].lower()

    def test_no_relation_if_pair_not_in_csv(self, nlp_processor):
        """
        Garante que nenhuma relação seja criada se o par (sintoma, doença)
        não existir no CSV, mesmo que apareçam na mesma frase.
        """
        # "dizziness" e "myocardial infarction" NÃO estão relacionados no CSV
        text = "The patient felt dizziness during the myocardial infarction event."
        relations = nlp_processor.extract_relations(text)
        assert len(relations) == 0

    def test_no_relation_if_entities_in_different_sentences(self, nlp_processor):
        """
        Garante que a relação só é extraída se as entidades estiverem na mesma frase.
        """
        # "chest pain" e "myocardial infarction" estão no CSV, mas em frases diferentes.
        text = "The patient reported severe chest pain. A later diagnosis confirmed myocardial infarction."
        relations = nlp_processor.extract_relations(text)
        assert len(relations) == 0