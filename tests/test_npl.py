import pytest
import spacy
from src.nlp import NLPProcessor

@pytest.fixture(scope="session")
def nlp_processor():
    
    nlp_model = spacy.load("en_core_news_sm")
    return NLPProcessor(nlp_model)

class TestExtractEntities: 

    @pytest.mark.parametrize("text, expected_entities", [
        (
            "Paciente relata dor no peito e falta de ar.", 
            [("dor no peito", "SINTOMA"), ("falta de ar", "SINTOMA")]
        ),
        (
            "O diagnóstico foi de infarto agudo do miocárdio.",
            [("infarto agudo do miocárdio", "DOENÇA")]
        ),
        (
            "O eletrocardiograma apresentou alterações significativas.",
            [("eletrocardiograma", "EXAME")]
        ),
        (
            "A principal suspeita é cardiopatia, pois o paciente sente palpitações.",
            [("cardiopatia", "DOENÇA"), ("palpitações", "SINTOMA")]
        )
    ])
    def test_extract_entities_success(self,nlp_processor, text, expected_entities):
        """
        Caso de Sucesso Parametrizado: Verifica a extração de diferentes tipos de
        entidades em diversas frases.
        """
        result_entities = nlp_processor.extract_entities(text)
        
        for expected in expected_entities:
            assert expected in result_entities

    def test_process_text_with_no_entities(self,nlp_processor):
        """
        Quando o texto que não contém entidades relevantes.
        """
        text = "O paciente tem 45 anos e é natural de Campina Grande."
        entities = nlp_processor.extract_entities(text)
        
        assert len(entities) == 0

    def test_process_empty_string(self,nlp_processor):
        """
        Quando recebemos uma string vazia como entrada.
        """
        text = ""
        entities = nlp_processor.extract_entities(text)
        
        assert len(entities) == 0

    def test_extract_entities_with_negation(self,nlp_processor):
        """
        Verifica se entidades são extraídas mesmo em um contexto de negação.
        """

        text = "O paciente nega dispneia e não apresentou febre."
        expected_entities = [("dispneia", "SINTOMA"), ("febre", "SINTOMA")]
        
        result_entities = nlp_processor.extract_entities(text)
        
        for expected in expected_entities:
            assert expected in result_entities

    def test_extract_entities_with_uncertainty(self,nlp_processor):
        """
        Verifica a extração de entidades em contextos de suspeita ou hipótese.
        """
        text = "A equipe médica suspeita de uma possível embolia pulmonar."
        expected_entities = [("embolia pulmonar", "DOENÇA")]
        
        result_entities = nlp_processor.extract_entities(text)
        
        assert expected_entities[0] in result_entities
    
    def test_extract_entities_with_abbreviations(self,nlp_processor):
        
        text = "O IAM foi confirmado pelo resultado do ECG."
        expected_entities = [("IAM", "DOENÇA"), ("ECG", "EXAME")]
        
        result_entities = nlp_processor.extract_entities(text)
        
        for expected in expected_entities:
            assert expected in result_entities

    @pytest.mark.parametrize("text, expected_entity", [
        ("O paciente tem Hipertensão.", ("Hipertensão", "DOENÇA")),
        ("O paciente tem hipertensão.", ("hipertensão", "DOENÇA")),
        ("O paciente tem HIPERTENSÃO.", ("HIPERTENSÃO", "DOENÇA"))
    ])
    def test_extract_entities_is_case_insensitive(self,nlp_processor, text, expected_entity):
       
        result_entities = nlp_processor.extract_entities(text)
        
        assert len(result_entities) >= 1
        assert expected_entity in result_entities

    def test_extract_entities_with_adjacent_punctuation(self,nlp_processor):
      
        text = "Sintomas incluem: cansaço, palpitações (leves) e sudorese."
        expected_entities = [
            ("cansaço", "SINTOMA"), 
            ("palpitações", "SINTOMA"), 
            ("sudorese", "SINTOMA")
        ]
        
        result_entities = nlp_processor.extract_entities(text)
        
        assert len(result_entities) == 3
        for expected in expected_entities:
            assert expected in result_entities

    
class TestExtractRelations:
        
    def test_extract_symptom_disease_relation(self,nlp_processor):
        """
        Verifica se uma relação direta entre sintoma e doença é identificada.
        """
        text = "A dispneia é um sintoma comum da insuficiência cardíaca."
        
        # Assumindo a estrutura da relação como uma tupla: (entidade_origem, tipo_relação, entidade_destino)
        expected_relation = ("dispneia", "É_SINTOMA_DE", "insuficiência cardíaca")
        
        relations = nlp_processor.extract_relations(text)
        
        assert expected_relation in relations

    def test_no_relation_between_entities(self,nlp_processor):
        """
        Verifica que entidades na mesma frase, mas sem ligação direta,
        não geram uma relação.
        """
        text = "O paciente apresentou febre. O médico suspeita de cardiomiopatia."
        relations = nlp_processor.extract_relations(text)
        
        assert len(relations) == 0

    def test_extract_direct_symptom_disease_relation(self,nlp_processor):
        """
        Verifica a extração de uma relação direta onde um substantivo é
        explicitamente ligado a outro (ex: 'sintoma de').
        """
        text = "A dor torácica é um sintoma clássico de infarto."
        
        # Assumimos que o NER já identifica 'dor torácica' e 'infarto'
        expected_relation = ("dor torácica", "É_SINTOMA_DE", "infarto")
        
        relations = nlp_processor.extract_relations(text)
        
        assert len(relations) > 0
        assert expected_relation in relations

    def test_extract_causal_verb_relation(self,nlp_processor):
        """
        Verifica a extração de uma relação baseada em um verbo causal (ex: 'pode causar').
        """
        text = "A hipertensão arterial não controlada pode causar danos renais."
        expected_relation = ("hipertensão arterial não controlada", "PODE_CAUSAR", "danos renais")

        relations = nlp_processor.extract_relations(text)

        assert expected_relation in relations

    def test_extract_confirmation_relation_by_exam(self,nlp_processor):
        """
        Verifica a extração de uma relação onde um exame confirma um diagnóstico.
        """
        text = "O eletrocardiograma confirmou a suspeita de arritmia cardíaca."
        expected_relation = ("eletrocardiograma", "CONFIRMA", "arritmia cardíaca")

        relations = nlp_processor.extract_relations(text)

        assert expected_relation in relations

    def test_extract_multiple_relations_in_sentence(self,nlp_processor):
        """
        Verifica se o sistema consegue extrair múltiplas relações de uma única frase.
        """
        text = "Febre é sintoma de virose, que por sua vez pode gerar desidratação."
        expected_relations = [
            ("Febre", "É_SINTOMA_DE", "virose"),
            ("virose", "PODE_GERAR", "desidratação")
        ]

        relations = nlp_processor.extract_relations(text)

        assert len(relations) == 2
        for expected in expected_relations:
            assert expected in relations

    def test_no_relation_extracted_when_entities_are_unconnected(self,nlp_processor):
        """
        Garante que o sistema não invente relações onde não existem.
        As entidades estão presentes, mas não há uma ligação sintática direta.
        """
        text = "O paciente com febre foi examinado. O diagnóstico foi pneumonia."
        relations = nlp_processor.extract_relations(text)
        
        assert len(relations) == 0

    def test_no_relation_extracted_on_negated_context(self,nlp_processor):
        """
        Garante que uma relação explicitamente negada não seja extraída.
        """
        text = "Apesar da dor de cabeça, o exame não indicou qualquer sinal de AVC."
        
        relations = nlp_processor.extract_relations(text)

        assert len(relations) == 0

def test_extract_entities_with_context_awareness(nlp_processor):
    """
    Testa se o NER é sensível ao contexto (ex: 'história de diabetes' vs 'diagnóstico de diabetes')
    """
    text = "História familiar de diabetes mellitus. Diagnóstico atual: hipertensão arterial."
    
    entities = nlp_processor.extract_entities(text)
    
    # Deve identificar ambas as doenças, mas talvez com marcas de contexto diferentes
    diabetes_entities = [ent for ent in entities if "diabetes" in ent[0].lower()]
    hypertension_entities = [ent for ent in entities if "hipertensão" in ent[0].lower()]
    
    assert len(diabetes_entities) > 0
    assert len(hypertension_entities) > 0

def test_extract_entities_with_temporal_information(nlp_processor):
    """
    Testa a extração de entidades com informações temporais
    """
    text = "O paciente apresentou dor torácica há 3 dias e febre desde ontem."
    
    entities = nlp_processor.extract_entities(text)
    temporal_info = nlp_processor.extract_temporal_information(text)
    
    assert any("dor torácica" in ent[0] for ent in entities)
    assert any("febre" in ent[0] for ent in entities)
    assert len(temporal_info) > 0  # Deve extrair informações temporais

def test_extract_entities_with_measurements(nlp_processor):
    """
    Testa a extração de entidades com valores de medição
    """
    text = "Pressão arterial: 150/95 mmHg. Frequência cardíaca: 120 bpm."
    
    entities = nlp_processor.extract_entities(text)
    measurements = nlp_processor.extract_measurements(text)
    
    assert any("Pressão arterial" in ent[0] for ent in entities)
    assert any("Frequência cardíaca" in ent[0] for ent in entities)
    assert len(measurements) >= 2  # Deve extrair ambas as medições

def test_extract_entities_with_negation_detection(nlp_processor):
    """
    Testa a detecção de negação no contexto de entidades
    """
    text = "Paciente nega dor torácica, mas relata dispneia."
    
    entities = nlp_processor.extract_entities(text)
    entities_with_negation = nlp_processor.extract_entities_with_negation(text)
    
    # Deve identificar ambas as entidades
    assert any("dor torácica" in ent[0] for ent in entities)
    assert any("dispneia" in ent[0] for ent in entities)
    
    # Deve marcar a dor torácica como negada
    negated_entities = [ent for ent in entities_with_negation if ent[2] is True]  # Assumindo que (entity, type, is_negated)
    assert any("dor torácica" in ent[0] for ent in negated_entities)