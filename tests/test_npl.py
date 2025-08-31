import pytest
import spacy
from src.nlp import NLPProcessor

@pytest.fixture(scope="session")
def nlp_processor():
    
    nlp_model = spacy.load("pt_core_news_lg") 
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
    def test_extract_entities_success(nlp_processor, text, expected_entities):
        """
        Caso de Sucesso Parametrizado: Verifica a extração de diferentes tipos de
        entidades em diversas frases.
        """
        result_entities = nlp_processor.extract_entities(text)
        
        for expected in expected_entities:
            assert expected in result_entities

    def test_process_text_with_no_entities(nlp_processor):
        """
        Quando o texto que não contém entidades relevantes.
        """
        text = "O paciente tem 45 anos e é natural de Campina Grande."
        entities = nlp_processor.extract_entities(text)
        
        assert len(entities) == 0

    def test_process_empty_string(nlp_processor):
        """
        Quando recebemos uma string vazia como entrada.
        """
        text = ""
        entities = nlp_processor.extract_entities(text)
        
        assert len(entities) == 0

    def test_extract_entities_with_negation(nlp_processor):
        """
        Verifica se entidades são extraídas mesmo em um contexto de negação.
        """

        text = "O paciente nega dispneia e não apresentou febre."
        expected_entities = [("dispneia", "SINTOMA"), ("febre", "SINTOMA")]
        
        result_entities = nlp_processor.extract_entities(text)
        
        for expected in expected_entities:
            assert expected in result_entities

    def test_extract_entities_with_uncertainty(nlp_processor):
        """
        Verifica a extração de entidades em contextos de suspeita ou hipótese.
        """
        text = "A equipe médica suspeita de uma possível embolia pulmonar."
        expected_entities = [("embolia pulmonar", "DOENÇA")]
        
        result_entities = nlp_processor.extract_entities(text)
        
        assert expected_entities[0] in result_entities
    
    def test_extract_entities_with_abbreviations(nlp_processor):
        
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
    def test_extract_entities_is_case_insensitive(nlp_processor, text, expected_entity):
       
        result_entities = nlp_processor.extract_entities(text)
        
        assert len(result_entities) >= 1
        assert expected_entity in result_entities

    def test_extract_entities_with_adjacent_punctuation(nlp_processor):
      
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
        
    def test_extract_symptom_disease_relation(nlp_processor):
        """
        Verifica se uma relação direta entre sintoma e doença é identificada.
        """
        text = "A dispneia é um sintoma comum da insuficiência cardíaca."
        
        # Assumindo a estrutura da relação como uma tupla: (entidade_origem, tipo_relação, entidade_destino)
        expected_relation = ("dispneia", "É_SINTOMA_DE", "insuficiência cardíaca")
        
        relations = nlp_processor.extract_relations(text)
        
        assert expected_relation in relations

    def test_no_relation_between_entities(nlp_processor):
        """
        Verifica que entidades na mesma frase, mas sem ligação direta,
        não geram uma relação.
        """
        text = "O paciente apresentou febre. O médico suspeita de cardiomiopatia."
        relations = nlp_processor.extract_relations(text)
        
        assert len(relations) == 0

    def test_extract_direct_symptom_disease_relation(nlp_processor):
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

    def test_extract_causal_verb_relation(nlp_processor):
        """
        Verifica a extração de uma relação baseada em um verbo causal (ex: 'pode causar').
        """
        text = "A hipertensão arterial não controlada pode causar danos renais."
        expected_relation = ("hipertensão arterial não controlada", "PODE_CAUSAR", "danos renais")

        relations = nlp_processor.extract_relations(text)

        assert expected_relation in relations

    def test_extract_confirmation_relation_by_exam(nlp_processor):
        """
        Verifica a extração de uma relação onde um exame confirma um diagnóstico.
        """
        text = "O eletrocardiograma confirmou a suspeita de arritmia cardíaca."
        expected_relation = ("eletrocardiograma", "CONFIRMA", "arritmia cardíaca")

        relations = nlp_processor.extract_relations(text)

        assert expected_relation in relations

    def test_extract_multiple_relations_in_sentence(nlp_processor):
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

    def test_no_relation_extracted_when_entities_are_unconnected(nlp_processor):
        """
        Garante que o sistema não invente relações onde não existem.
        As entidades estão presentes, mas não há uma ligação sintática direta.
        """
        text = "O paciente com febre foi examinado. O diagnóstico foi pneumonia."
        relations = nlp_processor.extract_relations(text)
        
        assert len(relations) == 0

    def test_no_relation_extracted_on_negated_context(nlp_processor):
        """
        Garante que uma relação explicitamente negada não seja extraída.
        """
        text = "Apesar da dor de cabeça, o exame não indicou qualquer sinal de AVC."
        
        relations = nlp_processor.extract_relations(text)

        assert len(relations) == 0