import pytest
import spacy
from src.nlp import NLPProcessor

@pytest.fixture(scope="session")
def nlp_processor():
    # Carrega o modelo small em inglês
    nlp_model = spacy.load("en_core_web_sm") 
    return NLPProcessor(nlp_model)

class TestExtractEntities: 

    @pytest.mark.parametrize("text, expected_entities", [
        ("Patient reports chest pain and shortness of breath.", 
         [("chest pain", "SYMPTOM"), ("shortness of breath", "SYMPTOM")]),
        ("The diagnosis was acute myocardial infarction.",
         [("acute myocardial infarction", "DISEASE")]),
        ("The electrocardiogram showed significant changes.",
         [("electrocardiogram", "EXAM")]),
        ("The main suspicion is heart disease, as the patient feels palpitations.",
         [("heart disease", "DISEASE"), ("palpitations", "SYMPTOM")])
    ])
    def test_extract_entities_success(self, nlp_processor, text, expected_entities):
        result_entities = nlp_processor.extract_entities(text)
        result_entities_lower = [(e[0].lower(), e[1]) for e in result_entities]
        for expected in expected_entities:
            assert (expected[0].lower(), expected[1]) in result_entities_lower

    def test_process_text_with_no_entities(self, nlp_processor):
        text = "The patient is 45 years old and is from New York."
        entities = nlp_processor.extract_entities(text)
        assert len(entities) == 0

    def test_process_empty_string(self, nlp_processor):
        text = ""
        entities = nlp_processor.extract_entities(text)
        assert len(entities) == 0

    def test_extract_entities_with_negation(self, nlp_processor):
        text = "The patient denies dyspnea and did not present fever."
        expected_entities = [("dyspnea", "SYMPTOM"), ("fever", "SYMPTOM")]
        result_entities = nlp_processor.extract_entities(text)
        result_entities_lower = [(e[0].lower(), e[1]) for e in result_entities]
        for expected in expected_entities:
            assert (expected[0].lower(), expected[1]) in result_entities_lower

    def test_extract_entities_with_uncertainty(self, nlp_processor):
        text = "The medical team suspects a possible pulmonary embolism."
        expected_entities = [("pulmonary embolism", "DISEASE")]
        result_entities = nlp_processor.extract_entities(text)
        result_entities_lower = [(e[0].lower(), e[1]) for e in result_entities]
        for expected in expected_entities:
            assert (expected[0].lower(), expected[1]) in result_entities_lower

    def test_extract_entities_with_abbreviations(self, nlp_processor):
        text = "The MI was confirmed by the ECG result."
        expected_entities = [("MI", "DISEASE"), ("ECG", "EXAM")]
        result_entities = nlp_processor.extract_entities(text)
        result_entities_lower = [(e[0].lower(), e[1]) for e in result_entities]
        for expected in expected_entities:
            assert (expected[0].lower(), expected[1]) in result_entities_lower

    @pytest.mark.parametrize("text, expected_entity", [
        ("The patient has Hypertension.", ("Hypertension", "DISEASE")),
        ("The patient has hypertension.", ("hypertension", "DISEASE")),
        ("The patient has HYPERTENSION.", ("HYPERTENSION", "DISEASE"))
    ])
    def test_extract_entities_is_case_insensitive(self, nlp_processor, text, expected_entity):
        result_entities = nlp_processor.extract_entities(text)
        result_entities_lower = [(e[0].lower(), e[1]) for e in result_entities]
        assert (expected_entity[0].lower(), expected_entity[1]) in result_entities_lower

    def test_extract_entities_with_adjacent_punctuation(self, nlp_processor):
        text = "Symptoms include: fatigue, palpitations (mild) and sweating."
        expected_entities = [
            ("fatigue", "SYMPTOM"),
            ("palpitations", "SYMPTOM"),
            ("sweating", "SYMPTOM")
        ]
        result_entities = nlp_processor.extract_entities(text)
        result_entities_lower = [(e[0].lower(), e[1]) for e in result_entities]
        for expected in expected_entities:
            assert (expected[0].lower(), expected[1]) in result_entities_lower


class TestExtractRelations:

    def _assert_relation_case_insensitive(self, expected_relation, relations):
        """ Helper para comparar relações ignorando maiúsculas/minúsculas """
        assert any(
            expected_relation[0].lower() == rel[0].lower() and
            expected_relation[1] == rel[1] and
            expected_relation[2].lower() == rel[2].lower()
            for rel in relations
        )

    def test_extract_symptom_disease_relation(self, nlp_processor):
        text = "Dyspnea is a common symptom of heart failure."
        expected_relation = ("dyspnea", "IS_SYMPTOM_OF", "heart failure")
        relations = nlp_processor.extract_relations(text)
        self._assert_relation_case_insensitive(expected_relation, relations)

    def test_no_relation_between_entities(self, nlp_processor):
        text = "The patient presented fever. The doctor suspects cardiomyopathy."
        relations = nlp_processor.extract_relations(text)
        assert len(relations) == 0

    def test_extract_direct_symptom_disease_relation(self, nlp_processor):
        text = "Chest pain is a classic symptom of heart attack."
        expected_relation = ("chest pain", "IS_SYMPTOM_OF", "heart attack")
        relations = nlp_processor.extract_relations(text)
        self._assert_relation_case_insensitive(expected_relation, relations)

    def test_extract_causal_verb_relation(self, nlp_processor):
        text = "Uncontrolled hypertension can cause kidney damage."
        expected_relation = ("uncontrolled hypertension", "CAN_CAUSE", "kidney damage")
        relations = nlp_processor.extract_relations(text)
        self._assert_relation_case_insensitive(expected_relation, relations)

    def test_extract_confirmation_relation_by_exam(self, nlp_processor):
        text = "The electrocardiogram confirmed the suspicion of cardiac arrhythmia."
        expected_relation = ("electrocardiogram", "CONFIRMS", "cardiac arrhythmia")
        relations = nlp_processor.extract_relations(text)
        self._assert_relation_case_insensitive(expected_relation, relations)

    def test_extract_multiple_relations_in_sentence(self, nlp_processor):
        text = "Fever is a symptom of viral infection, which in turn can generate dehydration."
        expected_relations = [
            ("Fever", "IS_SYMPTOM_OF", "viral infection"),
            ("viral infection", "CAN_GENERATE", "dehydration")
        ]
        relations = nlp_processor.extract_relations(text)
        assert len(relations) == 2
        for expected in expected_relations:
            self._assert_relation_case_insensitive(expected, relations)

    def test_no_relation_extracted_when_entities_are_unconnected(self, nlp_processor):
        text = "The patient with fever was examined. The diagnosis was pneumonia."
        relations = nlp_processor.extract_relations(text)
        assert len(relations) == 0

    def test_no_relation_extracted_on_negated_context(self, nlp_processor):
        text = "Despite the headache, the exam did not indicate any sign of stroke."
        relations = nlp_processor.extract_relations(text)
        assert len(relations) == 0
