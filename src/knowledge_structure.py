class KnowledgeOrganizer:
    def __init__(self, nlp_processor=None):
        self.nlp_processor = nlp_processor

    # só pra testar por enquanto
    def create_tables(self, all_entities, all_relations):
        import pandas as pd

        # separar sintomas e doenças
        symptom_rows = [e for e in all_entities if e[1] == "SINTOMA"]
        disease_rows = [e for e in all_entities if e[1] == "DOENÇA"]
        relation_rows = all_relations
        exam_rows = []

        symptom_df = pd.DataFrame(symptom_rows, columns=["symptom", "type", "source"])
        disease_df = pd.DataFrame(disease_rows, columns=["disease", "type", "source"])
        relation_df = pd.DataFrame(relation_rows, columns=["source", "relation", "target", "article"])
        exam_df = pd.DataFrame(exam_rows)

        return symptom_df, disease_df, exam_df, relation_df
