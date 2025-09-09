import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import json

from src.knowledge_structure import KnowledgeOrganizer

class TestKnowledgeOrganizer:
    # configurações
    @pytest.fixture
    def setup_organizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            organizer = KnowledgeOrganizer(output_dir=temp_dir)
            yield organizer
    
    @pytest.fixture
    def cardio_symptoms_dataset(self):
        return pd.read_csv("src/CardioSymptomsDataset.csv") # pegando os dados do dataset
    
    @pytest.fixture
    def sample_medbook_content(self):
        with open("src/medBook.txt", "r", encoding="utf-8") as f: # lendo o medbook
            return f.read()
    

    def test_initialization_with_real_data(self, cardio_symptoms_dataset):
        """o que tem que testar:
        - verificar se o dataset não está vazio
        - confirmar que existe coluna 'diagnostic'
        - verificar que há 20 sintomas no dataset, 20 porque é a quantidade que temos no real até agr"""

        assert not cardio_symptoms_dataset.empty
        assert 'diagnostic' in cardio_symptoms_dataset.columns
        
        symptom_cols = [col for col in cardio_symptoms_dataset.columns if col != 'diagnostic']
        assert len(symptom_cols) == 20
    
    def test_update_with_external_dataset_correct_format(self, setup_organizer):
        
        # cria dataset no formato esperado no método a ser implementado
        external_data = pd.DataFrame({
            'symptom': ['chest_pain', 'shortness_of_breath', 'fatigue', 'palpitations'],
            'disease': ['coronary artery disease', 'heart failure', 'hypertension', 'arrhythmia']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f: # salva o temporário por enquanto
            external_data.to_csv(f.name, index=False)
            dataset_path = f.name
        
        try:
            setup_organizer.update_with_external_dataset(dataset_path)
            
            # ve se os sintomas e doenças foram adicionados
            assert len(setup_organizer.symptoms) > 0
            assert len(setup_organizer.diseases) > 0
            print("formato correto")
        except Exception as e:
            print(e)
        finally:
            Path(dataset_path).unlink()

    def test_create_training_from_real_dataset(self, setup_organizer, cardio_symptoms_dataset):
        """Testa a criação de dataset de treinamento a partir do dataset real"""
        
        symptom_cols = [col for col in cardio_symptoms_dataset.columns if col != 'diagnostic']
        
        for symptom in symptom_cols:
            setup_organizer.symptoms.add(symptom)
            setup_organizer.symptom_frequency[symptom] = 1
        
        for disease in cardio_symptoms_dataset['diagnostic'].unique():
            setup_organizer.diseases.add(disease.lower())
            setup_organizer.disease_frequency[disease.lower()] = 1
        
        for _, row in cardio_symptoms_dataset.iterrows():
            disease = row['diagnostic'].lower()
            for symptom in symptom_cols:
                if row[symptom] == 1:
                    key = (symptom, disease)
                    setup_organizer.symptom_disease_associations[key] = 1
        
        # dataset de treinamento
        training_df = setup_organizer._create_training_dataset(
            pd.DataFrame({
                'symptom': list(setup_organizer.symptoms),
                'disease': list(setup_organizer.diseases),
                'association_strength': [1] * len(setup_organizer.symptoms) # é sempre 0 ou 1. * o len por conta do número de ocorrências
            }),
            pd.DataFrame({
                'symptom': list(setup_organizer.symptoms),
                'frequency': [1] * len(setup_organizer.symptoms)
            }),
            pd.DataFrame({
                'disease': list(setup_organizer.diseases),
                'frequency': [1] * len(setup_organizer.diseases)
            })
        )

        # Se chest_pain = 1 AND shortness_of_breath = 1 -> provavelmente coronary artery disease
        # Se fatigue = 1 AND palpitations = 1 -> provavelmente hypertension
        
        # passou
        assert not training_df.empty
        assert 'diagnostic' in training_df.columns
        
        # ver se todas as colunas de sintomas estão presentes
        for symptom in symptom_cols:
            assert symptom in training_df.columns
        
    @patch('src.nlp.NLPProcessor')
    def test_process_medbook_content(self, mock_nlp_class, setup_organizer, sample_medbook_content):
        """Testa o processamento do conteúdo real do medBook.txt"""
        
        mock_nlp = MagicMock()
        
        def mock_extract_entities(text):
            entities = []
            
            # manualmente por enquanto
            symptoms = [
                'chest pain', 'shortness of breath', 'fatigue', 'palpitations',
                'dizziness', 'sweating', 'chest tightness', 'dyspnea',
                'weakness', 'lightheadedness', 'syncope', 'edema',
                'ankle swelling', 'nausea', 'vomiting', 'headaches',
                'tachycardia', 'orthopnea', 'paroxysmal nocturnal dyspnea'
            ]
            diseases = [
                'coronary artery disease', 'myocardial infarction', 'angina pectoris',
                'heart failure', 'hypertension', 'arrhythmia', 'atrial fibrillation',
                'cardiomyopathy', 'congenital heart disease', 'peripheral artery disease',
                'stroke', 'deep vein thrombosis', 'pulmonary embolism', 'pericarditis',
                'endocarditis', 'aortic aneurysm', 'aortic dissection', 'cardiac arrest',
                'sudden cardiac death', 'ischemic heart disease'
            ]
            
            # criando as entidades manualmente
            for symptom in symptoms:
                if symptom.lower() in text.lower():
                    entities.append((symptom, 'SYMPTOM', False))            
            for disease in diseases:
                if disease.lower() in text.lower():
                    entities.append((disease, 'DISEASE', False))
            
            return entities
        
        def mock_extract_relations(text):
            relations = []
            
            # mapeando sintomas para doenças
            symptom_disease_map = {
                'chest pain': ['coronary artery disease', 'myocardial infarction', 'angina pectoris'],
                'shortness of breath': ['heart failure', 'pulmonary embolism', 'cardiomyopathy'],
                'fatigue': ['coronary artery disease', 'heart failure', 'hypertension'],
                'palpitations': ['arrhythmia', 'atrial fibrillation', 'hypertension']
            }
            
            # criando as relações manualmente
            for symptom, possible_diseases in symptom_disease_map.items():
                if symptom in text.lower():
                    for disease in possible_diseases:
                        if disease in text.lower():
                            relations.append((symptom, 'associated_with', disease))
            
            return relations
        
        mock_nlp.extract_entities_with_negation = mock_extract_entities
        mock_nlp.extract_relations = mock_extract_relations
        mock_nlp_class.return_value = mock_nlp
        
        setup_organizer.nlp_processor = mock_nlp
        
        # conteúdo do medbook
        metadata = pd.Series({
            'article_id': 'medbook_001',
            'source': 'Braunwald Textbook',
            'title': 'Cardiovascular Medicine Textbook'
        })
        
        # processando o conteudo (simuladooooo)
        setup_organizer._process_entities(mock_extract_entities(sample_medbook_content), metadata)
        setup_organizer._process_relations(mock_extract_relations(sample_medbook_content), metadata)
        
        # agora vai
        assert len(setup_organizer.symptoms) > 0
        assert len(setup_organizer.diseases) > 0
        assert len(setup_organizer.relations) > 0
        
        expected_diseases = ['coronary artery disease', 'myocardial infarction', 'heart failure']
        for disease in expected_diseases:
            assert disease in setup_organizer.diseases
    
    def test_export_format_compatibility(self, setup_organizer, cardio_symptoms_dataset):
        """Testa se o formato extraído é compatível com o formato esperado (real)"""
        
        # dataset real como base
        training_df = cardio_symptoms_dataset.copy()
        
        # formato binário para sintomas
        symptom_cols = [col for col in training_df.columns if col != 'diagnostic']
        for col in symptom_cols:
            training_df[col] = training_df[col].astype(int)
        
        with patch.object(setup_organizer, 'get_training_dataset', return_value=training_df):
            export_path = setup_organizer.export_for_decision_tree()
            
            assert export_path != ""
            assert Path(export_path).exists()
            
            exported_df = pd.read_csv(export_path)
            
            # compatibilidade de formato
            assert set(exported_df.columns) == set(cardio_symptoms_dataset.columns)
            assert 'diagnostic' in exported_df.columns
            
            for col in symptom_cols:
                assert exported_df[col].dtype in [np.int64, int]
            
    def test_knowledge_tables_creation_with_real_data(self, setup_organizer):
        """Testa a criação de tabelas de conhecimento com dados reais"""
        
        # dados de exemplo baseados no dataset real
        setup_organizer.symptoms = {
            'chest_pain', 'shortness_of_breath', 'fatigue', 'palpitations',
            'dizziness', 'sweating', 'dyspnea', 'edema'
        }
        setup_organizer.diseases = {
            'coronary artery disease', 'myocardial infarction', 'heart failure',
            'hypertension', 'arrhythmia'
        }
        
        # Adicionar frequências
        for symptom in setup_organizer.symptoms:
            setup_organizer.symptom_frequency[symptom] = 3
        for disease in setup_organizer.diseases:
            setup_organizer.disease_frequency[disease] = 2
        
        associations = {
            ('chest_pain', 'coronary artery disease'): 5,
            ('shortness_of_breath', 'heart failure'): 4,
            ('fatigue', 'coronary artery disease'): 3,
            ('palpitations', 'arrhythmia'): 6,
            ('dizziness', 'hypertension'): 2
        }
        
        setup_organizer.symptom_disease_associations.update(associations)
        
        setup_organizer.relations = [
            {
                'symptom': 'chest_pain',
                'relation': 'associated_with',
                'disease': 'coronary artery disease',
                'article_id': 'art1',
                'source': 'PubMed'
            },
            {
                'symptom': 'shortness_of_breath',
                'relation': 'symptom_of',
                'disease': 'heart failure',
                'article_id': 'art2',
                'source': 'Textbook'
            }
        ]
        
        tables = setup_organizer.create_knowledge_tables() # retorna um dict
        
        col = ['symptoms', 'diseases', 'exams', 'relations', 'associations', 'training_data']

        for table_name in col:
            assert table_name in tables
            if table_name != 'exams':  # exams pode estar vazio
                assert not tables[table_name].empty
        
        # Verificar estatísticas
        stats_path = Path(setup_organizer.output_dir) / "knowledge_stats.json"
        assert stats_path.exists()
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            assert stats['total_symptoms'] > 0
            assert stats['total_diseases'] > 0
        
        print("tabelas de conhecimento criadas com sucesso:")
        for name, df in tables.items():
            print(f"  {name}: {len(df)} registros")
    
    def test_integration_with_decision_tree_format(self, setup_organizer, cardio_symptoms_dataset):
        """Testa a integração com o formato esperado pela árvore de decisão"""
        
        # Simular que temos o dataset de treinamento
        with patch.object(setup_organizer, 'get_training_dataset', return_value=cardio_symptoms_dataset):
            export_path = setup_organizer.export_for_decision_tree()
            
            # ver se arquivo foi criado
            assert export_path != ""
            assert Path(export_path).exists()
            
            # carregar e verificar o conteúdo
            exported_df = pd.read_csv(export_path)
            
            # verificar se mantém o mesmo formato
            assert set(exported_df.columns) == set(cardio_symptoms_dataset.columns)
            assert len(exported_df) == len(cardio_symptoms_dataset)
            
            # verificar se os diagnósticos são os mesmos
            original_diagnostics = cardio_symptoms_dataset['diagnostic'].unique()
            exported_diagnostics = exported_df['diagnostic'].unique()
            assert set(original_diagnostics) == set(exported_diagnostics)
            
            print("integração com arvore passa")

class TestKnowledgeOrganizerIntegration:
    """testa a integração seguindo o nosso fluxograma"""
    
    def test_full_pipeline_with_real_data(self):
        """Testa o pipeline completo desde artigos até dataset estruturado"""
        with tempfile.TemporaryDirectory() as temp_dir:
            organizer = KnowledgeOrganizer(output_dir=temp_dir)
            organizer.symptom_disease_associations = {
                ('chest_pain', 'coronary artery disease'): 3,
                ('shortness_of_breath', 'heart failure'): 2,
                ('fatigue', 'coronary artery disease'): 1
            }
            
            cardio_df = pd.read_csv("src/CardioSymptomsDataset.csv")
            
            mock_nlp = MagicMock()
            
            def mock_extract_entities(text):
                entities = []
                symptoms = [
                    'chest_pain', 'chest_tightness', 'angina', 'shortness_of_breath',
                    'dyspnea', 'orthopnea', 'paroxysmal_nocturnal_dyspnea', 'palpitations',
                    'tachycardia', 'arrhythmia', 'fatigue', 'weakness', 'dizziness',
                    'syncope', 'lightheadedness', 'sweating', 'edema', 'ankle_swelling',
                    'nausea', 'vomiting'
                ]
                
                diseases = cardio_df['diagnostic'].unique()
                
                # limpando e padronizando
                for symptom in symptoms:
                    if symptom.replace('_', ' ') in text.lower():
                        entities.append((symptom, 'SYMPTOM', False))
                
                for disease in diseases:
                    if disease.lower() in text.lower():
                        entities.append((disease, 'DISEASE', False))
                
                return entities
            
            mock_nlp.extract_entities_with_negation = mock_extract_entities
            mock_nlp.extract_relations.return_value = []  # simplificar para teste
            
            organizer.nlp_processor = mock_nlp
            
            # metadados de artigos simulados
            metadata_df = pd.DataFrame({
                'article_id': ['medbook_001'],
                'title': ['Cardiovascular Medicine Textbook'],
                'download_path': [str(Path(temp_dir) / 'medbook.txt')],
                'source': ['Braunwald Textbook']
            })
            
            metadata_path = Path(temp_dir) / 'metadata.csv'
            metadata_df.to_csv(metadata_path, index=False)
            
            # arquivo de artigo simulado
            article_content = """
            Patients with chest pain and fatigue often have coronary artery disease.
            Shortness of breath and palpitations may indicate heart failure.
            Myocardial infarction typically presents with chest pain and sweating.
            """
            
            (Path(temp_dir) / 'medbook.txt').write_text(article_content)
            
            # processando o conteudo do artigo simulado
            with patch('pandas.read_csv', return_value=metadata_df):
                knowledge_tables = organizer.process_articles_from_metadata(str(metadata_path))
            
            assert 'training_data' in knowledge_tables
            assert not knowledge_tables['training_data'].empty
            
            # manda pra árvore de decisão
            export_path = organizer.export_for_decision_tree()
            assert export_path != ""
            
            # verifica compatibilidade
            exported_df = pd.read_csv(export_path)
            assert 'diagnostic' in exported_df.columns
            
            print("Pipeline completo testado com sucesso!")
