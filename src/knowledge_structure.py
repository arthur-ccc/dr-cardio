import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import json


class KnowledgeOrganizer:
    def __init__(self, nlp_processor=None, output_dir: str = "data/knowledge"):
        """
        Organizador de conhecimento para estruturar entidades e relações extraídas
        
        Args:
            nlp_processor: Instância do processador NLP (opcional)
            output_dir: Diretório para salvar as estruturas de conhecimento
        """
        self.nlp_processor = nlp_processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.symptoms: Set[str] = set()
        self.diseases: Set[str] = set()
        self.exams: Set[str] = set()
        self.relations: List[Dict] = []
        
        self.symptom_frequency: Dict[str, int] = defaultdict(int)
        self.disease_frequency: Dict[str, int] = defaultdict(int)
        self.symptom_disease_associations: Dict[Tuple[str, str], int] = defaultdict(int)
        
        self.min_association_strength = 2  # Mínimo de ocorrências para considerar associação

    def process_articles_from_metadata(self, metadata_path: str) -> Dict[str, pd.DataFrame]:
        """
        Processa artigos a partir de um arquivo de metadados
        - metadata_path: Caminho para o arquivo de metadados
        """
        
        try:
            metadata_df = pd.read_csv(metadata_path)
            articles_processed = 0
            
            for _, row in metadata_df.iterrows():
                article_path = row.get('download_path', '')
                if article_path and Path(article_path).exists():
                    self.process_article(article_path, row)
                    articles_processed += 1
            
            return self.create_knowledge_tables()
            
        except Exception as e:
            return {}

    def process_article(self, article_path: str, metadata: pd.Series) -> None:
        """
        Processa um artigo individual e extrai conhecimento
        """
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content:
                return
            
            if self.nlp_processor: # extraindo com o nlp
                entities = self.nlp_processor.extract_entities_with_negation(content)
                relations = self.nlp_processor.extract_relations(content)
                # Processar entidades e relações
                self._process_entities(entities, metadata)
                self._process_relations(relations, metadata)
            else: # fallback: processamento básico sem NLP
                self._basic_content_processing(content, metadata)
                
        except Exception as e:
            print(e)

    def _process_entities(self, entities: List[Tuple], metadata: pd.Series) -> None:
        """
        Processa entidades extraídas do texto
            - entities: Lista de entidades (texto, tipo, negação)
            - metadata: Metadados do artigo
        """
        article_id = metadata.get('article_id', 'unknown')
        source = metadata.get('source', 'unknown')
        
        for entity_text, entity_type, is_negated in entities:
            if is_negated:
                continue  # Ignorar entidades negadas
            entity_text = entity_text.lower().strip()
            if entity_type == "SYMPTOM":
                self.symptoms.add(entity_text)
                self.symptom_frequency[entity_text] += 1
            elif entity_type == "DISEASE":
                self.diseases.add(entity_text)
                self.disease_frequency[entity_text] += 1
            elif entity_type == "EXAM":
                self.exams.add(entity_text)

    def _process_relations(self, relations: List[Tuple], metadata: pd.Series) -> None:
        """
        Processa relações extraídas do texto
            - relations: Lista de relações (sintoma, relação, doença)
            - metadata: Metadados do artigo
        """
        article_id = metadata.get('article_id', 'unknown')
        source = metadata.get('source', 'unknown')
        for symptom, relation, disease in relations:
            symptom = symptom.lower().strip()
            disease = disease.lower().strip()
            # Registrar relação
            self.relations.append({
                'symptom': symptom,
                'relation': relation,
                'disease': disease,
                'article_id': article_id,
                'source': source
            })
            # Registrar associação para estatísticas
            self.symptom_disease_associations[(symptom, disease)] += 1

    def _basic_content_processing(self, content: str, metadata: pd.Series) -> None:
        """
        Processamento básico de conteúdo quando não há NLP disponível
            - content: Conteúdo do artigo
            - metadata: Metadados do artigo
        """
        article_id = metadata.get('article_id', 'unknown')
        source = metadata.get('source', 'unknown')
        
        # termos para busca
        symptom_keywords = {
            'chest pain', 'chest tightness', 'shortness of breath', 'dyspnea',
            'palpitations', 'tachycardia', 'arrhythmia', 'fatigue', 'weakness',
            'dizziness', 'syncope', 'lightheadedness', 'sweating', 'edema',
            'ankle swelling', 'nausea', 'vomiting', 'angina'
        }
        disease_keywords = {
            'coronary artery disease', 'myocardial infarction', 'heart failure',
            'hypertension', 'arrhythmia', 'atrial fibrillation', 'cardiomyopathy',
            'congenital heart disease', 'peripheral artery disease', 'stroke',
            'deep vein thrombosis', 'pulmonary embolism', 'pericarditis',
            'endocarditis', 'aortic aneurysm', 'aortic dissection',
            'cardiac arrest', 'ischemic heart disease'
        }
        
        # procurar termos no conteúdo
        content_lower = content.lower()
        for symptom in symptom_keywords:
            if symptom in content_lower:
                self.symptoms.add(symptom)
                self.symptom_frequency[symptom] += 1
        for disease in disease_keywords:
            if disease in content_lower:
                self.diseases.add(disease)
                self.disease_frequency[disease] += 1

    def create_knowledge_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Cria tabelas estruturadas de conhecimento
        
        Returns:
            Dicionário com DataFrames de sintomas, doenças, exames e relações
        """
        
        # DataFrame de sintomas
        symptom_data = []
        for symptom in self.symptoms:
            symptom_data.append({
                'symptom': symptom,
                'frequency': self.symptom_frequency[symptom],
                'type': 'symptom'
            })
        
        symptom_df = pd.DataFrame(symptom_data)
        
        # DataFrame de doenças
        disease_data = []
        for disease in self.diseases:
            disease_data.append({
                'disease': disease,
                'frequency': self.disease_frequency[disease],
                'type': 'disease'
            })
        
        disease_df = pd.DataFrame(disease_data)
        
        # DataFrame de exames
        exam_data = [{'exam': exam, 'type': 'exam'} for exam in self.exams]
        exam_df = pd.DataFrame(exam_data)
        
        # DataFrame de relações
        relation_df = pd.DataFrame(self.relations)
        
        # DataFrame de associações sintoma-doença (consolidado)
        association_data = []
        for (symptom, disease), strength in self.symptom_disease_associations.items():
            if strength >= self.min_association_strength:
                association_data.append({
                    'symptom': symptom,
                    'disease': disease,
                    'association_strength': strength,
                    'type': 'symptom_disease_association'
                })
        
        association_df = pd.DataFrame(association_data)
        
        # Criar dataset de treinamento para o modelo
        training_df = self._create_training_dataset(association_df, symptom_df, disease_df)
        
        # Salvar tabelas
        tables = {
            'symptoms': symptom_df,
            'diseases': disease_df,
            'exams': exam_df,
            'relations': relation_df,
            'associations': association_df,
            'training_data': training_df
        }
        
        self._save_knowledge_tables(tables)
        
        return tables

    def _create_training_dataset(self, association_df: pd.DataFrame,
                                 symptom_df: pd.DataFrame, 
                                 disease_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria dataset de treinamento para o modelo de árvore de decisão
        """
        if association_df.empty:
            return pd.DataFrame()
        
        # lista de todos os sintomas e doenças
        all_symptoms = sorted(list(self.symptoms))
        all_diseases = sorted(list(self.diseases))
        
        training_data = []
        
        # pra cada doença, cria exemplos com combinações de sintomas
        for disease in all_diseases:
            # sintomas associados a esta doença
            disease_symptoms = association_df[association_df['disease'] == disease]['symptom'].tolist()
            
            if not disease_symptoms:
                continue
    
            # exemplos com todos os sintomas "positive_example"
            positive_example = {symptom: 1 for symptom in disease_symptoms}
            positive_example['diagnostic'] = disease
            training_data.append(positive_example)
            
            # exemplos removendo uma doença "negative_example"
            for i in range(min(3, len(disease_symptoms))):
                negative_example = positive_example.copy()
                # removendo alguns sintomas aleatoriamente
                symptoms_to_remove = np.random.choice(disease_symptoms, 
                                                    size=max(1, len(disease_symptoms) // 2), 
                                                    replace=False)
                for symptom in symptoms_to_remove:
                    negative_example[symptom] = 0
                training_data.append(negative_example)
        
        training_df = pd.DataFrame(training_data)
        
        # troca NaN por 0 pros sintomas que não estão presentes
        for symptom in all_symptoms:
            if symptom not in training_df.columns:
                training_df[symptom] = 0
            else:
                training_df[symptom] = training_df[symptom].fillna(0).astype(int)
        if 'diagnostic' not in training_df.columns:
            training_df['diagnostic'] = 'unknown'
        return training_df

    def _save_knowledge_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        """
        Salva as tabelas de conhecimento em arquivos CSV
        """
        try:
            for name, df in tables.items():
                if not df.empty:
                    file_path = self.output_dir / f"{name}.csv"
                    df.to_csv(file_path, index=False, encoding='utf-8')
            stats = {
                'total_symptoms': len(self.symptoms),
                'total_diseases': len(self.diseases),
                'total_exams': len(self.exams),
                'total_relations': len(self.relations),
                'symptom_frequency': dict(self.symptom_frequency),
                'disease_frequency': dict(self.disease_frequency),
                'symptom_disease_associations': {
                    f"{symptom}_{disease}": count 
                    for (symptom, disease), count in self.symptom_disease_associations.items()
                }
            }
            stats_path = self.output_dir / "knowledge_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(e)

    def load_knowledge_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Carrega tabelas de conhecimento salvas anteriormente
        """
        tables = {}
        try:
            for file_type in ['symptoms', 'diseases', 'exams', 'relations', 'associations', 'training_data']:
                file_path = self.output_dir / f"{file_type}.csv"
                if file_path.exists():
                    tables[file_type] = pd.read_csv(file_path)
            return tables
        except Exception as e:
            return {}

    def get_training_dataset(self) -> pd.DataFrame:
        """
        Retorna o dataset de treinamento para o modelo
        """
        training_path = self.output_dir / "training_data.csv"
        
        if training_path.exists():
            return pd.read_csv(training_path)
        else:
            tables = self.load_knowledge_tables()
            if 'training_data' in tables and not tables['training_data'].empty:
                return tables['training_data']
            else:
                return pd.DataFrame()

    def update_with_external_dataset(self, dataset_path: str) -> None:
        """
        Atualiza o conhecimento com um dataset externo
        """   
        try:
            external_df = pd.read_csv(dataset_path)
            
            # verificar formato do dataset
            if 'symptom' in external_df.columns and 'disease' in external_df.columns:
                for _, row in external_df.iterrows():
                    symptom = str(row['symptom']).lower().strip()
                    disease = str(row['disease']).lower().strip()
                    
                    if symptom and disease:
                        self.symptoms.add(symptom)
                        self.diseases.add(disease)
                        self.symptom_disease_associations[(symptom, disease)] += 1
            else:
              print("erro ao acessar")
        except Exception as e:
          print(e)
    def export_for_decision_tree(self, output_path: Optional[str] = None) -> str:
        training_df = self.get_training_dataset()
        
        if training_df.empty:
            return ""
        
        if not output_path:
            output_path = self.output_dir / "cardio_training_dataset.csv"
        
        # Garantir formato binário para sintomas
        symptom_cols = [col for col in training_df.columns if col != 'diagnostic']
        for col in symptom_cols:
            training_df[col] = training_df[col].apply(lambda x: 1 if x else 0)
        
        training_df.to_csv(output_path, index=False)
        
        return str(output_path)
