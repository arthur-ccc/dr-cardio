import pandas as pd
from pathlib import Path

class DataIngestor:
    def __init__(self, dataset_path, output_dir):
        """
        dataset_path: caminho do dataset (string)
        output_dir: diretório de saída (Path)
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def download_metadata(self):
        # Retorna caminho fake de metadados
        return self.output_dir / "metadata.csv"

    def get_article_list_from_metadata(self, metadata_path):
        # Retorna DataFrame fake
        import pandas as pd
        return pd.DataFrame({
            "article_id": [],
            "title": [],
            "path": []
        })

    def read_article_content(self, article_path):
        # Retorna texto fake do artigo
        return ""

    def ingest(self):
        # Retorno fake da ingestão completa
        return []
