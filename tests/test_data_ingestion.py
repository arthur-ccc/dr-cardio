import os
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data_ingestion import DataIngestor


@pytest.fixture
def data_ingestor(tmp_path):
    
    return DataIngestor(kaggle_dataset_id="user/dataset", data_path=tmp_path)

class TestDowloadMetadata:
    def test_download_metadata_success(data_ingestor, mocker):
        """
        Quando o método de download é chamado corretamente
        e se o arquivo de metadados é "criado" no caminho esperado.
        """

        mocker.patch("kaggle.api.dataset_download_files", return_value=None)
        
        (data_ingestor.data_path / "metadata.csv").touch()

        metadata_path = data_ingestor.download_metadata("metadata.csv")

        assert os.path.exists(metadata_path)
        assert "metadata.csv" in str(metadata_path)

    def test_download_metadata_api_error(data_ingestor, mocker):
        """
        Quando lidamos com uma exceção da API do Kaggle
        """

        mocker.patch("kaggle.api.dataset_download_files", side_effect=Exception("API Error 404: Not Found"))

        with pytest.raises(RuntimeError) as excinfo:
            data_ingestor.download_metadata("metadata.csv")
        
        assert "Falha ao baixar dados do Kaggle: API Error 404: Not Found" in str(excinfo.value)

class TestGetArticleList:

    def test_get_article_list_from_metadata_success(data_ingestor, tmp_path):
        """
        Quando o CSV é lido e transformado em um DataFrame pandas.
        """
        
        csv_content = "article_id,title\narticle_01,Title One\narticle_02,Title Two"
        metadata_file = tmp_path / "metadata.csv"
        metadata_file.write_text(csv_content)

        df = data_ingestor.get_article_list_from_metadata(metadata_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "article_id" in df.columns
        assert df.iloc[0]["article_id"] == "article_01"

    def test_get_article_list_file_not_found(data_ingestor):
        """
        Quando o arquivo de metadados não existe.
        """
        non_existent_path = "/path/that/does/not/exist/metadata.csv"
        
        with pytest.raises(FileNotFoundError):
            data_ingestor.get_article_list_from_metadata(non_existent_path)

    def test_get_article_list_malformed_csv(data_ingestor, tmp_path):
        """
        Caso de Borda: Verifica o comportamento com um CSV sem a coluna esperada.
        """
        
        csv_content = "id,title\n01,Title One"
        metadata_file = tmp_path / "metadata.csv"
        metadata_file.write_text(csv_content)

        with pytest.raises(ValueError) as excinfo:
            data_ingestor.get_article_list_from_metadata(metadata_file)

        assert "Coluna 'article_id' não encontrada no arquivo de metadados" in str(excinfo.value)

class TestReadArticleContent:
    def test_read_article_content_success(data_ingestor, tmp_path):
        """
        Caso de Sucesso: Verifica se o conteúdo de um arquivo de texto é lido corretamente.
        """
        
        article_content = "Este é o conteúdo do artigo científico."
        article_file = tmp_path / "article_01.txt"
        article_file.write_text(article_content, encoding="utf-8")

        content = data_ingestor.read_article_content(article_file)

        assert isinstance(content, str)
        assert content == article_content

    def test_read_empty_article(data_ingestor, tmp_path):
        """
       Quando temos um arquivo vazio, ele deve retornar uma string vazia.
        """
        article_file = tmp_path / "empty_article.txt"
        article_file.touch() 

        content = data_ingestor.read_article_content(article_file)

        assert content == ""

        