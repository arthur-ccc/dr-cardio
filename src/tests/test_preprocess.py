import pandas as pd
import pytest
from preprocessing.preprocess import load_data

# pytest.fixture cria um recurso que pode ser usado pelos testes.
# Aqui, estamos criando um arquivo CSV temporário.
@pytest.fixture
def temp_csv_file(tmp_path):
    # tmp_path é uma funcionalidade do pytest que cria um diretório temporário
    directory = tmp_path / "data"
    directory.mkdir()
    file_path = directory / "test_data.csv"
    
    # Conteúdo do CSV de teste
    data = {
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C'],
        'class': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    return file_path

# Esta é a nossa primeira função de teste. O nome começa com "test_".
def test_load_data_success(temp_csv_file):
    """
    Testa se a função load_data carrega um arquivo CSV corretamente.
    """
    # 1. Executa a função que queremos testar
    df = load_data(temp_csv_file)

    # 2. Verifica (assert) se os resultados são os esperados
    assert isinstance(df, pd.DataFrame), "A função deve retornar um DataFrame."
    assert not df.empty, "O DataFrame não deve estar vazio."
    assert len(df) == 3, "O DataFrame deve conter 3 linhas."
    assert 'feature1' in df.columns, "A coluna 'feature1' deve estar presente."

def test_load_data_file_not_found():
    """
    Testa se a função levanta FileNotFoundError para um caminho inválido.
    """
    with pytest.raises(FileNotFoundError):
        load_data("caminho/para/arquivo_inexistente.csv")