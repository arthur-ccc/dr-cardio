import numpy as np
import pandas as pd


def generate_mock_dataset(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    sintomas = ["dor_peito", "falta_ar", "palpitacao", "cansaço", "tontura", "inchaço"]
    doencas = ["Infarto Agudo", "Cardiomiopatia", "Arritmia", "Insuficiencia Cardíaca"]
    
    # Criar valores binários para sintomas
    data = np.random.choice([0, 1], size=(n_samples, len(sintomas)))
    
    # Criar coluna de doença (string) de forma aleatória
    labels = np.random.choice(doencas, size=n_samples)
    
    df = pd.DataFrame(data, columns=sintomas)
    df["doenca_cardiaca"] = labels
    
    return df

def export(df: pd.DataFrame):
    return df.to_csv("data/dataset_mockado_bruto.csv", index=False)
