import pandas as pd
from sklearn.model_selection import train_test_split

# Mapeamento das classes para nomes de doenças
DISEASE_MAP = {
    1: "Psoríase",
    2: "Dermatite seborreica",
    3: "Líquen plano",
    4: "Pitiríase rósea",
    5: "Dermatite crônica",
    6: "Pitiríase rubra pilar"
}

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def preprocess(df: pd.DataFrame, target_col="class"):
    # Separar features e alvo
    X = df.drop(target_col, axis=1)
    y = df[target_col] - 1

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, DISEASE_MAP
