import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer


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

def preprocess(df, target_col="class", balance=True):
    X = df.drop(target_col, axis=1)
    y = df[target_col] - 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    if balance:
        smote = SMOTE(
                    sampling_strategy={0: 500, 1: 500, 2: 500, 3: 200, 4: 500, 5: 500},
                    random_state=42
                    )

        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, DISEASE_MAP
