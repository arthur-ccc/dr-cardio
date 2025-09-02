# -*- coding: utf-8 -*-

# Pandas e NumPy para manipulação de dados
import pandas as pd  # dataframes
import numpy as np   # operações numéricas
# Divisão treino/teste
from sklearn.model_selection import train_test_split  # separa dados em treino e teste
# Tipagem opcional
from typing import Tuple  # indicar tuplas no tipo de retorno

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e enriquece o dataset com passos simples e seguros para XGBoost.
    """
    # Faz uma cópia para não alterar o DF original por referência
    df = df.copy()  # evita efeitos colaterais

    # Remove duplicatas exatas, se existirem
    df = df.drop_duplicates()  # linhas duplicadas são removidas

    # Algumas versões do dataset possuem coluna 'id'; se existir, removemos (não agrega predição)
    if "id" in df.columns:  # checa existência
        df = df.drop(columns=["id"])  # descarta a coluna de identificador

    # Cria uma feature de IMC (Body Mass Index): peso / (altura_em_metros^2)
    # height está em centímetros e weight em kg no dataset
    if "height" in df.columns and "weight" in df.columns:  # garante que as colunas existem
        height_m = df["height"] / 100.0  # converte cm -> m
        df["bmi"] = df["weight"] / (height_m ** 2)  # calcula IMC

    # Trata valores estranhos de pressão arterial (ap_hi e ap_lo), limitando a um intervalo plausível
    if "ap_hi" in df.columns:  # checa coluna sistólica
        df["ap_hi"] = df["ap_hi"].clip(lower=80, upper=240)  # limita entre 80 e 240 mmHg
    if "ap_lo" in df.columns:  # checa coluna diastólica
        df["ap_lo"] = df["ap_lo"].clip(lower=40, upper=150)  # limita entre 40 e 150 mmHg

    # Converte 'age' (em dias) para 'age_years' (em anos), que é mais interpretável
    if "age" in df.columns:  # checa existência
        df["age_years"] = (df["age"] / 365.25).round(1)  # converte dias -> anos com 1 casa
        df = df.drop(columns=["age"])  # remove a coluna original em dias (evita redundância)

    # Remove linhas com NaN remanescentes (dataset geralmente não tem, mas por segurança)
    df = df.dropna()  # descarta linhas com valores ausentes

    # Retorna o DataFrame limpo/pronto para separar X e y
    return df  # dados limpos

def split_xy(df: pd.DataFrame, target_col: str = "cardio") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa features (X) e rótulo (y) a partir do DataFrame já limpo.
    """
    # Checa se a coluna target existe
    assert target_col in df.columns, f"Coluna alvo '{target_col}' não encontrada."  # falha cedo se não tiver alvo
    # X são todas as colunas menos a de destino
    X = df.drop(columns=[target_col])  # variáveis preditoras
    # y é apenas a coluna-alvo
    y = df[target_col]  # rótulo binário (0/1)
    # Retorna X e y
    return X, y  # separação feita

def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide X e y em conjuntos de treino e teste.
    """
    # Usa a função do scikit-learn para separar em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(  # retorna 4 subconjuntos
        X, y, test_size=test_size, random_state=random_state, stratify=y  # estratifica para manter proporções
    )
    # Retorna as quatro partições
    return X_train, X_test, y_train, y_test  # conjuntos prontos
