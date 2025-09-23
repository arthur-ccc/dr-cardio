import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from fairlearn.metrics import MetricFrame
from preprocessing.preprocess import load_data, preprocess
from data.mock import generate_mock_dataset, export
import matplotlib.pyplot as plt

df = load_data("data/demartology.csv")
X_train, X_test, y_train, y_test, le = preprocess(df, target_col="class")


modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

idades_teste = df.loc[X_test.index]['age']


# Definir os limites das faixas etárias
# Os limites são 0, 39, 59, e infinito (para pegar todos acima de 59)
bins = [0,18, 36, float('inf')]

labels = ['Criança(0-18)','Jovem Adulto(19-35)', 'Meia-idade/Idoso(40+)']

# Criar a nova coluna com os grupos etários
df['grupo_etario'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
grupo_etario_teste = pd.cut(idades_teste, bins=bins, labels=labels, right=True)

print("Distribuição das Faixas Etárias no Conjunto de Teste:")


# Contagem de amostras por grupo
'''
contagem_grupos = grupo_etario_teste.value_counts()
print("\n--- Contagem Absoluta ---")
print(contagem_grupos)

'''
# Proporção (porcentagem) de cada grupo
proporcao_grupos = grupo_etario_teste.value_counts(normalize=True)
print("\n--- Proporção (%) ---")
# Multiplicamos por 100 e formatamos para melhor visualização
print((proporcao_grupos * 100).round(2).astype(str) + ' %')
print("-" * 70)


if grupo_etario_teste.isnull().any():
   # print("Atenção: Existem valores nulos na coluna 'age' do conjunto de teste.")
 
    grupo_etario_teste = grupo_etario_teste.fillna(grupo_etario_teste.mode()[0])

print("=" * 70)
print("Análise de Justiça do Modelo por Faixa Etária")
print("=" * 70)


metricas = {
    'accuracy': accuracy_score,
    'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0)
}

# Criamos o MetricFrame com as variáveis corretas
# y_true: os rótulos verdadeiros do conjunto de teste
# y_pred: as predições do modelo para o conjunto de teste
# sensitive_features: os grupos etários correspondentes ao conjunto de teste
grouped_on_age = MetricFrame(metrics=metricas,
                             y_true=y_test,
                             y_pred=y_pred,
                             sensitive_features=grupo_etario_teste)


print("Métricas Gerais (Overall):\n", grouped_on_age.overall)
print("-" * 30)


print("Métricas por Grupo (By Group):\n", grouped_on_age.by_group)
print("-" * 30)

disparidades = grouped_on_age.difference()


print(f"Disparidade de Acurácia (Diferença): {disparidades['accuracy']:.3f}")
print(f"Disparidade de Recall (Diferença): {disparidades['recall_macro']:.3f}")
print("\n")
