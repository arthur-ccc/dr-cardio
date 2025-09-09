# src/decision_tree.py
# -*- coding: utf-8 -*-
"""
Implementação de um modelo de Árvore de Decisão para diagnóstico cardiovascular,
projetado para passar nos testes providos (TDD).

Principais requisitos cobertos:
- Treinamento com features binárias/contínuas e target categórico.
- Lidar com valores ausentes (NaN) via SimpleImputer.
- Ignorar colunas desconhecidas no momento de prever (sem quebrar).
- Expor explicabilidade básica: caminho de decisão e importâncias de features.
- Persistência do modelo (salvar/carregar) com joblib.
- Suporte a hiperparâmetros do DecisionTreeClassifier (max_depth, min_samples_split, etc.).

Observações:
- A classe abaixo não depende do dataset “real” enviado; ela é genérica o suficiente
  para treinar com os dados sintéticos dos testes.
- Os nomes de funções estão em português quando faz sentido, mas mantive a interface
  pedida pelos testes (em inglês).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import joblib
import os


@dataclass
class CardiovascularDiagnosisModel:
    """
    Wrapper para uma Árvore de Decisão com:
      - imputação de valores ausentes,
      - tolerância a colunas desconhecidas na predição,
      - explicabilidade básica (caminho e importâncias),
      - persistência (save/load).
    """
    # Hiperparâmetros comuns de árvore; aceitar **kwargs extras também.
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    random_state: Optional[int] = 42
    # Campo para armazenar outros kwargs passados nos testes (ex.: min_samples_leaf etc.)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Atributos preenchidos após o treino
    model: Optional[Pipeline] = field(init=False, default=None)
    feature_names_: List[str] = field(init=False, default_factory=list)
    classes_: List[str] = field(init=False, default_factory=list)

    def __init__(self, **kwargs: Any) -> None:
        """
        Construtor aceitando hiperparâmetros variáveis. Os testes passam
        dicionários com chaves como 'max_depth' e 'min_samples_split'.
        """
        # Extrai hiperparâmetros conhecidos
        self.max_depth = kwargs.pop("max_depth", None)
        self.min_samples_split = kwargs.pop("min_samples_split", 2)
        self.random_state = kwargs.pop("random_state", 42)
        # Armazena o restante para repassar ao estimador da árvore
        self.extra_params = kwargs

        # Inicializa atributos que serão definidos após treino
        self.model = None
        self.feature_names_ = []
        self.classes_ = []

    # =========================
    # Métodos utilitários
    # =========================
    def _ensure_is_fitted(self) -> None:
        """Garante que o modelo foi treinado antes de usar."""
        if self.model is None:
            raise NotFittedError(
                "O modelo ainda não foi treinado. Chame .train(X, y) antes."
            )

    def _align_columns_for_prediction(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Alinha o DataFrame X às colunas vistas no treino:
         - Mantém apenas colunas conhecidas (ignora desconhecidas).
         - Adiciona colunas faltantes com 0 (assumindo ‘ausência do sintoma’).
        Isso atende ao teste 'handle_unknown_symptoms'.
        """
        if not self.feature_names_:
            # Se isso acontecer, significa que predict foi chamado antes do train
            self._ensure_is_fitted()

        # Interseção: usa apenas colunas conhecidas
        known = [c for c in X.columns if c in self.feature_names_]
        X_known = X[known].copy()

        # Adiciona colunas que faltam (em ordem consistente)
        for col in self.feature_names_:
            if col not in X_known.columns:
                # Preenche com 0 como default (neutro para binárias e ok para imputação)
                X_known[col] = 0

        # Reordena as colunas exatamente como no treino
        X_known = X_known[self.feature_names_]

        # Garante tipo numérico quando possível; deixa objeto se não conversível
        # (SimpleImputer e DecisionTree lidam bem com numérico; converter ajuda).
        for col in X_known.columns:
            if X_known[col].dtype == object:
                # Tenta converter para float/int; valores não convertíveis viram NaN
                X_known[col] = pd.to_numeric(X_known[col], errors="coerce")

        return X_known

    # =========================
    # API principal requisitada
    # =========================
    def train(self, X: pd.DataFrame, y: Union[pd.Series, Iterable[Any]]) -> None:
        """
        Treina o pipeline (Imputer + DecisionTreeClassifier).

        - Lida com valores ausentes (SimpleImputer(strategy="most_frequent")):
          isso permite treinar mesmo com NaN, como exigido no teste.
        - Armazena a ordem e os nomes das features para alinhamento futuro.

        Parâmetros:
        X: DataFrame com as features.
        y: Série/iterável com as classes (strings).
        """
        # Garante DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Converte colunas do tipo objeto para numérico quando possível
        X_num = X.copy()
        for col in X_num.columns:
            if X_num[col].dtype == object:
                X_num[col] = pd.to_numeric(X_num[col], errors="coerce")

        # Salva nomes e ordem das colunas de treino
        self.feature_names_ = list(X_num.columns)

        # Constrói o classificador de árvore com hiperparâmetros fornecidos
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            **self.extra_params,  # permite hiperparâmetros adicionais
        )

        # Pipeline: imputação + árvore
        # - strategy="most_frequent": bom para binárias; funciona para contínuas também.
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("clf", tree),
            ]
        )

        # Ajusta o modelo
        self.model.fit(X_num, y)

        # Armazena classes para os testes
        # Convertendo para lista de strings para consistência
        self.classes_ = [str(c) for c in self.model.named_steps["clf"].classes_]

    def predict(self, X: pd.DataFrame) -> str:
        """
        Prediz a classe para um único exemplo (como usado nos testes).
        - Alinha colunas (ignora desconhecidas, adiciona faltantes).
        - Retorna string (não array), satisfez os asserts dos testes.
        """
        self._ensure_is_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Alinha colunas e tipa
        X_aligned = self._align_columns_for_prediction(X)

        # Realiza a predição
        pred = self.model.predict(X_aligned)

        # Os testes fornecem um único exemplo; retornamos apenas a string
        # Garante tipo str (não numpy.str_)
        return str(pred[0])

    def explain_prediction(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Retorna uma explicação simples contendo:
          - 'decision_path': lista de nós visitados com (feature, threshold, operacao, valor),
          - 'feature_importance': dict {feature: importância global na árvore}.

        Observação: é uma explicação baseada na estrutura da árvore do scikit-learn,
        adequada para fins didáticos do TDD.
        """
        self._ensure_is_fitted()

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Alinha features e calcula valores imputados
        X_aligned = self._align_columns_for_prediction(X)

        # Precisamos passar pelo imputer para obter os valores usados na árvore
        imputer: SimpleImputer = self.model.named_steps["imputer"]
        clf: DecisionTreeClassifier = self.model.named_steps["clf"]

        X_imputed = pd.DataFrame(
            imputer.transform(X_aligned),
            columns=self.feature_names_,
        )

        # Usaremos apenas a primeira linha (como nos testes)
        instance = X_imputed.iloc[0].values.reshape(1, -1)

        # decision_path retorna a matriz esparsa dos nós percorridos
        node_indicator = clf.decision_path(instance)
        # O índice do nó folha para a amostra
        leaf_id = clf.apply(instance)[0]

        # Acesso aos arrays internos da árvore
        tree_ = clf.tree_
        feature_index = tree_.feature
        threshold = tree_.threshold

        # Construção do caminho legível
        decision_steps: List[Dict[str, Any]] = []

        # Nós percorridos (exceto o nó folha final para evitar threshold = -2)
        node_index = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        for node_id in node_index:
            # Nó folha tem feature = _tree.TREE_UNDEFINED (-2), ignoramos.
            if feature_index[node_id] != -2:  # -2 indica que é folha
                feat_name = self.feature_names_[feature_index[node_id]]
                thresh = float(threshold[node_id])
                # Valor da feature naquela amostra
                feat_value = float(instance[0, feature_index[node_id]])
                # Determina se foi para esquerda (<=) ou direita (>)
                go_left = feat_value <= thresh
                op = "<=" if go_left else ">"
                decision_steps.append(
                    {
                        "node_id": int(node_id),
                        "feature": str(feat_name),
                        "threshold": thresh,
                        "value": feat_value,
                        "decision": f"{feat_name} {op} {thresh}",
                        "direction": "left" if go_left else "right",
                    }
                )

        explanation = {
            "decision_path": decision_steps,
            "leaf_id": int(leaf_id),
            "predicted_class": str(clf.predict(instance)[0]),
            "feature_importance": self.get_feature_importance(),
        }
        return explanation

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Retorna a importância global das features segundo a árvore treinada.
        - Formato: {nome_da_feature: importancia_float}
        - A soma das importâncias é ~1.0 (propriedade do scikit-learn).
        """
        self._ensure_is_fitted()
        clf: DecisionTreeClassifier = self.model.named_steps["clf"]
        importances = clf.feature_importances_
        return {
            feature: float(importance)
            for feature, importance in zip(self.feature_names_, importances)
        }

    def save_model(self, path: Union[str, os.PathLike]) -> None:
        """
        Salva o pipeline e os metadados necessários para predição futura.
        - Usa joblib para serializar com segurança objetos do scikit-learn.
        """
        self._ensure_is_fitted()
        bundle = {
            "pipeline": self.model,
            "feature_names_": self.feature_names_,
            "classes_": self.classes_,
            "init_params": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "random_state": self.random_state,
                **self.extra_params,
            },
        }
        joblib.dump(bundle, str(path))

    def load_model(self, path: Union[str, os.PathLike]) -> None:
        """
        Carrega o pipeline e metadados previamente salvos por save_model().
        - Após carregar, o objeto volta a estar pronto para predict/explicar.
        """
        bundle = joblib.load(str(path))
        self.model = bundle["pipeline"]
        self.feature_names_ = bundle["feature_names_"]
        # Garante que classes_ esteja disponível para os testes
        self.classes_ = [str(c) for c in bundle.get("classes_", [])]

        # Restaura hiperparâmetros (útil para consistência, não obrigatório p/ prever)
        init_params = bundle.get("init_params", {})
        self.max_depth = init_params.get("max_depth", None)
        self.min_samples_split = init_params.get("min_samples_split", 2)
        self.random_state = init_params.get("random_state", 42)
        # Tudo que sobrou vira extra_params
        known = {"max_depth", "min_samples_split", "random_state"}
        self.extra_params = {k: v for k, v in init_params.items() if k not in known}
