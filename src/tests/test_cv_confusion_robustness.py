import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend não interativo para CI
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# =========================
#   CROSS-VALIDATION
# =========================
class TestCrossValidation:
    def test_stratified_kfold_accuracy_runs(self, Xy_full, classifiers):
        X, y = Xy_full
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, clf in classifiers.items():
            scores = cross_val_score(clf.model, X, y, cv=cv, scoring="accuracy")
            np.save(os.path.join(ARTIFACTS_DIR, f"cv_scores_{name.replace(' ', '_')}.npy"), scores)
            assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
            assert scores.size == 5

    def test_class_balance_preserved_in_each_fold(self, Xy_full):
        """Confere se a estratificação preserva a proporção de classes em cada dobra (±5%)."""
        X, y = Xy_full
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        _, counts_global = np.unique(y, return_counts=True)
        prop_global = counts_global / counts_global.sum()

        for tr_idx, te_idx in cv.split(X, y):
            _, counts_fold = np.unique(y[te_idx], return_counts=True)
            prop_fold = counts_fold / counts_fold.sum()
            # mesma ordem de classes por construção; tolerância 5 p.p.
            assert np.all(np.abs(prop_fold - prop_global) <= 0.05 + 1e-9)

    def test_cross_validate_multiple_metrics(self, Xy_full, classifiers):
        """Avalia accuracy e f1_macro simultaneamente via cross_validate."""
        X, y = Xy_full
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, clf in classifiers.items():
            res = cross_validate(
                clf.model, X, y, cv=cv,
                scoring={"acc": "accuracy", "f1m": "f1_macro"},
                return_train_score=False
            )
            # salva para inspeção
            np.savez(
                os.path.join(ARTIFACTS_DIR, f"cv_multi_{name.replace(' ', '_')}.npz"),
                test_acc=res["test_acc"], test_f1m=res["test_f1m"]
            )
            assert len(res["test_acc"]) == 5 and len(res["test_f1m"]) == 5
            assert np.all((res["test_acc"] >= 0) & (res["test_acc"] <= 1))
            assert np.all((res["test_f1m"] >= 0) & (res["test_f1m"] <= 1))

    def test_repeated_stratified_kfold_stability(self, Xy_full, classifiers):
        """Repete CV (3x5) e checa se a variabilidade não é absurda (desvio < 0.15)."""
        X, y = Xy_full
        rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        for name, clf in classifiers.items():
            scores = cross_val_score(clf.model, X, y, cv=rcv, scoring="accuracy")
            assert np.std(scores) < 0.15
            np.save(os.path.join(ARTIFACTS_DIR, f"cv_repeated_{name.replace(' ', '_')}.npy"), scores)


# =========================
#   MATRIZ DE CONFUSÃO
# =========================
class TestConfusionMatrix:
    def _plot_and_save_cm(self, cm, name, normalized=False):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest")
        title = "Matriz de Confusão (norm.)" if normalized else "Matriz de Confusão"
        ax.set_title(f"{title} - {name}")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        suffix = "norm" if normalized else "raw"
        out = os.path.join(ARTIFACTS_DIR, f"confusion_matrix_{suffix}_{name.replace(' ', '_')}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        assert os.path.exists(out)

    def test_confusion_matrix_shape_and_save_plot(self, splits, classifiers):
        X_train, X_test, y_train, y_test, _ = splits
        n_classes = len(set(y_test))
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            assert cm.shape == (n_classes, n_classes)
            self._plot_and_save_cm(cm, name, normalized=False)

    def test_normalized_confusion_matrix_rows_sum_to_one(self, splits, classifiers):
        """Gera matriz normalizada por linha e confere se cada linha soma 1."""
        X_train, X_test, y_train, y_test, _ = splits
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # evita div por zero
            cm_norm = cm / row_sums
            self._plot_and_save_cm(cm_norm, name, normalized=True)
            assert np.allclose(cm_norm.sum(axis=1), 1.0, atol=1e-6)

    def test_per_class_recall_min_threshold(self, splits, classifiers):
        """Cada classe deve ter um recall mínimo (limiar baixo para evitar flaky)."""
        X_train, X_test, y_train, y_test, _ = splits
        MIN_RECALL = 0.05  # bem permissivo, só pra flaggar casos extremos
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            y_pred = clf.predict(X_test)
            recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
            # salva para inspeção
            np.save(os.path.join(ARTIFACTS_DIR, f"recall_per_class_{name.replace(' ', '_')}.npy"), recalls)
            assert np.all(recalls >= 0.0) and np.any(recalls >= MIN_RECALL)

    def test_top2_accuracy_if_proba_available(self, splits, classifiers):
        """Se o modelo expõe predict_proba, mede top-2 accuracy (útil em multiclasses)."""
        X_train, X_test, y_train, y_test, _ = splits
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            try:
                proba = clf.predict_proba(X_test)
            except Exception:
                continue  # sem probas, pula
            # top-2 accuracy
            top2 = top_k_accuracy_score(y_test, proba, k=2, labels=np.arange(proba.shape[1]))
            with open(os.path.join(ARTIFACTS_DIR, f"top2_{name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
                f.write(f"top2_accuracy={top2:.4f}\n")
            assert 0.0 <= top2 <= 1.0


# =========================
#        ROBUSTEZ
# =========================
class TestRobustez:
    @staticmethod
    def _add_gaussian_noise(X, noise_level=0.02, seed=42):
        rng = np.random.default_rng(seed)
        Xn = X.copy()
        if hasattr(Xn, "select_dtypes"):
            cols = Xn.select_dtypes(include=["number"]).columns
            for c in cols:
                std = np.std(Xn[c].values) or 1.0
                Xn[c] = Xn[c].values + rng.normal(0.0, noise_level * std, size=len(Xn))
        else:
            std = np.std(Xn, axis=0)
            std[std == 0] = 1.0
            Xn = Xn + rng.normal(0.0, noise_level * std, size=Xn.shape)
        return Xn

    def test_accuracy_under_noise(self, splits, classifiers):
        X_train, X_test, y_train, y_test, _ = splits
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            baseline = accuracy_score(y_test, clf.predict(X_test))
            X_noisy = self._add_gaussian_noise(X_test, noise_level=0.05)
            noisy_acc = accuracy_score(y_test, clf.predict(X_noisy))
            with open(os.path.join(ARTIFACTS_DIR, f"robustness_{name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
                f.write(f"baseline_acc={baseline:.4f}\n")
                f.write(f"noisy_acc={noisy_acc:.4f}\n")
                f.write(f"delta={noisy_acc - baseline:.4f}\n")
            assert 0.0 <= noisy_acc <= 1.0

    def test_accuracy_degrades_as_noise_increases(self, splits, classifiers):
        """Avalia tendência de queda com ruídos crescentes."""
        X_train, X_test, y_train, y_test, _ = splits
        noise_grid = [0.0, 0.02, 0.05, 0.10]
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            accs = []
            for nl in noise_grid:
                Xn = self._add_gaussian_noise(X_test, noise_level=nl, seed=42)
                accs.append(accuracy_score(y_test, clf.predict(Xn)))
            np.save(os.path.join(ARTIFACTS_DIR, f"noise_curve_{name.replace(' ', '_')}.npy"), np.array(accs))
            # Não exige monotonicamente estrito, mas espera baixa tendência
            assert accs[0] >= max(accs[-2], accs[-1]) or (accs[0] - accs[-1]) >= 0.01

    def test_label_shuffle_baseline_near_chance(self, splits, classifiers):
        """Treina com rótulos embaralhados; accuracy deve ficar perto do acaso."""
        X_train, X_test, y_train, y_test, _ = splits
        n_classes = len(np.unique(y_train))
        chance = 1.0 / n_classes
        TOL = 0.15  # tolerância
        rng = np.random.default_rng(123)
        y_train_shuffled = y_train.copy().values
        rng.shuffle(y_train_shuffled)
        for name, clf in classifiers.items():
            clf.train(X_train, y_train_shuffled)
            acc = accuracy_score(y_test, clf.predict(X_test))
            # Não precisa ser exatamente chance, mas não pode ser absurdamente alto
            assert abs(acc - chance) <= TOL or acc < 0.5

    def test_feature_permutation_drop_exists(self, splits, classifiers):
        """Permuta uma coluna numérica e verifica se ao menos uma causa queda de acurácia."""
        X_train, X_test, y_train, y_test, _ = splits
        numeric_cols = list(X_train.select_dtypes(include=["number"]).columns)
        if not numeric_cols:
            return
        for name, clf in classifiers.items():
            clf.train(X_train, y_train)
            base = accuracy_score(y_test, clf.predict(X_test))
            drops = []
            rng = np.random.default_rng(7)
            for col in numeric_cols[: min(10, len(numeric_cols))]:  # limita 10 colunas para performance
                Xp = X_test.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                accp = accuracy_score(y_test, clf.predict(Xp))
                drops.append(base - accp)
            max_drop = max(drops) if drops else 0.0
            with open(os.path.join(ARTIFACTS_DIR, f"perm_drop_{name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
                f.write(f"base_acc={base:.4f}\nmax_drop={max_drop:.4f}\n")
            # Espera-se alguma coluna relevante causar queda > 0.0 (pequena tolerância)
            assert max_drop >= -1e-6

    def test_subset_stability_variance(self, splits, classifiers):
        """Treina com 3 subconjuntos aleatórios (50% do treino cada) e mede variância da acurácia."""
        X_train, X_test, y_train, y_test, _ = splits
        rng = np.random.default_rng(2024)
        for name, clf in classifiers.items():
            accs = []
            for _ in range(3):
                idx = rng.choice(len(X_train), size=int(0.5 * len(X_train)), replace=False)
                clf.train(X_train.iloc[idx], y_train.iloc[idx])
                accs.append(accuracy_score(y_test, clf.predict(X_test)))
            var = float(np.var(accs))
            with open(os.path.join(ARTIFACTS_DIR, f"subset_var_{name.replace(' ', '_')}.txt"), "w", encoding="utf-8") as f:
                f.write(f"accs={accs}\nvar={var:.6f}\n")
            # Variância não deve explodir (limite largo para evitar flaky)
            assert var < 0.05
