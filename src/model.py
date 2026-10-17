"""Entraînement, scoring et explicabilité du modèle de risque de crédit.

Le modèle est une **régression logistique** sur variables standardisées
(numériques) et encodées en indicatrices (catégorielles). Ce choix est celui
retenu par la plupart des établissements pour un score de production : il est
stable, calibré et entièrement auditable.

Score de crédit (échelle type FICO, 300–850)
--------------------------------------------
Le score est dérivé des log-odds par la transformation classique
*points-to-double-the-odds* ::

    score = 600 + 20 * log2(odds_remboursement)

Concrètement : 600 points correspondent à une probabilité de défaut de 50 %,
et chaque doublement des chances de remboursement ajoute 20 points. La
décomposition est additive — la somme des contributions de chaque variable
plus le score de base redonne exactement le score final.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET

SCORE_ANCHOR = 600.0
POINTS_PER_DOUBLING = 20.0
SCORE_MIN = 300
SCORE_MAX = 850

# Points de score par unité de log-odds.
_K = POINTS_PER_DOUBLING / math.log(2.0)

MODEL_FILENAME = "credit_model.joblib"


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """Construit le pipeline scikit-learn (préprocessing + classifieur)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0,
        random_state=0,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def train_model(
    dataset: pd.DataFrame,
    *,
    source: str = "synthetic",
    test_size: float = 0.25,
    seed: int = 42,
) -> dict:
    """Entraîne le modèle et renvoie le bundle ``{pipeline, metadata}``."""
    if dataset.empty:
        raise ValueError("Le jeu de données d'entraînement est vide.")

    X = dataset[MODEL_FEATURES].copy()
    y = dataset[TARGET].astype(int)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    metadata = {
        "source": source,
        "n_rows": int(len(dataset)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(MODEL_FEATURES),
        "default_rate": float(y.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "metrics": {
            "auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else float("nan"),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "brier": float(brier_score_loss(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        },
        "coefficients": _coefficient_table(pipeline),
    }

    return {"pipeline": pipeline, "metadata": metadata}


def _coefficient_table(pipeline: Pipeline) -> list:
    """Table lisible des coefficients (odds ratios) pour l'audit."""
    preprocessor = pipeline.named_steps["pre"]
    classifier = pipeline.named_steps["clf"]
    names = list(preprocessor.get_feature_names_out())
    coefficients = classifier.coef_.ravel()

    rows = []
    for name, coefficient in zip(names, coefficients):
        column, category = split_output_name(name)
        rows.append(
            {
                "variable": column,
                "modalite": category,
                "coefficient": float(coefficient),
                "odds_ratio": float(math.exp(coefficient)),
            }
        )
    rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return rows


# --------------------------------------------------------------------------
# Persistance
# --------------------------------------------------------------------------


def save_model(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def as_record(payload: dict) -> pd.DataFrame:
    """Transforme un dictionnaire de saisie en DataFrame à une ligne."""
    record = {column: payload[column] for column in MODEL_FEATURES}
    return pd.DataFrame([record])


def split_output_name(name: str):
    """``num__age`` -> ``("age", None)`` ; ``cat__housing_rent`` -> ``("housing", "rent")``."""
    if name.startswith("num__"):
        return name[len("num__") :], None
    if name.startswith("cat__"):
        body = name[len("cat__") :]
        best = None
        for column in CATEGORICAL_FEATURES:
            prefix = column + "_"
            if body.startswith(prefix):
                if best is None or len(column) > len(best):
                    best = column
        if best is not None:
            return best, body[len(best) + 1 :]
        return body, None
    return name, None


def probability_of_default(pipeline: Pipeline, record: pd.DataFrame) -> float:
    """Probabilité de défaut (entre 0 et 1) pour un ou plusieurs dossiers."""
    return float(pipeline.predict_proba(record[MODEL_FEATURES])[:, 1][0])


def to_score(probability: float) -> float:
    """Convertit une probabilité de défaut en score de crédit 300–850."""
    probability = min(max(float(probability), 1e-6), 1 - 1e-6)
    odds_good = (1.0 - probability) / probability
    score = SCORE_ANCHOR + POINTS_PER_DOUBLING * math.log2(odds_good)
    return float(min(max(score, SCORE_MIN), SCORE_MAX))


def explain(pipeline: Pipeline, record: pd.DataFrame) -> dict:
    """Décompose une décision en contributions additives au score.

    Retourne notamment ``contributions`` : une entrée par variable, avec
    l'impact en points de score (positif = réduit le risque) et la valeur
    saisie, prête à être affichée.
    """
    preprocessor = pipeline.named_steps["pre"]
    classifier = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(record[MODEL_FEATURES])
    transformed = np.asarray(transformed, dtype=float).reshape(1, -1)[0]

    names = list(preprocessor.get_feature_names_out())
    coefficients = classifier.coef_.ravel()
    intercept = float(classifier.intercept_[0])

    # Agrégation par variable d'origine (une modalité active par catégorielle).
    per_column = {column: {"contribution_log_odds": 0.0, "category": None} for column in MODEL_FEATURES}
    for name, value, coefficient in zip(names, transformed, coefficients):
        column, category = split_output_name(name)
        if column not in per_column:
            continue
        if value != 0.0:
            per_column[column]["contribution_log_odds"] += coefficient * value
            if category is not None:
                per_column[column]["category"] = category

    contributions = []
    for column in MODEL_FEATURES:
        entry = per_column[column]
        contributions.append(
            {
                "column": column,
                "category": entry["category"],
                "value": record.iloc[0][column],
                "log_odds": entry["contribution_log_odds"],
                "score_points": -_K * entry["contribution_log_odds"],
            }
        )
    contributions.sort(key=lambda row: abs(row["score_points"]), reverse=True)

    base_score = SCORE_ANCHOR - _K * intercept
    probability = 1.0 / (1.0 + math.exp(-(intercept + sum(r["log_odds"] for r in contributions))))
    score = base_score + sum(r["score_points"] for r in contributions)

    return {
        "base_score": float(base_score),
        "score": float(score),
        "clamped_score": float(min(max(score, SCORE_MIN), SCORE_MAX)),
        "intercept": intercept,
        "contributions": contributions,
        "recomputed_probability": float(probability),
    }
