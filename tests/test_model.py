"""Tests du moteur de scoring (données, modèle, explicabilité).

Lancement :  pytest -q
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.data import build_dataset, generate_synthetic, load_german_credit
from src.model import (
    SCORE_MAX,
    SCORE_MIN,
    explain,
    probability_of_default,
    to_score,
    train_model,
)
from src.schema import CATEGORIES, MODEL_FEATURES, TARGET, risk_band, value_label

RECORD = {
    "age": 34,
    "credit_amount": 12000,
    "duration": 48,
    "installment_rate": 3,
    "present_residence": 2,
    "existing_credits": 2,
    "num_dependents": 2,
    "checking_account": "< 0",
    "credit_history": "past delay",
    "purpose": "business",
    "saving_accounts": "< 100",
    "employment_years": "1 - 4",
    "housing": "rent",
    "job": "unskilled resident",
}


@pytest.fixture(scope="module")
def bundle():
    dataset = build_dataset("synthetic", n_rows=6000, seed=42)
    return train_model(dataset, source="synthetic", seed=42)


# --------------------------------------------------------------------------
# Jeu de données synthétique
# --------------------------------------------------------------------------


def test_synthetic_respecte_le_schema():
    dataset = generate_synthetic(500, seed=1)
    assert list(dataset.columns) == MODEL_FEATURES + [TARGET]
    assert len(dataset) == 500
    assert set(dataset[TARGET].unique()) <= {0, 1}
    for column, allowed in CATEGORIES.items():
        assert set(dataset[column]) <= set(allowed), f"modalité inconnue dans {column}"


def test_taux_de_defaut_est_cale():
    """Le taux de défaut visé doit être respecté à quelques points près."""
    for target in (0.10, 0.29, 0.50):
        dataset = generate_synthetic(4000, seed=3, target_default_rate=target)
        observed = float(dataset[TARGET].mean())
        assert abs(observed - target) < 0.03, f"visé {target}, obtenu {observed}"


def test_taux_de_defaut_invalide():
    with pytest.raises(ValueError):
        generate_synthetic(100, target_default_rate=1.5)


def test_reproductibilite():
    first = generate_synthetic(300, seed=7)
    second = generate_synthetic(300, seed=7)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------
# German Credit
# --------------------------------------------------------------------------


def test_german_credit_normalise_sans_perte():
    """Toutes les modalités d'origine doivent être ramenées vers le schéma."""
    try:
        dataset = load_german_credit()
    except Exception as error:  # réseau indisponible
        pytest.skip(f"German Credit indisponible : {error}")

    assert len(dataset) == 1000, "aucun dossier ne doit être écarté"
    for column, allowed in CATEGORIES.items():
        assert set(dataset[column]) <= set(allowed), f"modalité non mappée dans {column}"
    assert abs(float(dataset[TARGET].mean()) - 0.30) < 0.01


# --------------------------------------------------------------------------
# Modèle et performance
# --------------------------------------------------------------------------


def test_performance_du_modele(bundle):
    metrics = bundle["metadata"]["metrics"]
    assert metrics["auc"] > 0.75, "un scorecard de démonstration doit dépasser 0.75 d'AUC"
    assert 0.0 < metrics["brier"] < 0.25


def test_probabilites_bornees(bundle):
    sample = build_dataset("synthetic", n_rows=300, seed=5)
    probabilities = bundle["pipeline"].predict_proba(sample[MODEL_FEATURES])[:, 1]
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_dossier_sain_mieux_note_que_dossier_risque(bundle):
    def score(**overrides):
        record = dict(RECORD)
        record.update(overrides)
        return explain(bundle["pipeline"], pd.DataFrame([record]))["clamped_score"]

    assert score(credit_history="all paid duly", checking_account=">= 200") > score()
    assert score(purpose="car", housing="own", installment_rate=1) > score()
    assert score(credit_amount=25000, duration=72) < score(credit_amount=3000, duration=12)


# --------------------------------------------------------------------------
# Échelle de score
# --------------------------------------------------------------------------


def test_echelle_de_score():
    assert math.isclose(to_score(0.5), 600.0, abs_tol=1e-9)
    assert math.isclose(to_score(0.25), 600 + 20 * math.log2(3), abs_tol=1e-6)
    assert to_score(0.01) > to_score(0.30) > to_score(0.75)
    assert SCORE_MIN <= to_score(1 - 1e-9) <= SCORE_MAX
    assert SCORE_MIN <= to_score(1e-9) <= SCORE_MAX


def test_bandes_de_risque():
    assert risk_band(0.01)[2] == "A"
    assert risk_band(0.10)[2] == "B"
    assert risk_band(0.25)[2] == "C"
    assert risk_band(0.40)[2] == "D"
    assert risk_band(0.80)[2] == "E"
    # les bandes doivent couvrir tout l'intervalle sans trou
    for probability in (0.0, 0.05, 0.15, 0.30, 0.50, 0.999):
        assert risk_band(probability) is not None


# --------------------------------------------------------------------------
# Explicabilité
# --------------------------------------------------------------------------


def test_explication_est_exactement_additive(bundle):
    """base + somme des impacts doit redonner le score, et la probabilité
    reconstruite doit coïncider avec celle de scikit-learn."""
    pipeline = bundle["pipeline"]
    sample = build_dataset("synthetic", n_rows=60, seed=11)

    for _, row in sample.iterrows():
        frame = pd.DataFrame([{column: row[column] for column in MODEL_FEATURES}])
        analysis = explain(pipeline, frame)

        total = analysis["base_score"] + sum(
            contribution["score_points"] for contribution in analysis["contributions"]
        )
        assert math.isclose(total, analysis["score"], abs_tol=1e-9)

        assert math.isclose(
            analysis["recomputed_probability"],
            probability_of_default(pipeline, frame),
            abs_tol=1e-9,
        )


def test_explication_couvre_toutes_les_variables(bundle):
    analysis = explain(bundle["pipeline"], pd.DataFrame([RECORD]))
    columns = {contribution["column"] for contribution in analysis["contributions"]}
    assert columns == set(MODEL_FEATURES)


def test_explication_signale_les_facteurs_defavorables(bundle):
    analysis = explain(bundle["pipeline"], pd.DataFrame([RECORD]))
    by_column = {contribution["column"]: contribution for contribution in analysis["contributions"]}

    # un compte courant négatif et un incident de paiement doivent desservir
    assert by_column["checking_account"]["score_points"] < 0
    assert by_column["credit_history"]["score_points"] < 0
    assert by_column["checking_account"]["category"] == "< 0"
    assert value_label("checking_account", "< 0") == "Solde négatif"


def test_explication_ignore_les_modalites_inactives(bundle):
    """Une seule modalité par variable catégorielle doit contribuer."""
    analysis = explain(bundle["pipeline"], pd.DataFrame([RECORD]))
    categorical = [c for c in analysis["contributions"] if c["category"] is not None]
    assert len(categorical) == 7, "7 variables catégorielles, une modalité active chacune"
