"""Tests de bout en bout de l'application Streamlit (via st.testing).

Lancement :  pytest -q
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")

RISKY = {
    "in_employment_years": "unemployed",
    "in_job": "unskilled non-resident",
    "in_checking_account": "< 0",
    "in_saving_accounts": "< 100",
    "in_installment_rate": 4,
    "in_purpose": "business",
    "in_credit_history": "critical account",
    "in_housing": "rent",
    "in_present_residence": 1,
    "in_existing_credits": 4,
    "in_num_dependents": 2,
}
HEALTHY = {
    "in_employment_years": ">= 7",
    "in_job": "highly qualified",
    "in_checking_account": ">= 200",
    "in_saving_accounts": ">= 1000",
    "in_installment_rate": 1,
    "in_purpose": "car",
    "in_credit_history": "all paid duly",
    "in_housing": "own",
    "in_present_residence": 4,
    "in_existing_credits": 1,
    "in_num_dependents": 1,
}
MEDIAN = {
    "in_employment_years": "1 - 4",
    "in_job": "skilled",
    "in_checking_account": "0 - 200",
    "in_saving_accounts": "< 100",
    "in_installment_rate": 2,
    "in_purpose": "car",
    "in_credit_history": "existing paid duly",
    "in_housing": "rent",
    "in_present_residence": 2,
    "in_existing_credits": 1,
    "in_num_dependents": 1,
}


# --------------------------------------------------------------------------
# Utilitaires de lecture de l'arbre de widgets
# --------------------------------------------------------------------------


def metric(app, label):
    for item in app.metric:
        if item.label == label:
            return item.value
    raise AssertionError(f"métrique absente : {label}")


def default_rate(app):
    return float(metric(app, "Probabilité de défaut").rstrip("%")) / 100


def verdict(app):
    """Score, classe et décision lus dans le bandeau de résultat."""
    for block in app.markdown:
        if "SCORE DE CRÉDIT" in block.value:
            score = re.search(r">(\d+)<span", block.value)
            band = re.search(r"Classe <b>(.*?)</b>", block.value)
            decision = re.search(r"Décision : <b>(.*?)</b>", block.value)
            return {
                "score": int(score.group(1)) if score else None,
                "classe": band.group(1) if band else None,
                "decision": decision.group(1) if decision else None,
            }
    return {}


def decomposition(app):
    """Score de base, somme des impacts et score final, lus dans la légende."""
    for caption in app.caption:
        match = re.search(
            r"Score de base : \*\*(-?[\d.]+)\*\*.*somme des impacts : \*\*([+-][\d.]+)\*\*"
            r".*score final : \*\*(-?[\d.]+)\*\*",
            caption.value,
            re.DOTALL,
        )
        if match:
            return tuple(float(group) for group in match.groups())
    raise AssertionError("légende de décomposition introuvable")


def submit(app, settings, *, age=35, amount=5000, duration=24):
    for key, value in settings.items():
        app.selectbox(key=key).select(value)
    app.number_input(key="in_age").set_value(age)
    app.number_input(key="in_credit_amount").set_value(amount)
    app.number_input(key="in_duration").set_value(duration)
    app.button(key="submit").click()
    app.run()
    return app


@pytest.fixture(scope="module")
def app():
    instance = AppTest.from_file(APP, default_timeout=300)
    instance.run()
    assert not instance.exception, [e.value for e in instance.exception]
    return instance


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_demarrage(app):
    assert "risque de crédit" in app.title[0].value
    assert app.selectbox(key="source").value == "synthetic"
    assert app.slider(key="threshold").value == 0.30
    # le formulaire expose bien les 14 variables du modèle
    form_keys = {widget.key for widget in app.selectbox} | {
        widget.key for widget in app.number_input
    }
    form_keys.discard("source")
    assert len(form_keys) == 14
    # aucun résultat avant la première évaluation
    assert len(app.info) == 1
    assert float(metric(app, "AUC (ROC)")) > 0.75


def test_dossier_risque_est_refuse(app):
    submit(app, RISKY, age=22, amount=25000, duration=72)
    result = verdict(app)
    assert result["decision"] == "Refuser"
    assert result["classe"] == "E"
    assert default_rate(app) > 0.5
    assert len(app.get("download_button")) == 1


def test_dossier_sain_est_accepte(app):
    submit(app, HEALTHY, age=45, amount=4000, duration=12)
    result = verdict(app)
    assert result["decision"] == "Accorder"
    assert result["classe"] == "A"
    assert default_rate(app) < 0.05


def test_explication_est_coherente(app):
    submit(app, MEDIAN)
    detail = app.dataframe[0].value
    assert list(detail.columns) == ["Variable", "Valeur", "Impact (points)"]
    assert len(detail) == 14

    base, impacts, final = decomposition(app)
    assert abs(base + impacts - final) < 1.0, "décomposition non additive"
    assert abs(final - verdict(app)["score"]) < 1.0


def test_resultat_persiste_et_seuil_s_applique(app):
    submit(app, MEDIAN)
    probability = default_rate(app)
    assert 0.20 < probability < 0.40, "le profil médian doit être un dossier intermédiaire"
    band = verdict(app)["classe"]

    attendu = {
        "A": "Accorder",
        "B": "Accorder",
        "C": "Accorder avec garanties",
    }.get(band, "Analyse manuelle")

    for threshold in (0.90, 0.50, 0.30):
        app.slider(key="threshold").set_value(threshold)
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        assert verdict(app)["decision"] == attendu, f"seuil {threshold}"

    # au-dessus de la probabilité de défaut, le dossier bascule en refus
    for threshold in (0.20, 0.05):
        app.slider(key="threshold").set_value(threshold)
        app.run()
        assert verdict(app)["decision"] == "Refuser", f"seuil {threshold}"


def test_bascule_de_portefeuille(app):
    app.selectbox(key="source").select("german")
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.slider(key="n_rows").disabled
    auc_german = float(metric(app, "AUC (ROC)"))
    assert 0.70 < auc_german < 0.90

    app.selectbox(key="source").select("both")
    app.run()
    assert float(metric(app, "AUC (ROC)")) > 0.75

    app.selectbox(key="source").select("synthetic")
    app.run()
    assert float(metric(app, "AUC (ROC)")) > 0.75


def test_reentrainement_sans_erreur(app):
    app.button(key="retrain").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert len(app.error) == 0
    assert float(metric(app, "AUC (ROC)")) > 0.75


def test_bornes_du_formulaire(app):
    submit(app, RISKY, age=18, amount=250, duration=4)
    assert not app.exception, [e.value for e in app.exception]
    assert verdict(app)["score"] is not None

    submit(app, HEALTHY, age=80, amount=25000, duration=72)
    assert not app.exception, [e.value for e in app.exception]
    assert verdict(app)["score"] is not None
