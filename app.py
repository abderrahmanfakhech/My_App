"""Application Streamlit — Évaluation intelligente du risque de crédit.

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import SOURCES, build_dataset, describe_dataset
from src.model import (
    MODEL_FILENAME,
    SCORE_MAX,
    SCORE_MIN,
    explain,
    load_model,
    save_model,
    to_score,
    train_model,
)
from src.schema import (
    CATEGORIES,
    NUMERIC_FIELDS,
    RISK_BANDS,
    column_label,
    risk_band,
    value_label,
)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / MODEL_FILENAME

SOURCE_LABELS = {
    "synthetic": "Portefeuille synthétique",
    "german": "German Credit (UCI, 1 000 dossiers réels)",
    "both": "Synthétique + German Credit",
}

ORDINAL_LABELS = {
    "installment_rate": {
        1: "1 — Faible",
        2: "2 — Modéré",
        3: "3 — Élevé",
        4: "4 — Très élevé",
    },
    "present_residence": {
        1: "Moins d'un an",
        2: "1 à 3 ans",
        3: "3 à 7 ans",
        4: "7 ans et plus",
    },
    "existing_credits": {
        1: "1 crédit",
        2: "2 crédits",
        3: "3 crédits",
        4: "4 crédits ou plus",
    },
    "num_dependents": {1: "Aucune", 2: "Une ou plus"},
}

# Organisation du formulaire : sections et variables qu'elles contiennent.
FORM_SECTIONS = [
    ("1 · Profil du demandeur", ["age", "employment_years", "job"]),
    ("2 · Situation financière", ["checking_account", "saving_accounts", "installment_rate"]),
    ("3 · Crédit demandé", ["credit_amount", "duration", "purpose"]),
    (
        "4 · Historique et stabilité",
        ["credit_history", "housing", "present_residence", "existing_credits", "num_dependents"],
    ),
]

# Valeur affichée par défaut pour les variables catégorielles.
CATEGORY_DEFAULTS = {
    "checking_account": "0 - 200",
    "credit_history": "existing paid duly",
    "purpose": "car",
    "saving_accounts": "< 100",
    "employment_years": "1 - 4",
    "housing": "rent",
    "job": "skilled",
}
ORDINAL_DEFAULTS = {
    "installment_rate": 2,
    "present_residence": 2,
    "existing_credits": 1,
    "num_dependents": 1,
}


def render_field(container, column: str):
    """Affiche le widget correspondant à une variable du modèle.

    Les variables à échelle ordinale et les variables nominales utilisent des
    listes déroulantes aux libellés explicites ; les variables continues
    utilisent une saisie numérique bornée par le schéma.
    """
    key = f"in_{column}"

    # Les variables ordinales passent avant les numériques : elles sont
    # techniquement numériques (1 à 4) mais gagnent à être présentées comme
    # des listes déroulantes explicitement libellées.
    if column in ORDINAL_LABELS:
        options = list(ORDINAL_LABELS[column])
        return container.selectbox(
            column_label(column),
            options=options,
            format_func=lambda value, c=column: ORDINAL_LABELS[c][value],
            index=options.index(ORDINAL_DEFAULTS[column]),
            key=key,
        )

    if column in NUMERIC_FIELDS:
        config = NUMERIC_FIELDS[column]
        return container.number_input(
            config["label"],
            min_value=config["min"],
            max_value=config["max"],
            value=config["default"],
            step=config["step"],
            help=config["help"],
            key=key,
        )

    options = CATEGORIES[column]
    return container.selectbox(
        column_label(column),
        options=options,
        format_func=lambda value, c=column: value_label(c, value),
        index=options.index(CATEGORY_DEFAULTS[column]),
        key=key,
    )

st.set_page_config(
    page_title="Évaluation du risque de crédit",
    page_icon="🏦",
    layout="wide",
)


# --------------------------------------------------------------------------
# Ressources mises en cache
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_dataset(source: str, n_rows: int, seed: int) -> pd.DataFrame:
    return build_dataset(source, n_rows=n_rows, seed=seed)


@st.cache_resource(show_spinner="Entraînement du modèle de scoring…")
def get_bundle(source: str, n_rows: int, seed: int, nonce: int = 0) -> dict:
    """Charge le modèle sauvegardé s'il correspond, sinon l'entraîne."""
    bundle = load_model(MODEL_PATH)
    metadata = (bundle or {}).get("metadata", {})
    if bundle is not None and metadata.get("source") == source and metadata.get("n_rows") == n_rows:
        return bundle

    bundle = train_model(get_dataset(source, n_rows, seed), source=source, seed=seed)
    save_model(bundle, MODEL_PATH)
    return bundle


# --------------------------------------------------------------------------
# Graphiques
# --------------------------------------------------------------------------


def score_gauge(score: float):
    """Jauge horizontale situant le score entre 300 et 850."""
    edges = [to_score(band[1]) for band in RISK_BANDS]
    bands = list(zip(edges, RISK_BANDS))

    fig, ax = plt.subplots(figsize=(6.6, 1.35), dpi=150)

    left = SCORE_MIN
    for right, band in bands:
        right = min(max(right, SCORE_MIN), SCORE_MAX)
        if right <= left:
            continue
        ax.barh(0, right - left, left=left, height=0.5, color=band[4], alpha=0.85, zorder=2)
        ax.text(
            (left + right) / 2,
            0,
            band[2],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        left = right

    ax.axvline(score, color="#111111", linewidth=2.5, zorder=5)
    ax.plot([score], [0.42], marker="v", color="#111111", markersize=9, zorder=6)

    ax.set_xlim(SCORE_MIN, SCORE_MAX)
    ax.set_ylim(-0.5, 0.75)
    ax.set_yticks([])
    ax.set_xticks([300, 400, 500, 600, 700, 800, 850])
    ax.tick_params(axis="x", labelsize=8, colors="#555555")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")
    fig.tight_layout(pad=0.2)
    return fig


def contribution_chart(contributions: list, top_n: int = 8):
    """Diagramme en barres des contributions au score."""
    rows = contributions[:top_n][::-1]
    labels = []
    for row in rows:
        detail = (
            value_label(row["column"], row["category"])
            if row["category"] is not None
            else f"{row['value']:g}"
        )
        label = f"{column_label(row['column'])} : {detail}"
        labels.append(label if len(label) <= 46 else label[:43] + "…")

    values = [row["score_points"] for row in rows]
    colors = ["#1a7f37" if value >= 0 else "#cf222e" for value in values]

    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 1.1), dpi=150)
    ax.barh(labels, values, color=colors, height=0.62)
    ax.axvline(0, color="#999999", linewidth=0.9)
    ax.set_xlabel("Impact sur le score (points)", fontsize=9, color="#555555")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8, colors="#555555")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    return fig


# --------------------------------------------------------------------------
# Barre latérale
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🏦 CréditScore")
    st.caption("Évaluation intelligente du risque de crédit")

    st.divider()

    source = st.selectbox(
        "Portefeuille d'entraînement",
        options=list(SOURCES),
        format_func=lambda key: SOURCE_LABELS[key],
        index=0,
        key="source",
        help=(
            "Le modèle est entraîné à la volée sur le portefeuille choisi. "
            "« German Credit » nécessite un accès Internet au premier lancement."
        ),
    )
    n_rows = st.slider(
        "Dossiers synthétiques",
        min_value=1000,
        max_value=20000,
        value=6000,
        step=1000,
        key="n_rows",
        disabled=source == "german",
    )
    threshold = st.slider(
        "Seuil de décision (probabilité de défaut)",
        min_value=0.05,
        max_value=0.90,
        value=0.30,
        step=0.01,
        key="threshold",
        help="Au-delà de ce seuil, le dossier est orienté vers un refus.",
    )

    if st.button("🔄 Réentraîner le modèle", key="retrain", width="stretch"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()

try:
    dataset = get_dataset(source, n_rows, 42)
    bundle = get_bundle(source, n_rows, 42)
except Exception as error:  # pragma: no cover - dépend du réseau
    st.error(
        "Impossible de préparer le modèle.\n\n"
        f"**Détail technique :** `{type(error).__name__}: {error}`\n\n"
        "Si vous avez choisi « German Credit », le téléchargement du jeu de données "
        "a peut-être échoué. Vérifiez votre connexion Internet ou revenez au "
        "portefeuille synthétique."
    )
    st.stop()

pipeline = bundle["pipeline"]
metadata = bundle["metadata"]
metrics = metadata["metrics"]
stats = describe_dataset(dataset)

with st.sidebar:
    st.subheader("Le modèle en bref")
    col_a, col_b = st.columns(2)
    col_a.metric("AUC (ROC)", f"{metrics['auc']:.3f}")
    col_b.metric("Exactitude", f"{metrics['accuracy']:.1%}")
    col_a.metric("Rappel (défauts)", f"{metrics['recall']:.1%}")
    col_b.metric("Précision", f"{metrics['precision']:.1%}")

    st.caption(
        f"{stats['n_rows']:,} dossiers — {stats['default_rate']:.1%} de défauts  \n"
        f"Entraîné le {metadata['trained_at'][:10]}"
    )

    with st.expander("Performance et calibration"):
        st.write(
            f"**Indice de Brier :** {metrics['brier']:.4f} "
            "(plus proche de 0 = mieux calibré)"
        )
        st.write(f"**F1 :** {metrics['f1']:.3f}")
        matrix = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Réel : remboursé", "Réel : défaut"],
            columns=["Prédit : remboursé", "Prédit : défaut"],
        )
        st.dataframe(matrix, width="stretch")

    with st.expander("Coefficients du modèle (audit)"):
        coefficients = pd.DataFrame(metadata["coefficients"])
        coefficients = coefficients.head(15).copy()
        coefficients["variable"] = coefficients["variable"].map(column_label)
        coefficients["modalite"] = coefficients["modalite"].fillna("—")
        coefficients["coefficient"] = coefficients["coefficient"].round(3)
        coefficients["odds_ratio"] = coefficients["odds_ratio"].round(3)
        coefficients.columns = ["Variable", "Modalité", "Coefficient", "Odds ratio"]
        st.dataframe(coefficients, width="stretch", hide_index=True)

    st.divider()
    st.caption(
        "Variables sensibles (sexe, origine, situation familiale) exclues du "
        "modèle, conformément à la directive 2004/113/CE et au RGPD."
    )


# --------------------------------------------------------------------------
# En-tête
# --------------------------------------------------------------------------

st.title("Évaluation intelligente du risque de crédit")
st.caption(
    "Renseignez le dossier du demandeur : le modèle estime sa probabilité de "
    "défaut, lui attribue un score et explique chaque décision variable par variable."
)

form_column, result_column = st.columns([1.25, 1], gap="large")


# --------------------------------------------------------------------------
# Formulaire
# --------------------------------------------------------------------------

with form_column:
    with st.form("scoring_form", border=True):
        st.markdown("#### Dossier du demandeur")

        values = {}
        for section_title, fields in FORM_SECTIONS:
            st.markdown(f"**{section_title}**")
            for start in range(0, len(fields), 3):
                slots = st.columns(3)
                for slot, column in zip(slots, fields[start : start + 3]):
                    values[column] = render_field(slot, column)

        submitted = st.form_submit_button(
            "⚖️ Évaluer le dossier",
            type="primary",
            width="stretch",
            key="submit",
        )

# Un `form_submit_button` ne renvoie True que sur le run du clic : sans
# mémorisation, le résultat disparaîtrait dès que l'utilisateur bouge le
# seuil de décision. On conserve donc le dernier dossier évalué.
if submitted:
    st.session_state["last_record"] = dict(values)

record = st.session_state.get("last_record")


# --------------------------------------------------------------------------
# Résultats
# --------------------------------------------------------------------------

with result_column:
    if record is None:
        st.info(
            "👈 Complétez le dossier puis cliquez sur **Évaluer le dossier**.\n\n"
            "Le résultat, le score et l'explication détaillée s'afficheront ici."
        )
    else:
        frame = pd.DataFrame([record])
        analysis = explain(pipeline, frame)
        probability = analysis["recomputed_probability"]
        score = analysis["clamped_score"]
        band = risk_band(probability)
        if probability >= threshold:
            decision = "Refuser"
        elif band[2] in ("A", "B"):
            decision = "Accorder"
        elif band[2] == "C":
            decision = "Accorder avec garanties"
        else:
            decision = "Analyse manuelle"

        st.metric("Probabilité de défaut", f"{probability:.1%}")
        st.pyplot(score_gauge(score), width="stretch")
        plt.close("all")

        st.markdown(
            f"<div style='padding:0.7rem 0.9rem;border-radius:0.5rem;"
            f"background:{band[4]}1a;border-left:5px solid {band[4]};margin-bottom:0.6rem'>"
            f"<div style='font-size:0.8rem;color:#555555;letter-spacing:0.04em'>"
            f"SCORE DE CRÉDIT</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{band[4]};line-height:1.1'>"
            f"{score:.0f}<span style='font-size:1rem;font-weight:500;color:#777777'>"
            f" / 850</span></div>"
            f"<div style='font-size:0.95rem;color:#333333;margin-top:0.2rem'>"
            f"Classe <b>{band[2]}</b> — risque {band[3].lower()} "
            f"&nbsp;•&nbsp; Décision : <b>{decision}</b></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("##### Pourquoi cette décision ?")
        st.caption(
            "Contributions additives au score. Vert = élément qui **réduit** le risque, "
            "rouge = élément qui l'**augmente**."
        )
        st.pyplot(contribution_chart(analysis["contributions"]), width="stretch")
        plt.close("all")

        with st.expander("Détail du calcul"):
            detail = pd.DataFrame(
                [
                    {
                        "Variable": column_label(row["column"]),
                        "Valeur": (
                            value_label(row["column"], row["category"])
                            if row["category"] is not None
                            else f"{row['value']:g}"
                        ),
                        "Impact (points)": round(row["score_points"], 1),
                    }
                    for row in analysis["contributions"]
                ]
            )
            st.dataframe(detail, width="stretch", hide_index=True)

            total = sum(row["score_points"] for row in analysis["contributions"])
            raw_score = analysis["base_score"] + total
            note = ""
            if abs(raw_score - analysis["clamped_score"]) > 0.5:
                note = (
                    f" Le résultat sort de l'échelle et est ramené à "
                    f"**{analysis['clamped_score']:.0f}**."
                )
            st.caption(
                f"Score de base : **{analysis['base_score']:.1f}** (situation neutre) — "
                f"somme des impacts : **{total:+.1f}** — score final : "
                f"**{raw_score:.1f}**.{note}"
            )

        export = frame.copy()
        export.insert(0, "score", round(score, 1))
        export.insert(1, "probabilite_defaut", round(probability, 4))
        export.insert(2, "classe_risque", band[2])
        export.insert(3, "decision", decision)
        st.download_button(
            "📥 Télécharger la fiche d'évaluation (CSV)",
            data=export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"evaluation_credit_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            width="stretch",
        )

        st.caption(
            "⚠️ Outil d'aide à la décision : le score ne se substitue pas à l'analyse "
            "d'un conseiller et doit pouvoir être contesté par le demandeur."
        )
