"""Jeux de données d'entraînement.

Deux sources sont disponibles :

* ``synthetic`` : un portefeuille de crédits généré localement, calibré pour
  reproduire un taux de défaut d'environ 28 % (proche du German Credit).
* ``german``   : le jeu de données public *German Credit* (UCI / OpenML,
  Hofmann, 1994), 1 000 dossiers réels, normalisé vers notre schéma.

``both`` concatène les deux portefeuilles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    CATEGORIES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

RNG_SEED = 42
DEFAULT_SYNTHETIC_ROWS = 6000
DEFAULT_DEFAULT_RATE = 0.29

# Dispersion du signal latent et bruit résiduel sur l'échelle log-odds.
# Ces deux valeurs fixent la difficulté du problème : elles produisent un
# scorecard à ~0.83 d'AUC, soit le niveau d'un bon modèle de production.
SIGNAL_STRENGTH = 1.65
LABEL_NOISE = 0.30


# --------------------------------------------------------------------------
# Synthétique
# --------------------------------------------------------------------------


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _choice(rng, values, weights, size):
    """Tirage aléatoire pondéré, sans dépendance à pandas."""
    values = np.asarray(list(values), dtype=object)
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(values, size=size, p=probabilities)


def _solve_intercept(signal, target_rate, iterations: int = 80) -> float:
    """Trouve l'ordonnée à l'origine donnant le taux de défaut visé.

    La recherche est dichotomique : la probabilité moyenne est strictement
    croissante avec l'intercept, donc la convergence est garantie.
    """
    low, high = -40.0, 40.0
    for _ in range(iterations):
        mid = (low + high) / 2.0
        if _sigmoid(mid + signal).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_synthetic(
    n_rows: int = DEFAULT_SYNTHETIC_ROWS,
    seed: int = RNG_SEED,
    target_default_rate: float = DEFAULT_DEFAULT_RATE,
) -> pd.DataFrame:
    """Génère un portefeuille de crédits synthétique au format canonique.

    Le défaut est simulé par un modèle logistique *ground truth* construit sur
    les mêmes variables que celles utilisées par le modèle entraîné. Les
    variables ont donc un effet réel, ordonné et interprétable.

    Les poids ci-dessous décrivent l'**importance relative** des facteurs de
    risque (un historique critique pèse davantage qu'un motif de crédit
    risqué, etc.). Le signal est ensuite recentré et réduit, puis l'intercept
    est calé automatiquement pour atteindre ``target_default_rate``.
    """
    if not 0.0 < target_default_rate < 1.0:
        raise ValueError("target_default_rate doit être strictement entre 0 et 1.")

    rng = np.random.default_rng(seed)

    # --- variables numériques ---
    age = np.clip(rng.normal(38, 12, n_rows), 18, 80).round().astype(int)
    credit_amount = np.clip(
        rng.lognormal(mean=8.1, sigma=0.65, size=n_rows), 250, 25000
    ).round().astype(int)
    duration = np.clip(rng.normal(24, 11, n_rows), 4, 72).round().astype(int)
    installment_rate = _choice(rng, [1, 2, 3, 4], [0.20, 0.45, 0.25, 0.10], n_rows).astype(int)
    present_residence = _choice(rng, [1, 2, 3, 4], [0.15, 0.35, 0.30, 0.20], n_rows).astype(int)
    existing_credits = _choice(rng, [1, 2, 3, 4], [0.60, 0.25, 0.10, 0.05], n_rows).astype(int)
    num_dependents = _choice(rng, [1, 2], [0.75, 0.25], n_rows).astype(int)

    # --- variables catégorielles ---
    checking_account = _choice(
        rng, CATEGORIES["checking_account"], [0.30, 0.27, 0.27, 0.16], n_rows
    )
    credit_history = _choice(
        rng, CATEGORIES["credit_history"], [0.04, 0.21, 0.53, 0.09, 0.13], n_rows
    )
    purpose = _choice(
        rng,
        CATEGORIES["purpose"],
        [0.30, 0.18, 0.14, 0.05, 0.04, 0.05, 0.01, 0.01, 0.10, 0.12],
        n_rows,
    )
    saving_accounts = _choice(
        rng, CATEGORIES["saving_accounts"], [0.55, 0.11, 0.06, 0.05, 0.23], n_rows
    )
    employment_years = _choice(
        rng, CATEGORIES["employment_years"], [0.06, 0.14, 0.34, 0.17, 0.29], n_rows
    )
    housing = _choice(rng, CATEGORIES["housing"], [0.70, 0.18, 0.12], n_rows)
    job = _choice(rng, CATEGORIES["job"], [0.02, 0.20, 0.63, 0.15], n_rows)

    # --- facteurs de risque (poids relatifs) ---
    signal = (
        1.20 * (checking_account == "< 0")
        + 0.55 * (checking_account == "0 - 200")
        - 0.70 * (checking_account == "no checking")
        + 0.85 * (credit_history == "critical account")
        + 0.45 * (credit_history == "past delay")
        + 0.35 * np.isin(purpose, ["business", "vacation", "education"])
        + 0.30 * np.isin(saving_accounts, ["< 100", "unknown"])
        + 0.25 * (employment_years == "unemployed")
        + 0.45 * ((credit_amount - 5000) / 5000.0)
        + 0.030 * (duration - 20)
        - 0.022 * (age - 40)
        + 0.35 * (installment_rate - 2)
        + 0.30 * (housing == "rent")
        - 0.20 * np.isin(job, ["skilled", "highly qualified"])
        + rng.normal(0.0, LABEL_NOISE, n_rows)
    )

    # Recentrage/réduction : la dispersion du signal devient contrôlée.
    signal = (signal - signal.mean()) / signal.std()

    intercept = _solve_intercept(SIGNAL_STRENGTH * signal, target_default_rate)
    probability_default = _sigmoid(intercept + SIGNAL_STRENGTH * signal)
    default = rng.binomial(1, probability_default)

    return pd.DataFrame(
        {
            "age": age,
            "credit_amount": credit_amount,
            "duration": duration,
            "installment_rate": installment_rate,
            "present_residence": present_residence,
            "existing_credits": existing_credits,
            "num_dependents": num_dependents,
            "checking_account": checking_account,
            "credit_history": credit_history,
            "purpose": purpose,
            "saving_accounts": saving_accounts,
            "employment_years": employment_years,
            "housing": housing,
            "job": job,
            TARGET: default.astype(int),
        }
    )[MODEL_FEATURES + [TARGET]]


# --------------------------------------------------------------------------
# German Credit (UCI / OpenML)
# --------------------------------------------------------------------------


def _map_checking(value) -> str:
    s = str(value).strip().lower().replace(" ", "")
    if "nochecking" in s or s == "none":
        return "no checking"
    if s.startswith(">="):
        return ">= 200"
    if s.startswith("<0"):
        return "< 0"
    return "0 - 200"


def _map_savings(value):
    s = str(value).strip().lower().replace(" ", "")
    # « unknown/no savings account » et « no known savings » désignent la même
    # situation : aucune épargne connue.
    if "unknown" in s or "nosavings" in s or ("no" in s and "savings" in s):
        return "unknown"
    if s.startswith(">="):
        return ">= 1000"
    if s.startswith("<100"):
        return "< 100"
    if s.startswith("500"):
        return "500 - 1000"
    if s.startswith("100"):
        return "100 - 500"
    return None


def _map_credit_history(value):
    s = str(value).strip().lower()
    if "critical" in s:
        return "critical account"
    if "delay" in s:
        return "past delay"
    if "no credit" in s:
        return "no credits taken"
    if "existing" in s and "paid" in s:
        return "existing paid duly"
    if "all paid" in s:
        return "all paid duly"
    return None


def _map_purpose(value) -> str:
    s = str(value).strip().lower()
    for key, label in (
        ("radio/tv", "radio/TV"),
        ("domestic appliance", "domestic appliances"),
        ("furniture", "furniture/equipment"),
        ("retrain", "retraining"),
        ("education", "education"),
        ("vacation", "vacation"),
        ("repair", "repairs"),
        ("business", "business"),
        ("car", "car"),
    ):
        if key in s:
            return label
    return "others"


def _map_employment(value):
    s = str(value).strip().lower().replace(" ", "")
    if "unemployed" in s:
        return "unemployed"
    if s.startswith(">="):
        return ">= 7"
    if s.startswith("<1"):
        return "< 1"
    if s.startswith("1"):
        return "1 - 4"
    if s.startswith("4"):
        return "4 - 7"
    return None


def _map_housing(value):
    s = str(value).strip().lower()
    if "own" in s:
        return "own"
    if "rent" in s:
        return "rent"
    if "free" in s:  # « free » ou « for free »
        return "free"
    return None


def _map_job(value):
    s = str(value).strip().lower()
    if "non res" in s or "non-res" in s or "non_res" in s:
        return "unskilled non-resident"
    if "unskilled" in s or "unemp" in s:
        return "unskilled resident"
    if "high" in s or "mgmt" in s or "manag" in s or "self emp" in s:
        return "highly qualified"
    if "skilled" in s:
        return "skilled"
    return None


def _pick(frame: pd.DataFrame, *candidates: str):
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    raise KeyError(
        f"Colonne introuvable parmi {candidates}. Colonnes disponibles : {list(frame.columns)}"
    )


def load_german_credit() -> pd.DataFrame:
    """Télécharge (avec cache) et normalise le jeu *German Credit*.

    Les colonnes d'origine sont renommées et les modalités ramenées vers le
    vocabulaire canonique du schéma. Les dossiers dont une modalité reste
    inconnue sont écartés.
    """
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    raw = bunch.frame

    frame = pd.DataFrame(
        {
            "age": pd.to_numeric(_pick(raw, "age"), errors="coerce"),
            "credit_amount": pd.to_numeric(_pick(raw, "credit_amount"), errors="coerce"),
            "duration": pd.to_numeric(_pick(raw, "duration", "month"), errors="coerce"),
            "installment_rate": pd.to_numeric(
                _pick(raw, "installment_commitment", "installment_rate"), errors="coerce"
            ),
            "present_residence": pd.to_numeric(
                _pick(raw, "present_residence", "residence_since"), errors="coerce"
            ),
            "existing_credits": pd.to_numeric(
                _pick(raw, "existing_credits", "num_credits"), errors="coerce"
            ),
            "num_dependents": pd.to_numeric(_pick(raw, "num_dependents"), errors="coerce"),
            "checking_account": _pick(raw, "checking_status", "checking_account").map(
                _map_checking
            ),
            "credit_history": _pick(raw, "credit_history").map(_map_credit_history),
            "purpose": _pick(raw, "purpose").map(_map_purpose),
            "saving_accounts": _pick(raw, "savings_status", "saving_accounts").map(
                _map_savings
            ),
            "employment_years": _pick(raw, "employment", "employment_years").map(
                _map_employment
            ),
            "housing": _pick(raw, "housing").map(_map_housing),
            "job": _pick(raw, "job").map(_map_job),
        }
    )

    target = _pick(raw, "class", "target").astype(str).str.strip().str.lower()
    frame[TARGET] = (~target.isin(["good", "0", "false"])).astype(int)

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    frame = frame.dropna(subset=required).reset_index(drop=True)

    for column in NUMERIC_FEATURES:
        frame[column] = frame[column].astype(int)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype(str)

    if frame.empty:
        raise RuntimeError(
            "Aucun dossier German Credit n'a pu être normalisé : "
            "les libellés du jeu de données ont peut-être changé."
        )

    return frame[MODEL_FEATURES + [TARGET]]


# --------------------------------------------------------------------------
# Assemblage
# --------------------------------------------------------------------------

SOURCES = ("synthetic", "german", "both")


def build_dataset(
    source: str = "synthetic",
    n_rows: int = DEFAULT_SYNTHETIC_ROWS,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Construit le portefeuille d'entraînement demandé."""
    if source not in SOURCES:
        raise ValueError(f"source doit être l'une de {SOURCES}, reçu {source!r}")

    frames = []
    if source in ("synthetic", "both"):
        frames.append(generate_synthetic(n_rows=n_rows, seed=seed))
    if source in ("german", "both"):
        frames.append(load_german_credit())

    if len(frames) == 1:
        dataset = frames[0]
    else:
        dataset = pd.concat(frames, ignore_index=True)

    dataset = dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    for column in CATEGORICAL_FEATURES:
        dataset[column] = dataset[column].astype(str)
    for column in NUMERIC_FEATURES:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    dataset[TARGET] = dataset[TARGET].astype(int)

    return dataset.dropna(subset=NUMERIC_FEATURES).reset_index(drop=True)


def describe_dataset(dataset: pd.DataFrame) -> dict:
    """Quelques statistiques descriptives pour l'interface."""
    positives = int(dataset[TARGET].sum())
    total = int(len(dataset))
    return {
        "n_rows": total,
        "n_defaults": positives,
        "default_rate": (positives / total) if total else 0.0,
        "mean_credit_amount": float(dataset["credit_amount"].mean()) if total else 0.0,
        "mean_duration": float(dataset["duration"].mean()) if total else 0.0,
    }
