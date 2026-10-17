"""Schéma de données de l'application.

Ce module est la source de vérité unique : il décrit les variables du modèle,
les catégories autorisées, les libellés français et les bornes des champs.

Aucune variable sensible (sexe, origine, situation familiale) n'est utilisée
comme variable explicative, conformément à la réglementation européenne
(directive 2004/113/CE, RGPD art. 22, lignes directrices ACPR/EBA).
"""

from __future__ import annotations

TARGET = "default"

# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age",
    "credit_amount",
    "duration",
    "installment_rate",
    "present_residence",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_account",
    "credit_history",
    "purpose",
    "saving_accounts",
    "employment_years",
    "housing",
    "job",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Les valeurs ci-dessous sont les valeurs *canoniques* utilisées en interne.
# Les libellés affichés à l'utilisateur sont définis dans VALUE_LABELS_FR.
CATEGORIES = {
    "checking_account": ["< 0", "0 - 200", ">= 200", "no checking"],
    "credit_history": [
        "no credits taken",
        "all paid duly",
        "existing paid duly",
        "past delay",
        "critical account",
    ],
    "purpose": [
        "car",
        "furniture/equipment",
        "radio/TV",
        "domestic appliances",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business",
        "others",
    ],
    "saving_accounts": ["< 100", "100 - 500", "500 - 1000", ">= 1000", "unknown"],
    "employment_years": ["unemployed", "< 1", "1 - 4", "4 - 7", ">= 7"],
    "housing": ["own", "rent", "free"],
    "job": [
        "unskilled non-resident",
        "unskilled resident",
        "skilled",
        "highly qualified",
    ],
}

# --------------------------------------------------------------------------
# Libellés français
# --------------------------------------------------------------------------

COLUMN_LABELS_FR = {
    "age": "Âge",
    "credit_amount": "Montant du crédit",
    "duration": "Durée du crédit",
    "installment_rate": "Taux d'endettement",
    "present_residence": "Ancienneté au domicile",
    "existing_credits": "Crédits en cours",
    "num_dependents": "Personnes à charge",
    "checking_account": "Compte courant",
    "credit_history": "Historique de remboursement",
    "purpose": "Objet du crédit",
    "saving_accounts": "Épargne disponible",
    "employment_years": "Ancienneté professionnelle",
    "housing": "Logement",
    "job": "Qualification professionnelle",
    "default": "Défaut de paiement",
}

VALUE_LABELS_FR = {
    "checking_account": {
        "< 0": "Solde négatif",
        "0 - 200": "Faible (0 à 200)",
        ">= 200": "Confortable (200 et plus)",
        "no checking": "Pas de compte courant",
    },
    "credit_history": {
        "no credits taken": "Aucun crédit antérieur",
        "all paid duly": "Tous les crédits remboursés",
        "existing paid duly": "Crédits en cours, remboursés à date",
        "past delay": "Retard de paiement passé",
        "critical account": "Incident de paiement / compte critique",
    },
    "purpose": {
        "car": "Achat d'un véhicule",
        "furniture/equipment": "Meubles / équipement",
        "radio/TV": "Radio / télévision",
        "domestic appliances": "Électroménager",
        "repairs": "Réparations / rénovation",
        "education": "Éducation / formation",
        "vacation": "Vacances",
        "retraining": "Reconversion professionnelle",
        "business": "Projet professionnel",
        "others": "Autre",
    },
    "saving_accounts": {
        "< 100": "Moins de 100",
        "100 - 500": "100 à 500",
        "500 - 1000": "500 à 1000",
        ">= 1000": "1000 et plus",
        "unknown": "Inconnue / pas d'épargne",
    },
    "employment_years": {
        "unemployed": "Sans emploi",
        "< 1": "Moins d'un an",
        "1 - 4": "1 à 4 ans",
        "4 - 7": "4 à 7 ans",
        ">= 7": "7 ans et plus",
    },
    "housing": {
        "own": "Propriétaire",
        "rent": "Locataire",
        "free": "Logé à titre gratuit",
    },
    "job": {
        "unskilled non-resident": "Non qualifié (travailleur non résident)",
        "unskilled resident": "Non qualifié (résident)",
        "skilled": "Qualifié / employé",
        "highly qualified": "Cadre / hautement qualifié",
    },
}

# --------------------------------------------------------------------------
# Champs du formulaire
# --------------------------------------------------------------------------

NUMERIC_FIELDS = {
    "age": {
        "label": "Âge",
        "min": 18,
        "max": 80,
        "default": 35,
        "step": 1,
        "help": "Âge du demandeur au moment de la demande.",
    },
    "credit_amount": {
        "label": "Montant du crédit demandé",
        "min": 250,
        "max": 25000,
        "default": 5000,
        "step": 250,
        "help": "Montant en euros (ou unité monétaire du jeu de données).",
    },
    "duration": {
        "label": "Durée du crédit (mois)",
        "min": 4,
        "max": 72,
        "default": 24,
        "step": 1,
        "help": "Durée de remboursement demandée.",
    },
    "installment_rate": {
        "label": "Taux d'endettement (1 = faible, 4 = élevé)",
        "min": 1,
        "max": 4,
        "default": 2,
        "step": 1,
        "help": "Part du revenu disponible consacrée aux mensualités.",
    },
    "present_residence": {
        "label": "Ancienneté au domicile (1 = moins d'1 an, 4 = 7 ans et plus)",
        "min": 1,
        "max": 4,
        "default": 2,
        "step": 1,
        "help": "Stabilité résidentielle du demandeur.",
    },
    "existing_credits": {
        "label": "Crédits déjà en cours dans l'établissement",
        "min": 1,
        "max": 4,
        "default": 1,
        "step": 1,
        "help": "Nombre de crédits en cours dans la banque.",
    },
    "num_dependents": {
        "label": "Personnes à charge (1 = aucune, 2 = une ou plus)",
        "min": 1,
        "max": 2,
        "default": 1,
        "step": 1,
        "help": "Personnes dépendant financièrement du demandeur.",
    },
}

# --------------------------------------------------------------------------
# Classes de risque et décision
# --------------------------------------------------------------------------

# (borne inférieure, borne supérieure, note, libellé, couleur, décision)
RISK_BANDS = [
    (0.00, 0.05, "A", "Très faible", "#1a7f37", "Accorder"),
    (0.05, 0.15, "B", "Faible", "#2da44e", "Accorder"),
    (0.15, 0.30, "C", "Modéré", "#bf8700", "Accorder avec garanties"),
    (0.30, 0.50, "D", "Élevé", "#d1560a", "Analyse manuelle"),
    (0.50, 1.01, "E", "Très élevé", "#cf222e", "Refuser"),
]


def risk_band(probability: float):
    """Retourne le tuple de classe de risque correspondant à une probabilité."""
    for band in RISK_BANDS:
        if band[0] <= probability < band[1]:
            return band
    return RISK_BANDS[-1]


def value_label(column: str, value: str) -> str:
    """Libellé français d'une valeur catégorielle (repli sur la valeur brute)."""
    return VALUE_LABELS_FR.get(column, {}).get(value, str(value))


def column_label(column: str) -> str:
    """Libellé français d'une variable."""
    return COLUMN_LABELS_FR.get(column, column)
