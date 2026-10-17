"""Entraîne le modèle de scoring et l'enregistre sur disque.

Exemples :
    python train.py
    python train.py --source both
    python train.py --source german --rows 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data import DEFAULT_SYNTHETIC_ROWS, SOURCES, build_dataset, describe_dataset
from src.model import MODEL_FILENAME, save_model, train_model

MODEL_DIR = Path(__file__).parent / "models"

# La console Windows utilise cp1252 par défaut, qui ne sait pas afficher
# certains caractères (flèches, accents composés). On force l'UTF-8.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement du score de crédit.")
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default="synthetic",
        help="Portefeuille d'entraînement (défaut : synthetic).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_SYNTHETIC_ROWS,
        help="Nombre de dossiers synthétiques à générer.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    parser.add_argument(
        "--out",
        type=Path,
        default=MODEL_DIR / MODEL_FILENAME,
        help="Chemin de sortie du modèle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"-> Construction du portefeuille '{args.source}'...")
    dataset = build_dataset(source=args.source, n_rows=args.rows, seed=args.seed)
    stats = describe_dataset(dataset)
    print(
        f"   {stats['n_rows']} dossiers, {stats['n_defaults']} defauts "
        f"({stats['default_rate']:.1%})"
    )

    print("-> Entrainement de la regression logistique...")
    bundle = train_model(dataset, source=args.source, seed=args.seed)

    metrics = bundle["metadata"]["metrics"]
    print(f"  AUC (ROC)   : {metrics['auc']:.4f}")
    print(f"  Exactitude  : {metrics['accuracy']:.1%}")
    print(f"  Rappel      : {metrics['recall']:.1%}")
    print(f"  Précision   : {metrics['precision']:.1%}")
    print(f"  Brier       : {metrics['brier']:.4f}")

    path = save_model(bundle, args.out)
    print(f"[OK] Modele enregistre : {path}")


if __name__ == "__main__":
    main()
