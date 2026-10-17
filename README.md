# 🏦 CréditScore — Évaluation intelligente du risque de crédit

Application **Streamlit** qui évalue le risque de défaut d'un demandeur de crédit :
elle calcule sa probabilité de défaut, lui attribue un **score de 300 à 850**, le
classe par niveau de risque et **explique chaque décision variable par variable**.

Le moteur est une **régression logistique** — le modèle de référence des
établissements bancaires pour un score de production : stable, calibré et
entièrement auditable.

---

## Sommaire

- [Aperçu](#aperçu)
- [Démarrage rapide](#démarrage-rapide)
- [L'application en détail](#lapplication-en-détail)
- [Comment fonctionne le modèle](#comment-fonctionne-le-modèle)
- [Données d'entraînement](#données-dentraînement)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Conformité et bonnes pratiques](#conformité-et-bonnes-pratiques)
- [Dépannage](#dépannage)

---

## Aperçu

| Fonction | Description |
|---|---|
| **Score d'un dossier** | Un formulaire en 4 sections (14 variables) → probabilité de défaut, score /850, classe de risque A→E et décision. |
| **Explication** | Décomposition additive du score : chaque variable affiche son impact en points, en vert (réduit le risque) ou rouge (l'augmente). |
| **Seuil ajustable** | Le seuil de décision est un curseur : on voit en direct la décision basculer de « accorder » à « refuser ». |
| **Performance du modèle** | AUC, exactitude, rappel, précision, indice de Brier, matrice de confusion. |
| **Audit** | Table des coefficients et des *odds ratios* du modèle, triée par influence. |
| **Export** | Téléchargement de la fiche d'évaluation du dossier au format CSV. |

---

## Démarrage rapide

### 1. Prérequis : Python

L'application nécessite **Python 3.10 ou plus récent**.

> ⚠️ Sur cette machine, Python n'était pas installé (« python » ouvrait le
> Microsoft Store). Python **3.12.10** a été installé dans
> `%LOCALAPPDATA%\Programs\Python\Python312` et un environnement virtuel a été
> créé dans `.venv/`. Si vous repartez d'une machine vierge :
>
> ```powershell
> winget install --id Python.Python.3.12 -e
> ```
> Si l'installation via `winget` échoue (invite UAC), téléchargez l'installeur sur
> <https://www.python.org/downloads/windows/> et cochez **« Add python.exe to PATH »**.

### 2. Installer les dépendances

```powershell
cd C:\Users\user\Desktop\My_App
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Lancer l'application

```powershell
.\run.ps1
```

Le lanceur crée l'environnement virtuel s'il manque, installe les dépendances,
puis démarre le serveur sur <http://127.0.0.1:8501>.

<details>
<summary>Pourquoi un script plutôt que <code>streamlit run app.py</code> ?</summary>

`run.ps1` lie le serveur explicitement à IPv4. Sur Windows, la double pile IPv6
utilisée par défaut (`:::8501`) peut faire tomber la boucle `asyncio` sur
`WinError 64 — Le nom réseau spécifié n'est plus disponible` : le processus
reste vivant mais n'accepte plus aucune connexion, et la page affiche
`Failed to fetch dynamically imported module`.

Le script local fait donc :

```powershell
streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

L'adresse n'est volontairement **pas** fixée dans `.streamlit/config.toml` :
ce fichier étant versionné, il s'appliquerait aussi au déploiement Streamlit
Cloud, où la plateforme impose sa propre liaison réseau.

</details>

> Le modèle s'entraîne automatiquement au premier lancement (environ 2 secondes)
> et est mis en cache dans `models/credit_model.joblib`.

### 4. (Optionnel) Entraîner le modèle en ligne de commande

```powershell
python train.py                       # portefeuille synthétique (par défaut)
python train.py --source german       # German Credit, 1 000 dossiers réels
python train.py --source both         # les deux combinés
python train.py --source synthetic --rows 20000 --seed 7
```

---

## L'application en détail

### Barre latérale

- **Portefeuille d'entraînement** : synthétique, German Credit, ou les deux.
- **Dossiers synthétiques** : taille du portefeuille généré (désactivé pour German Credit).
- **Seuil de décision** : au-delà de cette probabilité de défaut, le dossier est refusé.
- **Réentraîner le modèle** : vide les caches et relance l'entraînement.
- **Le modèle en bref** : AUC, exactitude, rappel, précision, matrice de confusion
  et table des coefficients (repliable).

### Formulaire (à gauche)

| Section | Variables |
|---|---|
| 1 · Profil du demandeur | âge, ancienneté professionnelle, qualification |
| 2 · Situation financière | compte courant, épargne, taux d'endettement |
| 3 · Crédit demandé | montant, durée, objet |
| 4 · Historique et stabilité | historique de remboursement, logement, ancienneté au domicile, crédits en cours, personnes à charge |

### Résultat (à droite)

1. **Probabilité de défaut** en pourcentage.
2. **Jauge de score** de 300 à 850, colorée par bande de risque.
3. **Bandeau de décision** : score, classe (A→E) et décision
   (*Accorder*, *Accorder avec garanties*, *Analyse manuelle*, *Refuser*).
4. **Diagramme des contributions** : le « pourquoi » de la décision.
5. **Détail du calcul** (repliable) : la table complète des impacts et la
   vérification que *score de base + somme des impacts = score final*.

Le dernier dossier évalué reste affiché lorsque vous bougez le seuil :
la décision se recalcule, le score non.

### Classes de risque

| Classe | Probabilité de défaut | Score | Lecture |
|---|---|---|---|
| **A** | < 5 % | ≥ 685 | Risque très faible |
| **B** | 5 % – 15 % | 650 – 685 | Risque faible |
| **C** | 15 % – 30 % | 624 – 650 | Risque modéré |
| **D** | 30 % – 50 % | 600 – 624 | Risque élevé |
| **E** | ≥ 50 % | < 600 | Risque très élevé |

---

## Comment fonctionne le modèle

### Pipeline

```
variables numériques  →  StandardScaler      ┐
                                             ├→  Régression logistique  →  probabilité de défaut
variables catégorielles → OneHotEncoder      ┘
```

- **Variables numériques** : 7 (âge, montant, durée, taux d'endettement,
  ancienneté au domicile, crédits en cours, personnes à charge).
- **Variables catégorielles** : 7 (compte courant, historique, objet, épargne,
  ancienneté professionnelle, logement, qualification).
- **Ordre de grandeur** : environ 45 indicatrices après encodage.

### Du log-odds au score

Le score est dérivé des *log-odds* par la transformation classique
*points-to-double-the-odds* :

```
score = 600 + 20 × log₂( chances de remboursement )
```

Autrement dit : **600 points** correspondent à une probabilité de défaut de 50 %,
et **chaque doublement des chances de remboursement ajoute 20 points**. Le score
est borné à l'échelle [300, 850].

### L'explication est exacte, pas approximative

Le score est une fonction affine des variables transformées, donc la
décomposition est **rigoureusement additive** :

```
score = score_de_base  +  Σ (impact de chaque variable)
```

Chaque impact vaut `−K × coefficient × valeur_transformée`, avec
`K = 20 / ln(2) ≈ 28,85`. La somme affichée dans « Détail du calcul » redonne
exactement le score : il n'y a aucune approximation, contrairement à un modèle
à base d'arbres qui exigerait SHAP.

Un impact **positif** (vert) signifie que la variable *améliore* le score.

### Performance obtenue

| Portefeuille | Dossiers | AUC | Exactitude | Brier |
|---|---|---|---|---|
| Synthétique | 6 000 | **0,824** | 78,6 % | 0,147 |
| German Credit | 1 000 | **0,772** | 73,6 % | 0,171 |
| Les deux | 7 000 | **0,801** | 77,2 % | 0,155 |

Ces niveaux sont représentatifs d'un bon scorecard de crédit en production.

---

## Données d'entraînement

### Portefeuille synthétique (par défaut)

6 000 dossiers générés localement par `src/data.py`, sans aucune donnée client
réelle et **sans accès Internet**.

La génération est **auto-calibrée** :

1. tirage des 14 variables selon des distributions réalistes (montants
   log-normaux, durées, profils de compte…) ;
2. calcul d'un signal de risque pondéré — un incident de paiement pèse plus
   lourd qu'un motif de crédit à risque ;
3. recentrage/réduction du signal, pour que sa dispersion soit un paramètre
   explicite (`SIGNAL_STRENGTH`) ;
4. **recherche dichotomique de l'intercept** pour atteindre exactement le taux
   de défaut visé (`DEFAULT_DEFAULT_RATE = 0,29`), puis bruit résiduel sur
   l'échelle log-odds (`LABEL_NOISE`).

```python
from src.data import generate_synthetic

portfolio = generate_synthetic(n_rows=8000, seed=7, target_default_rate=0.20)
```

Les variables ont donc un effet réel, ordonné et interprétable : le modèle
apprend une relation qui existe vraiment.

### German Credit (UCI / OpenML)

1 000 dossiers réels (Hofmann, 1994), téléchargés via
`sklearn.datasets.fetch_openml("credit-g")` puis **normalisés** vers le schéma
canonique de l'application : renommage des colonnes et traduction de chaque
modalité d'origine (`'no known savings'` → `unknown`, `'unemp/unskilled non res'`
→ `unskilled non-resident`, `'for free'` → `free`…).

Les 1 000 dossiers sont conservés (taux de défaut de 30 %), ce qui est vérifié
par les tests.

---

## Structure du projet

```
My_App/
├── app.py                  # application Streamlit
├── train.py                # entraînement en ligne de commande
├── run.ps1                 # lanceur local (liaison IPv4)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml         # thème et télémétrie (sans adresse réseau)
├── models/
│   └── credit_model.joblib # modèle entraîné (regénéré automatiquement)
├── src/
│   ├── schema.py           # variables, modalités, libellés FR, classes de risque
│   ├── data.py             # portefeuille synthétique + German Credit
│   └── model.py            # entraînement, scoring, persistance, explicabilité
└── tests/
    ├── conftest.py
    ├── test_model.py       # données, modèle, échelle de score, explicabilité
    └── test_app.py         # parcours complets de l'interface
```

---

## Tests

```powershell
pip install pytest
pytest -q
```

Résultat attendu :

```
22 passed
```

La suite couvre :

- **Données** : respect du schéma, calibration du taux de défaut, reproductibilité,
  couverture complète du mapping German Credit.
- **Modèle** : AUC, probabilités bornées, cohérence métier (un dossier sain
  score mieux qu'un dossier risqué).
- **Échelle** : monotonie et bornes du score, continuité des bandes de risque.
- **Explicabilité** : additivité exacte sur 60 dossiers, couverture des 14
  variables, signe des facteurs défavorables.
- **Interface** : démarrage, dossier risqué refusé, dossier sain accepté,
  cohérence de la décomposition affichée, application du seuil, bascule entre
  portefeuilles, réentraînement, bornes extrêmes du formulaire.

---

## Conformité et bonnes pratiques

- **Aucune variable sensible** n'est utilisée comme variable explicative : ni le
  sexe, ni l'origine, ni la situation familiale (directive 2004/113/CE, RGPD
  art. 22, lignes directrices ACPR/EBA). L'âge reste utilisé, comme dans la
  pratique bancaire, mais il conviendrait de mesurer son impact
  différencié lors d'un déploiement réel.
- **Explicabilité individuelle** : le demandeur peut se voir communiquer les
  facteurs de la décision.
- **Traçabilité** : la table des coefficients est consultable et le modèle est
  daté et versionné sur disque.
- ⚠️ **Outil d'aide à la décision**, pas un décideur automatique : le score ne
  se substitue pas à l'analyse d'un conseiller et doit pouvoir être contesté.

---

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `python` ouvre le Microsoft Store | Python non installé, seul l'alias Store existe | Installer Python (voir [Démarrage rapide](#1-prérequis--python)) |
| `streamlit : commande introuvable` | Environnement virtuel non activé | `.\.venv\Scripts\Activate.ps1` |
| `ModuleNotFoundError: No module named 'src'` | Lancement depuis un autre dossier | Se placer dans `My_App/` avant `streamlit run app.py` |
| Erreur au choix « German Credit » | Pas d'accès Internet au premier lancement | Se connecter, ou revenir au portefeuille synthétique |
| `ScriptRunContext` dans la console | Avertissement bénin en mode test | Sans gravité, ignorable |
| `WinError 64 — Le nom réseau spécifié n'est plus disponible`, puis page morte / `Failed to fetch dynamically imported module` | Bug `asyncio` intermittent de Windows avec la double pile IPv6 par défaut (`:::8501`) : le serveur cesse d'accepter les connexions | Lancez `.\run.ps1` (liaison IPv4). Si le serveur tombe déjà, arrêtez-le et relancez |
| PowerShell bloque `Activate.ps1` | Politique d'exécution | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

### Afficher l'application sur votre réseau local

`run.ps1` n'écoute que sur `127.0.0.1` : rien n'est exposé à l'extérieur. Pour la
montrer à un collègue, lancez :

```powershell
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

puis communiquez l'adresse `http://<votre-IP-locale>:8501`.

---

## Déployer sur Streamlit Community Cloud

1. Poussez le dépôt sur GitHub (voir ci-dessous).
2. Rendez-vous sur <https://share.streamlit.io> → **New app**.
3. Renseignez le dépôt (`abderrahmanfakhech/My_App`), la branche `main` et le
   fichier principal `app.py`.
4. Cliquez sur **Deploy**.

Aucune configuration supplémentaire n'est nécessaire :

- les dépendances sont lues dans `requirements.txt` ;
- le modèle s'entraîne au premier chargement, puis est mis en cache ;
- `.streamlit/config.toml` ne fixe ni adresse ni port, donc la plateforme reste
  libre d'imposer sa propre liaison réseau.

> Le portefeuille **German Credit** télécharge le jeu de données au premier
> lancement via `fetch_openml` : l'accès Internet est disponible sur Streamlit
> Cloud, mais prévoyez quelques secondes pour la première ouverture. Le
> portefeuille synthétique, lui, ne nécessite aucun réseau.

---

## Pousser le dépôt sur GitHub

Le dépôt distant est `https://github.com/abderrahmanfakhech/My_App.git`. Depuis
`My_App/` :

```powershell
git init
git add .
git commit -m "Application d'évaluation du risque de crédit"
git remote add origin https://github.com/abderrahmanfakhech/My_App.git
git branch -M main
git push -u origin main
```

Le `.gitignore` exclut l'environnement virtuel, les caches Python et le modèle
entraîné — le dépôt ne contient que le code source.
