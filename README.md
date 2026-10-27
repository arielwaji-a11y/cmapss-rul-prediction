# Prédiction de la durée de vie restante de turboréacteurs (NASA C-MAPSS)

> Projet de fin de formation — Certificat en science des données, TÉLUQ  
> Auteur : Ariel Manfouo Waji

À partir des capteurs d'un turboréacteur, prédire combien de cycles de vol il lui reste avant défaillance (**RUL**, *Remaining Useful Life*), puis traduire cette prédiction en **décision de maintenance chiffrée**.

## Résultats clés

<!-- À remplir après l'exécution des notebooks -->

| Modèle | RMSE FD001 | Score NASA FD001 | RMSE FD004 | Score NASA FD004 |
|---|---|---|---|---|
| Référence naïve (moyenne) | – | – | – | – |
| Régression linéaire | – | – | – | – |
| XGBoost | – | – | – | – |
| LSTM (PyTorch) | – | – | – | – |
| CNN 1D (PyTorch) | – | – | – | – |

**Économie estimée** de la maintenance prédictive par rapport au meilleur intervalle fixe : **[X] $ par moteur** (hypothèses de coûts documentées dans le notebook 05, avec analyse de sensibilité).

![RUL prédite vs réelle](reports/figures/rul_predite_vs_reelle_FD001.png)

## Démarche

1. **Exploration** : durée de vie, évolution des capteurs, corrélations, retrait des capteurs constants justifié par leur variance.
2. **Cible RUL plafonnée** (125 cycles, choix testé entre 120 et 130).
3. **Préparation** : normalisation apprise sur l'entraînement seulement ; pour FD002/FD004, regroupement des 6 conditions de vol (k-moyennes) et normalisation par condition.
4. **Validation sans fuite** : séparation **par moteur** (`GroupKFold`) ; le jeu de test officiel n'est utilisé qu'une fois.
5. **Modèles** : référence naïve, régression linéaire, forêt aléatoire, XGBoost, puis **LSTM et CNN 1D en PyTorch** sur fenêtres de 30 cycles.
6. **Interprétation** : valeurs SHAP, erreur selon la RUL réelle, intervalle d'incertitude par MC Dropout.
7. **Analyse de rentabilisation** : comparaison de trois stratégies (intervalle fixe, prédictive, idéale) et choix du seuil qui minimise le coût total.

### Métriques

Avec *d* = RUL prédite − RUL réelle : la **RMSE**, et le **score asymétrique de la NASA**, qui pénalise davantage une prédiction trop tardive (*e*^(d/10) − 1) qu'une prédiction trop précoce (*e*^(−d/13) − 1) — une panne imprévue coûte plus cher qu'un remplacement anticipé.

## Structure du dépôt

```
cmapss-rul-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── download_data.py        # téléchargement des données NASA
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preparation.ipynb
│   ├── 03_modeles_classiques.ipynb
│   ├── 04_modeles_pytorch.ipynb
│   └── 05_analyse_rentabilisation.ipynb
├── src/
│   ├── data.py                 # chargement, RUL plafonnée, conditions de vol, normalisation
│   ├── features.py             # variables mobiles, fenêtres de séquences
│   ├── models.py               # LSTM et CNN 1D en PyTorch, entraînement, MC Dropout
│   ├── evaluate.py             # RMSE, score NASA, erreur par tranche de RUL
│   └── cost_model.py           # simulation des stratégies de maintenance
└── reports/
    ├── figures/
    └── rapport_final.pdf
```

## Exécution

> **Python 3.10 à 3.12 requis.** PyTorch n'est pas toujours disponible pour les versions plus récentes de Python : avec une version trop récente, `pip install` échoue avec « No matching distribution found for torch ». Sous Windows, si plusieurs versions sont installées, crée l'environnement avec `py -3.12 -m venv .venv`.
>
> Sous Windows PowerShell, si l'activation est bloquée (« l'exécution de scripts est désactivée »), lance d'abord `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (valable pour la fenêtre en cours seulement).

```bash
git clone https://github.com/arielwaji-a11y/cmapss-rul-prediction.git
cd cmapss-rul-prediction
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python data/download_data.py
jupyter notebook notebooks/
```

Exécuter les notebooks dans l'ordre. Changer `SUBSET = "FD001"` en `"FD004"` en tête de chaque notebook pour le sous-ensemble difficile.  
L'entraînement des modèles PyTorch peut se faire sur **Databricks** ; les expériences sont suivies avec **MLflow**.

## Données

[CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data), NASA. Quatre sous-ensembles de moteurs simulés suivis jusqu'à la panne ; ce projet traite FD001 (1 condition, 1 mode de défaillance) et FD004 (6 conditions, 2 modes).

> A. Saxena, K. Goebel, D. Simon et N. Eklund, « Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation », *PHM08*, Denver, 2008.

## Limites

Les données sont simulées, propres et sans valeurs manquantes. Avec de vraies données d'avions connectés, il faudrait gérer des capteurs manquants, des flottes hétérogènes et un historique de maintenance incomplet. Les montants de l'analyse de rentabilisation sont des hypothèses, pas des coûts réels.
