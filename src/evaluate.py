"""Métriques d'évaluation : RMSE et score asymétrique de la NASA (PHM08).

Avec d = RUL prédite − RUL réelle :
    s = Σ exp(−d/13) − 1   si d < 0  (prédiction trop précoce)
        Σ exp( d/10) − 1   si d ≥ 0  (prédiction trop tardive, plus pénalisée)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred, name: str = "modèle") -> dict:
    return {"modèle": name, "RMSE": rmse(y_true, y_pred), "score NASA": nasa_score(y_true, y_pred)}


def error_by_rul_bucket(y_true, y_pred, bins=(0, 25, 50, 75, 100, 125, np.inf)) -> pd.DataFrame:
    """Erreur selon la RUL réelle : le modèle est-il fiable quand la panne approche ?"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    df = pd.DataFrame({"réel": y_true, "erreur": y_pred - y_true})
    df["tranche RUL réelle"] = pd.cut(df["réel"], bins=bins, right=False)
    return (
        df.groupby("tranche RUL réelle", observed=True)["erreur"]
        .agg(n="size", biais="mean", RMSE=lambda e: float(np.sqrt(np.mean(e**2))))
        .round(2)
    )


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("modèle").round(2)
