"""Variables pour les modèles classiques et fenêtres de séquences pour PyTorch."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    """Pente de la régression linéaire d'une fenêtre (unité : par cycle)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n) - (n - 1) / 2
    return float(np.dot(x, values - values.mean()) / np.dot(x, x))


def rolling_features(df: pd.DataFrame, sensors: list[str], window: int = 10) -> pd.DataFrame:
    """Moyenne, écart-type et pente mobiles sur les `window` derniers cycles,
    calculés moteur par moteur (aucune information ne passe d'un moteur à l'autre)."""
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")[sensors]
    roll = g.rolling(window, min_periods=1)
    mean = roll.mean().reset_index(level=0, drop=True).add_suffix("_mean")
    std = roll.std().reset_index(level=0, drop=True).fillna(0).add_suffix("_std")
    # Pente = cov(cycle, capteur) / var(cycle) sur la fenêtre, calculée de façon vectorisée.
    t = df["cycle"].astype(float)
    tx = df[sensors].mul(t, axis=0)
    tmp = pd.concat([df[["unit"]], tx.add_suffix("_tx"), t.rename("t"), (t**2).rename("t2")], axis=1)
    r = tmp.groupby("unit").rolling(window, min_periods=2).mean().reset_index(level=0, drop=True)
    var_t = (r["t2"] - r["t"] ** 2).replace(0, np.nan)
    slope = pd.DataFrame(
        {s + "_slope": (r[s + "_tx"] - r["t"] * mean[s + "_mean"]) / var_t for s in sensors}
    ).fillna(0)
    return pd.concat([df, mean, std, slope], axis=1)


def feature_columns(sensors: list[str]) -> list[str]:
    cols = list(sensors)
    for suffix in ("_mean", "_std", "_slope"):
        cols += [s + suffix for s in sensors]
    return cols


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par moteur : son dernier cycle observé (pour évaluer sur le test)."""
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).set_index("unit")


def _pad(arr: np.ndarray, length: int) -> np.ndarray:
    """Complète une série trop courte en répétant sa première ligne."""
    if len(arr) >= length:
        return arr
    pad = np.repeat(arr[:1], length - len(arr), axis=0)
    return np.vstack([pad, arr])


def make_sequences(df: pd.DataFrame, sensors: list[str], window: int = 30, target: str = "RUL"):
    """Fenêtres glissantes de longueur `window` pour l'entraînement.

    Retourne X de forme (n, window, n_capteurs), y de forme (n,) et le numéro
    de moteur de chaque fenêtre (pour la validation par groupe).
    """
    xs, ys, units = [], [], []
    for unit, grp in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = _pad(grp[sensors].to_numpy(np.float32), window)
        rul = grp[target].to_numpy(np.float32)
        rul = np.concatenate([np.full(len(values) - len(rul), rul[0], np.float32), rul])
        for end in range(window, len(values) + 1):
            xs.append(values[end - window : end])
            ys.append(rul[end - 1])
            units.append(unit)
    return np.stack(xs), np.array(ys, np.float32), np.array(units)


def last_sequences(df: pd.DataFrame, sensors: list[str], window: int = 30):
    """Dernière fenêtre de chaque moteur (jeu de test). Retourne X et les numéros de moteur."""
    xs, units = [], []
    for unit, grp in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = _pad(grp[sensors].to_numpy(np.float32), window)
        xs.append(values[-window:])
        units.append(unit)
    return np.stack(xs), np.array(units)
