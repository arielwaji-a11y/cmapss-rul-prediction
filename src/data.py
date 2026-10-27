"""Chargement des données C-MAPSS, construction de la cible RUL et normalisation.

Chaque fichier brut (train_FD00x.txt, test_FD00x.txt) contient, sans en-tête et
séparées par des espaces : numéro du moteur, numéro du cycle, 3 réglages
opérationnels et 21 capteurs, soit 26 colonnes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
COLUMNS = ID_COLS + SETTING_COLS + SENSOR_COLS  # 26 colonnes

SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Certains fichiers ont des espaces en fin de ligne : on retire les colonnes vides.
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name} : {df.shape[1]} colonnes trouvées, {len(COLUMNS)} attendues."
        )
    df.columns = COLUMNS
    df[ID_COLS] = df[ID_COLS].astype(int)
    return df


def load_subset(subset: str = "FD001", data_dir: Path | str = DATA_DIR):
    """Retourne (train, test, rul_test).

    rul_test est une Series indexée par numéro de moteur : la RUL réelle
    au dernier cycle observé de chaque moteur de test.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Sous-ensemble inconnu : {subset}")
    data_dir = Path(data_dir)
    train = _read_raw(data_dir / f"train_{subset}.txt")
    test = _read_raw(data_dir / f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", sep=r"\s+", header=None)
    rul_test = pd.Series(rul.iloc[:, 0].to_numpy(), index=np.arange(1, len(rul) + 1), name="RUL")
    rul_test.index.name = "unit"
    return train, test, rul_test


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Ajoute la colonne RUL = dernier cycle du moteur − cycle courant, plafonnée à `cap`."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = last - df["cycle"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(test: pd.DataFrame, rul_test: pd.Series, cap: int | None = 125) -> pd.DataFrame:
    """RUL de chaque ligne de test, reconstruite à partir de la RUL au dernier cycle."""
    test = test.copy()
    last = test.groupby("unit")["cycle"].transform("max")
    test["RUL"] = test["unit"].map(rul_test) + (last - test["cycle"])
    if cap is not None:
        test["RUL"] = test["RUL"].clip(upper=cap)
    return test


def low_variance_sensors(df: pd.DataFrame, threshold: float = 1e-4, cols=SENSOR_COLS) -> list[str]:
    """Capteurs (quasi) constants : écart-type relatif sous `threshold`.

    Le choix est justifié par la variance observée, pas par une liste toute faite.
    """
    std = df[cols].std()
    mean = df[cols].mean().abs().replace(0, 1)
    rel = std / mean
    return rel[rel < threshold].index.tolist()


def fit_operating_conditions(train: pd.DataFrame, n_conditions: int = 6, seed: int = 42) -> KMeans:
    """Regroupe les conditions de vol (FD002, FD004) à partir des 3 réglages."""
    km = KMeans(n_clusters=n_conditions, n_init=10, random_state=seed)
    km.fit(train[SETTING_COLS].round(1))
    return km


def assign_conditions(df: pd.DataFrame, km: KMeans | None) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = 0 if km is None else km.predict(df[SETTING_COLS].round(1))
    return df


class ConditionScaler:
    """Normalisation des capteurs apprise sur l'entraînement seulement,
    une échelle par condition de vol (une seule pour FD001/FD003)."""

    def __init__(self, sensors: list[str], method: str = "zscore"):
        self.sensors = list(sensors)
        self.method = method
        self.scalers_: dict[int, object] = {}

    def _new(self):
        return StandardScaler() if self.method == "zscore" else MinMaxScaler()

    def fit(self, df: pd.DataFrame) -> "ConditionScaler":
        for cond, grp in df.groupby("condition"):
            self.scalers_[cond] = self._new().fit(grp[self.sensors])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        out = np.array(df[self.sensors], dtype=float, copy=True)
        for cond, idx in df.groupby("condition").indices.items():
            out[idx] = self.scalers_[cond].transform(df.iloc[idx][self.sensors])
        df[self.sensors] = out
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def prepare(subset: str = "FD001", cap: int = 125, data_dir: Path | str = DATA_DIR):
    """Chaîne complète : chargement, RUL, conditions, retrait des capteurs constants,
    normalisation. Retourne (train, test, rul_test, sensors)."""
    train, test, rul_test = load_subset(subset, data_dir)
    train = add_rul(train, cap)
    n_cond = 6 if subset in ("FD002", "FD004") else 1
    km = fit_operating_conditions(train, n_cond) if n_cond > 1 else None
    train, test = assign_conditions(train, km), assign_conditions(test, km)

    # Variance mesurée par condition : un capteur constant à l'intérieur de chaque condition
    # n'apporte rien même s'il varie entre conditions.
    dropped = set(SENSOR_COLS)
    for _, grp in train.groupby("condition"):
        dropped &= set(low_variance_sensors(grp))
    sensors = [c for c in SENSOR_COLS if c not in dropped]

    scaler = ConditionScaler(sensors).fit(train)
    return scaler.transform(train), scaler.transform(test), rul_test, sensors
